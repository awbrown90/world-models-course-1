import os
import argparse
import colorsys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import cv2
import pygame

from vector_quantize_pytorch import VectorQuantize
from game_library import ENV_REGISTRY

PALETTE_RGB = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)


def make_token_palette(num_embeddings: int) -> np.ndarray:
    """
    Create a fixed RGB color for each discrete code index.
    Same index -> same color every frame.
    """
    palette = np.zeros((num_embeddings, 3), dtype=np.uint8)
    for i in range(num_embeddings):
        h = i / max(1, num_embeddings)
        s = 0.9
        v = 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        palette[i] = [int(255 * r), int(255 * g), int(255 * b)]
    return palette


def render_token_map(tokens_hw: np.ndarray, token_palette: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Convert a [H_t, W_t] token index map into a color image and upscale it.
    """
    token_rgb = token_palette[tokens_hw]  # [H_t, W_t, 3]
    vis = cv2.resize(token_rgb, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    # Draw grid lines so the discrete latent cells are easy to see
    h_t, w_t = tokens_hw.shape
    cell_w = max(1, out_w // w_t)
    cell_h = max(1, out_h // h_t)

    grid_color = (30, 30, 30)
    for x in range(0, out_w, cell_w):
        cv2.line(vis, (x, 0), (x, out_h - 1), grid_color, 1)
    for y in range(0, out_h, cell_h):
        cv2.line(vis, (0, y), (out_w - 1, y), grid_color, 1)

    return vis


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=16, num_embeddings=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

        self.vq = VectorQuantize(
            dim=embedding_dim,
            codebook_size=num_embeddings,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        b, c, h_t, w_t = z.shape
        z_flat = z.view(b, c, -1).transpose(1, 2)  # [B, H_t*W_t, C]
        quantized, indices, vq_loss = self.vq(z_flat)
        quantized = quantized.transpose(1, 2).view(b, c, h_t, w_t)
        recon = self.decoder(quantized)
        return recon, vq_loss, None

    @torch.no_grad()
    def get_tokens(self, x):
        z = self.encoder(x)
        b, c, h_t, w_t = z.shape
        z_flat = z.view(b, c, -1).transpose(1, 2)
        _, indices, _ = self.vq(z_flat)
        return indices.view(b, h_t, w_t)

    @torch.no_grad()
    def decode_tokens(self, indices):
        if hasattr(self.vq, "codebook"):
            cb = self.vq.codebook
        elif hasattr(self.vq, "_codebook") and hasattr(self.vq._codebook, "embed"):
            cb = self.vq._codebook.embed
        else:
            cb = self.vq._codebook.codebook

        quantized = cb[indices]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)


def train_live(env_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = ENV_REGISTRY[env_name]()
    vqvae = VQVAE().to(device)
    optimizer = optim.AdamW(vqvae.parameters(), lr=1e-3)

    replay_buffer = deque(maxlen=2000)
    batch_size = 64

    save_dir = os.path.join("models", "vqvae-models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vqvae_{env_name}_live.pth")

    token_palette = make_token_palette(vqvae.num_embeddings)

    pygame.init()
    pygame.font.init()

    render_w, render_h = 80 * 6, 64 * 6
    header_h = 36
    screen = pygame.display.set_mode((render_w * 3, render_h + header_h))
    pygame.display.set_caption(
        f'Interactive {env_name.capitalize()} VQ-VAE: GT | Recon | Discrete Latent Codes'
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    step = 0
    running = True

    print("\n" + "=" * 50)
    print(env.get_instructions())
    print("CRITICAL: Deflect/Move constantly during the first 500 steps to map tokens!")
    print("Display panels: [GT] [Recon] [Discrete Latent Code Map]")
    print(f"When you close the window, the model will be saved to: {save_path}")
    print("=" * 50 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = env.get_action(keys)

        frame = env.step(action)
        frame_tensor = torch.as_tensor(frame, dtype=torch.float32, device=device).unsqueeze(0)
        replay_buffer.append(frame_tensor.squeeze(0))

        loss_val = 0.0
        if len(replay_buffer) >= batch_size:
            batch_idx = np.random.randint(0, len(replay_buffer), size=batch_size)
            batch = torch.stack([replay_buffer[i] for i in batch_idx]).to(device)

            recon, vq_loss, _ = vqvae(batch)
            recon_loss = F.mse_loss(recon, batch)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()

        if len(replay_buffer) >= batch_size and step % 100 == 0:
            with torch.no_grad():
                tokens = vqvae.get_tokens(batch)
                flat_tokens = tokens.flatten().cpu().numpy()
                unique_tokens, counts = np.unique(flat_tokens, return_counts=True)

                bg_percent = (np.max(counts) / len(flat_tokens)) * 100
                num_used = len(unique_tokens)

                print(
                    f"Step {step} | Loss: {loss_val:.4f} | "
                    f"Codebook Used: {num_used}/{vqvae.num_embeddings} | "
                    f"Background: {bg_percent:.1f}%"
                )

        with torch.no_grad():
            recon_single, _, _ = vqvae(frame_tensor)
            tokens_single = vqvae.get_tokens(frame_tensor)[0].cpu().numpy()

        gt_class = (frame.squeeze() > 0.5).astype(np.int64)
        recon_class = (recon_single.squeeze().cpu().numpy() > 0.5).astype(np.int64)

        gt_img = cv2.resize(
            (PALETTE_RGB[gt_class] * 255).astype(np.uint8),
            (render_w, render_h),
            interpolation=cv2.INTER_NEAREST
        )
        recon_img = cv2.resize(
            (PALETTE_RGB[recon_class] * 255).astype(np.uint8),
            (render_w, render_h),
            interpolation=cv2.INTER_NEAREST
        )
        token_img = render_token_map(tokens_single, token_palette, render_w, render_h)

        combined = np.hstack((gt_img, recon_img, token_img))
        surface = pygame.surfarray.make_surface(np.transpose(combined, (1, 0, 2)))

        screen.fill((0, 0, 0))
        screen.blit(surface, (0, header_h))

        labels = ["GT", "Recon", "Latent Codes"]
        for i, label_text in enumerate(labels):
            label = font.render(label_text, True, (255, 255, 255))
            center_x = i * render_w + render_w // 2
            rect = label.get_rect(center=(center_x, header_h // 2))
            screen.blit(label, rect)

        pygame.display.flip()

        clock.tick(30)
        step += 1

    torch.save(vqvae.state_dict(), save_path)
    print(f"\nSaved final VQ-VAE weights successfully to {save_path}")
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE for a specific arcade environment.")
    parser.add_argument(
        '--env',
        type=str,
        required=True,
        choices=list(ENV_REGISTRY.keys()),
        help="Which environment to load (e.g., pong, platformer)"
    )
    args = parser.parse_args()

    train_live(args.env)