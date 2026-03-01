import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.cuda.amp as amp   # ← mixed precision for speed

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

HEIGHT = 64
WIDTH = 80
BALL_SIZE = 5

# (BouncingBallEnv, ConvTokenizer, SpatioTemporalWorldModel classes unchanged — exactly the same as before)
class BouncingBallEnv:
    def __init__(self):
        self.height = HEIGHT
        self.width = WIDTH
        self.ball_size = BALL_SIZE
        self.reset()

    def reset(self):
        self.x = np.random.randint(0, self.width - self.ball_size + 1)
        self.y = np.random.randint(0, self.height - self.ball_size + 1)
        self.vx = np.random.choice([-1, 1])
        self.vy = np.random.choice([-1, 1])
        return self._get_frame()

    def step(self):
        self.x += self.vx
        self.y += self.vy
        if self.x < 0 or self.x > self.width - self.ball_size:
            self.vx = -self.vx
            self.x = max(0, min(self.width - self.ball_size, self.x))
        if self.y < 0 or self.y > self.height - self.ball_size:
            self.vy = -self.vy
            self.y = max(0, min(self.height - self.ball_size, self.y))
        return self._get_frame()

    def _get_frame(self):
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        frame[self.y:self.y + self.ball_size, self.x:self.x + self.ball_size] = 1.0
        return frame


class ConvTokenizer(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.down_h, self.down_w = HEIGHT // 8, WIDTH // 8
        self.num_patches = self.down_h * self.down_w

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, embed_dim, 4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def patchify(self, x):
        tokens = self.encoder(x)
        return tokens.view(-1, self.num_patches, self.embed_dim)

    def unpatchify(self, x):
        B = x.shape[0]
        x = x.transpose(1, 2).view(B, self.embed_dim, self.down_h, self.down_w)
        return self.decoder(x)


class SpatioTemporalWorldModel(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4, history_len=2):
        super().__init__()
        self.tokenizer = ConvTokenizer(embed_dim)
        self.history_len = history_len
        self.num_patches = self.tokenizer.num_patches
        self.embed_dim = embed_dim

        self.spatial_pos = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, history_len * self.num_patches, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.predictor = nn.Linear(embed_dim, embed_dim)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        patches = self.tokenizer.patchify(x)
        patches = patches.reshape(B, T * self.num_patches, self.embed_dim)
        patches = patches + self.spatial_pos.repeat(1, T, 1) + self.temporal_pos
        out = self.transformer(patches)
        next_tokens = self.predictor(out[:, -self.num_patches:, :])
        return self.tokenizer.unpatchify(next_tokens)


def main():
    device = torch.device('cuda')
    model_path = 'world_model_64x80_fast.pth'

    model = SpatioTemporalWorldModel(history_len=2).to(device)
    scaler = amp.GradScaler()   # ← mixed precision speed boost

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ Loaded saved model from {model_path}")
    else:
        print("Training optimized model on 64×80 rectangle...")
        data = generate_sequences(num_seqs=10000, seq_len=16)   # balanced size

        X, Y = [], []
        for t in range(data.shape[1] - 2):
            X.append(data[:, t:t+2])
            Y.append(data[:, t+2:t+3].squeeze(1))
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)

        loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True,
                            num_workers=2, pin_memory=True)   # faster settings

        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(30):
            model.train()
            loss_avg = 0.0
            for bx, by in tqdm(loader, desc=f"Epoch {epoch}"):
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)

                opt.zero_grad()
                with amp.autocast():   # ← huge speedup on RTX 5080
                    noisy = bx + torch.randn_like(bx) * 0.12
                    pred = model(noisy)
                    loss = criterion(pred, by)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_avg += loss.item()
            print(f"Epoch {epoch} loss: {loss_avg/len(loader):.6f}")
        torch.save(model.state_dict(), model_path)
        print("Training finished and saved.")

    # === 5000-step rollout (fixed shapes) ===
    model.eval()
    env = BouncingBallEnv()
    hist = [env.reset(), env.step()]
    current = torch.tensor(np.array(hist)).unsqueeze(0).unsqueeze(2).to(device)  # (1, 2, 1, 64, 80)

    frames_gt = []
    frames_gen = []
    for _ in tqdm(range(5000), desc="5000-step rollout"):
        with torch.no_grad():
            pred = model(current)                                 # (1, 1, 64, 80)
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            pred_bin = pred_bin.unsqueeze(1)                      # ← FIX: make it (1, 1, 1, 64, 80)
        current = torch.cat([current[:, 1:], pred_bin], dim=1)

        gt = env.step()
        frames_gt.append((gt * 255).astype(np.uint8))
        frames_gen.append((pred_bin[0, 0, 0].cpu().numpy() * 255).astype(np.uint8))

    render_h, render_w = 512, 640
    side_frames = []
    for g, p in zip(frames_gt, frames_gen):
        gt_up = cv2.resize(g, (render_w, render_h), cv2.INTER_NEAREST)
        gen_up = cv2.resize(p, (render_w, render_h), cv2.INTER_NEAREST)
        combined = np.hstack((gt_up, gen_up))
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        side_frames.append(combined_bgr)

    out = cv2.VideoWriter('perfect_bouncing_ball_64x80_5000_steps.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 512))
    for f in side_frames: out.write(f)
    out.release()

    print("✅ Saved perfect_bouncing_ball_64x80_5000_steps.mp4")


def generate_sequences(num_seqs=10000, seq_len=16):
    env = BouncingBallEnv()
    seqs = []
    for _ in tqdm(range(num_seqs), desc="Generating data"):
        env.reset()
        seq = [env._get_frame()]
        for _ in range(seq_len - 1):
            seq.append(env.step())
        seqs.append(seq)
    return torch.tensor(np.array(seqs)).unsqueeze(2)


if __name__ == "__main__":
    main()