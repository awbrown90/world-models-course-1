import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pygame

# ==========================================
# 1. ENVIRONMENT CLASSES (Matched to Training)
# ==========================================
class Character(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image_data = [
            "  BBB  ", "   B   ", "BBBKBBB", "B BBB B",
            "  BBB  ", " BB BB ", "B     B", "RR   RR"
        ]
        self.BLUE = (0, 0, 255); self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255); self.BLACK = (0, 0, 0)
        self.WIDTH, self.HEIGHT = 800, 600
        self.platforms = [
            Platform(100, 400, 200, 20), Platform(400, 300, 200, 20), Platform(650, 200, 100, 20)
        ]
        self.image = self.create_image()
        self.rect = self.image.get_rect()
        self.rect.x = self.WIDTH // 2; self.rect.y = self.HEIGHT - self.image.get_height() - 55 
        self.speed_x = 0; self.speed_y = 0
        self.jumping = False; self.on_ground = False
    
    def create_image(self):
        unscaled_image = pygame.Surface((len(self.image_data[0]), len(self.image_data)), pygame.SRCALPHA)
        for y, row in enumerate(self.image_data):
            for x, color_code in enumerate(row):
                color = self.BLUE if color_code == "B" else self.RED if color_code == "R" else self.WHITE if color_code == "W" else self.BLACK if color_code == "K" else None
                if color: unscaled_image.set_at((x, y), color)
        return pygame.transform.scale(unscaled_image, (unscaled_image.get_width() * 6, unscaled_image.get_height() * 6))
    
    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        self.rect.x = max(0, min(self.WIDTH - self.image.get_width(), self.rect.x))
        if not self.on_ground: self.speed_y += 1
        self.on_ground = False
        if self.rect.y >= self.HEIGHT - 100 and not self.on_ground:
            self.speed_y = 0; self.rect.y = self.HEIGHT - 100; self.on_ground = True
        for platform in self.platforms:
            if self.rect.colliderect(platform.rect):
                if self.speed_y > 0:  
                    self.rect.y = platform.rect.y - self.rect.height; self.speed_y = 0; self.on_ground = True
                elif self.speed_y < 0: 
                    self.rect.y = platform.rect.y + platform.rect.height; self.speed_y = 0
            else:
                for platform in self.platforms:
                    if self.rect.copy().move(0, 1).colliderect(platform.rect):
                        self.on_ground = True; break
    def jump(self):
        if self.on_ground:
            self.jumping = True; self.on_ground = False; self.speed_y = -20

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill((255, 165, 0))
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y

class ColorfulPlatformerEnv:
    def __init__(self, action_repeat=4):
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.WHITE, self.GREEN = (255, 255, 255), (0, 255, 0)
        self.action_repeat = action_repeat
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.character = Character()
        self.platforms = self.character.platforms

    def _render(self):
        self.screen.fill(self.WHITE)
        self.screen.blit(self.character.image, self.character.rect)
        pygame.draw.rect(self.screen, self.GREEN, (0, self.HEIGHT - 55, self.WIDTH, 55))
        for platform in self.platforms:
            self.screen.blit(platform.image, platform.rect)

    def reset(self):
        self.character = Character()
        self.platforms = self.character.platforms
        self._render()
        return self.get_state()

    def get_state(self):
        image = pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])
        # NEAREST neighbor scaling to prevent color blending
        scaled = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)  
        return (scaled.astype(np.float32) / 255.0).transpose(2, 0, 1)

    def step(self, action):
        # FRAME SKIPPING
        for _ in range(self.action_repeat):
            if action == 0: self.character.speed_x = -5
            elif action == 1: self.character.speed_x = 5
            elif action == 2: self.character.speed_x = 0
            elif action == 3: self.character.jump()
            self.character.update()

        self._render()
        return self.get_state()

# ==========================================
# 2. HIGH-CAPACITY VQ-VAE CLASSES
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings; self.embedding_dim = embedding_dim; self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        loss = F.mse_loss(quantized, inputs.detach()) + self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), loss

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=32, num_embeddings=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            ResBlock(128), ResBlock(128),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            ResBlock(128), ResBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )
    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq(z)
        return self.decoder(quantized), vq_loss

# ==========================================
# 3. DIAGNOSTIC RUNNER
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Diagnostics initializing on {device}...")
    
    env = ColorfulPlatformerEnv(action_repeat=4)
    env.reset()
    
    vqvae_path = 'vqvae_supervised_highcap.pth'
    vqvae = VQVAE().to(device)
    if os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device, weights_only=True))
        print(f"Successfully loaded {vqvae_path}")
    else:
        print(f"ERROR: Could not find {vqvae_path}. Run capstone_world_model_supervised.py first.")
        return
        
    vqvae.eval()
    
    pygame.init()
    render_width, render_height = 512, 512
    screen = pygame.display.set_mode((render_width * 2, render_height))
    pygame.display.set_caption('High-Cap VQ-VAE Reconstruction Diagnostic (15 FPS AI Vision)')
    clock = pygame.time.Clock()
    
    print("Controls: 'A'=Left, 'D'=Right, 'W'=Jump, 'S'=Still. 'Q' to quit.")
    print("Note: The game will feel fast/choppy because you are viewing exactly what the World Model sees (action_repeat=4).")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]: running = False
            
        action = 2 
        if keys[pygame.K_a]: action = 0 
        if keys[pygame.K_d]: action = 1 
        if keys[pygame.K_w] and env.character.on_ground: action = 3 
            
        # Get Ground Truth Frame (This executes 4 underlying physics steps!)
        gt_frame = env.step(action)
        
        # Pass through VQ-VAE
        with torch.no_grad():
            tensor_frame = torch.tensor(gt_frame).unsqueeze(0).to(device)
            recon_frame, _ = vqvae(tensor_frame)
            recon_frame = recon_frame.squeeze(0).cpu().numpy()
            
        # Render Ground Truth
        gt_img = (np.clip(gt_frame.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
        gt_upscaled = cv2.resize(gt_img, (render_width, render_height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(gt_upscaled, 'Ground Truth', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Render Reconstruction
        recon_img = (np.clip(recon_frame.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)
        recon_upscaled = cv2.resize(recon_img, (render_width, render_height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(recon_upscaled, 'High-Cap VQ-VAE', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display side-by-side
        combined_bgr = np.hstack((gt_upscaled, recon_upscaled))
        combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB).transpose(1, 0, 2)
        screen.blit(pygame.surfarray.make_surface(combined_rgb), (0, 0))
        pygame.display.flip()
        
        # We cap the visualizer at 30fps. Since every tick = 4 physics steps, 
        # the game is actually running at 120 internal physics steps per second.
        clock.tick(30)
        
    pygame.quit()

if __name__ == '__main__':
    main()