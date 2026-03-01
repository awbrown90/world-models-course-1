import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- 1. Platformer Simulator (High Jump Edition) ---
class PlatformerEnv:
    def __init__(self, height=64, width=80, ball_size=5):
        self.height = height
        self.width = width
        self.ball_size = ball_size
        self.reset()

    def reset(self):
        self.x = self.width // 2
        self.y = self.height - self.ball_size
        self.vx = 0
        self.vy = 0
        self.on_ground = True
        return self._get_frame()

    def step(self, action):
        if action == 0:
            self.vx = -2
        elif action == 1:
            self.vx = 2
        elif action == 2:
            self.vx = 0
            
        if action == 3 and self.on_ground:
            self.vy = -8
            self.on_ground = False
            
        self.vy += 1 
        self.vy = min(self.vy, 4)
        
        self.x += self.vx
        self.y += self.vy
        
        if self.y >= self.height - self.ball_size:
            self.y = self.height - self.ball_size
            self.vy = 0
            self.on_ground = True
            
        if self.x < 0:
            self.x = 0
        elif self.x > self.width - self.ball_size:
            self.x = self.width - self.ball_size
            
        return self._get_frame()

    def _get_frame(self):
        # We use a 1-channel float tensor for the VQ-VAE
        frame = np.zeros((1, self.height, self.width), dtype=np.float32)
        frame[0, self.y:self.y+self.ball_size, self.x:self.x+self.ball_size] = 1.0 
        return frame

def generate_static_frames(num_frames, height=64, width=80):
    # VQ-VAEs don't care about time! We just need a massive pile of static images.
    env = PlatformerEnv(height=height, width=width)
    frames = []
    
    for _ in range(num_frames // 100):
        env.reset()
        for _ in range(100):
            act = np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.1, 0.2])
            frames.append(env.step(act))
            
    return torch.tensor(np.array(frames))

# --- 2. VQ-VAE Architecture ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs shape: (B, C, H, W) -> Convert to (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to find nearest codebook vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss computation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # The Straight-Through Estimator trick
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate Perplexity (Codebook Usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=16, num_embeddings=32):
        super().__init__()
        # Downsample 64x80 to 16x20 discrete tokens (a 16x spatial compression!)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1),
        )
        
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity

# --- 3. Stage 1 Training Loop ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = 'vqvae_platformer2.pth'
    
    print("Generating 100,000 static frames for VQ-VAE Codebook training...")
    train_data = generate_static_frames(num_frames=100000)
    
    dataset = TensorDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # We use a vocabulary of 32 discrete "words" to describe our platformer universe
    model = VQVAE(in_channels=1, embedding_dim=16, num_embeddings=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # We use BCE because the pixels are strictly 0.0 (black) or 1.0 (white)
    recon_criterion = nn.BCEWithLogitsLoss()
    
    epochs = 10
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_recon_loss = 0
        total_vq_loss = 0
        total_perplexity = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            imgs = batch[0].to(device)
            
            optimizer.zero_grad()
            
            x_recon, vq_loss, perplexity = model(imgs)
            recon_loss = recon_criterion(x_recon, imgs)
            
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            
            pbar.set_postfix({
                'Recon Loss': f"{recon_loss.item():.4f}", 
                'Perplexity': f"{perplexity.item():.1f}"
            })
            
        avg_loss = (total_recon_loss + total_vq_loss) / len(dataloader)
        avg_perplexity = total_perplexity / len(dataloader)
        
        print(f"Epoch {epoch+1} Summary | Total Loss: {avg_loss:.4f} | Avg Perplexity (Active Tokens): {avg_perplexity:.1f}/32")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            
    print(f"\nVQ-VAE Training Complete! Model saved to '{model_path}'")

if __name__ == '__main__':
    main()