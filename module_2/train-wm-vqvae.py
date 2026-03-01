import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
from tqdm import tqdm

PALETTE_RGB = np.array([
    [0.0, 0.0, 0.0], 
    [1.0, 1.0, 1.0]  
], dtype=np.float32)

# --- 1. Platformer Simulator & Frozen VQ-VAE Classes ---
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
        frame = np.zeros((1, self.height, self.width), dtype=np.float32)
        frame[0, self.y:self.y+self.ball_size, self.x:self.x+self.ball_size] = 1.0 
        return frame

def generate_sequences(num_seqs, seq_len, height=64, width=80):
    env = PlatformerEnv(height=height, width=width)
    frames_seq = []
    actions_seq = []
    
    for _ in range(num_seqs):
        #env.reset()
        frames = [env._get_frame()]
        actions = []
        
        step_idx = 0
        while step_idx < seq_len - 1:
            act = np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.1, 0.2])
            duration = np.random.randint(1, 6)
            for _ in range(duration):
                if step_idx >= seq_len - 1:
                    break
                frames.append(env.step(act))
                actions.append(act)
                step_idx += 1
                
        actions.append(2) 
        frames_seq.append(frames)
        actions_seq.append(actions)
        
    return torch.tensor(np.array(frames_seq)), torch.tensor(np.array(actions_seq))

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, None

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=16, num_embeddings=32):
        super().__init__()
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
        quantized, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, None
        
    # --- NEW HELPER METHODS FOR STAGE 2 ---
    @torch.no_grad()
    def get_tokens(self, x):
        """Converts pixels directly to discrete token integers"""
        z = self.encoder(x)
        inputs = z.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.vq.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.vq.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.vq.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices.view(inputs.shape[0], inputs.shape[1], inputs.shape[2])

    @torch.no_grad()
    def decode_tokens(self, indices):
        """Converts discrete token integers back into pixels"""
        B, H_t, W_t = indices.shape
        flat_indices = indices.view(-1)
        quantized = self.vq.embedding(flat_indices).view(B, H_t, W_t, self.vq.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)

# --- 2. Stage 2: Latent Transformer Architecture ---
class LatentWorldModel(nn.Module):
    def __init__(self, vocab_size=32, embed_dim=128, num_heads=4, num_layers=4, history_len=4, h_t=16, w_t=20):
        super().__init__()
        self.num_patches = h_t * w_t
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # 1. We replace the ConvTokenizer with a simple NLP-style Embedding lookup!
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.action_embedding = nn.Embedding(4, embed_dim)
        
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, history_len * self.num_patches, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, 
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. Output is logits over the VQ-VAE codebook vocabulary
        self.predictor = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, token_seq, actions):
        # token_seq shape: (B, T, h_t, w_t)
        B, T, h_t, w_t = token_seq.shape
        
        # Look up continuous embeddings for our discrete tokens
        latents = self.token_embedding(token_seq).view(B, T, self.num_patches, self.embed_dim)
        
        action_embeds = self.action_embedding(actions) 
        
        latents = latents + self.spatial_pos_embed.unsqueeze(1)
        latents = latents + action_embeds.unsqueeze(2) 
        
        latents = latents.reshape(B, T * self.num_patches, self.embed_dim)
        latents = latents + self.temporal_pos_embed
        
        out = self.transformer(latents)
        last_frame_out = out[:, -self.num_patches:, :]
        
        # Predict the exact token IDs for the next frame
        return self.predictor(last_frame_out) # Shape: (B, num_patches, vocab_size)

# --- 3. Training & Interactive Rollout ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    seq_len = 12 
    history_len = 4 
    vqvae_path = 'vqvae_platformer.pth'
    wm_path = 'latent_world_model2.pth'
    
    # 1. Load the frozen VQ-VAE (The "Eyes")
    vqvae = VQVAE().to(device)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device, weights_only=True))
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False
    
    # 2. Initialize the Latent World Model (The "Brain")
    model = LatentWorldModel(history_len=history_len).to(device)
    
    if os.path.exists(wm_path):
        print(f"Found saved model at '{wm_path}'. Loading weights...")
        model.load_state_dict(torch.load(wm_path, map_location=device, weights_only=True))
    else:
        print("Generating 10000 sequences of gameplay...")
        raw_frames, raw_actions = generate_sequences(num_seqs=10000, seq_len=seq_len)
        
        print("Pre-tokenizing entire dataset through VQ-VAE (Translating pixels to language)...")
        all_tokens = []
        batch_size = 500
        for i in tqdm(range(0, len(raw_frames), batch_size)):
            b_pixels = raw_frames[i:i+batch_size].to(device)
            B, T, C, H, W = b_pixels.shape
            b_pixels_flat = b_pixels.view(B*T, C, H, W)
            
            # This is where the magic happens: pixels become integers!
            tokens_flat = vqvae.get_tokens(b_pixels_flat) 
            tokens = tokens_flat.view(B, T, 16, 20)
            all_tokens.append(tokens.cpu())
            
        train_tokens = torch.cat(all_tokens, dim=0)
        train_actions = raw_actions
        
        X_tokens, X_actions, Y_tokens = [], [], []
        for t in range(seq_len - history_len):
            X_tokens.append(train_tokens[:, t : t + history_len])
            X_actions.append(train_actions[:, t : t + history_len]) 
            Y_tokens.append(train_tokens[:, t + history_len])
            
        X_tokens = torch.cat(X_tokens, dim=0)
        X_actions = torch.cat(X_actions, dim=0)
        Y_tokens = torch.cat(Y_tokens, dim=0)
        
        dataset = TensorDataset(X_tokens, X_actions, Y_tokens)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        epochs = 20
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # NO MORE HACKY WEIGHTS! The VQ-VAE balanced the data for us.
        criterion = nn.CrossEntropyLoss() 
        
        best_loss = float('inf')
        
        print("\nTraining Transformer purely on discrete tokens...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for b_tokens, b_actions, b_y in pbar:
                b_tokens = b_tokens.to(device)
                b_actions = b_actions.to(device)
                b_y = b_y.to(device).view(-1) # Flatten target for CrossEntropy
                
                optimizer.zero_grad()
                
                # Predict next tokens
                logits = model(b_tokens, b_actions) 
                logits = logits.view(-1, 32) # Flatten logits: (B * 320, 32)
                
                loss = criterion(logits, b_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), wm_path)
                
            scheduler.step()
            
    print("\n--- INTERACTIVE ROLLOUT STARTING ---")
    print("Click on the PyGame window to focus it.")
    print("Controls: 'W'=Jump, 'A'=Left, 'D'=Right, 'S'=Still.")
    print("Press 'Q' to quit and save the MP4.")
    
    model.load_state_dict(torch.load(wm_path, map_location=device, weights_only=True))
    model.eval()
    
    env = PlatformerEnv()
    
    # Generate initial history dynamically
    hist_frames = [env.reset()]
    hist_actions = [2] 
    for _ in range(history_len - 1):
        hist_frames.append(env.step(2))
        hist_actions.append(2)
        
    hist_tensor = torch.tensor(np.array(hist_frames)).unsqueeze(0).to(device)
    current_hist_actions = torch.tensor(np.array(hist_actions)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        current_hist_tokens = vqvae.get_tokens(hist_tensor.view(-1, 1, 64, 80)).view(1, history_len, 16, 20)
    
    rendered_frames = []
    render_height = 512 
    render_width = int(512 * (80/64)) 
    
    # THE FIX: Initialize a visible PyGame window and a clock for FPS
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((render_width * 2, render_height))
    pygame.display.set_caption('Latent World Model (PyGame)')
    clock = pygame.time.Clock()
    
    total_steps = 0
    running = True
    
    with torch.no_grad():
        while total_steps < 1000 and running:
            # 1. Predict Next Tokens
            logits = model(current_hist_tokens, current_hist_actions)
            pred_tokens = torch.argmax(logits, dim=2).view(1, 1, 16, 20)
            
            # 2. Decode Tokens Back to Pixels
            pred_pixels = vqvae.decode_tokens(pred_tokens.squeeze(1)) 
            pred_class = (pred_pixels.squeeze() > 0.5).int().cpu().numpy()
            
            pred_rgb = PALETTE_RGB[pred_class]
            sim_img = (pred_rgb * 255).astype(np.uint8)
            sim_img = cv2.cvtColor(sim_img, cv2.COLOR_RGB2BGR)
            sim_upscaled = cv2.resize(sim_img, (render_width, render_height), interpolation=cv2.INTER_NEAREST)
            cv2.putText(sim_upscaled, 'Latent World Model (W/A/S/D)', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 3. Ground Truth Step
            # Read Pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                break
                
            action = 2 
            if keys[pygame.K_a]:
                action = 0 
            if keys[pygame.K_d]:
                action = 1 
            if keys[pygame.K_w] and env.on_ground:
                action = 3 
                
            gt_frame = env.step(action)
            total_steps += 1
            
            gt_rgb = PALETTE_RGB[gt_frame.squeeze().astype(np.int64)]
            gt_img = (gt_rgb * 255).astype(np.uint8)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            gt_upscaled = cv2.resize(gt_img, (render_width, render_height), interpolation=cv2.INTER_NEAREST)
            cv2.putText(gt_upscaled, 'Ground Truth', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 4. Render to PyGame Window
            combined_bgr = np.hstack((gt_upscaled, sim_upscaled))
            rendered_frames.append(combined_bgr)
            
            # Convert OpenCV BGR to PyGame RGB, and transpose (H, W, C) to (W, H, C)
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            combined_rgb = np.transpose(combined_rgb, (1, 0, 2))
            
            surface = pygame.surfarray.make_surface(combined_rgb)
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            
            # Maintain exactly 30 FPS
            clock.tick(30)
            
            # 5. Update Autoregressive History
            current_hist_tokens = torch.cat([current_hist_tokens[:, 1:], pred_tokens], dim=1)
            action_tensor = torch.tensor([[action]]).to(device)
            current_hist_actions = torch.cat([current_hist_actions[:, 1:], action_tensor], dim=1)
            
    pygame.quit()
    
    print("\nSaving Video...")
    video_path = 'latent_world_model.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    out = cv2.VideoWriter(video_path, fourcc, fps, (render_width * 2, render_height))
    
    for frame in rendered_frames:
        out.write(frame)
    out.release()
    print(f"Video saved to: {video_path}")

if __name__ == '__main__':
    main()