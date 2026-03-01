import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
from tqdm import tqdm
import pygame

# ==========================================
# 1. UNIFIED ENVIRONMENT
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
        scaled = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)  
        return (scaled.astype(np.float32) / 255.0).transpose(2, 0, 1)

    def step(self, action):
        for _ in range(self.action_repeat):
            if action == 0: self.character.speed_x = -5
            elif action == 1: self.character.speed_x = 5
            elif action == 2: self.character.speed_x = 0
            elif action == 3 and self.character.on_ground: self.character.jump()
            self.character.update()
        self._render()
        return self.get_state()

@torch.no_grad()
@torch.no_grad()
# CHANGED: Added `start_indices` as a parameter, removed `num_episodes`
def run_long_horizon_eval(wm, vqvae, all_tokens, raw_actions, tensor_raw_frames, device, history_len, start_indices, rollout_steps=50):
    """
    Simulates playing the game using strictly autoregressive predictions.
    Uses a FIXED set of start_indices for true apples-to-apples comparison.
    """
    wm.eval()
    vqvae.eval()
    
    num_episodes = len(start_indices) # Derive count from the provided list
    
    # 1. Prepare initial perfect history for all episodes simultaneously
    hist_tokens = torch.stack([all_tokens[idx : idx + history_len] for idx in start_indices]).to(device)
    hist_actions = torch.stack([torch.tensor(raw_actions[idx + 1 : idx + history_len + 1]) for idx in start_indices]).to(device)
    
    all_pred_tokens = []
    current_tokens = hist_tokens
    current_actions = hist_actions
    
    # 2. Open-Loop Autoregressive Generation (Batched for speed)
    for step in range(rollout_steps):
        # Extract the Ground Truth action for this specific time step
        step_actions = torch.tensor([raw_actions[idx + history_len + step] for idx in start_indices]).unsqueeze(1).to(device)
        actions_in = torch.cat([current_actions[:, 1:], step_actions], dim=1)
        
        # Predict next frame
        logits = wm(current_tokens, actions_in)
        pred_tokens = torch.argmax(logits, dim=2).view(num_episodes, 1, 16, 16)
        all_pred_tokens.append(pred_tokens.cpu())
        
        # Update history with the PREDICTED frame (Open-Loop)
        current_tokens = torch.cat([current_tokens[:, 1:], pred_tokens], dim=1)
        current_actions = actions_in
        
    all_pred_tokens = torch.cat(all_pred_tokens, dim=1) # Shape: (num_episodes, rollout_steps, 16, 16)
    
    # 3. Decode to Pixels and Calculate MSE against Ground Truth
    mse_scores = []
    for i in range(num_episodes):
        idx = start_indices[i]
        
        # Get Ground Truth frames
        gt_frames = tensor_raw_frames[idx + history_len : idx + history_len + rollout_steps].to(device)
        
        # Decode the model's hallucinated tokens back into pixels
        pred_toks = all_pred_tokens[i].to(device) 
        pred_pixels = vqvae.decode_tokens(pred_toks) 
        
        # Compare visual accuracy
        mse = F.mse_loss(pred_pixels, gt_frames).item()
        mse_scores.append(mse)
        
    wm.train() # Return to training mode
    return np.mean(mse_scores), np.min(mse_scores), np.max(mse_scores)

# ==========================================
# 2. HIGH-CAPACITY VQ-VAE & WORLD MODEL
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
    @torch.no_grad()
    def get_tokens(self, x):
        z = self.encoder(x)
        inputs = z.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.vq.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.vq.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.vq.embedding.weight.t()))
        return torch.argmin(distances, dim=1).view(inputs.shape[0], inputs.shape[1], inputs.shape[2])
    @torch.no_grad()
    def decode_tokens(self, indices):
        B, H_t, W_t = indices.shape
        quantized = self.vq.embedding(indices.view(-1)).view(B, H_t, W_t, self.vq.embedding_dim).permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)

class LatentWorldModel(nn.Module):
    def __init__(self, vocab_size=512, embed_dim=256, num_heads=8, num_layers=6, history_len=8):
        super().__init__()
        self.num_patches = 16 * 16; self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.action_embedding = nn.Embedding(4, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, history_len * self.num_patches, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.predictor = nn.Linear(embed_dim * 2, vocab_size)
    def forward(self, token_seq, actions):
        B, T, h_t, w_t = token_seq.shape
        latents = self.token_embedding(token_seq).view(B, T, self.num_patches, self.embed_dim)
        
        # Get action embeddings and expand to match patches
        action_embeds = self.action_embedding(actions) 
        
        latents = latents + self.spatial_pos_embed.unsqueeze(1) + action_embeds.unsqueeze(2) 
        
        # --- THE FIX: Slice the temporal embedding to match the current T (7 or 8) ---
        temporal_pos = self.temporal_pos_embed[:, :T * self.num_patches, :]
        latents = latents.reshape(B, T * self.num_patches, self.embed_dim) + temporal_pos
        # -----------------------------------------------------------------------------
        
        transformer_out = self.transformer(latents)[:, -self.num_patches:, :]
        
        # Grab the action embedding for the exact frame we are predicting
        current_action = action_embeds[:, -1, :].unsqueeze(1).expand(-1, self.num_patches, -1)
        
        # Concatenate the Transformer output with the raw Action embedding
        combined_features = torch.cat([transformer_out, current_action], dim=2)
        
        return self.predictor(combined_features)
    def get_transformer_features(self, latents, actions):
        """
        Extracts the internal physics state for the Actor head.
        latents: [Batch, 8, 16, 16]
        actions: [Batch, 8]
        """
        B, T, H, W = latents.shape
        latents = self.token_embedding(latents).view(B, T, self.num_patches, self.embed_dim) # Add this!
        action_embeds = self.action_embedding(actions) 
        
        # Match your exact forward pass math here!
        latents = latents + self.spatial_pos_embed.unsqueeze(1) + action_embeds.unsqueeze(2)
        latents = latents.reshape(B, T * self.num_patches, self.embed_dim) + self.temporal_pos_embed
        
        # Pass through attention layers
        transformer_out = self.transformer(latents) # [B, 2048, 256]
        
        # Slice out the very last frame (the 8th frame)
        last_frame_features = transformer_out[:, -self.num_patches:, :] # [B, 256, 256]
        
        # FIX: Do NOT average pool! Return the full spatial grid.
        return last_frame_features # Shape is now [B, 256 patches, 256 dims]

# ==========================================
# 3. SUPERVISED PIPELINE
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 8-FRAME CONTEXT PIPELINE INITIALIZED ON {device} ---")
    
    seq_len = 16 
    history_len = 8
    
    vqvae_path = 'vqvae_supervised_highcap.pth'
    #wm_path = 'wm_supervised_highcap_big.pth'
    wm_path = 'wm_supervised_highcap_big_best.pth'
    
    vqvae = VQVAE().to(device)
    wm = LatentWorldModel(history_len=history_len).to(device)
    
    if os.path.exists(vqvae_path) and os.path.exists(wm_path):
        print("Found checkpoints. Skipping to rollout!")
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device, weights_only=True))
        wm.load_state_dict(torch.load(wm_path, map_location=device, weights_only=True))
    else:
        print("Loading manual dataset...")
        frames_dir = 'dataset/frames'
        action_path = 'dataset/actions.txt'
        
        if not os.path.exists(frames_dir) or not os.path.exists(action_path):
            print("ERROR: Could not find dataset/. Run test_env_record.py first.")
            return
            
        with open(action_path, 'r') as f:
            raw_actions = [int(line.strip()) for line in f.readlines() if line.strip()]
            
        raw_frames = []
        for i in tqdm(range(len(raw_actions)), desc="Loading PNGs"):
            img = cv2.imread(os.path.join(frames_dir, f'frame_{i:05d}.png'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor_img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1) 
            raw_frames.append(tensor_img)
            
        print(f"Loaded {len(raw_frames)} frames.")
        
        if os.path.exists(vqvae_path):
            print(f"[*] Found pre-trained VQ-VAE at {vqvae_path}. Loading...")
            vqvae.load_state_dict(torch.load(vqvae_path, map_location=device, weights_only=True))
            #wm.load_state_dict(torch.load(wm_path, map_location=device, weights_only=True))
        else:
            print("\n--- STAGE 1: Training High-Cap VQ-VAE ---")
            flat_train_frames = torch.tensor(np.array(raw_frames))
            vq_loader = DataLoader(TensorDataset(flat_train_frames), batch_size=256, shuffle=True)
            optimizer_vq = optim.AdamW(vqvae.parameters(), lr=1e-3)
            scheduler_vq = optim.lr_scheduler.CosineAnnealingLR(optimizer_vq, T_max=300, eta_min=1e-5)
            
            best_recon_loss = float('inf')
            vq_patience, vq_patience_counter = 20, 0
            
            for epoch in range(1000): 
                epoch_recon = 0.0
                current_lr = optimizer_vq.param_groups[0]['lr']
                pbar_vq = tqdm(vq_loader, desc=f"VQ Epoch {epoch+1}/1000 [LR: {current_lr:.5f}]")
                for batch in pbar_vq:
                    imgs = batch[0].to(device)
                    optimizer_vq.zero_grad()
                    x_recon, vq_loss = vqvae(imgs)
                    
                    recon_loss = F.mse_loss(x_recon, imgs)
                    total_loss = (10.0 * recon_loss) + vq_loss
                    
                    total_loss.backward()
                    optimizer_vq.step()
                    
                    epoch_recon += recon_loss.item()
                    pbar_vq.set_postfix({'recon': f"{recon_loss.item():.5f}", 'vq': f"{vq_loss.item():.5f}"})
                    
                avg_recon = epoch_recon / len(vq_loader)
                if avg_recon < best_recon_loss:
                    best_recon_loss = avg_recon
                    torch.save(vqvae.state_dict(), vqvae_path)
                    vq_patience_counter = 0
                else:
                    vq_patience_counter += 1
                    if vq_patience_counter >= vq_patience:
                        print(f"\n   -> VQ-VAE Early stopping triggered. Best Recon Loss: {best_recon_loss:.5f}")
                        break
                scheduler_vq.step()
                        
            vqvae.load_state_dict(torch.load(vqvae_path, map_location=device, weights_only=True))
            
        vqvae.eval()
        for param in vqvae.parameters(): param.requires_grad = False
        
        print("\n--- STAGE 2: Tokenizing Data & Training World Model ---")
        print("Tokenizing entire dataset into 512-word Vocabulary...")
        
        # Tokenize the entire dataset first to save memory and make alignment easy
        all_tokens = []
        tensor_raw_frames = torch.tensor(np.array(raw_frames))
        for i in range(0, len(tensor_raw_frames), 100):
            b_pixels = tensor_raw_frames[i:i+100].to(device)
            all_tokens.append(vqvae.get_tokens(b_pixels).cpu())
        all_tokens = torch.cat(all_tokens, dim=0) # Shape: (Total_Frames, 16, 16)
        
        # Build perfectly aligned histories and targets
        X_tokens, X_actions, Y_tokens = [], [], []
        H = history_len
        
        for t in range(len(all_tokens) - H - 1):
            # History frames: f[t] to f[t+H-1]
            X_tokens.append(all_tokens[t : t + H])
            
            # Actions that caused the transitions: a[t+1] to a[t+H]
            X_actions.append(torch.tensor(raw_actions[t + 1 : t + H + 1])) 
            
            # Target frame: f[t+H]
            Y_tokens.append(all_tokens[t + H])
            
        X_tokens = torch.stack(X_tokens)
        X_actions = torch.stack(X_actions)
        Y_tokens = torch.stack(Y_tokens)
        
        dataloader = DataLoader(TensorDataset(X_tokens, X_actions, Y_tokens), batch_size=32, shuffle=True)
        
        wm.train()
        # --- NEW: LOAD EXISTING WEIGHTS FOR WARM RESTART ---
        if os.path.exists(wm_path):
            print(f"[*] Found existing World Model at {wm_path}. Loading weights to resume training...")
            wm.load_state_dict(torch.load(wm_path, map_location=device, weights_only=True))

        # CHANGED: Lower Learning Rate (1e-4) for fine-tuning!
        best_eval_mse = float('inf') # NEW: Track the best evaluation score
        epochs = 500
        optimizer_wm = optim.AdamW(wm.parameters(), lr=3e-4) 
        scheduler_wm = optim.lr_scheduler.CosineAnnealingLR(optimizer_wm, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        # --- NEW: PRE-SELECT FIXED EVALUATION INDICES ---
        eval_rollout_steps = 50
        max_start_idx = len(all_tokens) - history_len - eval_rollout_steps - 1
        
        # Use a fixed manual seed (e.g., 42) so the validation set never changes!
        eval_gen = torch.Generator()
        eval_gen.manual_seed(42) 
        eval_start_indices = torch.randint(0, max_start_idx, (100,), generator=eval_gen)
        # ------------------------------------------------
        
        print(f"\nResuming Training for {epochs} Epochs. SCHEDULED SAMPLING LOCKED AT 30%...")
        for epoch in range(epochs):
            current_lr = optimizer_wm.param_groups[0]['lr']
            pbar = tqdm(dataloader, desc=f"WM Epoch {epoch+1}/{epochs} [LR: {current_lr:.5f}]")
            
            prob_use_pred = min(0.3, (epoch / 30.0) * 0.3)
            
            for b_tokens, b_actions, b_y in pbar:
                b_tokens, b_actions, b_y = b_tokens.to(device), b_actions.to(device), b_y.to(device)
                
                # Scheduled Sampling Injection
                if torch.rand(1).item() < prob_use_pred:
                    with torch.no_grad():
                        partial_history = b_tokens[:, :-1, :, :] 
                        partial_actions = b_actions[:, :-1]
                        
                        intermediate_logits = wm(partial_history, partial_actions)
                        pred_frame_8 = torch.argmax(intermediate_logits, dim=2).view(-1, 1, 16, 16)
                        
                        corrupted_history = torch.cat([partial_history, pred_frame_8], dim=1)
                        b_tokens = corrupted_history.detach()

                optimizer_wm.zero_grad()
                logits = wm(b_tokens, b_actions).view(-1, 512) 
                loss = criterion(logits, b_y.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
                optimizer_wm.step()
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'p_sample': f"{prob_use_pred:.2f}"})
                
            scheduler_wm.step()
            torch.save(wm.state_dict(), wm_path)

            # --- NEW: LONG-HORIZON EVALUATION EVERY 5 EPOCHS ---
            if (epoch + 1) % 5 == 0:
                print(f"\n--- Running 50-Step Long-Horizon Eval (Fixed 100 Episodes) ---")
                
                avg_mse, min_mse, max_mse = run_long_horizon_eval(
                    wm, vqvae, all_tokens, raw_actions, tensor_raw_frames, device, 
                    history_len=history_len, start_indices=eval_start_indices, rollout_steps=eval_rollout_steps
                )
                print(f"-> Result: AVG Pixel MSE: {avg_mse:.5f} | Best: {min_mse:.5f} | Worst: {max_mse:.5f}")
                
                # --- NEW: SAVE THE BEST MODEL ---
                if avg_mse < best_eval_mse:
                    best_eval_mse = avg_mse
                    best_path = wm_path.replace('.pth', '_best.pth')
                    torch.save(wm.state_dict(), best_path)
                    print(f"🌟 NEW BEST MODEL SAVED! (Improved to {best_eval_mse:.5f}) 🌟\n")
                else:
                    print(f"Current Best remains: {best_eval_mse:.5f}\n")
            # ---------------------------------------------------
                
        torch.save(wm.state_dict(), wm_path)

    # ==========================================
    # 4. INTERACTIVE ROLLOUT
    # ==========================================
    print("\n--- 8-FRAME CONTEXT ROLLOUT STARTING ---")
    # NEW: Load the golden weights if they exist!
    best_path = wm_path.replace('.pth', '_best.pth')
    if os.path.exists(best_path):
        print("Loading BEST evaluation weights for rollout...")
        wm.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    env = ColorfulPlatformerEnv(action_repeat=4)
    render_width, render_height = 512, 512
    screen = pygame.display.set_mode((render_width * 2, render_height))
    pygame.display.set_caption('Supervised World Model (History=8)')
    clock = pygame.time.Clock()
    
    wm.eval()
    vqvae.eval()
    
    hist_frames = [env.reset()]
    hist_actions = [2] 
    for _ in range(history_len - 1):
        hist_frames.append(env.step(2)); hist_actions.append(2)
        
    hist_tensor = torch.tensor(np.array(hist_frames)).unsqueeze(0).to(device)
    current_hist_actions = torch.tensor(np.array(hist_actions)).unsqueeze(0).to(device)
    current_hist_tokens = vqvae.get_tokens(hist_tensor.view(-1, 3, 64, 64)).view(1, history_len, 16, 16)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]: running = False
            
        # 1. READ KEYBOARD INPUT FIRST
        action = 2 
        if keys[pygame.K_a]: action = 0 
        if keys[pygame.K_d]: action = 1 
        if keys[pygame.K_w]: action = 3 
            
        # 2. INJECT CURRENT ACTION INTO HISTORY
        action_tensor = torch.tensor([[action]]).to(device)
        actions_in = torch.cat([current_hist_actions[:, 1:], action_tensor], dim=1)
        
        # 3. PREDICT NEXT FRAME USING THE ACTION WE JUST PRESSED
        logits = wm(current_hist_tokens, actions_in)
        pred_tokens = torch.argmax(logits, dim=2).view(1, 1, 16, 16)
        pred_pixels = vqvae.decode_tokens(pred_tokens.squeeze(1)).squeeze(0).cpu().numpy()
        
        # 4. STEP GROUND TRUTH ENVIRONMENT
        gt_frame = env.step(action)
        
        # Render Simulated Frame
        sim_img = cv2.resize(cv2.cvtColor((np.clip(pred_pixels.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), (render_width, render_height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(sim_upscaled:=sim_img, 'Latent World Model', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Render Ground Truth Frame
        gt_img = cv2.resize(cv2.cvtColor((np.clip(gt_frame.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), (render_width, render_height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(gt_upscaled:=gt_img, 'Ground Truth', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display side-by-side
        screen.blit(pygame.surfarray.make_surface(cv2.cvtColor(np.hstack((gt_upscaled, sim_upscaled)), cv2.COLOR_BGR2RGB).transpose(1, 0, 2)), (0, 0))
        pygame.display.flip()
        clock.tick(30)
        
        # 5. UPDATE HISTORIES FOR NEXT LOOP
        current_hist_tokens = torch.cat([current_hist_tokens[:, 1:], pred_tokens], dim=1)
        current_hist_actions = actions_in
        
    pygame.quit()

if __name__ == '__main__':
    main()