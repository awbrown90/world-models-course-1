import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import pygame
import random
from collections import deque
import csv

# ==========================================
# IMPORT YOUR ARCHITECTURES AND ENVIRONMENT
# ==========================================
from capstone_world_model_supervised import VQVAE, LatentWorldModel 
from generate_expert_dataset import ColorfulPlatformerEnv, get_expert_action

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE ACTOR AGENT WRAPPER
# ==========================================
class TransDreamerActor(nn.Module):
    def __init__(self, world_model, freeze_backbone=True):
        super().__init__()
        self.backbone = world_model
        self.is_frozen = freeze_backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.train()
            
        self.channel_compressor = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1)
        self.actor_head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
        
    def forward(self, tokens, actions):
        if self.is_frozen:
            with torch.no_grad():
                spatial_features = self.backbone.get_transformer_features(tokens, actions)
        else:
            spatial_features = self.backbone.get_transformer_features(tokens, actions)
            
        B = spatial_features.shape[0]
        x = spatial_features.transpose(1, 2).reshape(B, 256, 16, 16)
        x = self.channel_compressor(x) 
        x = x.reshape(B, -1) 
        logits = self.actor_head(x)
        return logits

# ==========================================
# 2. THE REPLAY BUFFER
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, tokens, expert_action):
        self.buffer.append((tokens.cpu(), expert_action))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        tokens_batch = torch.cat([item[0] for item in batch], dim=0) 
        actions_batch = torch.tensor([item[1] for item in batch], dtype=torch.long) 
        return tokens_batch.to(DEVICE), actions_batch.to(DEVICE)
        
    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. LIVE TRAINING LOOP
# ==========================================
def run_live_dagger():
    # ==========================================
    # TOGGLE YOUR AGENT HERE ("A", "B", or "C")
    # A = Pre-trained Frozen World Model
    # B = Scratch Unfrozen Backbone
    # C = Random Frozen World Model (The Baseline)
    # ==========================================
    AGENT_TYPE = "B" 
    
    print(f"Loading Models for Live Training (Model {AGENT_TYPE})...")
    
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load("vqvae_supervised_highcap.pth", map_location=DEVICE))
    vqvae.eval()
    
    if AGENT_TYPE == "A":
        wm = LatentWorldModel().to(DEVICE)
        wm.load_state_dict(torch.load("wm_supervised_highcap_big_best.pth", map_location=DEVICE))
        actor = TransDreamerActor(wm, freeze_backbone=True).to(DEVICE)
        BATCH_SIZE, TRAIN_FREQ, LR = 64, 8, 3e-4
        
    elif AGENT_TYPE == "B":
        wm = LatentWorldModel().to(DEVICE)
        actor = TransDreamerActor(wm, freeze_backbone=False).to(DEVICE)
        BATCH_SIZE, TRAIN_FREQ, LR = 16, 16, 1e-4
        
    elif AGENT_TYPE == "C":
        wm = LatentWorldModel().to(DEVICE) 
        # WE DO NOT LOAD WEIGHTS. WE JUST FREEZE THE RANDOM GARBAGE.
        actor = TransDreamerActor(wm, freeze_backbone=True).to(DEVICE)
        BATCH_SIZE, TRAIN_FREQ, LR = 64, 8, 3e-4 # Same fast hardware settings as A
    
    actor.train() 
    optimizer = optim.AdamW(actor.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    buffer = ReplayBuffer(capacity=5000)
    
    env = ColorfulPlatformerEnv(action_repeat=4)
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(f"LIVE DAgger - Model {AGENT_TYPE}")
    clock = pygame.time.Clock()
    
    global_step = 0
    episodes = 0
    font = pygame.font.SysFont(None, 36)
    
    save_filename = f"live_actor_champion_{AGENT_TYPE}.pth"
    log_filename = f"training_log_{AGENT_TYPE}.csv"

    # --- SETUP CSV LOGGER ---
    with open(log_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Steps", "Success"])

    print(f"\nStarting Live Interaction for Model {AGENT_TYPE}!")
    print(f"Logging data to {log_filename}")
    
    while True:
        obs = env.reset()
        done = False
        steps = 0
        episodes += 1
        
        obs_buffer = deque([obs for _ in range(8)], maxlen=8)
        
        while not done and steps < 300:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # --- 1. OBSERVE ---
            img_batch = torch.tensor(np.array(obs_buffer)).to(DEVICE)
            with torch.no_grad():
                tokens = vqvae.get_tokens(img_batch).unsqueeze(0)
            input_actions = torch.full((1, 8), 2, dtype=torch.long, device=DEVICE)
            
            # --- 2. AGENT ACTS ---
            with torch.no_grad():
                logits = actor(tokens, input_actions)
                probs = torch.softmax(logits, dim=1)
                agent_action = torch.distributions.Categorical(probs).sample().item()
                
            # --- 3. EXPERT LABELS ---
            expert_action, _ = get_expert_action(env.character, noise_prob=0.0)
            
            # --- 4. STORE IN BUFFER ---
            buffer.push(tokens, expert_action)
            
            # --- 5. STEP ENVIRONMENT ---
            obs, done = env.step(agent_action)
            obs_buffer.append(obs)
            
            # --- 6. MICRO-UPDATE (TRAINING) ---
            if len(buffer) >= BATCH_SIZE and global_step % TRAIN_FREQ == 0:
                batch_tokens, batch_expert_actions = buffer.sample(BATCH_SIZE)
                batch_history = torch.full((BATCH_SIZE, 8), 2, dtype=torch.long, device=DEVICE)
                
                optimizer.zero_grad()
                pred_logits = actor(batch_tokens, batch_history)
                loss = criterion(pred_logits, batch_expert_actions)
                loss.backward()
                optimizer.step()
                
            # --- 7. RENDER ---
            screen.blit(env.screen, (0, 0))
            
            agent_text = ["LEFT", "RIGHT", "STILL", "JUMP"][agent_action]
            expert_text = ["LEFT", "RIGHT", "STILL", "JUMP"][expert_action]
            
            img1 = font.render(f"Agent: {agent_text}", True, (255, 0, 0))
            img2 = font.render(f"Expert: {expert_text}", True, (0, 150, 0))
            img3 = font.render(f"Buffer: {len(buffer)} | Ep: {episodes}", True, (0, 0, 0))
            
            screen.blit(img1, (20, 20))
            screen.blit(img2, (20, 60))
            screen.blit(img3, (20, 100))
            
            pygame.display.flip()
            clock.tick(30)
            
            global_step += 1
            steps += 1
            
        # --- 8. LOG AND SAVE DATA ---
        success = 1 if steps < 300 else 0
        with open(log_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episodes, steps, success])
            
        if episodes % 5 == 0:
            torch.save(actor.state_dict(), save_filename)
            print(f"Ep {episodes}: Logged {steps} steps. Saved weights to {save_filename}")


if __name__ == "__main__":
    run_live_dagger()