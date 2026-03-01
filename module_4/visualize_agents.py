import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import pygame
from collections import deque

# Import your classes
from capstone_world_model_supervised import VQVAE, LatentWorldModel 
from generate_expert_dataset import ColorfulPlatformerEnv 

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
            
        # 1. Feature Compressor: Shrinks 256 channels to 16, KEEPS 16x16 grid!
        self.channel_compressor = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1)
        
        # 2. MLP Brain
        # 16 channels * 16 height * 16 width = 4096 input features
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
        
        # Compress channels, preserve spatial awareness
        x = self.channel_compressor(x) # [B, 16, 16, 16]
        
        # Flatten and pass to the logic brain
        x = x.reshape(B, -1) # [B, 4096]
        logits = self.actor_head(x)
        
        return logits

# ==========================================
# 2. EVALUATION LOOP WITH RENDERING
# ==========================================
def evaluate_agent_visually(actor, vqvae, num_episodes=5, name="Agent", fps=15):
    actor.eval()
    vqvae.eval()
    env = ColorfulPlatformerEnv(action_repeat=4)
    
    # Initialize Pygame Display
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(f"Evaluating: {name} (Use WASD to Intervene!)")
    clock = pygame.time.Clock()
    
    print(f"\n--- Launching Visual Evaluation for {name} ---")
    print(">>> INTERACTIVE MODE ACTIVE: Press W, A, S, D or Arrow Keys to override the AI! <<<")
    
    for ep in range(num_episodes):
        obs = env.reset()
        if False:
            # ==========================================
            # THE TROLL MODIFICATION
            # ==========================================
            if len(env.platforms) > 1:
                # Shift the middle platform (index 1)
                env.platforms[1].rect.x -= 24
                env.platforms[1].rect.y += 2
                print(f'--- TROLL MOD ACTIVATED: Shifted Platform 2 in Episode {ep+1} ---')
                
                # Re-render and grab the updated observation so the 
                # 8-frame starting buffer doesn't see a "teleporting" platform
                env._render()
                obs = env.get_state()
            # ==========================================
        done = False
        steps = 0
        
        # Initialize the 8-frame sliding window
        obs_buffer = deque([obs for _ in range(8)], maxlen=8)
        
        while not done:
            # Allow you to quit the window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # 1. Prepare visual tokens
            img_batch = torch.tensor(np.array(obs_buffer)).to(DEVICE)
            with torch.no_grad():
                tokens = vqvae.get_tokens(img_batch).unsqueeze(0) 
            
            # 2. Prepare dummy action history
            input_actions = torch.full((1, 8), 2, dtype=torch.long, device=DEVICE)
            
            # 3. Get Model Prediction
            with torch.no_grad():
                logits = actor(tokens, input_actions)
                probs = torch.softmax(logits, dim=1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                
            # --- HUMAN OVERRIDE INJECTION ---
            keys = pygame.key.get_pressed()
            human_override = False
            
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action = 0 # LEFT
                human_override = True
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action = 1 # RIGHT
                human_override = True
            elif keys[pygame.K_w] or keys[pygame.K_UP] or keys[pygame.K_SPACE]:
                action = 3 # JUMP
                human_override = True
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                action = 2 # STILL
                human_override = True
            # --------------------------------
                
            # 4. Step Environment
            obs, done = env.step(action)
            
            # --- RENDERING ---
            screen.blit(env.screen, (0, 0))
            
            font = pygame.font.SysFont(None, 36)
            action_text = ["LEFT", "RIGHT", "STILL", "JUMP"][action]
            
            # Visual feedback for when you are driving
            if human_override:
                img = font.render(f"HUMAN OVERRIDE: {action_text}", True, (255, 0, 0)) # Red
            else:
                img = font.render(f"AI Action: {action_text}", True, (0, 0, 0)) # Black
                
            screen.blit(img, (20, 20))
            pygame.display.flip()
            clock.tick(fps)
            # -----------------
            
            obs_buffer.append(obs)
            steps += 1
            
        print(f"Episode {ep+1}: SUCCESS in {steps} steps")
  

# ==========================================
# 3. MAIN SCRIPT
# ==========================================
def main():
    print("Loading Models...")
    # Load VQVAE
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load("vqvae_supervised_highcap.pth", map_location=DEVICE))
    
    # Load Model A (Champion)
    wm_champion = LatentWorldModel().to(DEVICE)
    actor_a = TransDreamerActor(wm_champion, freeze_backbone=True).to(DEVICE)
    try:
        actor_a.load_state_dict(torch.load("live_actor_champion_A.pth", map_location=DEVICE))
        #actor_a.load_state_dict(torch.load("live_actor_champion_B.pth", map_location=DEVICE))
        #actor_a.load_state_dict(torch.load("live_actor_champion_C.pth", map_location=DEVICE))
        print("Model A loaded successfully.")
    except Exception as e:
        print(f"Could not load Model A: {e}")
        return

    '''
    # Load Model B (Baseline)
    wm_scratch = LatentWorldModel().to(DEVICE)
    actor_b = TransDreamerActor(wm_scratch, freeze_backbone=False).to(DEVICE)
    try:
        actor_b.load_state_dict(torch.load("actor_baseline_B.pth", map_location=DEVICE))
        print("Model B loaded successfully.")
    except Exception as e:
        print(f"Could not load Model B: {e}")
        return
    '''

    # Let's watch 3 episodes of each!
    evaluate_agent_visually(actor_a, vqvae, num_episodes=20, name="Model A (Champion)", fps=15)
    #evaluate_agent_visually(actor_b, vqvae, num_episodes=3, name="Model B (Baseline)", fps=15)
    
    pygame.quit()

if __name__ == "__main__":
    main()