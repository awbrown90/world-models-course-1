import os
import numpy as np
import cv2
import pygame
import random
from tqdm import tqdm

# ==========================================
# 1. ENVIRONMENT (Modified for Episode Logic)
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
            Platform(100, 400, 200, 20), # P1
            Platform(400, 300, 200, 20), # P2
            Platform(650, 200, 100, 20)  # P3 (Target)
        ]
        self.image = self.create_image()
        self.rect = self.image.get_rect()
        
        # CHANGED: Spawn at a random X position on the floor
        self.rect.x = random.randint(50, self.WIDTH - 100) 
        self.rect.y = self.HEIGHT - self.image.get_height() - 55 
        
        self.speed_x = 0; self.speed_y = 0
        self.jumping = False; self.on_ground = True
    
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
        
        # Floor collision
        if self.rect.y >= self.HEIGHT - 100 and not self.on_ground:
            self.speed_y = 0; self.rect.y = self.HEIGHT - 100; self.on_ground = True
            
        # Platform collision
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
            
        # Optional: Draw a faint target zone so we know where P3 center is
        #target_rect = pygame.Rect(690, 150, 20, 50)
        #pygame.draw.rect(self.screen, (200, 200, 200), target_rect, 2)

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
        
        # Check if episode is done (Standing on P3 near center)
        done = False
        char = self.character
        if char.on_ground and char.rect.y < 250: # On top platform
            if 680 <= char.rect.x <= 710: # Near center
                done = True
                
        return self.get_state(), done

# ==========================================
# 2. THE EXPERT BOT
# ==========================================
def _get_expert_action_internal(char):
    """
    Internal phase machine. Only returns integer actions (0, 1, 2, 3).
    """
    p1, p2, p3 = char.platforms
    W = char.rect.width

    def on_platform(p, y_tol=5):
        if not char.on_ground: return False
        overlap = (char.rect.right > p.rect.left + 2) and (char.rect.left < p.rect.right - 2)
        return overlap and abs(char.rect.bottom - p.rect.top) <= y_tol

    def move_toward_x(target_x, tol=15): 
        if char.rect.x < target_x - tol: return 1
        if char.rect.x > target_x + tol: return 0
        return 2

    def stop_then_jump():
        if abs(char.speed_x) > 0: return 2
        return 3

    if not hasattr(_get_expert_action_internal, "phase"):
        _get_expert_action_internal.phase = 0

    if char.on_ground and char.rect.y >= 470:
        _get_expert_action_internal.phase = 0

    if on_platform(p1): _get_expert_action_internal.phase = max(_get_expert_action_internal.phase, 1)
    if on_platform(p2): _get_expert_action_internal.phase = max(_get_expert_action_internal.phase, 2)
    if on_platform(p3): _get_expert_action_internal.phase = max(_get_expert_action_internal.phase, 3)

    phase = _get_expert_action_internal.phase

    if phase == 0:
        jump_x = p1.rect.right + 20 
        if char.on_ground:
            a = move_toward_x(jump_x)
            if a != 2: return a
            return stop_then_jump()

        if char.rect.bottom > p1.rect.top and char.rect.x > p1.rect.right:
            return 2 
            
        target_x = p1.rect.centerx - W // 2
        return move_toward_x(target_x)

    if phase == 1:
        jump_x = p1.rect.right - W - 10 
        if char.on_ground:
            a = move_toward_x(jump_x)
            if a != 2: return a
            return stop_then_jump()

        if char.rect.bottom > p2.rect.top and char.rect.x < p2.rect.left:
            return 2
            
        target_x = p2.rect.centerx - W // 2
        return move_toward_x(target_x)

    if phase == 2:
        jump_x = p2.rect.right - W - 10
        if char.on_ground:
            a = move_toward_x(jump_x)
            if a != 2: return a
            return stop_then_jump()

        if char.rect.bottom > p3.rect.top and char.rect.x < p3.rect.left:
            return 2
            
        target_x = p3.rect.centerx - W // 2
        return move_toward_x(target_x)

    target_x = (695+12) - W // 2
    return move_toward_x(target_x, tol=2)

def get_expert_action(char, noise_prob=0.3):
    """
    Wrapper function to handle noise injection safely.
    Always returns a tuple: (action_int, is_noise_bool)
    """
    if random.random() < noise_prob:
        return random.choice([0, 1, 2]), True  # Noisy action
        
    expert_action = _get_expert_action_internal(char)
    return expert_action, False  # Clean expert action
# ==========================================
# 3. RECORDING LOOP
# ==========================================
def main():
    # --- DEBUG TOGGLE ---
    DEBUG_RENDER = False   # Set to False for lightning-fast dataset generation
    DEBUG_FPS = 60        # Slow it down to 15 FPS so you can watch clearly
    # --------------------

    save_dir = "expert_dataset"
    frames_dir = os.path.join(save_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    env = ColorfulPlatformerEnv(action_repeat=4)
    
    num_episodes = 1000
    total_frames = 0
    
    actions_record = []
    episodes_record = [] 
    
    print(f"Generating {num_episodes} expert episodes...")
    
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock() # Initialize the clock
    
    for ep in tqdm(range(num_episodes)):
        frame = env.reset()
        done = False
        
        ep_start_idx = total_frames
        
        frame_uint8 = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{total_frames:05d}.png"), cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
        
        timeout_counter = 0
        while not done:
            if DEBUG_RENDER:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            else:
                pygame.event.pump() 

            # Catch the tuple
            action, is_noise = get_expert_action(env.character)
            frame, done = env.step(action)
            
            # Log both the action and the noise flag (1 for True, 0 for False)
            actions_record.append(f"{action},{int(is_noise)}")
            
            total_frames += 1
            frame_uint8 = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(frames_dir, f"frame_{total_frames:05d}.png"), cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
            
            if DEBUG_RENDER:
                screen.blit(env.screen, (0, 0))
                pygame.display.flip()
                clock.tick(DEBUG_FPS)
                
            timeout_counter += 1
            if timeout_counter > 200:
                print(f"Timeout on episode {ep}. Bot got stuck.")
                break
                
        ep_end_idx = total_frames
        episodes_record.append(f"{ep_start_idx},{ep_end_idx}")

    # Save Actions with Headers
    with open(os.path.join(save_dir, 'actions.txt'), 'w') as f:
        f.write("action,is_noise\n")
        for a in actions_record:
            f.write(f"{a}\n")
            
    with open(os.path.join(save_dir, 'episodes.txt'), 'w') as f:
        f.write("start_idx,end_idx\n")
        for ep in episodes_record:
            f.write(f"{ep}\n")
            
    print(f"\nDone! Saved {total_frames} total frames across {num_episodes} episodes.")
    pygame.quit()

if __name__ == "__main__":
    main()