import os
import numpy as np
import cv2
import pygame

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
        # IMPORTANT: INTER_NEAREST prevents blurry gradient tokens!
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

def main():
    pygame.init()
    render_width, render_height = 512, 512
    screen = pygame.display.set_mode((render_width, render_height))
    pygame.display.set_caption('Data Recorder (Press R to toggle)')
    clock = pygame.time.Clock()
    
    env = ColorfulPlatformerEnv(action_repeat=4)
    env.reset()
    
    os.makedirs('dataset/frames', exist_ok=True)
    action_log_path = 'dataset/actions.txt'
    recording = False
    recorded_frames = 0
    r_key_down = False
    
    print("Controls: 'A'=Left, 'D'=Right, 'W'=Jump, 'S'=Still.")
    print("Press 'R' to toggle recording ON/OFF.")
    
    open(action_log_path, 'w').close()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]: running = False
        
        if keys[pygame.K_r] and not r_key_down:
            recording = not recording
            if recording:
                print(f"🔴 Recording STARTED... (Current frames: {recorded_frames})")
            else:
                print(f"⏹️ Recording PAUSED. Total frames: {recorded_frames}")
        r_key_down = keys[pygame.K_r]
            
        action = 2 
        if keys[pygame.K_a]: action = 0 
        if keys[pygame.K_d]: action = 1 
        if keys[pygame.K_w]: action = 3 
            
        frame = env.step(action)
        
        if recording:
            img_bgr = cv2.cvtColor((np.clip(frame.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'dataset/frames/frame_{recorded_frames:05d}.png', img_bgr)
            with open(action_log_path, 'a') as f:
                f.write(f"{action}\n")
            recorded_frames += 1
        
        img_rgb = (np.clip(frame.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        img_upscaled = cv2.resize(img_rgb, (render_width, render_height), interpolation=cv2.INTER_NEAREST)
        
        if recording:
            cv2.circle(img_upscaled, (30, 30), 10, (255, 0, 0), -1)
            cv2.putText(img_upscaled, f"REC: {recorded_frames}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        screen.blit(pygame.surfarray.make_surface(np.transpose(img_upscaled, (1, 0, 2))), (0, 0))
        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()
    print(f"\nFinal dataset contains {recorded_frames} frames.")

if __name__ == '__main__':
    main()