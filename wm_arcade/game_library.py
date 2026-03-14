import numpy as np
import pygame


class PlatformerEnv:
    def __init__(self, height=64, width=80, ball_size=5):
        self.height = height
        self.width = width
        self.ball_size = ball_size
        self.num_actions = 4  # left, right, stay, jump
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

        self.vy = min(self.vy + 1, 4)
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
        x0 = int(self.x)
        y0 = int(self.y)
        frame[0, y0:y0 + self.ball_size, x0:x0 + self.ball_size] = 1.0
        return frame

    def get_action(self, keys):
        action = 2  # stay
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = 0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = 1
        if (keys[pygame.K_w] or keys[pygame.K_UP]) and self.on_ground:
            action = 3
        return action

    def get_instructions(self):
        return "Controls: 'W'=Jump, 'A'=Left, 'D'=Right. Default is Stay."


class PongEnv:
    def __init__(
        self,
        height=64,
        width=80,
        ball_size=5,
        paddle_height=16,
        paddle_width=4,
        miss_behavior="tile",  # "tile" or "respawn"
    ):
        self.height = height
        self.width = width
        self.ball_size = ball_size
        self.paddle_height = paddle_height
        self.paddle_width = paddle_width
        self.speed = 2
        self.num_actions = 3  # up, down, stay

        if miss_behavior not in ("tile", "respawn"):
            raise ValueError("miss_behavior must be 'tile' or 'respawn'")
        self.miss_behavior = miss_behavior

        self.reset()

    def _spawn_ball_random(self, force_right_to_left=False):
        self.ball_x = self.width // 2
        self.ball_y = np.random.randint(0, self.height - self.ball_size + 1)

        if force_right_to_left:
            self.vx = -self.speed
        else:
            self.vx = self.speed if np.random.rand() > 0.5 else -self.speed

        self.vy = self.speed if np.random.rand() > 0.5 else -self.speed

    def reset(self):
        self.paddle_y = (self.height - self.paddle_height) // 2
        self._spawn_ball_random()
        return self._get_frame()

    def step(self, action):
        if action == 0:
            self.paddle_y -= 4
        elif action == 1:
            self.paddle_y += 4

        self.paddle_y = max(0, min(self.height - self.paddle_height, self.paddle_y))

        self.ball_x += self.vx
        self.ball_y += self.vy

        # Top / bottom bounce
        if self.ball_y <= 0:
            self.ball_y = 0
            self.vy = -self.vy
        elif self.ball_y >= self.height - self.ball_size:
            self.ball_y = self.height - self.ball_size
            self.vy = -self.vy

        # Right wall bounce
        if self.ball_x >= self.width - self.ball_size:
            self.ball_x = self.width - self.ball_size
            self.vx = -self.vx

        # Paddle collision test only while ball is moving left and overlapping paddle x-range
        if self.vx < 0 and self.ball_x <= self.paddle_width:
            overlaps_paddle = (
                self.ball_y + self.ball_size >= self.paddle_y
                and self.ball_y <= self.paddle_y + self.paddle_height
            )

            if overlaps_paddle:
                self.ball_x = self.paddle_width
                self.vx = -self.vx

        # If the ball has fully exited the left side, apply miss behavior
        if self.ball_x + self.ball_size <= 0:
            if self.miss_behavior == "tile":
                # Preserve continuity: if ball_x is -ball_size, it reappears at width-ball_size
                self.ball_x += self.width
            else:  # respawn
                self._spawn_ball_random(force_right_to_left=True)

        return self._get_frame()

    def _get_frame(self):
        frame = np.zeros((1, self.height, self.width), dtype=np.float32)

        py = int(self.paddle_y)
        bx = int(self.ball_x)
        by = int(self.ball_y)

        # Paddle
        frame[0, py:py + self.paddle_height, 0:self.paddle_width] = 1.0

        # Ball with clipped drawing so partial off-screen positions still render correctly
        x1 = max(0, bx)
        x2 = min(self.width, bx + self.ball_size)
        y1 = max(0, by)
        y2 = min(self.height, by + self.ball_size)

        if x1 < x2 and y1 < y2:
            frame[0, y1:y2, x1:x2] = 1.0

        return frame

    def get_action(self, keys):
        action = 2  # stay
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action = 0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = 1
        return action

    def get_instructions(self):
        mode_text = (
            "Misses wrap from left edge to right edge (deterministic)."
            if self.miss_behavior == "tile"
            else "Misses randomly respawn the ball."
        )
        return f"Controls: 'W'=Up, 'S'=Down. Default is Stay. {mode_text}"


class SpaceInvadersEnv:
    """
    Simplified black-and-white Space Invaders.
    Actions:
      0 = left
      1 = right
      2 = stay
      3 = shoot
      4 = left + shoot
      5 = right + shoot
    """
    def __init__(self, height=64, width=80):
        self.height = height
        self.width = width
        self.num_actions = 6

        # Player
        self.player_w = 7
        self.player_h = 3
        self.player_speed = 3

        # Player bullets
        self.player_bullet_w = 1
        self.player_bullet_h = 3
        self.player_bullet_speed = 4
        self.player_cooldown_frames = 6

        # Invaders
        self.enemy_w = 5
        self.enemy_h = 3
        self.enemy_rows = 3
        self.enemy_cols = 6
        self.enemy_gap_x = 3
        self.enemy_gap_y = 4
        self.enemy_move_interval_base = 5
        self.enemy_drop = 3

        # Enemy bullets
        self.enemy_bullet_w = 1
        self.enemy_bullet_h = 3
        self.enemy_bullet_speed = 2
        self.max_enemy_bullets = 2
        self.enemy_fire_prob = 0.08

        self.reset()

    def reset(self):
        self.player_x = (self.width - self.player_w) // 2
        self.player_y = self.height - 6

        self.player_bullets = []
        self.enemy_bullets = []
        self.player_cooldown = 0

        self.wave = 0
        self.score = 0
        self.tick = 0

        self._spawn_wave()
        return self._get_frame()

    def _spawn_wave(self):
        total_w = self.enemy_cols * self.enemy_w + (self.enemy_cols - 1) * self.enemy_gap_x
        start_x = (self.width - total_w) // 2
        start_y = 6

        self.invaders = []
        for r in range(self.enemy_rows):
            for c in range(self.enemy_cols):
                x = start_x + c * (self.enemy_w + self.enemy_gap_x)
                y = start_y + r * (self.enemy_h + self.enemy_gap_y)
                self.invaders.append([x, y])

        self.invader_dx = 1
        self.enemy_move_interval = max(2, self.enemy_move_interval_base - min(self.wave, 3))

    @staticmethod
    def _rect_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
        return (
            x1 < x2 + w2 and
            x1 + w1 > x2 and
            y1 < y2 + h2 and
            y1 + h1 > y2
        )

    def _choose_bottom_invader(self):
        if not self.invaders:
            return None

        # Choose bottom-most invader from a random column
        cols = {}
        for x, y in self.invaders:
            if x not in cols or y > cols[x][1]:
                cols[x] = [x, y]

        choices = list(cols.values())
        return choices[np.random.randint(len(choices))]

    def step(self, action):
        # --------------------
        # 1. Player movement
        # --------------------
        if action in (0, 4):
            self.player_x -= self.player_speed
        elif action in (1, 5):
            self.player_x += self.player_speed

        self.player_x = max(0, min(self.width - self.player_w, self.player_x))

        # --------------------
        # 2. Player shooting
        # --------------------
        if self.player_cooldown > 0:
            self.player_cooldown -= 1

        wants_shoot = action in (3, 4, 5)
        if wants_shoot and self.player_cooldown == 0 and len(self.player_bullets) < 1:
            bullet_x = self.player_x + self.player_w // 2
            bullet_y = self.player_y - self.player_bullet_h
            self.player_bullets.append([bullet_x, bullet_y])
            self.player_cooldown = self.player_cooldown_frames

        # --------------------
        # 3. Move player bullets
        # --------------------
        new_player_bullets = []
        for bx, by in self.player_bullets:
            by -= self.player_bullet_speed
            if by + self.player_bullet_h >= 0:
                new_player_bullets.append([bx, by])
        self.player_bullets = new_player_bullets

        # --------------------
        # 4. Move invaders
        # --------------------
        self.tick += 1
        if self.invaders and self.tick % self.enemy_move_interval == 0:
            left_edge = min(x for x, _ in self.invaders)
            right_edge = max(x + self.enemy_w for x, _ in self.invaders)

            hit_edge = (right_edge + self.invader_dx > self.width) or (left_edge + self.invader_dx < 0)

            if hit_edge:
                self.invader_dx = -self.invader_dx
                for inv in self.invaders:
                    inv[1] += self.enemy_drop
            else:
                for inv in self.invaders:
                    inv[0] += self.invader_dx

        # --------------------
        # 5. Enemy shooting
        # --------------------
        if self.invaders and len(self.enemy_bullets) < self.max_enemy_bullets and np.random.rand() < self.enemy_fire_prob:
            shooter = self._choose_bottom_invader()
            if shooter is not None:
                sx, sy = shooter
                self.enemy_bullets.append([sx + self.enemy_w // 2, sy + self.enemy_h])

        # --------------------
        # 6. Move enemy bullets
        # --------------------
        new_enemy_bullets = []
        for bx, by in self.enemy_bullets:
            by += self.enemy_bullet_speed
            if by < self.height:
                new_enemy_bullets.append([bx, by])
        self.enemy_bullets = new_enemy_bullets

        # --------------------
        # 7. Player bullet hits invader
        # --------------------
        hit_invader_indices = set()
        surviving_player_bullets = []

        for bx, by in self.player_bullets:
            hit = False
            for i, (ix, iy) in enumerate(self.invaders):
                if self._rect_overlap(
                    bx, by, self.player_bullet_w, self.player_bullet_h,
                    ix, iy, self.enemy_w, self.enemy_h
                ):
                    hit = True
                    hit_invader_indices.add(i)
                    self.score += 1
                    break

            if not hit:
                surviving_player_bullets.append([bx, by])

        self.player_bullets = surviving_player_bullets
        if hit_invader_indices:
            self.invaders = [
                inv for i, inv in enumerate(self.invaders)
                if i not in hit_invader_indices
            ]

        # --------------------
        # 8. Enemy bullet hits player -> reset
        # --------------------
        for bx, by in self.enemy_bullets:
            if self._rect_overlap(
                bx, by, self.enemy_bullet_w, self.enemy_bullet_h,
                self.player_x, self.player_y, self.player_w, self.player_h
            ):
                return self.reset()

        # --------------------
        # 9. Invaders reach player line -> reset
        # --------------------
        if self.invaders:
            lowest_enemy = max(y + self.enemy_h for _, y in self.invaders)
            if lowest_enemy >= self.player_y:
                return self.reset()

        # --------------------
        # 10. Clear wave -> spawn next wave
        # --------------------
        if not self.invaders:
            self.wave += 1
            self.player_bullets = []
            self.enemy_bullets = []
            self._spawn_wave()

        return self._get_frame()

    def _get_frame(self):
        frame = np.zeros((1, self.height, self.width), dtype=np.float32)

        # Player
        px = int(self.player_x)
        py = int(self.player_y)
        frame[0, py:py + self.player_h, px:px + self.player_w] = 1.0

        # Invaders
        for ix, iy in self.invaders:
            ix = int(ix)
            iy = int(iy)
            if 0 <= iy < self.height and 0 <= ix < self.width:
                y1 = max(0, iy)
                y2 = min(self.height, iy + self.enemy_h)
                x1 = max(0, ix)
                x2 = min(self.width, ix + self.enemy_w)
                frame[0, y1:y2, x1:x2] = 1.0

        # Player bullets
        for bx, by in self.player_bullets:
            bx = int(bx)
            by = int(by)
            if 0 <= bx < self.width:
                y1 = max(0, by)
                y2 = min(self.height, by + self.player_bullet_h)
                if y1 < y2:
                    frame[0, y1:y2, bx:bx + self.player_bullet_w] = 1.0

        # Enemy bullets
        for bx, by in self.enemy_bullets:
            bx = int(bx)
            by = int(by)
            if 0 <= bx < self.width:
                y1 = max(0, by)
                y2 = min(self.height, by + self.enemy_bullet_h)
                if y1 < y2:
                    frame[0, y1:y2, bx:bx + self.enemy_bullet_w] = 1.0

        return frame

    def get_action(self, keys):
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        shoot = keys[pygame.K_SPACE]

        if left and not right and shoot:
            return 4
        if right and not left and shoot:
            return 5
        if left and not right:
            return 0
        if right and not left:
            return 1
        if shoot:
            return 3
        return 2

    def get_instructions(self):
        return "Controls: 'A/D'=Move, 'SPACE'=Shoot. You can move and shoot together."


# The Arcade Registry
ENV_REGISTRY = {
    "platformer": PlatformerEnv,
    "pong": PongEnv,
    "space_invaders": SpaceInvadersEnv,
}