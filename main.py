import pygame
import random
import numpy as np
import pickle
import os

# Initialize Pygame
pygame.init()

# Game settings
WIDTH, HEIGHT = 400, 600
BIRD_X = 50
GRAVITY = 0.8
JUMP_STRENGTH = -10
PIPE_WIDTH = 60
PIPE_SPEED = 2.5
FPS = 30
LIVES = 3

# Game Over display time (in milliseconds)
GAME_OVER_WAIT_TIME = 3000  # 3 seconds

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Load images
def load_image(path, default_size=None):
    if os.path.exists(path):
        img = pygame.image.load(path).convert_alpha()
        if default_size:
            img = pygame.transform.scale(img, default_size)
        return img
    else:
        print(f"Image not found: {path}")
        return pygame.Surface((50, 50))

BACKGROUND_IMG = load_image("background.png", (WIDTH, HEIGHT))
BIRD_IMG = load_image("bird.png", (50, 35))
PIPE_IMG = load_image("pipe.png")
PIPE_IMG_INV = pygame.transform.flip(PIPE_IMG, False, True)
HEART_IMG = load_image("heart.png", (30, 30))

# Load fonts (using Arial)
pygame.font.init()
game_over_font = pygame.font.SysFont("Arial", 50, bold=True)
default_font = pygame.font.SysFont("Arial", 30)

# Q-Learning
STATE_BINS = (20, 20, 20)
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_PROB = 1.0
EXPLORATION_DECAY = 0.998
EXPLORATION_MIN = 0.01
ACTIONS = [0, 1]  # 0: do nothing, 1: jump

# Q-table save file
Q_TABLE_FILE = "q_table.pkl"
try:
    with open(Q_TABLE_FILE, "rb") as f:
        Q_table = pickle.load(f)
except FileNotFoundError:
    Q_table = np.zeros(STATE_BINS + (len(ACTIONS),))

def discretize_state(bird_y, pipe_x, pipe_gap, pipe_y):
    pipe_gap_center = pipe_y + pipe_gap / 2
    bird_bin = min(STATE_BINS[0] - 1, max(0, int(bird_y / HEIGHT * STATE_BINS[0])))
    pipe_x_bin = min(STATE_BINS[1] - 1, max(0, int(pipe_x / WIDTH * STATE_BINS[1])))
    pipe_gap_bin = min(STATE_BINS[2] - 1, max(0, int(pipe_gap_center / HEIGHT * STATE_BINS[2])))
    return (bird_bin, pipe_x_bin, pipe_gap_bin)

def show_game_over(score):
    # Light blue background
    screen.fill((173, 216, 230))

    # Basic position (adjust as needed)
    base_y = HEIGHT // 2 - 150  # Start higher for more space

    # Display score
    score_text = default_font.render(f"Score: {score}", True, (255, 255, 255))
    score_rect = score_text.get_rect(center=(WIDTH // 2, base_y))
    screen.blit(score_text, score_rect)

    # Display "GAME OVER" text with a 60-pixel gap below the score
    game_over_text = game_over_font.render("GAME OVER", True, (0, 0, 0))
    game_over_rect = game_over_text.get_rect(center=(WIDTH // 2, base_y + 100))
    screen.blit(game_over_text, game_over_rect)

    # Display "Flappy bird" text with a 60-pixel gap below "GAME OVER"
    fb_text = default_font.render("Flappy bird", True, (255, 255, 255))
    fb_rect = fb_text.get_rect(center=(WIDTH // 2, base_y + 150))
    screen.blit(fb_text, fb_rect)

    # Display the image (bird.png) with an 80-pixel gap below "Flappy bird"
    if os.path.exists("bird.png"):
        bird_img = pygame.image.load("bird.png").convert_alpha()
        bird_img = pygame.transform.scale(bird_img, (100, 100))
        bird_rect = bird_img.get_rect(center=(WIDTH // 2, base_y + 250))
        screen.blit(bird_img, bird_rect)

    pygame.display.flip()
    pygame.time.wait(GAME_OVER_WAIT_TIME)

class FlappyBird:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_velocity = 0
        self.pipe_x = WIDTH
        # Random vertical pipe position considering the gap
        self.pipe_y = random.randint(50, HEIGHT - 300)
        self.pipe_gap = random.randint(150, 300)
        self.score = 0
        self.lives = LIVES
        return discretize_state(self.bird_y, self.pipe_x, self.pipe_gap, self.pipe_y)

    def check_collision(self):
        bird_rect = pygame.Rect(BIRD_X, self.bird_y, 50, 35)
        pipe_rect_top = pygame.Rect(self.pipe_x, 0, PIPE_WIDTH, self.pipe_y)
        pipe_rect_bottom = pygame.Rect(self.pipe_x,
                                       self.pipe_y + self.pipe_gap,
                                       PIPE_WIDTH,
                                       HEIGHT - (self.pipe_y + self.pipe_gap))
        return (bird_rect.colliderect(pipe_rect_top) or
                bird_rect.colliderect(pipe_rect_bottom) or
                self.bird_y < 0 or
                self.bird_y > HEIGHT - 35)

    def step(self, action):
        collision = False
        # Action: jump or do nothing
        if action == 1:
            self.bird_velocity = JUMP_STRENGTH

        # Physics update
        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        # Move pipes
        self.pipe_x -= PIPE_SPEED
        pipe_passed = False
        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = WIDTH
            self.pipe_y = random.randint(50, HEIGHT - 300)
            self.pipe_gap = random.randint(150, 300)
            self.score += 1
            pipe_passed = True

        # Collision check
        if self.check_collision():
            self.lives -= 1
            collision = True
            if self.lives > 0:
                # Partial reset after collision
                self.bird_y = HEIGHT // 2
                self.bird_velocity = 0
                self.pipe_x = WIDTH
                self.pipe_y = random.randint(50, HEIGHT - 300)
                self.pipe_gap = random.randint(150, 300)

        # Reward assignment
        if collision:
            reward = -25
        else:
            reward = 1
            if pipe_passed:
                reward += 15  # Reward for passing a pipe

        done = self.lives <= 0

        return discretize_state(self.bird_y, self.pipe_x, self.pipe_gap, self.pipe_y), reward, done

    def render(self):
        # Display background and bird
        screen.blit(BACKGROUND_IMG, (0, 0))
        screen.blit(BIRD_IMG, (BIRD_X, self.bird_y))

        # Upper pipe (dynamic resizing)
        upper_pipe_height = self.pipe_y
        scaled_upper = pygame.transform.scale(PIPE_IMG, (PIPE_WIDTH, upper_pipe_height))
        scaled_upper_inv = pygame.transform.flip(scaled_upper, False, True)
        screen.blit(scaled_upper_inv, (self.pipe_x, 0))

        # Lower pipe
        lower_pipe_height = HEIGHT - (self.pipe_y + self.pipe_gap)
        scaled_lower = pygame.transform.scale(PIPE_IMG, (PIPE_WIDTH, lower_pipe_height))
        screen.blit(scaled_lower, (self.pipe_x, self.pipe_y + self.pipe_gap))

        # Display lives
        for i in range(self.lives):
            screen.blit(HEART_IMG, (WIDTH - 40 - (i * 35), 10))

        # Display score
        score_text = default_font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

# ----------------------------
# Training phase (Q-learning)
# ----------------------------
game = FlappyBird()
num_episodes = 70000
print("Training started...")

for episode in range(num_episodes):
    state = game.reset()
    done = False
    while not done:
        pygame.event.pump()  # Process events
        # Choose action: exploration or exploitation
        if random.uniform(0, 1) < EXPLORATION_PROB:
            action = random.choice(ACTIONS)
        else:
            action = int(np.argmax(Q_table[state]))

        new_state, reward, done = game.step(action)

        # Update Q-table
        Q_table[state][action] = (1 - LEARNING_RATE) * Q_table[state][action] + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * np.max(Q_table[new_state])
        )
        state = new_state

    # Exploration probability decay
    EXPLORATION_PROB = max(EXPLORATION_MIN, EXPLORATION_PROB * EXPLORATION_DECAY)

    # Display score every 100 episodes
    if episode % 100 == 0:
        print(f"Episode {episode} - Score: {game.score} - Exploration: {EXPLORATION_PROB:.3f}")

# Save Q-table
with open(Q_TABLE_FILE, "wb") as f:
    pickle.dump(Q_table, f)
print("Training completed. Q-table saved.")

# ----------------------------
# Game mode (exploitation only)
# ----------------------------
running = True
game = FlappyBird()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Choose action based on learned Q-table
    current_state = discretize_state(game.bird_y, game.pipe_x, game.pipe_gap, game.pipe_y)
    action = int(np.argmax(Q_table[current_state]))
    _, _, done = game.step(action)
    game.render()

    if done:
        # Display Game Over screen with the score, "GAME OVER" text,
        # "Flappy bird" text, and the image
        show_game_over(game.score)
        game.reset()

pygame.quit()
