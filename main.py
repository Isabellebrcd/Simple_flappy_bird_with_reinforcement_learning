import pygame
import random
import numpy as np
import pickle
import os

# Initialisation de Pygame
pygame.init()

# Paramètres du jeu
WIDTH, HEIGHT = 400, 600
BIRD_X = 50
GRAVITY = 0.8
JUMP_STRENGTH = -10
PIPE_GAP = 220
PIPE_WIDTH = 60
PIPE_SPEED = 2.5
FPS = 30
LIVES = 3

# Initialisation de l'écran
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


# Chargement des images
def load_image(path, default_size=None):
    if os.path.exists(path):
        img = pygame.image.load(path).convert_alpha()
        if default_size:
            img = pygame.transform.scale(img, default_size)
        return img
    else:
        print(f"Image introuvable : {path}")
        return pygame.Surface((50, 50))


BACKGROUND_IMG = load_image("background.png", (WIDTH, HEIGHT))
BIRD_IMG = load_image("bird.png", (50, 35))
PIPE_IMG = load_image("pipe.png")  # Chargement sans redimensionnement initial
PIPE_IMG_INV = pygame.transform.flip(PIPE_IMG, False, True)
HEART_IMG = load_image("heart.png", (30, 30))

# Chargement de la police
pygame.font.init()
font = pygame.font.SysFont("Arial", 30)

# Q-Learning
STATE_BINS = (10, 10)
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_PROB = 1.0
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.01
ACTIONS = [0, 1]

# Chargement du Q-table
Q_TABLE_FILE = "q_table.pkl"
try:
    with open(Q_TABLE_FILE, "rb") as f:
        Q_table = pickle.load(f)
except FileNotFoundError:
    Q_table = np.zeros(STATE_BINS + (len(ACTIONS),))


def discretize_state(bird_y, pipe_x, pipe_y):
    return (
        min(STATE_BINS[0] - 1, max(0, int(bird_y / HEIGHT * STATE_BINS[0]))),
        min(STATE_BINS[1] - 1, max(0, int(pipe_x / WIDTH * STATE_BINS[1]))),
    )


class FlappyBird:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_velocity = 0
        self.pipe_x = WIDTH
        self.pipe_y = random.randint(50, HEIGHT - PIPE_GAP - 50)  # Plage élargie
        self.score = 0
        self.lives = LIVES
        return discretize_state(self.bird_y, self.pipe_x, self.pipe_y)

    def check_collision(self):
        bird_rect = pygame.Rect(BIRD_X, self.bird_y, 50, 35)

        # Rectangles de collision ajustés
        pipe_rect_top = pygame.Rect(self.pipe_x, 0, PIPE_WIDTH, self.pipe_y)
        pipe_rect_bottom = pygame.Rect(self.pipe_x,
                                       self.pipe_y + PIPE_GAP,
                                       PIPE_WIDTH,
                                       HEIGHT - (self.pipe_y + PIPE_GAP))

        return (bird_rect.colliderect(pipe_rect_top) or
                bird_rect.colliderect(pipe_rect_bottom) or
                self.bird_y < 0 or
                self.bird_y > HEIGHT - 35)

    def step(self, action):
        collision = False
        if action == 1:
            self.bird_velocity = JUMP_STRENGTH

        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        # Déplacement et régénération des tuyaux
        self.pipe_x -= PIPE_SPEED
        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = WIDTH
            self.pipe_y = random.randint(50, HEIGHT - PIPE_GAP - 50)
            self.score += 1

        if self.check_collision():
            self.lives -= 1
            collision = True
            if self.lives > 0:
                self.bird_y = HEIGHT // 2
                self.bird_velocity = 0
                self.pipe_x = WIDTH
                self.pipe_y = random.randint(50, HEIGHT - PIPE_GAP - 50)

        reward = 2 if not collision else -20
        done = self.lives <= 0

        return discretize_state(self.bird_y, self.pipe_x, self.pipe_y), reward, done

    def render(self):
        screen.blit(BACKGROUND_IMG, (0, 0))
        screen.blit(BIRD_IMG, (BIRD_X, self.bird_y))

        # Tuyau supérieur avec redimensionnement dynamique
        upper_pipe_height = self.pipe_y
        scaled_upper = pygame.transform.scale(PIPE_IMG, (PIPE_WIDTH, upper_pipe_height))
        scaled_upper_inv = pygame.transform.flip(scaled_upper, False, True)
        screen.blit(scaled_upper_inv, (self.pipe_x, 0))

        # Tuyau inférieur avec redimensionnement dynamique
        lower_pipe_height = HEIGHT - (self.pipe_y + PIPE_GAP)
        scaled_lower = pygame.transform.scale(PIPE_IMG, (PIPE_WIDTH, lower_pipe_height))
        screen.blit(scaled_lower, (self.pipe_x, self.pipe_y + PIPE_GAP))

        # Affichage des vies
        for i in range(self.lives):
            screen.blit(HEART_IMG, (WIDTH - 40 - (i * 35), 10))

        # Affichage du score
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)


# Entraînement
game = FlappyBird()
for episode in range(3000):
    state = game.reset()
    done = False
    while not done:
        pygame.event.pump()
        if random.uniform(0, 1) < EXPLORATION_PROB:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(Q_table[state])

        new_state, reward, done = game.step(action)
        Q_table[state][action] = (1 - LEARNING_RATE) * Q_table[state][action] + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * np.max(Q_table[new_state])
        )
        state = new_state

    EXPLORATION_PROB = max(EXPLORATION_MIN, EXPLORATION_PROB * EXPLORATION_DECAY)

# Sauvegarde du Q-table
with open(Q_TABLE_FILE, "wb") as f:
    pickle.dump(Q_table, f)

# Mode jeu
running = True
game = FlappyBird()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = np.argmax(Q_table[discretize_state(game.bird_y, game.pipe_x, game.pipe_y)])
    _, _, done = game.step(action)
    game.render()

    if done:
        print("Game Over! Score:", game.score)
        game.reset()

pygame.quit()