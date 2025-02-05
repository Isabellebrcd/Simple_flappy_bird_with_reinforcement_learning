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
PIPE_IMG = load_image("pipe.png")
PIPE_IMG_INV = pygame.transform.flip(PIPE_IMG, False, True)
HEART_IMG = load_image("heart.png", (30, 30))

# Chargement de la police
pygame.font.init()
font = pygame.font.SysFont("Arial", 30)

# Q-Learning

STATE_BINS = (20, 20, 20)
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_PROB = 1.0
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.01
ACTIONS = [0, 1]  # 0: ne rien faire, 1: sauter

# Chargement du Q-table
Q_TABLE_FILE = "q_table.pkl"
try:
    with open(Q_TABLE_FILE, "rb") as f:
        Q_table = pickle.load(f)
except FileNotFoundError:
    Q_table = np.zeros(STATE_BINS + (len(ACTIONS),))


def discretize_state(bird_y, pipe_x, pipe_y):

    pipe_gap_center = pipe_y + PIPE_GAP / 2
    bird_bin = min(STATE_BINS[0] - 1, max(0, int(bird_y / HEIGHT * STATE_BINS[0])))
    pipe_x_bin = min(STATE_BINS[1] - 1, max(0, int(pipe_x / WIDTH * STATE_BINS[1])))
    pipe_gap_bin = min(STATE_BINS[2] - 1, max(0, int(pipe_gap_center / HEIGHT * STATE_BINS[2])))
    return (bird_bin, pipe_x_bin, pipe_gap_bin)


class FlappyBird:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_velocity = 0
        self.pipe_x = WIDTH
        self.pipe_y = random.randint(50, HEIGHT - PIPE_GAP - 50)
        self.score = 0
        self.lives = LIVES
        return discretize_state(self.bird_y, self.pipe_x, self.pipe_y)

    def check_collision(self):
        bird_rect = pygame.Rect(BIRD_X, self.bird_y, 50, 35)
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
        # Action : sauter ou ne rien faire
        if action == 1:
            self.bird_velocity = JUMP_STRENGTH

        # Mise à jour de la physique
        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        # Déplacement des tuyaux
        self.pipe_x -= PIPE_SPEED
        pipe_passed = False
        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = WIDTH
            self.pipe_y = random.randint(50, HEIGHT - PIPE_GAP - 50)
            self.score += 1
            pipe_passed = True

        # Vérification de collision
        if self.check_collision():
            self.lives -= 1
            collision = True
            if self.lives > 0:
                # Réinitialisation partielle après collision (l'oiseau repart du centre)
                self.bird_y = HEIGHT // 2
                self.bird_velocity = 0
                self.pipe_x = WIDTH
                self.pipe_y = random.randint(50, HEIGHT - PIPE_GAP - 50)

        # Attribution de la récompense
        if collision:
            reward = -20
        else:
            reward = 1
            if pipe_passed:
                reward += 10  # Récompense pour avoir passé un tuyau

        done = self.lives <= 0

        return discretize_state(self.bird_y, self.pipe_x, self.pipe_y), reward, done

    def render(self):
        screen.blit(BACKGROUND_IMG, (0, 0))
        screen.blit(BIRD_IMG, (BIRD_X, self.bird_y))

        # Tuyau supérieur (redimensionnement dynamique)
        upper_pipe_height = self.pipe_y
        scaled_upper = pygame.transform.scale(PIPE_IMG, (PIPE_WIDTH, upper_pipe_height))
        scaled_upper_inv = pygame.transform.flip(scaled_upper, False, True)
        screen.blit(scaled_upper_inv, (self.pipe_x, 0))

        # Tuyau inférieur (redimensionnement dynamique)
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
num_episodes = 3000

for episode in range(num_episodes):
    state = game.reset()
    done = False
    while not done:
        pygame.event.pump()
        # Choix de l'action : exploration ou exploitation
        if random.uniform(0, 1) < EXPLORATION_PROB:
            action = random.choice(ACTIONS)
        else:
            action = int(np.argmax(Q_table[state]))

        new_state, reward, done = game.step(action)

        # Mise à jour de la Q-table
        Q_table[state][action] = (1 - LEARNING_RATE) * Q_table[state][action] + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * np.max(Q_table[new_state])
        )
        state = new_state

    # Décroissance de la probabilité d'exploration
    EXPLORATION_PROB = max(EXPLORATION_MIN, EXPLORATION_PROB * EXPLORATION_DECAY)

    # Afficher le score tous les 100 épisodes pour suivre l'entraînement
    if episode % 100 == 0:
        print(f"Episode {episode} - Score: {game.score} - Exploration: {EXPLORATION_PROB:.3f}")

# Sauvegarde du Q-table
with open(Q_TABLE_FILE, "wb") as f:
    pickle.dump(Q_table, f)

# Mode jeu (exploitation uniquement)
running = True
game = FlappyBird()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Choix de l'action selon la Q-table apprise
    current_state = discretize_state(game.bird_y, game.pipe_x, game.pipe_y)
    action = int(np.argmax(Q_table[current_state]))
    _, _, done = game.step(action)
    game.render()

    if done:
        print("Game Over! Score:", game.score)
        game.reset()

pygame.quit()
