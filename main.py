import pygame
import random
import numpy as np
import pickle

# Initialisation de Pygame
pygame.init()

# Paramètres du jeu
WIDTH, HEIGHT = 400, 600
BIRD_X = 50
GRAVITY = 1
JUMP_STRENGTH = -10
PIPE_GAP = 150
PIPE_WIDTH = 70
PIPE_SPEED = 3
FPS = 30

# Couleurs
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialisation de l'écran
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Définition des états pour Q-Learning
STATE_BINS = (10, 10)  # Discrétisation de l'espace d'état
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_PROB = 1.0  # Épsilon
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.01

# Actions possibles
ACTIONS = [0, 1]  # 0 = ne rien faire, 1 = sauter

# Chargement/sauvegarde du Q-table
Q_TABLE_FILE = "q_table.pkl"
try:
    with open(Q_TABLE_FILE, "rb") as f:
        Q_table = pickle.load(f)
except FileNotFoundError:
    Q_table = np.zeros(STATE_BINS + (len(ACTIONS),))


def discretize_state(bird_y, bird_velocity, pipe_x, pipe_y):
    """Convertit l'état continu en un état discret."""
    state = (
        min(STATE_BINS[0] - 1, max(0, int(bird_y / HEIGHT * STATE_BINS[0]))),
        min(STATE_BINS[1] - 1, max(0, int(pipe_x / WIDTH * STATE_BINS[1]))),
    )
    return state


class FlappyBird:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_velocity = 0
        self.pipe_x = WIDTH
        self.pipe_y = random.randint(100, HEIGHT - PIPE_GAP - 100)
        self.score = 0
        return discretize_state(self.bird_y, self.bird_velocity, self.pipe_x, self.pipe_y)

    def step(self, action):
        # Appliquer la physique
        if action == 1:
            self.bird_velocity = JUMP_STRENGTH
        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        # Déplacer le tuyau
        self.pipe_x -= PIPE_SPEED
        if self.pipe_x < -PIPE_WIDTH:
            self.pipe_x = WIDTH
            self.pipe_y = random.randint(100, HEIGHT - PIPE_GAP - 100)
            self.score += 1

        # Vérifier la collision
        if self.bird_y <= 0 or self.bird_y >= HEIGHT:
            return discretize_state(self.bird_y, self.bird_velocity, self.pipe_x, self.pipe_y), -100, True
        if self.pipe_x < BIRD_X < self.pipe_x + PIPE_WIDTH and not (self.pipe_y < self.bird_y < self.pipe_y + PIPE_GAP):
            return discretize_state(self.bird_y, self.bird_velocity, self.pipe_x, self.pipe_y), -100, True

        return discretize_state(self.bird_y, self.bird_velocity, self.pipe_x, self.pipe_y), 1, False

    def render(self):
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLUE, (BIRD_X, self.bird_y, 30, 30))
        pygame.draw.rect(screen, GREEN, (self.pipe_x, 0, PIPE_WIDTH, self.pipe_y))
        pygame.draw.rect(screen, GREEN, (self.pipe_x, self.pipe_y + PIPE_GAP, PIPE_WIDTH, HEIGHT))
        pygame.display.flip()
        clock.tick(FPS)


# Entraînement avec Q-Learning
game = FlappyBird()
for episode in range(10000):
    state = game.reset()
    done = False
    while not done:
        if random.uniform(0, 1) < EXPLORATION_PROB:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(Q_table[state])

        new_state, reward, done = game.step(action)
        Q_table[state][action] = (1 - LEARNING_RATE) * Q_table[state][action] + LEARNING_RATE * (
                    reward + DISCOUNT_FACTOR * np.max(Q_table[new_state]))
        state = new_state

    EXPLORATION_PROB = max(EXPLORATION_MIN, EXPLORATION_PROB * EXPLORATION_DECAY)

# Sauvegarde de la Q-table
with open(Q_TABLE_FILE, "wb") as f:
    pickle.dump(Q_table, f)

# Mode jeu
running = True
game.reset()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = np.argmax(Q_table[discretize_state(game.bird_y, game.bird_velocity, game.pipe_x, game.pipe_y)])
    game.step(action)
    game.render()

pygame.quit()