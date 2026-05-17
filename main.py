import pygame
import random
import colorsys
import numpy as np

pygame.init()

WIDTH, HEIGHT = 900, 700
BRAIN_PANEL_WIDTH = 560
TOTAL_WIDTH = WIDTH + BRAIN_PANEL_WIDTH

MOVE_SPEED = 12
WALL_SPEED = 18
WALL_WIDTH = 240

POPULATION = 800
FPS = 2000

STARTING_LAYERS = [6, 9, 4, 4, 9, 4]

START_X = 40
START_Y = HEIGHT // 2

MUTATION_RATE = 0.08
MUTATION_STRENGTH = 0.25

MIN_HIDDEN = 2
MAX_NEURONS_PER_LAYER = 16

ADD_NEURON_CHANCE = 0.06
REMOVE_NEURON_CHANCE = 0.06

BRAIN_UPDATE_INTERVAL = 30
INFO_UPDATE_INTERVAL = 15

INPUT_LABELS = (
    "X Pos",
    "Y Pos",
    "Wall X",
    "Gap Y",
    "Gap Size",
    "Wall Speed",
)

OUTPUT_LABELS = ("LEFT", "RIGHT", "UP", "DOWN")

NODE_CACHE = {}

screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 18)

brain_surface = pygame.Surface((BRAIN_PANEL_WIDTH, HEIGHT))

cached_input_labels = [
    font.render(label, True, (0, 0, 0))
    for label in INPUT_LABELS
]

cached_output_labels = [
    font.render(label, True, (0, 0, 0))
    for label in OUTPUT_LABELS
]


def brain_to_color(brain):
    values = np.concatenate(
        [
            *(w.flatten() for w in brain.weights),
            *(b.flatten() for b in brain.biases),
        ]
    )

    total = np.sum(values)
    avg = np.mean(np.abs(values))
    variance = np.var(values)

    h = float(abs(total * 0.1) % 1.0)
    s = float(min(1.0, 0.4 + avg * 0.5))
    v = float(min(1.0, 0.6 + (variance % 1.0) * 0.4))

    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    return (
        int(r * 255),
        int(g * 255),
        int(b * 255),
    )


class Brain:
    def __init__(self, layer_sizes=None, data=None):
        if layer_sizes is None:
            layer_sizes = STARTING_LAYERS

        self.layer_sizes = list(layer_sizes)

        if data is not None:
            self.weights, self.biases = data
            return

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            inp = layer_sizes[i]
            out = layer_sizes[i + 1]

            self.weights.append(
                np.random.uniform(-1, 1, (inp, out)).astype(np.float32)
            )

            self.biases.append(
                np.random.uniform(-1, 1, out).astype(np.float32)
            )

    def copy(self):
        return Brain(
            self.layer_sizes[:],
            data=(
                [w.copy() for w in self.weights],
                [b.copy() for b in self.biases],
            ),
        )

    def forward(self, activations):
        for weights, biases in zip(self.weights, self.biases):
            activations = np.tanh(activations @ weights + biases)

        return activations

    def mutate(self):
        child = self.copy()

        for i in range(len(child.weights)):
            weights = child.weights[i]
            biases = child.biases[i]

            weight_mask = np.random.random(weights.shape) < MUTATION_RATE
            bias_mask = np.random.random(biases.shape) < MUTATION_RATE

            weights += (
                weight_mask
                * np.random.uniform(
                    -MUTATION_STRENGTH,
                    MUTATION_STRENGTH,
                    weights.shape,
                ).astype(np.float32)
            )

            biases += (
                bias_mask
                * np.random.uniform(
                    -MUTATION_STRENGTH,
                    MUTATION_STRENGTH,
                    biases.shape,
                ).astype(np.float32)
            )

        hidden_count = len(child.layer_sizes) - 2

        if hidden_count > 0:
            target_layer = random.randint(1, len(child.layer_sizes) - 2)
            weights_index = target_layer - 1
            target_size = child.layer_sizes[target_layer]

            if (
                random.random() < ADD_NEURON_CHANCE
                and target_size < MAX_NEURONS_PER_LAYER
            ):
                child.layer_sizes[target_layer] += 1

                prev = child.weights[weights_index]
                nxt = child.weights[weights_index + 1]

                child.weights[weights_index] = np.hstack([
                    prev,
                    np.random.uniform(
                        -1,
                        1,
                        (prev.shape[0], 1),
                    ).astype(np.float32),
                ])

                child.biases[weights_index] = np.append(
                    child.biases[weights_index],
                    np.float32(random.uniform(-1, 1)),
                )

                child.weights[weights_index + 1] = np.vstack([
                    nxt,
                    np.random.uniform(
                        -1,
                        1,
                        (1, nxt.shape[1]),
                    ).astype(np.float32),
                ])

            if (
                random.random() < REMOVE_NEURON_CHANCE
                and target_size > MIN_HIDDEN
            ):
                remove_index = random.randint(0, target_size - 1)

                child.layer_sizes[target_layer] -= 1

                child.weights[weights_index] = np.delete(
                    child.weights[weights_index],
                    remove_index,
                    axis=1,
                )

                child.biases[weights_index] = np.delete(
                    child.biases[weights_index],
                    remove_index,
                )

                child.weights[weights_index + 1] = np.delete(
                    child.weights[weights_index + 1],
                    remove_index,
                    axis=0,
                )

        return child

    def get_activations(self, inputs):
        activations = [inputs]
        current = inputs

        for weights, biases in zip(self.weights, self.biases):
            current = np.tanh(current @ weights + biases)
            activations.append(current)

        return activations


def draw_brain(surface, brain, inputs, x=140, y=40):
    activations = brain.get_activations(inputs)

    key = tuple(brain.layer_sizes)

    if key not in NODE_CACHE:
        layer_nodes = []

        layer_spacing = 60
        max_nodes = max(len(layer) for layer in activations)
        step = 40 if max_nodes <= 12 else 26

        for layer_index, layer in enumerate(activations):
            nodes = []
            base_x = x + layer_index * layer_spacing

            for i in range(len(layer)):
                nodes.append((base_x, y + i * step))

            layer_nodes.append(nodes)

        NODE_CACHE[key] = layer_nodes

    layer_nodes = NODE_CACHE[key]

    for layer_index in range(len(layer_nodes) - 1):
        from_nodes = layer_nodes[layer_index]
        to_nodes = layer_nodes[layer_index + 1]
        weights = brain.weights[layer_index]

        for i, (x1, y1) in enumerate(from_nodes):
            row = weights[i]

            for j, (x2, y2) in enumerate(to_nodes):
                weight = row[j]
                strength = min(255, int(abs(weight) * 255))

                color = (
                    (0, strength, 0)
                    if weight > 0
                    else (strength, 0, 0)
                )

                pygame.draw.line(surface, color, (x1, y1), (x2, y2), 1)

    for i, pos in enumerate(layer_nodes[0]):
        brightness = min(255, int(abs(activations[0][i]) * 255))

        pygame.draw.circle(
            surface,
            (brightness, brightness, brightness),
            pos,
            10,
        )

        if i < len(cached_input_labels):
            surface.blit(cached_input_labels[i], (pos[0] - 110, pos[1] - 10))

    for layer_index in range(1, len(layer_nodes) - 1):
        for i, pos in enumerate(layer_nodes[layer_index]):
            brightness = min(
                255,
                int(abs(activations[layer_index][i]) * 255),
            )

            pygame.draw.circle(
                surface,
                (brightness, brightness, 255),
                pos,
                12,
            )

    outputs = activations[-1]
    strongest = int(np.argmax(outputs))

    for i, pos in enumerate(layer_nodes[-1]):
        brightness = min(255, int(abs(outputs[i]) * 255))

        color = (
            (255, brightness, 0)
            if i == strongest
            else (180, 180, 180)
        )

        pygame.draw.circle(surface, color, pos, 14)

        if i < len(cached_output_labels):
            surface.blit(cached_output_labels[i], (pos[0] + 25, pos[1] - 10))


class Wall:
    def __init__(self):
        self.width = WALL_WIDTH
        self.reset()

    def reset(self):
        self.x = WIDTH
        self.gap_size = random.randint(60, 230)
        self.speed = random.randint(12, 18)

        margin = 35

        self.gap_y = random.randint(
            margin,
            HEIGHT - margin - self.gap_size,
        )

    @property
    def gap_center(self):
        return self.gap_y + self.gap_size * 0.5

    def update(self):
        self.x -= self.speed

        if self.x + self.width < 0:
            self.reset()

    def collides(self, agent):
        if not (
            agent.x < self.x + self.width
            and agent.x + agent.width > self.x
        ):
            return False

        return not (
            agent.y >= self.gap_y
            and agent.y + agent.width <= self.gap_y + self.gap_size
        )

    def draw(self, surface):
        pygame.draw.rect(
            surface,
            (220, 220, 220),
            (self.x, 0, self.width, self.gap_y),
        )

        pygame.draw.rect(
            surface,
            (220, 220, 220),
            (
                self.x,
                self.gap_y + self.gap_size,
                self.width,
                HEIGHT - (self.gap_y + self.gap_size),
            ),
        )


class Agent:
    def __init__(self, x, y, brain=None, width=10):
        self.x = x
        self.y = y
        self.width = width

        self.brain = brain if brain else Brain()
        self.color = brain_to_color(self.brain)

        self.alive = True
        self.score = 0

        self.inputs = np.zeros(6, dtype=np.float32)

    def clamp(self):
        self.x = max(0, min(WIDTH - self.width, self.x))
        self.y = max(0, min(HEIGHT - self.width, self.y))

    def think(self, wall):
        self.inputs[0] = self.x / WIDTH
        self.inputs[1] = self.y / HEIGHT
        self.inputs[2] = wall.x / WIDTH
        self.inputs[3] = wall.gap_center / HEIGHT
        self.inputs[4] = wall.gap_size / HEIGHT
        self.inputs[5] = wall.speed / WALL_SPEED

        outputs = self.brain.forward(self.inputs)
        action = int(np.argmax(outputs))

        if action == 0:
            self.x -= MOVE_SPEED
        elif action == 1:
            self.x += MOVE_SPEED
        elif action == 2:
            self.y -= MOVE_SPEED
        else:
            self.y += MOVE_SPEED

    def update(self, wall):
        if not self.alive:
            return

        self.think(wall)
        self.clamp()

        self.score += 1

        if wall.collides(self):
            self.alive = False

    def draw(self, surface):
        if self.alive:
            pygame.draw.rect(
                surface,
                self.color,
                (self.x, self.y, self.width, self.width),
            )

    def spawn_child(self):
        return Agent(
            START_X,
            START_Y,
            self.brain.mutate(),
            self.width,
        )


def weighted_choice(agents):
    weights = [max(1.0, a.score) for a in agents]
    total = sum(weights)

    r = random.uniform(0, total)

    upto = 0

    for agent, weight in zip(agents, weights):
        upto += weight

        if upto >= r:
            return agent

    return agents[-1]


def evolve(population, generation):
    survivors = [a for a in population if a.alive]

    if not survivors:
        return (
            [Agent(START_X, START_Y) for _ in range(POPULATION)],
            generation + 1,
        )

    new_population = []

    while len(new_population) < POPULATION:
        parent = weighted_choice(survivors)
        new_population.append(parent.spawn_child())

    return new_population, generation + 1


agents = [Agent(START_X, START_Y) for _ in range(POPULATION)]

wall = Wall()

running = True
frame_count = 0
generation = 1

cached_info1 = None
cached_info2 = None
cached_info3 = None

last_best_id = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    wall.update()

    alive_count = 0

    for agent in agents:
        agent.update(wall)

        if agent.alive:
            alive_count += 1

    best_agent = max(agents, key=lambda a: a.score)

    if alive_count <= POPULATION * 0.04:
        agents, generation = evolve(agents, generation)
        wall = Wall()

    frame_count += 1

    screen.fill((255, 255, 255))

    wall.draw(screen)

    for agent in agents:
        agent.draw(screen)

    if frame_count % INFO_UPDATE_INTERVAL == 0:
        cached_info1 = font.render(
            f"Generation: {generation}",
            True,
            (0, 0, 0),
        )

        cached_info2 = font.render(
            f"Alive: {alive_count}/{POPULATION}",
            True,
            (0, 0, 0),
        )

        cached_info3 = font.render(
            f"Layers: {best_agent.brain.layer_sizes}",
            True,
            (0, 0, 0),
        )

    if cached_info1:
        screen.blit(cached_info1, (10, 10))
        screen.blit(cached_info2, (10, 35))
        screen.blit(cached_info3, (10, 60))

    if (
        frame_count % BRAIN_UPDATE_INTERVAL == 0
        or id(best_agent.brain) != last_best_id
    ):
        brain_surface.fill((240, 240, 240))

        draw_brain(
            brain_surface,
            best_agent.brain,
            best_agent.inputs,
        )

        last_best_id = id(best_agent.brain)

    screen.blit(brain_surface, (WIDTH, 0))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()