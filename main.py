import pygame
import random
import colorsys
import math
from copy import deepcopy
from numba import njit
import numpy as np
import colorsys

pygame.init()

WIDTH, HEIGHT = 900, 700
BRAIN_PANEL_WIDTH = 560
TOTAL_WIDTH = WIDTH + BRAIN_PANEL_WIDTH

screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 18)


brain_surface = pygame.Surface((BRAIN_PANEL_WIDTH, HEIGHT))

POPULATION = 8000
STARTING_LAYERS = [6, 9, 4, 4, 9, 4]
START_X = 40
START_Y = HEIGHT // 2

MOVE_SPEED = 8
WALL_SPEED = 18
WALL_WIDTH = 240
GAP_SIZE = 160

MUTATION_RATE = 0.08
MUTATION_STRENGTH = 0.35

MIN_HIDDEN = 2
MAX_NEURONS_PER_LAYER = 16
MAX_LAYERS = 4

ADD_NEURON_CHANCE = 0.06
REMOVE_NEURON_CHANCE = 0.06
ADD_LAYER_CHANCE = 0.06
REMOVE_LAYER_CHANCE = 0.06

running = True


@njit
def compute_brain_stats(values):
    total = np.sum(values)
    avg = np.mean(np.abs(values))
    variance = np.var(values)
    return total, avg, variance

def brain_to_color(brain):
    values = np.array(
        [
            value
            for layer in brain.weights
            for row in layer
            for value in row
        ] + [
            value
            for layer in brain.biases
            for value in layer
        ],
        dtype=np.float32
    )

    total, avg, variance = compute_brain_stats(values)

    h = float(abs(total * 0.1) % 1.0)
    s = float(min(1.0, 0.4 + avg * 0.5))
    v = float(min(1.0, 0.6 + (variance % 1.0) * 0.4))

    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    return (
        int(r * 255),
        int(g * 255),
        int(b * 255)
    )

class Brain:
    def __init__(self, layer_sizes=None, data=None):
        if layer_sizes is None:
            layer_sizes = STARTING_LAYERS

        self.layer_sizes = list(layer_sizes)

        if data is not None:
            self.weights, self.biases = data
        else:
            self.weights = []
            self.biases = []

            for i in range(len(layer_sizes) - 1):
                input_size = layer_sizes[i]
                output_size = layer_sizes[i + 1]

                layer_weights = [
                    [random.uniform(-1, 1) for _ in range(output_size)]
                    for _ in range(input_size)
                ]

                layer_biases = [random.uniform(-1, 1) for _ in range(output_size)]

                self.weights.append(layer_weights)
                self.biases.append(layer_biases)

    def copy(self):
        return Brain(
            deepcopy(self.layer_sizes),
            data=(
                deepcopy(self.weights),
                deepcopy(self.biases),
            ),
        )

    def activate(self, x):
        return math.tanh(x)

    def forward(self, inputs):
        activations = inputs

        for layer_weights, layer_biases in zip(self.weights, self.biases):
            next_activations = []

            for j in range(len(layer_biases)):
                total = layer_biases[j]

                for i in range(len(activations)):
                    total += activations[i] * layer_weights[i][j]

                next_activations.append(self.activate(total))

            activations = next_activations

        return activations

    def mutate(self, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
        child = self.copy()

        for layer_index in range(len(child.weights)):
            layer_weights = child.weights[layer_index]
            layer_biases = child.biases[layer_index]

            for i in range(len(layer_weights)):
                for j in range(len(layer_weights[i])):

                    if random.random() < rate:
                        layer_weights[i][j] += random.uniform(-strength, strength)

            for j in range(len(layer_biases)):

                if random.random() < rate:
                    layer_biases[j] += random.uniform(-strength, strength)

        hidden_layer_count = len(child.layer_sizes) - 2

        if hidden_layer_count > 0:
            target_layer_index = random.randint(1, len(child.layer_sizes) - 2)
            target_size = child.layer_sizes[target_layer_index]
            weights_layer_index = target_layer_index - 1

            if (
                random.random() < ADD_NEURON_CHANCE
                and target_size < MAX_NEURONS_PER_LAYER
            ):

                child.layer_sizes[target_layer_index] += 1

                prev_weights = child.weights[weights_layer_index]
                for row in prev_weights:
                    row.append(random.uniform(-1, 1))

                child.biases[weights_layer_index].append(random.uniform(-1, 1))

                next_weights = child.weights[weights_layer_index + 1]
                next_weights.append([
                    random.uniform(-1, 1)
                    for _ in range(child.layer_sizes[target_layer_index + 1])
                ])

            if (
                random.random() < REMOVE_NEURON_CHANCE
                and target_size > MIN_HIDDEN
            ):

                remove_index = random.randint(0, target_size - 1)
                child.layer_sizes[target_layer_index] -= 1

                prev_weights = child.weights[weights_layer_index]
                for row in prev_weights:
                    del row[remove_index]

                del child.biases[weights_layer_index][remove_index]

                next_weights = child.weights[weights_layer_index + 1]
                del next_weights[remove_index]

        if (
            random.random() < ADD_LAYER_CHANCE
            and len(child.layer_sizes) - 2 < MAX_LAYERS
        ):

            insert_index = random.randint(1, len(child.layer_sizes) - 1)
            weights_layer_index = insert_index - 1
            new_size = random.randint(MIN_HIDDEN, MAX_NEURONS_PER_LAYER)

            prev_size = child.layer_sizes[insert_index - 1]
            next_size = child.layer_sizes[insert_index]

            child.layer_sizes.insert(insert_index, new_size)

            new_weights_prev = [
                [random.uniform(-1, 1) for _ in range(new_size)]
                for _ in range(prev_size)
            ]

            new_biases_prev = [random.uniform(-1, 1) for _ in range(new_size)]

            new_weights_next = [
                [random.uniform(-1, 1) for _ in range(next_size)]
                for _ in range(new_size)
            ]

            child.weights[weights_layer_index] = new_weights_prev
            child.biases[weights_layer_index] = new_biases_prev
            child.weights.insert(weights_layer_index + 1, new_weights_next)
            child.biases.insert(weights_layer_index + 1, [random.uniform(-1, 1) for _ in range(next_size)])

        if (
            random.random() < REMOVE_LAYER_CHANCE
            and len(child.layer_sizes) - 2 > 1
        ):

            remove_index = random.randint(1, len(child.layer_sizes) - 2)
            weights_layer_index = remove_index - 1

            prev_size = child.layer_sizes[remove_index - 1]
            next_size = child.layer_sizes[remove_index + 1]

            del child.layer_sizes[remove_index]

            new_weights_prev = [
                [random.uniform(-1, 1) for _ in range(next_size)]
                for _ in range(prev_size)
            ]

            child.weights[weights_layer_index] = new_weights_prev
            child.biases[weights_layer_index] = [random.uniform(-1, 1) for _ in range(next_size)]

            del child.weights[weights_layer_index + 1]
            del child.biases[weights_layer_index + 1]

        return child
    
    def get_activations(self, inputs):

        activations = [inputs]

        current = inputs

        for layer_weights, layer_biases in zip(self.weights, self.biases):
            next_activations = []

            for j in range(len(layer_biases)):
                total = layer_biases[j]

                for i in range(len(current)):
                    total += current[i] * layer_weights[i][j]

                next_activations.append(self.activate(total))

            activations.append(next_activations)
            current = next_activations

        return activations
    
def draw_brain(surface, brain, inputs, x=500, y=50):

    activations = brain.get_activations(inputs)

    input_labels = [
        "X Pos",
        "Y Pos",
        "Wall X",
        "Gap Y",
        "Gap Size",
        "Wall Speed"
    ]

    output_labels = [
        "LEFT",
        "RIGHT",
        "UP",
        "DOWN"
    ]

    layer_nodes = []
    layer_spacing = 60

    max_nodes = max(len(layer) for layer in activations)
    for layer_index, layer in enumerate(activations):
        nodes = []
        step = 40 if max_nodes <= 12 else 26
        for i in range(len(layer)):
            nodes.append((x + layer_index * layer_spacing, y + i * step))
        layer_nodes.append(nodes)

    for layer_index in range(len(layer_nodes) - 1):
        from_nodes = layer_nodes[layer_index]
        to_nodes = layer_nodes[layer_index + 1]
        weights = brain.weights[layer_index]

        for i, (x1, y1) in enumerate(from_nodes):
            for j, (x2, y2) in enumerate(to_nodes):
                weight = weights[i][j]
                strength = min(255, int(abs(weight) * 255))
                color = (0, strength, 0) if weight > 0 else (strength, 0, 0)

                pygame.draw.line(surface, color, (x1, y1), (x2, y2), 2)

    for i, pos in enumerate(layer_nodes[0]):
        value = abs(activations[0][i])
        brightness = min(255, int(value * 255))

        pygame.draw.circle(
            surface,
            (brightness, brightness, brightness),
            pos,
            12
        )

        if i < len(input_labels):
            text = font.render(
                f"{input_labels[i]}: {round(activations[0][i], 2)}",
                True,
                (0, 0, 0)
            )
            surface.blit(text, (pos[0] - 110, pos[1] - 10))

    for layer_index in range(1, len(layer_nodes) - 1):
        for i, pos in enumerate(layer_nodes[layer_index]):
            value = abs(activations[layer_index][i])
            brightness = min(255, int(value * 255))

            pygame.draw.circle(
                surface,
                (brightness, brightness, 255),
                pos,
                14
            )

            if len(layer_nodes[layer_index]) > 12:
                if i not in [0, len(layer_nodes[layer_index]) - 1]:
                    continue

            text = font.render(
                f"H{layer_index}.{i}",
                True,
                (0, 0, 0)
            )
            surface.blit(text, (pos[0] - 10, pos[1] - 30))

    outputs = activations[-1]
    strongest_output = max(range(len(outputs)), key=lambda i: outputs[i])

    for i, pos in enumerate(layer_nodes[-1]):
        value = abs(outputs[i])
        brightness = min(255, int(value * 255))

        if i == strongest_output:
            color = (255, brightness, 0)
            radius = 18
        else:
            color = (180, 180, 180)
            radius = 14

        pygame.draw.circle(
            surface,
            color,
            pos,
            radius
        )

        label = output_labels[i] if i < len(output_labels) else f"O{i}"
        text = font.render(
            f"{label}: {round(outputs[i], 2)}",
            True,
            (0, 0, 0)
        )

        surface.blit(text, (pos[0] + 25, pos[1] - 10))


    # legend = [
    #     "Green line = positive weight",
    #     "Red line = negative weight",
    #     "Bright node = strong activation",
    #     "Yellow output = chosen action"
    # ]

    # for i, line in enumerate(legend):
    #     text = font.render(line, True, (0, 0, 0))
    #     surface.blit(text, (x, y + 360 + i * 20))


class Wall:
    def __init__(self):
        self.width = WALL_WIDTH
        self.speed = WALL_SPEED
        self.gap_size = GAP_SIZE
        self.reset()

    def reset(self):
        self.x = WIDTH
        margin = 35
        self.gap_size = random.randint(60, 230)
        self.speed = random.randint(12, 18)
        self.gap_y = random.randint(margin, HEIGHT - margin - self.gap_size)

    @property
    def gap_center(self):
        return self.gap_y + self.gap_size / 2

    def update(self):
        self.x -= self.speed
        if self.x + self.width < 0:
            self.reset()

    def collides(self, agent):
        horizontal_overlap = (
            agent.x < self.x + self.width
            and agent.x + agent.width > self.x
        )
        if not horizontal_overlap:
            return False

        inside_gap = (
            agent.y >= self.gap_y
            and agent.y + agent.width <= self.gap_y + self.gap_size
        )
        return not inside_gap

    def draw(self, surface):
        pygame.draw.rect(surface, (0, 0, 0), (self.x, 0, self.width, self.gap_y))
        pygame.draw.rect(
            surface,
            (0, 0, 0),
            (self.x, self.gap_y + self.gap_size, self.width, HEIGHT - (self.gap_y + self.gap_size)),
        )


class Agent:
    def __init__(self, x, y, brain=None, width=10):
        self.x = x
        self.y = y
        self.brain = brain if brain is not None else Brain()
        self.color = brain_to_color(self.brain)
        self.width = width
        self.alive = True
        self.score = 0

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def clamp_to_screen(self):
        self.x = max(0, min(WIDTH - self.width, self.x))
        self.y = max(0, min(HEIGHT - self.width, self.y))

    def think(self, wall):
        inputs = [
            self.x / WIDTH,
            self.y / HEIGHT,
            wall.x / WIDTH,
            wall.gap_center / HEIGHT,
            wall.gap_size / HEIGHT,
            wall.speed / float(WALL_SPEED),
        ]

        outputs = self.brain.forward(inputs)
        action = max(range(len(outputs)), key=lambda i: outputs[i])

        if action == 0:
            self.x -= MOVE_SPEED
        elif action == 1:
            self.x += MOVE_SPEED
        elif action == 2:
            self.y -= MOVE_SPEED
        elif action == 3:
            self.y += MOVE_SPEED

    def update(self, wall):
        if not self.alive:
            return

        self.think(wall)
        self.clamp_to_screen()
        self.score += 1

        if wall.collides(self):
            self.alive = False

    def draw(self, surface):
        if not self.alive:
            return
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.width))

    def spawn_child(self, mutate=True):

        if mutate:
            child_brain = self.brain.mutate()
            # child_color = mutate_color(self.color)
        else:
            child_brain = self.brain.copy()
            # child_color = self.color

        child = Agent(
            START_X,
            START_Y,
            child_brain,
            self.width
        )

        return child


def weighted_choice(agents):
    weights = [max(1.0, agent.score) for agent in agents]
    total = sum(weights)
    roll = random.uniform(0, total)

    upto = 0.0
    for agent, weight in zip(agents, weights):
        upto += weight
        if upto >= roll:
            return agent

    return agents[-1]


def evolve(population, generation):
    survivors = [agent for agent in population if agent.alive]

    # extinction event
    if len(survivors) == 0:
        print("All agents Died")

        new_population = [
            Agent(
                START_X,
                START_Y,
            )
            for _ in range(POPULATION)
        ]

        return new_population, generation + 1

    new_population = []

    # ONLY survivors reproduce
    while len(new_population) < POPULATION:

        parent = random.choice(survivors)

        child = parent.spawn_child(mutate=True)

        child.set_position(START_X, START_Y)
        child.alive = True
        child.score = 0

        new_population.append(child)

    return new_population, generation + 1


agents = [
    Agent(START_X, START_Y)
    for _ in range(POPULATION)
]
best_agent = None
wall = Wall()
generation = 1

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for agent in agents:
        agent.update(wall)

    alive_count = sum(1 for a in agents if a.alive)
    dead_count = POPULATION - alive_count

    best_agent = max(agents, key=lambda a: a.score)
    inputs = [
        best_agent.x / WIDTH,
        best_agent.y / HEIGHT,
        wall.x / WIDTH,
        wall.gap_center / HEIGHT,
        wall.gap_size / HEIGHT,
        wall.speed / float(WALL_SPEED),
    ]
    

    wall.update()

    if dead_count >= int(POPULATION * 0.96) or alive_count == 0: #  
        agents, generation = evolve(agents, generation)
        wall = Wall()

    screen.fill((255, 255, 255))
    brain_surface.fill((240, 240, 240))
    wall.draw(screen)

    for agent in agents:
        agent.draw(screen)

    info1 = font.render(f"Generation: {generation}", True, (0, 0, 0))
    info2 = font.render(f"Alive: {alive_count} / {POPULATION}", True, (0, 0, 0))
    # info3 = font.render(f"Dead: {dead_count}", True, (0, 0, 0))

    screen.blit(info1, (10, 10))
    screen.blit(info2, (10, 32))
    draw_brain(
        brain_surface,
        best_agent.brain,
        inputs,
        x=140,
        y=40
    )
    screen.blit(brain_surface, (WIDTH, 0))
    
    brain_size_text = font.render(
    f"Best Brain Layers: {best_agent.brain.layer_sizes}",
    True,
    (0, 0, 0)
)

    screen.blit(brain_size_text, (10, 54))
    




    pygame.display.flip()
    clock.tick(120)

pygame.quit()