import pygame
import random
import colorsys
import math
from copy import deepcopy

pygame.init()

WIDTH, HEIGHT = 900, 700
BRAIN_PANEL_WIDTH = 560
TOTAL_WIDTH = WIDTH + BRAIN_PANEL_WIDTH

screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 18)


brain_surface = pygame.Surface((BRAIN_PANEL_WIDTH, HEIGHT))

POPULATION = 2000
START_X = 400
START_Y = HEIGHT // 2

MOVE_SPEED = 5
WALL_SPEED = 10
WALL_WIDTH = 240
GAP_SIZE = 90

MUTATION_RATE = 0.08
MUTATION_STRENGTH = 0.35

running = True


def color_pool(count=256):
    colors = []
    for i in range(count):
        h = i / count
        s = random.uniform(0.25, 1.0)
        v = random.uniform(0.8, 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    random.shuffle(colors)
    return colors


def mutate_color(color):
    r, g, b = [c / 255 for c in color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    h = (h + random.uniform(-0.03, 0.03)) % 1.0
    s = min(1.0, max(0.25, s + random.uniform(-0.03, 0.03)))
    v = min(1.0, max(0.5, v + random.uniform(-0.03, 0.03)))

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


class Brain:
    def __init__(self, input_size=5, hidden_size=8, output_size=4, data=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if data is not None:
            self.w1, self.b1, self.w2, self.b2 = data
        else:
            self.w1 = [
                [random.uniform(-1, 1) for _ in range(hidden_size)]
                for _ in range(input_size)
            ]
            self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]

            self.w2 = [
                [random.uniform(-1, 1) for _ in range(output_size)]
                for _ in range(hidden_size)
            ]
            self.b2 = [random.uniform(-1, 1) for _ in range(output_size)]

    def copy(self):
        return Brain(
            self.input_size,
            self.hidden_size,
            self.output_size,
            data=(
                deepcopy(self.w1),
                deepcopy(self.b1),
                deepcopy(self.w2),
                deepcopy(self.b2),
            ),
        )

    def activate(self, x):
        return math.tanh(x)

    def forward(self, inputs):
        hidden = []
        for j in range(len(self.b1)):
            total = self.b1[j]
            for i in range(len(inputs)):
                total += inputs[i] * self.w1[i][j]
            hidden.append(self.activate(total))

        outputs = []
        for j in range(len(self.b2)):
            total = self.b2[j]
            for i in range(len(hidden)):
                total += hidden[i] * self.w2[i][j]
            outputs.append(self.activate(total))

        return outputs

    def mutate(self, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
        child = self.copy()

        for i in range(len(child.w1)):
            for j in range(len(child.w1[i])):
                if random.random() < rate:
                    child.w1[i][j] += random.uniform(-strength, strength)

        for j in range(len(child.b1)):
            if random.random() < rate:
                child.b1[j] += random.uniform(-strength, strength)

        for i in range(len(child.w2)):
            for j in range(len(child.w2[i])):
                if random.random() < rate:
                    child.w2[i][j] += random.uniform(-strength, strength)

        for j in range(len(child.b2)):
            if random.random() < rate:
                child.b2[j] += random.uniform(-strength, strength)

        return child
    
    def get_activations(self, inputs):

        hidden = []

        for j in range(len(self.b1)):
            total = self.b1[j]

            for i in range(len(inputs)):
                total += inputs[i] * self.w1[i][j]

            hidden.append(self.activate(total))

        outputs = []

        for j in range(len(self.b2)):
            total = self.b2[j]

            for i in range(len(hidden)):
                total += hidden[i] * self.w2[i][j]

            outputs.append(self.activate(total))

        return hidden, outputs
    
def draw_brain(surface, brain, inputs, x=500, y=50):

    hidden, outputs = brain.get_activations(inputs)

    input_labels = [
        "X Pos",
        "Y Pos",
        "Wall X",
        "Gap Y",
        "Gap Size"
    ]

    output_labels = [
        "LEFT",
        "RIGHT",
        "UP",
        "DOWN"
    ]

    input_nodes = []
    hidden_nodes = []
    output_nodes = []

    # positions
    for i in range(len(inputs)):
        input_nodes.append((x, y + i * 60))

    for i in range(len(hidden)):
        hidden_nodes.append((x + 140, y + i * 40))

    for i in range(len(outputs)):
        output_nodes.append((x + 280, y + i * 80))

    # input -> hidden
    for i, (x1, y1) in enumerate(input_nodes):
        for j, (x2, y2) in enumerate(hidden_nodes):

            weight = brain.w1[i][j]

            strength = min(255, int(abs(weight) * 255))

            # GREEN = positive influence
            # RED = negative influence
            color = (
                0,
                strength,
                0
            ) if weight > 0 else (
                strength,
                0,
                0
            )

            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 2)

    # hidden -> output
    for i, (x1, y1) in enumerate(hidden_nodes):
        for j, (x2, y2) in enumerate(output_nodes):

            weight = brain.w2[i][j]

            strength = min(255, int(abs(weight) * 255))

            color = (
                0,
                strength,
                0
            ) if weight > 0 else (
                strength,
                0,
                0
            )

            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 2)

    for i, pos in enumerate(input_nodes):

        value = abs(inputs[i])

        brightness = min(255, int(value * 255))

        pygame.draw.circle(
            surface,
            (brightness, brightness, brightness),
            pos,
            12
        )

        # label
        text = font.render(
            f"{input_labels[i]}: {round(inputs[i], 2)}",
            True,
            (0, 0, 0)
        )

        surface.blit(text, (pos[0] - 110, pos[1] - 10))


    for i, pos in enumerate(hidden_nodes):

        value = abs(hidden[i])

        brightness = min(255, int(value * 255))

        pygame.draw.circle(
            surface,
            (brightness, brightness, 255),
            pos,
            14
        )

        if i not in [0, len(hidden_nodes) - 1]:
            continue

        text = font.render(
            f"H{i}",
            True,
            (0, 0, 0)
        )

        surface.blit(text, (pos[0] - 10, pos[1] - 30))


    strongest_output = max(range(len(outputs)), key=lambda i: outputs[i])

    for i, pos in enumerate(output_nodes):

        value = abs(outputs[i])

        brightness = min(255, int(value * 255))

        # highlight chosen action
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

        text = font.render(
            f"{output_labels[i]}: {round(outputs[i], 2)}",
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
    def __init__(self, x, y, color, brain=None, width=10):
        self.x = x
        self.y = y
        self.color = color
        self.brain = brain if brain is not None else Brain()
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
            child_color = mutate_color(self.color)
        else:
            child_brain = self.brain.copy()
            child_color = self.color

        child = Agent(
            START_X,
            START_Y,
            child_color,
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


def evolve(population, generation, base_colors):
    survivors = [agent for agent in population if agent.alive]

    # extinction event
    if len(survivors) == 0:
        print("All agents Died")

        new_population = [
            Agent(
                START_X,
                START_Y,
                random.choice(base_colors)
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


base_colors = color_pool()
agents = [
    Agent(START_X, START_Y, random.choice(base_colors))
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
    ]
    

    wall.update()

    if dead_count >= int(POPULATION * 0.96) or alive_count == 0: #  
        agents, generation = evolve(agents, generation, base_colors)
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
    




    pygame.display.flip()
    clock.tick(60)

pygame.quit()