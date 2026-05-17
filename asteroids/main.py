import math
import random
import colorsys

import numpy as np
import pygame

pygame.init()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

WIDTH, HEIGHT = 700, 600
BRAIN_PANEL_WIDTH = 850
TOTAL_WIDTH = WIDTH + BRAIN_PANEL_WIDTH

POPULATION = 120
MAX_GENERATION_FRAMES = 1400

ROAD_WIDTH = 80
TRACK_EDGE_PADDING = 142

# Track controls
TRACK_RADIUS = 290.0
TRACK_CHECKPOINT_DENSITY = 72
OBSTACLE_COUNT = 46
CHECKPOINT_HIT_RADIUS = 10.0
LAP_PROGRESS_MIN_RATIO = 0.8

START_SPEED = 5.0
MIN_SPEED = 1.5
MAX_SPEED = 10.5
STEER_SPEED = 0.085
ACCEL_SPEED = 0.18
FRICTION = 0.02

MUTATION_RATE = 0.08
MUTATION_STRENGTH = 0.35

MIN_HIDDEN_NEURONS = 4
MAX_HIDDEN_NEURONS = 24
ADD_NEURON_CHANCE = 0.07
REMOVE_NEURON_CHANCE = 0.06

RAY_COUNT = 5
RAY_MAX_DISTANCE = 170
RAY_STEP = 4

INPUT_SIZE = 6
OUTPUT_SIZE = 4
LAYER_SIZES = [INPUT_SIZE, 14, 12, 8, OUTPUT_SIZE]

MOVE_LABELS = ["STEER L", "STEER R", "ACCEL", "BRAKE"]
INPUT_LABELS = ["Ray 1", "Ray 2", "Ray 3", "Ray 4", "Ray 5", "Speed"]

WHITE = (245, 245, 245)
BLACK = (0, 0, 0)
GRASS = (86, 130, 82)
ROAD = (70, 70, 74)
ROAD_EDGE = (238, 238, 238)
ROAD_BOUNDARY = (36, 36, 40)
ROAD_CENTERLINE = (235, 210, 120)
OBSTACLE_COLOR = (190, 85, 60)

window = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
pygame.display.set_caption("Procedural F1 Evolution")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 18)

brain_screen = pygame.Surface((BRAIN_PANEL_WIDTH, HEIGHT))

running = True


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def split_length(total, parts):
    if parts <= 1 or total <= 1:
        return [total]

    parts = min(parts, total)
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0

    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut

    return values


def brain_to_color(brain):
    values = np.concatenate(
        [w.ravel() for w in brain.weights] + [b.ravel() for b in brain.biases]
    ).astype(np.float32)

    total = float(np.sum(values))
    avg = float(np.mean(np.abs(values)))
    variance = float(np.var(values))

    h = float(abs(total * 0.1) % 1.0)
    s = float(clamp(0.4 + avg * 0.5, 0.0, 1.0))
    v = float(clamp(0.6 + (variance % 1.0) * 0.4, 0.0, 1.0))

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def weighted_choice(population):
    weights = [max(1.0, agent.score) for agent in population]
    total = sum(weights)
    roll = random.uniform(0, total)

    upto = 0.0
    for agent, weight in zip(population, weights):
        upto += weight
        if upto >= roll:
            return agent

    return population[-1]


# -----------------------------------------------------------------------------
# BRAIN
# -----------------------------------------------------------------------------

class Brain:
    def __init__(self, layer_sizes=None, weights=None, biases=None):
        self.layer_sizes = list(layer_sizes if layer_sizes is not None else LAYER_SIZES)

        if weights is not None and biases is not None:
            self.weights = [np.array(w, dtype=np.float32) for w in weights]
            self.biases = [np.array(b, dtype=np.float32) for b in biases]
        else:
            self.weights = []
            self.biases = []

            for i in range(len(self.layer_sizes) - 1):
                in_size = self.layer_sizes[i]
                out_size = self.layer_sizes[i + 1]

                self.weights.append(
                    np.random.uniform(-1, 1, (in_size, out_size)).astype(np.float32)
                )
                self.biases.append(
                    np.random.uniform(-1, 1, out_size).astype(np.float32)
                )

    def copy(self):
        return Brain(
            layer_sizes=self.layer_sizes[:],
            weights=[w.copy() for w in self.weights],
            biases=[b.copy() for b in self.biases],
        )

    def forward_with_activations(self, inputs):
        activations = [np.asarray(inputs, dtype=np.float32)]
        current = activations[0]

        for weights, biases in zip(self.weights, self.biases):
            current = np.tanh(np.dot(current, weights) + biases).astype(np.float32)
            activations.append(current)

        return activations

    def forward(self, inputs):
        return self.forward_with_activations(inputs)[-1]

    def mutate(self, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
        child = self.copy()

        for i in range(len(child.weights)):
            w = child.weights[i]
            b = child.biases[i]

            weight_mask = np.random.random(w.shape) < rate
            weight_delta = np.random.uniform(-strength, strength, w.shape).astype(np.float32)
            w += weight_mask * weight_delta

            bias_mask = np.random.random(b.shape) < rate
            bias_delta = np.random.uniform(-strength, strength, b.shape).astype(np.float32)
            b += bias_mask * bias_delta

        hidden_layer_count = len(child.layer_sizes) - 2
        if hidden_layer_count > 0:
            target_layer = random.randint(1, len(child.layer_sizes) - 2)
            layer_size = child.layer_sizes[target_layer]
            in_matrix_index = target_layer - 1
            out_matrix_index = target_layer

            if random.random() < ADD_NEURON_CHANCE and layer_size < MAX_HIDDEN_NEURONS:
                child.layer_sizes[target_layer] += 1

                in_weights = child.weights[in_matrix_index]
                new_column = np.random.uniform(-1, 1, (in_weights.shape[0], 1)).astype(np.float32)
                child.weights[in_matrix_index] = np.hstack([in_weights, new_column])

                in_biases = child.biases[in_matrix_index]
                new_bias = np.array([random.uniform(-1, 1)], dtype=np.float32)
                child.biases[in_matrix_index] = np.concatenate([in_biases, new_bias])

                out_weights = child.weights[out_matrix_index]
                new_row = np.random.uniform(-1, 1, (1, out_weights.shape[1])).astype(np.float32)
                child.weights[out_matrix_index] = np.vstack([out_weights, new_row])

                layer_size += 1

            if random.random() < REMOVE_NEURON_CHANCE and layer_size > MIN_HIDDEN_NEURONS:
                remove_index = random.randint(0, layer_size - 1)
                child.layer_sizes[target_layer] -= 1

                child.weights[in_matrix_index] = np.delete(child.weights[in_matrix_index], remove_index, axis=1)
                child.biases[in_matrix_index] = np.delete(child.biases[in_matrix_index], remove_index)
                child.weights[out_matrix_index] = np.delete(child.weights[out_matrix_index], remove_index, axis=0)

        return child


# -----------------------------------------------------------------------------
# TRACK
# -----------------------------------------------------------------------------

class Track:
    def __init__(self):
        self.centerline = None
        self.checkpoints = []
        self.obstacles = []
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_angle = 0.0
        self.lap_length = 0.0
        self.road_surface = None
        self.drive_surface = None
        self.mask = None
        self.generate()

    def _generate_circle_track(self):
        cx = WIDTH * 0.5
        cy = HEIGHT * 0.5
        max_radius = min(WIDTH, HEIGHT) * 0.5 - ROAD_WIDTH * 0.65 - 8.0
        radius = clamp(TRACK_RADIUS, 70.0, max_radius)

        samples = 360
        centerline = []
        for i in range(samples):
            angle = (i / samples) * (2.0 * math.pi)
            px = cx + math.cos(angle) * radius
            py = cy + math.sin(angle) * radius
            centerline.append(np.array([px, py], dtype=np.float32))

        self.centerline = np.asarray(centerline, dtype=np.float32)

    def _generate_obstacles(self):
        self.obstacles = []
        n = len(self.centerline)
        start_guard = 34
        attempts = 0
        max_attempts = 1200

        while len(self.obstacles) < OBSTACLE_COUNT and attempts < max_attempts:
            attempts += 1
            idx = random.randint(0, n - 1)

            # Keep start area clear.
            if idx < start_guard or idx > n - start_guard:
                continue

            p = self.centerline[idx]
            prev_pt = self.centerline[idx - 1]
            next_pt = self.centerline[(idx + 1) % n]
            tangent = normalize(next_pt - prev_pt)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

            radial_offset = random.uniform(-ROAD_WIDTH * 0.2, ROAD_WIDTH * 0.2)
            pos = p + normal * radial_offset
            r = random.uniform(9.0, 15.0)

            ok = True
            for obstacle in self.obstacles:
                if np.linalg.norm(pos - obstacle["pos"]) < (r + obstacle["r"] + 16.0):
                    ok = False
                    break

            if ok:
                self.obstacles.append({"pos": pos, "r": r})

    def generate(self):
        self._generate_circle_track()
        self._generate_obstacles()
        self.lap_length = 2.0 * math.pi * float(np.linalg.norm(self.centerline[0] - np.array([WIDTH * 0.5, HEIGHT * 0.5], dtype=np.float32)))

        self.start_x = float(self.centerline[0][0])
        self.start_y = float(self.centerline[0][1])
        start_dir = normalize(self.centerline[1] - self.centerline[0])
        self.start_angle = math.atan2(float(start_dir[1]), float(start_dir[0]))

        self.checkpoints = []
        step = max(8, len(self.centerline) // TRACK_CHECKPOINT_DENSITY)
        for i in range(0, len(self.centerline), step):
            p = self.centerline[i]
            nvec = self.centerline[(i + 1) % len(self.centerline)] - self.centerline[i - 1]
            d = normalize(nvec)
            self.checkpoints.append(
                {"x": float(p[0]), "y": float(p[1]), "angle": math.atan2(float(d[1]), float(d[0]))}
            )

        self.build_surface()

    def _draw_track_body(self):
        # Draw as a connected ribbon, not stamped circles.
        pygame.draw.lines(
            self.road_surface,
            (*ROAD, 255),
            True,
            [(int(p[0]), int(p[1])) for p in self.centerline],
            int(ROAD_WIDTH),
        )

    def _draw_track_edges(self):
        n = len(self.centerline)
        left = []
        right = []
        half_w = ROAD_WIDTH * 0.5
        for i in range(n):
            prev_pt = self.centerline[i - 1]
            next_pt = self.centerline[(i + 1) % n]
            tangent = normalize(next_pt - prev_pt)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            curr = self.centerline[i]
            left.append(curr + normal * half_w)
            right.append(curr - normal * half_w)

        pygame.draw.lines(
            self.road_surface,
            (*ROAD_EDGE, 255),
            True,
            [(int(p[0]), int(p[1])) for p in left],
            2,
        )
        pygame.draw.lines(
            self.road_surface,
            (*ROAD_EDGE, 255),
            True,
            [(int(p[0]), int(p[1])) for p in right],
            2,
        )

    def build_surface(self):
        self.drive_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.drive_surface.fill((0, 0, 0, 0))
        pygame.draw.lines(
            self.drive_surface,
            (255, 255, 255, 255),
            True,
            [(int(p[0]), int(p[1])) for p in self.centerline],
            int(ROAD_WIDTH),
        )

        # Carve obstacles out of drivable mask for fast collision/raycast checks.
        for obstacle in self.obstacles:
            pos = obstacle["pos"]
            r = obstacle["r"]
            pygame.draw.circle(
                self.drive_surface,
                (0, 0, 0, 0),
                (int(pos[0]), int(pos[1])),
                int(r + 2.0),
            )

        self.mask = pygame.mask.from_surface(self.drive_surface)

        self.road_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self.road_surface.fill((0, 0, 0, 0))

        # Visual-only bounds: not part of collision mask.
        pygame.draw.lines(
            self.road_surface,
            (*ROAD_BOUNDARY, 255),
            True,
            [(int(p[0]), int(p[1])) for p in self.centerline],
            int(ROAD_WIDTH + 12),
        )

        self._draw_track_body()
        self._draw_track_edges()

        # Center guide line to make curvature and lane center obvious.
        for i in range(0, len(self.centerline), 12):
            a = self.centerline[i]
            b = self.centerline[(i + 6) % len(self.centerline)]
            pygame.draw.line(
                self.road_surface,
                (*ROAD_CENTERLINE, 210),
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                2,
            )

        for obstacle in self.obstacles:
            pos = obstacle["pos"]
            r = obstacle["r"]
            pygame.draw.circle(
                self.road_surface,
                OBSTACLE_COLOR,
                (int(pos[0]), int(pos[1])),
                int(r),
            )
            pygame.draw.circle(
                self.road_surface,
                (30, 30, 30),
                (int(pos[0]), int(pos[1])),
                int(r),
                2,
            )

    def is_on_track(self, x, y):
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= WIDTH or iy >= HEIGHT:
            return False
        return self.mask.get_at((ix, iy)) == 1

    def ray_distance(self, x, y, angle, max_distance=RAY_MAX_DISTANCE, step=RAY_STEP):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        for dist in range(0, max_distance + 1, step):
            px = int(x + cos_a * dist)
            py = int(y + sin_a * dist)
            if px < 0 or py < 0 or px >= WIDTH or py >= HEIGHT:
                return dist / max_distance
            if not self.is_on_track(px, py):
                return dist / max_distance
        return 1.0

    def draw(self, surface):
        surface.fill(GRASS)
        surface.blit(self.road_surface, (0, 0))
        p0 = self.centerline[0]
        p1 = self.centerline[1]
        tangent = normalize(p1 - p0)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        a = p0 + normal * (ROAD_WIDTH * 0.45)
        b = p0 - normal * (ROAD_WIDTH * 0.45)
        pygame.draw.line(surface, (255, 255, 255), (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 5)


# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------

class Agent:
    def __init__(self, x, y, angle, brain=None, width=10):
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)
        self.width = width

        self.brain = brain if brain is not None else Brain()
        self.color = brain_to_color(self.brain)

        self.speed = START_SPEED
        self.alive = True
        self.score = 0.0
        self.frames = 0
        self.checkpoint_index = 0
        self.laps_completed = 0
        self.distance_this_lap = 0.0
        self.inputs = np.zeros(INPUT_SIZE, dtype=np.float32)

    def reset_to_start(self, track):
        self.x = track.start_x
        self.y = track.start_y
        self.angle = track.start_angle
        self.speed = START_SPEED
        self.alive = True
        self.score = 0.0
        self.frames = 0
        self.checkpoint_index = 0
        self.laps_completed = 0
        self.distance_this_lap = 0.0

    def clamp(self):
        self.x = clamp(self.x, 0, WIDTH - 1)
        self.y = clamp(self.y, 0, HEIGHT - 1)

    def sense(self, track):
        ray_angles = np.linspace(-1.15, 1.15, RAY_COUNT)
        ray_values = []

        for offset in ray_angles:
            dist = track.ray_distance(self.x, self.y, self.angle + float(offset))
            ray_values.append(dist)

        speed_norm = clamp((self.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED), 0.0, 1.0)

        self.inputs = np.asarray(ray_values + [speed_norm], dtype=np.float32)
        return self.inputs

    def think(self, track):
        inputs = self.sense(track)
        outputs = self.brain.forward(inputs)

        steer = float(outputs[1] - outputs[0])
        throttle = float(outputs[2] - outputs[3])

        self.angle += steer * STEER_SPEED
        self.speed += throttle * ACCEL_SPEED
        self.speed -= FRICTION
        self.speed = clamp(self.speed, MIN_SPEED, MAX_SPEED)

        old_x, old_y = self.x, self.y
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.distance_this_lap += math.hypot(self.x - old_x, self.y - old_y)

    def update(self, track):
        if not self.alive:
            return

        self.think(track)
        self.clamp()

        if not track.is_on_track(self.x, self.y):
            self.alive = False
            return

        self.frames += 1
        self.score += 1.0

        checkpoint = track.checkpoints[self.checkpoint_index]
        dx = self.x - checkpoint["x"]
        dy = self.y - checkpoint["y"]
        dist = math.hypot(dx, dy)

        if dist < CHECKPOINT_HIT_RADIUS:
            self.checkpoint_index += 1
            if self.checkpoint_index >= len(track.checkpoints):
                if self.distance_this_lap >= track.lap_length * LAP_PROGRESS_MIN_RATIO:
                    self.laps_completed += 1
                    self.score += 2500.0
                self.checkpoint_index = 0
                self.distance_this_lap = 0.0

    def draw(self, surface):
        if not self.alive:
            return

        forward = np.array([math.cos(self.angle), math.sin(self.angle)], dtype=np.float32)
        right = np.array([-forward[1], forward[0]], dtype=np.float32)

        center = np.array([self.x, self.y], dtype=np.float32)
        nose = center + forward * 12
        left = center - forward * 7 + right * 6
        right_pt = center - forward * 7 - right * 6

        pygame.draw.polygon(
            surface,
            self.color,
            [
                (int(nose[0]), int(nose[1])),
                (int(left[0]), int(left[1])),
                (int(right_pt[0]), int(right_pt[1])),
            ],
        )

    def spawn_child(self, track):
        return Agent(
            track.start_x,
            track.start_y,
            track.start_angle,
            brain=self.brain.mutate(),
            width=self.width,
        )


# -----------------------------------------------------------------------------
# BRAIN VISUALISATION
# -----------------------------------------------------------------------------

def draw_brain(surface, brain, inputs, x=24, y=42):
    activations = brain.forward_with_activations(inputs)
    layer_sizes = brain.layer_sizes

    surface.fill((232, 232, 232))

    surface.blit(font.render("Best agent brain", True, BLACK), (12, 10))

    panel_w = surface.get_width()
    panel_h = surface.get_height()

    left_margin = x + 80
    right_margin = 140
    top_margin = y
    bottom_margin = 182

    x_positions = np.linspace(left_margin, panel_w - right_margin, len(layer_sizes))

    layer_positions = []
    for layer_index, layer_size in enumerate(layer_sizes):
        if layer_size == 1:
            ys = [panel_h * 0.5]
        else:
            usable_h = panel_h - top_margin - bottom_margin
            step = min(42.0, usable_h / max(layer_size - 1, 1))
            span = step * (layer_size - 1)
            top = top_margin + (usable_h - span) * 0.5
            ys = [top + i * step for i in range(layer_size)]

        xs = [x_positions[layer_index]] * layer_size
        layer_positions.append(list(zip(xs, ys)))

    for layer_index in range(len(brain.weights)):
        weights = brain.weights[layer_index]

        for i, (x1, y1) in enumerate(layer_positions[layer_index]):
            for j, (x2, y2) in enumerate(layer_positions[layer_index + 1]):
                w = float(weights[i][j])
                strength = int(min(255, abs(w) * 220))
                color = (0, strength, 0) if w >= 0 else (strength, 0, 0)
                pygame.draw.line(surface, color, (x1, y1), (x2, y2), 2)

    for layer_index, nodes in enumerate(layer_positions):
        is_input = layer_index == 0
        is_output = layer_index == len(layer_positions) - 1

        layer_name = "INPUT" if is_input else "OUTPUT" if is_output else f"H{layer_index}"
        surface.blit(font.render(layer_name, True, BLACK), (int(nodes[0][0]) - 18, 18))

        for node_index, (nx, ny) in enumerate(nodes):
            activation = float(abs(activations[layer_index][node_index]))
            brightness = int(min(255, activation * 255))

            if is_input:
                node_color = (brightness, brightness, brightness)
                radius = 11
            elif is_output:
                best_action = int(np.argmax(activations[-1]))
                node_color = (255, brightness, 0) if node_index == best_action else (175, 175, 175)
                radius = 14 if node_index == best_action else 11
            else:
                node_color = (brightness, brightness, 255)
                radius = 12

            pygame.draw.circle(surface, node_color, (int(nx), int(ny)), radius)

            if is_input:
                label = font.render(f"{INPUT_LABELS[node_index]}: {inputs[node_index]:.2f}", True, BLACK)
                surface.blit(label, (int(nx) - 90, int(ny) - 10))

            elif is_output:
                label = font.render(
                    f"{MOVE_LABELS[node_index]}: {activations[-1][node_index]:.2f}",
                    True,
                    BLACK,
                )
                surface.blit(label, (int(nx) + 18, int(ny) - 10))

    legend_y = panel_h - 88
    legend_lines = [
        "Green line = positive weight",
        "Red line = negative weight",
        "Yellow output = chosen action",
    ]
    for i, line in enumerate(legend_lines):
        surface.blit(font.render(line, True, BLACK), (12, legend_y + i * 20))


# -----------------------------------------------------------------------------
# EVOLUTION
# -----------------------------------------------------------------------------

def evolve(population, track):

    # winners = agents that completed at least one full lap
    winners = [
        agent
        for agent in population
        if agent.laps_completed > 0
    ]

    # fallback if nobody completed it
    if not winners:
        winners = sorted(
            population,
            key=lambda a: a.score,
            reverse=True
        )[:5]

    # only keep top 5
    winners = sorted(
        winners,
        key=lambda a: a.score,
        reverse=True
    )[:5]

    new_track = Track()

    new_population = []

    # preserve winners exactly
    for winner in winners:

        elite = Agent(
            new_track.start_x,
            new_track.start_y,
            new_track.start_angle,
            brain=winner.brain.copy()
        )

        elite.reset_to_start(new_track)

        new_population.append(elite)

    # mutated children
    while len(new_population) < POPULATION:

        parent = random.choice(winners)

        child = parent.spawn_child(new_track)

        child.reset_to_start(new_track)

        new_population.append(child)

    return new_population, new_track


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    global running

    track = Track()
    agents = [
        Agent(track.start_x, track.start_y, track.start_angle)
        for _ in range(POPULATION)
    ]

    generation = 1
    frame_in_generation = 0
    best_ever_score = -1.0
    best_ever = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for agent in agents:
            agent.update(track)

        alive_agents = [a for a in agents if a.alive]
        best_agent = max(agents, key=lambda a: a.score)

        if best_agent.score > best_ever_score:
            best_ever_score = best_agent.score
            best_ever = best_agent

        frame_in_generation += 1

        if not alive_agents or frame_in_generation >= MAX_GENERATION_FRAMES:

            agents, track = evolve(agents, track)

            generation += 1
            frame_in_generation = 0

        # RENDER
        window.fill(WHITE)
        track.draw(window)

        for agent in agents:
            agent.draw(window)

        alive_count = sum(1 for a in agents if a.alive)

        window.blit(font.render(f"Generation: {generation}", True, BLACK), (12, 10))
        window.blit(font.render(f"Alive: {alive_count}/{POPULATION}", True, BLACK), (12, 30))
        window.blit(font.render(f"Best score: {best_agent.score:.0f}", True, BLACK), (12, 50))

        if best_agent is not None:
            brain_inputs = best_agent.sense(track)
            draw_brain(brain_screen, best_agent.brain, brain_inputs)
        else:
            brain_screen.fill((232, 232, 232))

        window.blit(brain_screen, (WIDTH, 0))
        pygame.draw.line(window, (30, 30, 30), (WIDTH, 0), (WIDTH, HEIGHT), 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    raise SystemExit


if __name__ == "__main__":
    main()
