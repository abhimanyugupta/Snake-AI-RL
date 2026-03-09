import math
import random
from dataclasses import dataclass
from enum import Enum

import pygame


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass(frozen=True)
class Point:
    x: int
    y: int


class SnakeGameAI:
    """Playable snake game that also exposes helper methods for RL training."""

    def __init__(
        self,
        w=640,
        h=560,
        block_size=20,
        speed=12,
        render=True,
        sidebar_width=680,
        window_h=None,
    ):
        pygame.init()

        self.board_w = w
        self.board_h = h
        self.w = w
        self.h = h
        self.block_size = block_size
        self.speed = speed
        self.render = render
        self.sidebar_width = sidebar_width if render else 0
        requested_window_h = window_h if window_h is not None else self.board_h
        self.window_w = self.board_w + self.sidebar_width
        self.window_h = max(self.board_h, requested_window_h)
        self.quit_requested = False
        self.dashboard_data = {}
        self.reward_config = {"food": 10.0, "death": -10.0, "step": 0.0}

        if self.render:
            self.display = pygame.display.set_mode((self.window_w, self.window_h))
            pygame.display.set_caption("Snake RL Dashboard")
            self.title_font = pygame.font.SysFont("arial", 26, bold=True)
            self.font = pygame.font.SysFont("arial", 20)
            self.small_font = pygame.font.SysFont("arial", 16)
            self.tiny_font = pygame.font.SysFont("arial", 14)
            # Pre-allocate reusable alpha surfaces to avoid per-frame allocation
            self._overlay_surface = pygame.Surface((self.board_w, self.board_h), pygame.SRCALPHA)
            glow_size = self.block_size * 3
            self._glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        else:
            self.display = None
            self.title_font = None
            self.font = None
            self.small_font = None
            self.tiny_font = None
            self._overlay_surface = None
            self._glow_surface = None

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Reset the game so training can start a fresh episode."""
        start_x = (self.board_w // 2 // self.block_size) * self.block_size
        start_y = (self.board_h // 2 // self.block_size) * self.block_size

        self.direction = Direction.RIGHT
        self.head = Point(start_x, start_y)
        self.snake = [
            self.head,
            Point(start_x - self.block_size, start_y),
            Point(start_x - (2 * self.block_size), start_y),
        ]

        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()

    def set_dashboard_data(self, data):
        self.dashboard_data = dict(data or {})

    def set_reward_config(self, reward_config):
        self.reward_config = {
            "food": float(reward_config.get("food", 10.0)),
            "death": float(reward_config.get("death", -10.0)),
            "step": float(reward_config.get("step", 0.0)),
        }

    def handle_system_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.quit_requested = True

    def draw(self):
        if self.render:
            self._draw_scene()

    def get_relative_points(self):
        return {
            "straight": self._point_from_direction(self.head, self.direction),
            "right": self._point_from_direction(self.head, self._turn_right(self.direction)),
            "left": self._point_from_direction(self.head, self._turn_left(self.direction)),
        }

    def _place_food(self):
        """Place food on a free grid cell."""
        max_x = (self.board_w - self.block_size) // self.block_size
        max_y = (self.board_h - self.block_size) // self.block_size

        while True:
            x = random.randint(0, max_x) * self.block_size
            y = random.randint(0, max_y) * self.block_size
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action=None, events=None, draw_frame=True):
        """
        Advance the game by one frame.

        - If action is None, the game uses keyboard input.
        - If action is [1, 0, 0], [0, 1, 0], or [0, 0, 1], the snake is
          controlled by the agent.
        """
        if self.quit_requested:
            return 0.0, True, self.score

        self.frame_iteration += 1

        if events is None:
            events = pygame.event.get() if self.render else []

        self.handle_system_events(events)
        if self.quit_requested:
            return 0.0, True, self.score

        if action is None:
            self._handle_human_input(events)
            self._move()
        else:
            self._move(action)

        self.snake.insert(0, self.head)
        reward = self.reward_config["step"]
        game_over = False

        # End the game if the snake hits a wall, itself, or loops for too long.
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = self.reward_config["death"]
            game_over = True
        elif self.head == self.food:
            self.score += 1
            reward = self.reward_config["food"]
            self._place_food()
        else:
            self.snake.pop()

        if self.render and draw_frame:
            self._draw_scene()

        self.clock.tick(self.speed)
        return reward, game_over, self.score

    def is_collision(self, point=None):
        """Check whether a point hits a wall or the snake body."""
        if point is None:
            point = self.head

        if point.x < 0 or point.x >= self.board_w or point.y < 0 or point.y >= self.board_h:
            return True

        if point in self.snake[1:]:
            return True

        return False

    def _handle_human_input(self, events):
        """Allow the player to move with arrow keys or WASD."""
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue

            if event.key in (pygame.K_LEFT, pygame.K_a):
                self._change_direction(Direction.LEFT)
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self._change_direction(Direction.RIGHT)
            elif event.key in (pygame.K_UP, pygame.K_w):
                self._change_direction(Direction.UP)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self._change_direction(Direction.DOWN)

    def _change_direction(self, new_direction):
        opposite = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
        }

        if new_direction != opposite[self.direction]:
            self.direction = new_direction

    def _move(self, action=None):
        """
        Move the snake.

        The agent action is a one-hot list:
        [1, 0, 0] = keep going straight
        [0, 1, 0] = turn right
        [0, 0, 1] = turn left
        """
        if action is not None:
            action = list(action)
            if len(action) != 3 or sum(action) != 1:
                raise ValueError("Action must be a one-hot list like [1, 0, 0].")
            self.direction = self._direction_for_action(action)

        self.head = self._point_from_direction(self.head, self.direction)

    def _direction_for_action(self, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_index = clockwise.index(self.direction)
        action_index = list(action).index(1)

        if action_index == 0:
            return clockwise[current_index]
        if action_index == 1:
            return clockwise[(current_index + 1) % 4]
        return clockwise[(current_index - 1) % 4]

    def _point_from_direction(self, point, direction):
        if direction == Direction.RIGHT:
            return Point(point.x + self.block_size, point.y)
        if direction == Direction.LEFT:
            return Point(point.x - self.block_size, point.y)
        if direction == Direction.UP:
            return Point(point.x, point.y - self.block_size)
        return Point(point.x, point.y + self.block_size)

    def _turn_right(self, direction):
        turns = {
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP,
            Direction.UP: Direction.RIGHT,
        }
        return turns[direction]

    def _turn_left(self, direction):
        turns = {
            Direction.RIGHT: Direction.UP,
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
        }
        return turns[direction]

    def _draw_scene(self):
        if not self.render or self.display is None:
            return

        self.display.fill((18, 18, 18))
        self._draw_board_background()
        self._draw_danger_overlays()
        self._draw_snake_and_food()
        self._draw_action_arrows()
        self._draw_sidebar()
        self._draw_overlay_message()
        pygame.display.flip()

    def _draw_board_background(self):
        board_rect = pygame.Rect(0, 0, self.board_w, self.board_h)
        # Deep space premium navy/slate background
        pygame.draw.rect(self.display, (16, 18, 24), board_rect)

        # Subtle grid lines
        for x in range(0, self.board_w, self.block_size):
            pygame.draw.line(self.display, (28, 32, 40), (x, 0), (x, self.board_h), 1)
        for y in range(0, self.board_h, self.block_size):
            pygame.draw.line(self.display, (28, 32, 40), (0, y), (self.board_w, y), 1)

        # Premium outer board border
        pygame.draw.rect(self.display, (50, 60, 80), board_rect, width=2)
        pygame.draw.rect(self.display, (35, 45, 60), board_rect.inflate(4, 4), width=2)

    def _draw_snake_and_food(self):
        # --- Draw Premium Food (Pulsing Glow) ---
        pulse = (math.sin(pygame.time.get_ticks() / 200.0) + 1) / 2  # 0 to 1
        food_center = (self.food.x + self.block_size // 2, self.food.y + self.block_size // 2)
        
        # Glow effect (only drawn if not in turbo mode to save FPS)
        toggles = self.dashboard_data.get("toggles", []) if self.dashboard_data else []
        is_turbo = toggles[4].get("value", False) if len(toggles) > 4 else False
        if not is_turbo and self._glow_surface is not None:
            self._glow_surface.fill((0, 0, 0, 0))  # Clear cached surface
            glow_radius = int(self.block_size * 0.8 + pulse * self.block_size * 0.4)
            pygame.draw.circle(self._glow_surface, (255, 60, 60, 40 + int(pulse * 30)), (self.block_size * 1.5, self.block_size * 1.5), glow_radius)
            self.display.blit(self._glow_surface, (self.food.x - self.block_size, self.food.y - self.block_size))

        # Core food apple
        food_rect = pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size)
        pygame.draw.rect(self.display, (255, 70, 70), food_rect, border_radius=self.block_size // 2)
        pygame.draw.rect(self.display, (255, 180, 180), food_rect.inflate(-8, -8), border_radius=self.block_size // 2)

        # --- Draw Premium Snake (Gradients & Joints) ---
        n = len(self.snake)
        for index, part in enumerate(reversed(self.snake)):
            # Reversed so head is drawn last and on top
            true_idx = n - 1 - index
            
            # Gradient: Head is bright neon green/cyan, tail is dark teal
            ratio = true_idx / max(1, n - 1)
            r = int(50 * (1 - ratio) + 20 * ratio)
            g = int(240 * (1 - ratio) + 120 * ratio)
            b = int(140 * (1 - ratio) + 180 * ratio)
            color = (r, g, b)
            
            rect = pygame.Rect(part.x, part.y, self.block_size, self.block_size)
            
            if true_idx == 0:
                # Head
                pygame.draw.rect(self.display, (255, 255, 255), rect, border_radius=6)
                pygame.draw.rect(self.display, color, rect.inflate(-4, -4), border_radius=4)
                
                # Draw Eyes
                eye_color = (20, 20, 30)
                cx, cy = part.x + self.block_size // 2, part.y + self.block_size // 2
                offset = 4
                if self.direction == Direction.RIGHT:
                    pygame.draw.circle(self.display, eye_color, (cx + offset, cy - offset), 2)
                    pygame.draw.circle(self.display, eye_color, (cx + offset, cy + offset), 2)
                elif self.direction == Direction.LEFT:
                    pygame.draw.circle(self.display, eye_color, (cx - offset, cy - offset), 2)
                    pygame.draw.circle(self.display, eye_color, (cx - offset, cy + offset), 2)
                elif self.direction == Direction.UP:
                    pygame.draw.circle(self.display, eye_color, (cx - offset, cy - offset), 2)
                    pygame.draw.circle(self.display, eye_color, (cx + offset, cy - offset), 2)
                elif self.direction == Direction.DOWN:
                    pygame.draw.circle(self.display, eye_color, (cx - offset, cy + offset), 2)
                    pygame.draw.circle(self.display, eye_color, (cx + offset, cy + offset), 2)
                    
            else:
                # Body segment
                pygame.draw.rect(self.display, color, rect.inflate(-2, -2), border_radius=4)
                
            # Connect the joints with a circle for a continuous tube look
            if true_idx > 0:
                prev_part = self.snake[true_idx - 1]
                joint_x = (part.x + prev_part.x) // 2 + self.block_size // 2
                joint_y = (part.y + prev_part.y) // 2 + self.block_size // 2
                
                # Use the color of the segment closer to the head for the joint
                joint_ratio = (true_idx - 0.5) / max(1, n - 1)
                jr = int(50 * (1 - joint_ratio) + 20 * joint_ratio)
                jg = int(240 * (1 - joint_ratio) + 120 * joint_ratio)
                jb = int(140 * (1 - joint_ratio) + 180 * joint_ratio)
                
                pygame.draw.circle(self.display, (jr, jg, jb), (joint_x, joint_y), self.block_size // 2 - 1)

    def _draw_danger_overlays(self):
        data = self.dashboard_data
        if not data or not data.get("show_dangers") or self._overlay_surface is None:
            return

        self._overlay_surface.fill((0, 0, 0, 0))  # Clear cached surface
        candidate_points = data.get("candidate_points", {})
        deadly_moves = data.get("deadly_moves", {})

        for key, point in candidate_points.items():
            if not deadly_moves.get(key):
                continue

            draw_point = self._clamp_point_to_board(point)
            rect = pygame.Rect(draw_point.x, draw_point.y, self.block_size, self.block_size)
            pygame.draw.rect(self._overlay_surface, (220, 70, 70, 130), rect, border_radius=5)
            pygame.draw.rect(self._overlay_surface, (255, 180, 180, 180), rect, width=2, border_radius=5)

            label_surface = self.small_font.render(key[0].upper(), True, (255, 255, 255))
            self._overlay_surface.blit(label_surface, (rect.x + 4, rect.y + 2))

        self.display.blit(self._overlay_surface, (0, 0))

    def _draw_action_arrows(self):
        data = self.dashboard_data
        if not data or not data.get("show_arrows"):
            return

        candidate_points = data.get("candidate_points", {})
        deadly_moves = data.get("deadly_moves", {})
        action_key = data.get("action_key")
        q_values = data.get("q_values", [0.0, 0.0, 0.0])
        decision_type = data.get("decision_type", "")

        start = (self.head.x + self.block_size // 2, self.head.y + self.block_size // 2)
        key_order = ["straight", "right", "left"]
        key_labels = {"straight": "S", "right": "R", "left": "L"}

        for index, key in enumerate(key_order):
            point = candidate_points.get(key)
            if point is None:
                continue

            end_point = self._clamp_point_to_board(point)
            end = (end_point.x + self.block_size // 2, end_point.y + self.block_size // 2)

            color = (110, 160, 255)
            if deadly_moves.get(key):
                color = (255, 90, 90)
            if key == action_key:
                color = (255, 196, 68) if decision_type == "explore" else (95, 236, 124)
            if decision_type == "policy preview" and key == action_key:
                color = (120, 210, 255)

            self._draw_arrow(start, end, color)

            label = f"{key_labels[key]} {q_values[index]:.2f}"
            label_surface = self.tiny_font.render(label, True, color)
            label_rect = label_surface.get_rect(center=(end[0], end[1] - 14))
            self.display.blit(label_surface, label_rect)

    def _draw_arrow(self, start, end, color):
        pygame.draw.line(self.display, color, start, end, 4)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return

        ux = dx / length
        uy = dy / length
        size = 10
        left = (end[0] - ux * size - uy * size / 2, end[1] - uy * size + ux * size / 2)
        right = (end[0] - ux * size + uy * size / 2, end[1] - uy * size - ux * size / 2)
        pygame.draw.polygon(self.display, color, [end, left, right])

    def _draw_sidebar(self):
        if self.sidebar_width <= 0:
            return

        data = self.dashboard_data
        panel_x = self.board_w
        panel_rect = pygame.Rect(panel_x, 0, self.sidebar_width, self.window_h)
        
        # Premium background gradient effect (simulated with slightly lighter top)
        pygame.draw.rect(self.display, (22, 23, 28), panel_rect)
        top_glow = pygame.Rect(panel_x, 0, self.sidebar_width, 100)
        # We can't easily draw true gradients in raw pygame without surfaces, but we can do a solid color 
        # that looks a bit richer. Let's stick to a solid premium dark color and use borders to define shape.
        
        # Draw a subtle separator line with a shadow effect
        pygame.draw.line(self.display, (15, 15, 18), (panel_x - 1, 0), (panel_x - 1, self.window_h), 2)
        pygame.draw.line(self.display, (45, 48, 56), (panel_x, 0), (panel_x, self.window_h), 1)

        # Setup 2-column grid layout
        padding = 20
        col_gap = 20
        col_w = (self.sidebar_width - (padding * 2) - col_gap) // 2
        
        left_col = pygame.Rect(panel_x + padding, padding, col_w, self.window_h - padding * 2)
        right_col = pygame.Rect(panel_x + padding + col_w + col_gap, padding, col_w, self.window_h - padding * 2)

        # Title
        title = data.get("panel_title", "Snake Dashboard")
        title_surface = self.title_font.render(title, True, (250, 252, 255))
        self.display.blit(title_surface, (left_col.x, left_col.y))
        
        # Move y down for both columns after title
        title_h = title_surface.get_height() + 20
        left_col.y += title_h
        left_col.height -= title_h
        right_col.y += title_h
        right_col.height -= title_h

        if not data:
            manual_lines = [
                f"Score: {self.score}",
                "Use Arrow keys or WASD.",
                "Run train.py for the RL dashboard.",
                "Model: Q-table dictionary.",
            ]
            for index, line in enumerate(manual_lines):
                surface = self.small_font.render(line, True, (200, 205, 215))
                self.display.blit(surface, (left_col.x, left_col.y + index * 28))
            return

        # --- LEFT COLUMN: Metrics -> Controls ---
        # Draw Metrics Card
        metrics_h = len(data.get("metrics", [])) * 26 + 15
        metrics_rect = pygame.Rect(left_col.x, left_col.y, left_col.width, metrics_h)
        self._draw_card_background(metrics_rect)
        
        metrics_y = metrics_rect.y + 10
        for label, value in data.get("metrics", []):
            line = f"{label}: {value}"
            # Make label slightly dimmer than value for premium look
            label_surf = self.small_font.render(f"{label}:", True, (160, 165, 175))
            val_surf = self.small_font.render(str(value), True, (240, 245, 255))
            self.display.blit(label_surf, (metrics_rect.x + 12, metrics_y))
            self.display.blit(val_surf, (metrics_rect.x + 12 + label_surf.get_width() + 6, metrics_y))
            metrics_y += 24
            
        left_col.y += metrics_h + 15
        left_col.height -= (metrics_h + 15)

        # Draw Controls (will render themselves relative to their data dicts)
        self._draw_controls(data)

        # --- RIGHT COLUMN: Q-Values -> Graph -> State ---
        # Q-Values
        q_h = 130
        q_rect = pygame.Rect(right_col.x, right_col.y, right_col.width, q_h)
        if data.get("q_values") is not None:
            self._draw_q_values(q_rect, data)
            
        right_col.y += q_h + 15
        right_col.height -= (q_h + 15)
            
        # Graph
        graph_h = 240
        graph_rect = pygame.Rect(right_col.x, right_col.y, right_col.width, graph_h)
        self._draw_graph(graph_rect, data)
        
        right_col.y += graph_h + 15
        right_col.height -= (graph_h + 15)
        
        # State block - takes remaining space
        if data.get("state_lines"):
            self._draw_state_block(right_col, data)

    def _draw_card_background(self, rect):
        """Draw a premium dark card with subtle borders."""
        pygame.draw.rect(self.display, (30, 32, 38), rect, border_radius=8)
        pygame.draw.rect(self.display, (55, 60, 70), rect, width=1, border_radius=8)
        # Subtle drop shadow at bottom
        pygame.draw.line(self.display, (15, 16, 20), (rect.x + 4, rect.bottom), (rect.right - 4, rect.bottom), 2)

    def _draw_controls(self, data):
        for slider in data.get("sliders", []):
            label_surface = self.tiny_font.render(
                f"{slider['label']}: {slider['value_text']}",
                True,
                (210, 215, 225),
            )
            self.display.blit(label_surface, (slider["x"], slider["y"]))

            track_rect = pygame.Rect(
                slider["track_x"], slider["track_y"], slider["track_w"], slider["track_h"]
            )
            fill_w = max(0, min(track_rect.width, int(track_rect.width * slider["ratio"])))
            fill_rect = pygame.Rect(track_rect.x, track_rect.y, fill_w, track_rect.height)

            # Premium slider track
            pygame.draw.rect(self.display, (45, 48, 55), track_rect, border_radius=6)
            pygame.draw.rect(self.display, (30, 32, 36), track_rect, width=1, border_radius=6)
            
            # Premium slider fill (gradient-like via inner rect, but solid for now)
            pygame.draw.rect(self.display, (80, 190, 255), fill_rect, border_radius=6)
            
            # Premium knob (larger, layered for glow effect)
            knob_center = (slider["knob_x"], slider["knob_y"])
            pygame.draw.circle(self.display, (40, 140, 220), knob_center, slider["knob_radius"] + 2)
            pygame.draw.circle(self.display, (250, 252, 255), knob_center, slider["knob_radius"])

        for toggle in data.get("toggles", []):
            rect = pygame.Rect(toggle["x"], toggle["y"], toggle["w"], toggle["h"])
            # Premium toggle buttons
            base_color = (65, 180, 100) if toggle["value"] else (50, 52, 60)
            hover_color = (80, 200, 115) if toggle["value"] else (60, 62, 70)
            border_color = (45, 140, 75) if toggle["value"] else (40, 42, 48)
            
            pygame.draw.rect(self.display, base_color, rect, border_radius=6)
            pygame.draw.rect(self.display, border_color, rect, width=2, border_radius=6)

            text_color = (255, 255, 255) if toggle["value"] else (180, 185, 195)
            label_surface = self.tiny_font.render(toggle["label"], True, text_color)
            # Center text in rect
            text_rect = label_surface.get_rect(center=rect.center)
            self.display.blit(label_surface, text_rect)

        for input_box in data.get("inputs", []):
            label_surface = self.tiny_font.render(input_box["label"], True, (210, 215, 225))
            self.display.blit(label_surface, (input_box["x"], input_box["y"] - 20))

            rect = pygame.Rect(input_box["x"], input_box["y"], input_box["w"], input_box["h"])
            # Premium text input
            fill_color = (20, 22, 26) if input_box.get("active") else (28, 30, 36)
            border_color = (100, 210, 255) if input_box.get("active") else (60, 65, 75)
            pygame.draw.rect(self.display, fill_color, rect, border_radius=6)
            pygame.draw.rect(self.display, border_color, rect, width=2, border_radius=6)

            value_text = input_box.get("text") or input_box.get("hint", "")
            value_color = (250, 252, 255) if input_box.get("text") else (120, 125, 135)
            value_surface = self.small_font.render(value_text, True, value_color)
            
            # Simple cursor blink effect
            if input_box.get("active") and pygame.time.get_ticks() % 1000 < 500:
                cursor_x = rect.x + 10 + value_surface.get_width()
                pygame.draw.line(self.display, (250, 252, 255), (cursor_x, rect.y + 6), (cursor_x, rect.bottom - 6), 2)
            
            self.display.blit(value_surface, (rect.x + 10, rect.y + 4))

    def _draw_q_values(self, rect, data):
        self._draw_card_background(rect)
        
        y = rect.y + 12
        heading = self.small_font.render("Q-Values & Policy", True, (250, 252, 255))
        self.display.blit(heading, (rect.x + 12, y))
        y += 28

        q_values = data.get("q_values", [0.0, 0.0, 0.0])
        action_labels = data.get("action_labels", ["Straight", "Right", "Left"])
        action_index = data.get("action_index")
        decision_type = data.get("decision_type", "")

        max_abs = max(1.0, max(abs(value) for value in q_values))
        
        # Calculate layut dynamically to fit rect
        label_w = 60
        val_w = 45
        bar_x = rect.x + 12 + label_w
        bar_w = rect.width - 24 - label_w - val_w
        
        for index, label in enumerate(action_labels):
            bar_y = y + index * 26
            value = q_values[index]
            fill_w = int((abs(value) / max_abs) * bar_w)
            
            # Premium colors
            color = (80, 230, 120) if value >= 0 else (255, 100, 100)
            if index == action_index:
                color = (255, 190, 60) if decision_type == "explore" else color
                if decision_type == "policy preview":
                    color = (100, 200, 255)

            label_surface = self.tiny_font.render(label, True, (200, 205, 215))
            self.display.blit(label_surface, (rect.x + 12, bar_y + 2))
            
            # Track
            pygame.draw.rect(self.display, (20, 22, 26), (bar_x, bar_y + 6, bar_w, 10), border_radius=5)
            # Fill
            pygame.draw.rect(self.display, color, (bar_x, bar_y + 6, fill_w, 10), border_radius=5)

            value_surface = self.tiny_font.render(f"{value:.2f}", True, (240, 245, 255))
            self.display.blit(value_surface, (bar_x + bar_w + 8, bar_y + 2))

    def _draw_state_block(self, rect, data):
        self._draw_card_background(rect)
        
        y = rect.y + 12
        heading = self.small_font.render("State Representation", True, (250, 252, 255))
        self.display.blit(heading, (rect.x + 12, y))
        y += 26

        for line in data.get("state_lines", []):
            surface = self.tiny_font.render(line, True, (190, 195, 205))
            self.display.blit(surface, (rect.x + 12, y))
            y += 20

        y += 10
        for line in data.get("help_lines", []):
            surface = self.tiny_font.render(line, True, (130, 200, 255))
            self.display.blit(surface, (rect.x + 12, y))
            y += 18

    def _draw_graph(self, rect, data):
        if not data.get("show_graph"):
            return

        self._draw_card_background(rect)

        title = self.tiny_font.render("Learning Graph: Score & Moving Avg", True, (250, 252, 255))
        self.display.blit(title, (rect.x + 12, rect.y + 12))

        scores = data.get("graph_scores", [])
        averages = data.get("graph_averages", [])
        if len(scores) < 2:
            no_data = self.tiny_font.render("Waiting for data...", True, (120, 125, 135))
            self.display.blit(no_data, (rect.x + 12, rect.y + 40))
            return

        plot_rect = pygame.Rect(rect.x + 15, rect.y + 35, rect.width - 30, rect.height - 50)
        
        # Plot area background
        pygame.draw.rect(self.display, (20, 22, 26), plot_rect, border_radius=4)
        pygame.draw.rect(self.display, (40, 42, 48), plot_rect, width=1, border_radius=4)
        
        # Baseline
        pygame.draw.line(
            self.display,
            (60, 65, 75),
            (plot_rect.x, plot_rect.bottom),
            (plot_rect.right, plot_rect.bottom),
            1,
        )

        max_value = max(max(scores), max(averages), 1)
        self._draw_graph_line(plot_rect, scores, max_value, (80, 190, 255), thickness=2)
        self._draw_graph_line(plot_rect, averages, max_value, (80, 230, 120), thickness=3)

    def _draw_graph_line(self, rect, values, max_value, color, thickness=2):
        if len(values) < 2:
            return

        points = []
        for index, value in enumerate(values):
            ratio_x = index / (len(values) - 1)
            ratio_y = 0 if max_value == 0 else value / max_value
            x = rect.x + int(ratio_x * rect.width)
            y = rect.bottom - int(ratio_y * rect.height)
            
            # Keep inside plot area
            y = max(rect.top, min(rect.bottom, y))
            points.append((x, y))

        if len(points) >= 2:
            # Draw premium smoothed line
            pygame.draw.lines(self.display, color, False, points, thickness)

    def _draw_overlay_message(self):
        title = self.dashboard_data.get("overlay_title")
        subtitle = self.dashboard_data.get("overlay_subtitle")
        if not title:
            return

        overlay = pygame.Surface((self.board_w, self.board_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.display.blit(overlay, (0, 0))

        title_surface = self.title_font.render(title, True, (255, 255, 255))
        subtitle_surface = self.small_font.render(subtitle or "", True, (235, 235, 235))
        title_rect = title_surface.get_rect(center=(self.board_w // 2, self.board_h // 2 - 14))
        subtitle_rect = subtitle_surface.get_rect(center=(self.board_w // 2, self.board_h // 2 + 16))
        self.display.blit(title_surface, title_rect)
        self.display.blit(subtitle_surface, subtitle_rect)

    def _clamp_point_to_board(self, point):
        max_x = self.board_w - self.block_size
        max_y = self.board_h - self.block_size
        return Point(
            min(max(point.x, 0), max_x),
            min(max(point.y, 0), max_y),
        )

    def show_game_over_screen(self):
        """Simple restart screen for manual play."""
        if not self.render:
            return False

        self.set_dashboard_data(
            {
                "overlay_title": "Game Over",
                "overlay_subtitle": "Press R to restart or Q to quit",
            }
        )

        while not self.quit_requested:
            events = pygame.event.get()
            self.handle_system_events(events)

            for event in events:
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key in (pygame.K_r, pygame.K_RETURN):
                    self.set_dashboard_data({})
                    return True
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.quit_requested = True
                    return False

            self._draw_scene()
            self.clock.tick(10)

        return False

    def close(self):
        pygame.quit()


def run_human_game():
    game = SnakeGameAI(render=True, speed=12)

    try:
        while not game.quit_requested:
            game.set_dashboard_data(
                {
                    "panel_title": "Manual Play",
                    "metrics": [
                        ("Score", game.score),
                        ("Model", "Human control"),
                    ],
                    "state_lines": [
                        "Use Arrow keys or WASD to move.",
                        "Run train.py for the RL dashboard.",
                    ],
                    "help_lines": [],
                }
            )
            _, game_over, _ = game.play_step()
            if game_over:
                if game.show_game_over_screen():
                    game.reset()
                else:
                    break
    finally:
        game.close()


if __name__ == "__main__":
    run_human_game()




