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
        sidebar_width=360,
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
        else:
            self.display = None
            self.title_font = None
            self.font = None
            self.small_font = None
            self.tiny_font = None

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
        pygame.draw.rect(self.display, (24, 32, 24), board_rect)

        for x in range(0, self.board_w, self.block_size):
            pygame.draw.line(self.display, (36, 48, 36), (x, 0), (x, self.board_h), 1)
        for y in range(0, self.board_h, self.block_size):
            pygame.draw.line(self.display, (36, 48, 36), (0, y), (self.board_w, y), 1)

        pygame.draw.rect(self.display, (70, 90, 70), board_rect, width=2)

    def _draw_snake_and_food(self):
        food_rect = pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size)
        pygame.draw.rect(self.display, (231, 76, 60), food_rect, border_radius=5)
        pygame.draw.rect(self.display, (120, 30, 30), food_rect, width=2, border_radius=5)

        for index, part in enumerate(self.snake):
            body_color = (79, 220, 130) if index == 0 else (46, 175, 99)
            rect = pygame.Rect(part.x, part.y, self.block_size, self.block_size)
            pygame.draw.rect(self.display, body_color, rect, border_radius=5)
            pygame.draw.rect(self.display, (12, 70, 32), rect, width=2, border_radius=5)

    def _draw_danger_overlays(self):
        data = self.dashboard_data
        if not data or not data.get("show_dangers"):
            return

        overlay = pygame.Surface((self.board_w, self.board_h), pygame.SRCALPHA)
        candidate_points = data.get("candidate_points", {})
        deadly_moves = data.get("deadly_moves", {})

        for key, point in candidate_points.items():
            if not deadly_moves.get(key):
                continue

            draw_point = self._clamp_point_to_board(point)
            rect = pygame.Rect(draw_point.x, draw_point.y, self.block_size, self.block_size)
            pygame.draw.rect(overlay, (220, 70, 70, 130), rect, border_radius=5)
            pygame.draw.rect(overlay, (255, 180, 180, 180), rect, width=2, border_radius=5)

            label_surface = self.small_font.render(key[0].upper(), True, (255, 255, 255))
            overlay.blit(label_surface, (rect.x + 4, rect.y + 2))

        self.display.blit(overlay, (0, 0))

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
        pygame.draw.rect(self.display, (27, 27, 31), panel_rect)
        pygame.draw.line(self.display, (70, 70, 80), (panel_x, 0), (panel_x, self.window_h), 2)

        title = data.get("panel_title", "Snake Dashboard")
        title_surface = self.title_font.render(title, True, (245, 245, 245))
        self.display.blit(title_surface, (panel_x + 18, 14))

        if not data:
            manual_lines = [
                f"Score: {self.score}",
                "Use Arrow keys or WASD.",
                "Run train.py for the RL dashboard.",
                "Model: Q-table dictionary.",
            ]
            for index, line in enumerate(manual_lines):
                surface = self.small_font.render(line, True, (220, 220, 220))
                self.display.blit(surface, (panel_x + 18, 60 + index * 24))
            return

        metrics_y = 54
        for label, value in data.get("metrics", []):
            line = f"{label}: {value}"
            surface = self.small_font.render(line, True, (225, 225, 225))
            self.display.blit(surface, (panel_x + 18, metrics_y))
            metrics_y += 22

        self._draw_controls(panel_x + 18, data)
        if data.get("q_values") is not None:
            self._draw_q_values(panel_x + 18, data)
        self._draw_graph(panel_x + 18, data)
        if data.get("state_lines"):
            self._draw_state_block(panel_x + 18, data)

    def _draw_controls(self, panel_x, data):
        for slider in data.get("sliders", []):
            label_surface = self.tiny_font.render(
                f"{slider['label']}: {slider['value_text']}",
                True,
                (235, 235, 235),
            )
            self.display.blit(label_surface, (slider["x"], slider["y"]))

            track_rect = pygame.Rect(
                slider["track_x"], slider["track_y"], slider["track_w"], slider["track_h"]
            )
            fill_w = max(0, min(track_rect.width, int(track_rect.width * slider["ratio"])))
            fill_rect = pygame.Rect(track_rect.x, track_rect.y, fill_w, track_rect.height)

            pygame.draw.rect(self.display, (60, 64, 72), track_rect, border_radius=5)
            pygame.draw.rect(self.display, (87, 200, 255), fill_rect, border_radius=5)
            pygame.draw.circle(
                self.display,
                (245, 245, 245),
                (slider["knob_x"], slider["knob_y"]),
                slider["knob_radius"],
            )

        for toggle in data.get("toggles", []):
            rect = pygame.Rect(toggle["x"], toggle["y"], toggle["w"], toggle["h"])
            color = (78, 196, 111) if toggle["value"] else (70, 70, 80)
            pygame.draw.rect(self.display, color, rect, border_radius=8)
            pygame.draw.rect(self.display, (25, 25, 30), rect, width=1, border_radius=8)

            label_surface = self.tiny_font.render(toggle["label"], True, (245, 245, 245))
            self.display.blit(label_surface, (rect.x + 8, rect.y + 6))

        for input_box in data.get("inputs", []):
            label_surface = self.tiny_font.render(input_box["label"], True, (235, 235, 235))
            self.display.blit(label_surface, (input_box["x"], input_box["y"] - 18))

            rect = pygame.Rect(input_box["x"], input_box["y"], input_box["w"], input_box["h"])
            fill_color = (42, 46, 58) if input_box.get("active") else (30, 33, 41)
            border_color = (100, 210, 255) if input_box.get("active") else (85, 85, 95)
            pygame.draw.rect(self.display, fill_color, rect, border_radius=8)
            pygame.draw.rect(self.display, border_color, rect, width=2, border_radius=8)

            value_text = input_box.get("text") or input_box.get("hint", "")
            value_color = (245, 245, 245) if input_box.get("text") else (140, 140, 150)
            value_surface = self.small_font.render(value_text, True, value_color)
            self.display.blit(value_surface, (rect.x + 8, rect.y + 5))

    def _draw_q_values(self, panel_x, data):
        y = data.get("q_values_y", 350)
        heading = self.small_font.render("Q-values and policy", True, (245, 245, 245))
        self.display.blit(heading, (panel_x, y))
        y += 24

        q_values = data.get("q_values", [0.0, 0.0, 0.0])
        action_labels = data.get("action_labels", ["Straight", "Right", "Left"])
        action_index = data.get("action_index")
        decision_type = data.get("decision_type", "")

        max_abs = max(1.0, max(abs(value) for value in q_values))
        for index, label in enumerate(action_labels):
            bar_y = y + index * 22
            bar_x = panel_x + 95
            bar_w = 170
            value = q_values[index]
            fill_w = int((abs(value) / max_abs) * bar_w)
            color = (95, 236, 124) if value >= 0 else (255, 110, 110)
            if index == action_index:
                color = (255, 196, 68) if decision_type == "explore" else color
                if decision_type == "policy preview":
                    color = (120, 210, 255)

            label_surface = self.tiny_font.render(label, True, (220, 220, 220))
            self.display.blit(label_surface, (panel_x, bar_y + 2))
            pygame.draw.rect(self.display, (58, 58, 68), (bar_x, bar_y + 4, bar_w, 12), border_radius=6)
            pygame.draw.rect(self.display, color, (bar_x, bar_y + 4, fill_w, 12), border_radius=6)

            value_surface = self.tiny_font.render(f"{value:.2f}", True, (235, 235, 235))
            self.display.blit(value_surface, (bar_x + bar_w + 8, bar_y + 1))

    def _draw_state_block(self, panel_x, data):
        y = data.get("state_y", 424)
        heading = self.small_font.render("State representation", True, (245, 245, 245))
        self.display.blit(heading, (panel_x, y))
        y += 22

        for line in data.get("state_lines", []):
            surface = self.tiny_font.render(line, True, (215, 215, 220))
            self.display.blit(surface, (panel_x, y))
            y += 18

        for line in data.get("help_lines", []):
            surface = self.tiny_font.render(line, True, (160, 200, 255))
            self.display.blit(surface, (panel_x, y))
            y += 17

    def _draw_graph(self, panel_x, data):
        if not data.get("show_graph"):
            return

        graph_y = data.get("graph_y", self.window_h - 110)
        graph_h = data.get("graph_h", 90)
        graph_rect = pygame.Rect(panel_x, graph_y, self.sidebar_width - 36, graph_h)
        pygame.draw.rect(self.display, (34, 34, 40), graph_rect, border_radius=10)
        pygame.draw.rect(self.display, (70, 70, 80), graph_rect, width=1, border_radius=10)

        title = self.tiny_font.render("Learning graph: score and moving average", True, (230, 230, 230))
        self.display.blit(title, (graph_rect.x + 8, graph_rect.y + 6))

        scores = data.get("graph_scores", [])
        averages = data.get("graph_averages", [])
        if len(scores) < 2:
            return

        plot_rect = pygame.Rect(graph_rect.x + 8, graph_rect.y + 24, graph_rect.width - 16, graph_rect.height - 32)
        pygame.draw.line(
            self.display,
            (90, 90, 100),
            (plot_rect.x, plot_rect.bottom),
            (plot_rect.right, plot_rect.bottom),
            1,
        )

        max_value = max(max(scores), max(averages), 1)
        self._draw_graph_line(plot_rect, scores, max_value, (120, 200, 255))
        self._draw_graph_line(plot_rect, averages, max_value, (120, 255, 150))

    def _draw_graph_line(self, rect, values, max_value, color):
        if len(values) < 2:
            return

        points = []
        for index, value in enumerate(values):
            ratio_x = index / (len(values) - 1)
            ratio_y = 0 if max_value == 0 else value / max_value
            x = rect.x + int(ratio_x * rect.width)
            y = rect.bottom - int(ratio_y * rect.height)
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(self.display, color, False, points, 2)

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




