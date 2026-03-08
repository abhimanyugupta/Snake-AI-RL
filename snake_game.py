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

    def __init__(self, w=640, h=480, block_size=20, speed=12, render=True):
        pygame.init()

        self.w = w
        self.h = h
        self.block_size = block_size
        self.speed = speed
        self.render = render
        self.quit_requested = False

        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake AI Project")
            self.font = pygame.font.SysFont("arial", 24)
            self.small_font = pygame.font.SysFont("arial", 18)
        else:
            self.display = None
            self.font = None
            self.small_font = None

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Reset the game so training can start a fresh episode."""
        start_x = (self.w // 2 // self.block_size) * self.block_size
        start_y = (self.h // 2 // self.block_size) * self.block_size

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

    def _place_food(self):
        """Place food on a free grid cell."""
        max_x = (self.w - self.block_size) // self.block_size
        max_y = (self.h - self.block_size) // self.block_size

        while True:
            x = random.randint(0, max_x) * self.block_size
            y = random.randint(0, max_y) * self.block_size
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action=None, events=None):
        """
        Advance the game by one frame.

        - If action is None, the game uses keyboard input.
        - If action is [1, 0, 0], [0, 1, 0], or [0, 0, 1], the snake is
          controlled by the agent.
        """
        if self.quit_requested:
            return 0, True, self.score

        self.frame_iteration += 1

        if events is None:
            events = pygame.event.get() if self.render else []

        self._handle_quit_events(events)
        if self.quit_requested:
            return 0, True, self.score

        if action is None:
            self._handle_human_input(events)
            self._move()
        else:
            self._move(action)

        self.snake.insert(0, self.head)
        reward = 0
        game_over = False

        # End the game if the snake hits a wall, itself, or loops for too long.
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = -10
            game_over = True
        elif self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        if self.render:
            self._draw_scene()

        self.clock.tick(self.speed)
        return reward, game_over, self.score

    def is_collision(self, point=None):
        """Check whether a point hits a wall or the snake body."""
        if point is None:
            point = self.head

        if point.x < 0 or point.x >= self.w or point.y < 0 or point.y >= self.h:
            return True

        if point in self.snake[1:]:
            return True

        return False

    def _handle_quit_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.quit_requested = True

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

            clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            current_index = clockwise.index(self.direction)
            action_index = action.index(1)

            if action_index == 0:
                new_direction = clockwise[current_index]
            elif action_index == 1:
                new_direction = clockwise[(current_index + 1) % 4]
            else:
                new_direction = clockwise[(current_index - 1) % 4]

            self.direction = new_direction

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

    def _draw_scene(self, overlay_title=None, overlay_subtitle=None):
        if not self.render or self.display is None:
            return

        self.display.fill((20, 20, 20))

        for index, part in enumerate(self.snake):
            body_color = (46, 204, 113) if index == 0 else (39, 174, 96)
            rect = pygame.Rect(part.x, part.y, self.block_size, self.block_size)
            pygame.draw.rect(self.display, body_color, rect, border_radius=4)
            pygame.draw.rect(self.display, (10, 80, 30), rect, width=2, border_radius=4)

        food_rect = pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size)
        pygame.draw.rect(self.display, (231, 76, 60), food_rect, border_radius=4)

        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        controls_text = self.small_font.render(
            "Move: Arrow keys or WASD", True, (200, 200, 200)
        )

        self.display.blit(score_text, (10, 10))
        self.display.blit(controls_text, (10, 40))

        if overlay_title:
            overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            self.display.blit(overlay, (0, 0))

            title_surface = self.font.render(overlay_title, True, (255, 255, 255))
            subtitle_surface = self.small_font.render(
                overlay_subtitle or "",
                True,
                (220, 220, 220),
            )

            title_rect = title_surface.get_rect(center=(self.w // 2, self.h // 2 - 15))
            subtitle_rect = subtitle_surface.get_rect(center=(self.w // 2, self.h // 2 + 20))

            self.display.blit(title_surface, title_rect)
            self.display.blit(subtitle_surface, subtitle_rect)

        pygame.display.flip()

    def show_game_over_screen(self):
        """Simple restart screen for manual play."""
        if not self.render:
            return False

        while not self.quit_requested:
            events = pygame.event.get()
            self._handle_quit_events(events)

            for event in events:
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key in (pygame.K_r, pygame.K_RETURN):
                    return True
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.quit_requested = True
                    return False

            self._draw_scene("Game Over", "Press R to restart or Q to quit")
            self.clock.tick(10)

        return False

    def close(self):
        pygame.quit()


def run_human_game():
    game = SnakeGameAI(render=True, speed=12)

    try:
        while not game.quit_requested:
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
