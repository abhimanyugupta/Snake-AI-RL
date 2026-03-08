import pickle
import random
from collections import defaultdict

from snake_game import Direction, Point


class QLearningAgent:
    """
    Small tabular Q-learning agent.

    This is a friendly starting point before moving to a neural network.
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_rate=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_games = 0

        # Every state maps to 3 action values:
        # 0 = straight, 1 = turn right, 2 = turn left
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])

    def get_state(self, game):
        """
        Build a small state tuple from the game.

        The state keeps only a few useful facts:
        - danger straight / right / left
        - current direction
        - whether food is left / right / up / down
        """
        head = game.head
        block = game.block_size

        direction = game.direction
        point_straight = self._next_point(head, direction, block)
        point_right = self._next_point(head, self._turn_right(direction), block)
        point_left = self._next_point(head, self._turn_left(direction), block)

        state = (
            int(game.is_collision(point_straight)),
            int(game.is_collision(point_right)),
            int(game.is_collision(point_left)),
            int(direction == Direction.LEFT),
            int(direction == Direction.RIGHT),
            int(direction == Direction.UP),
            int(direction == Direction.DOWN),
            int(game.food.x < game.head.x),
            int(game.food.x > game.head.x),
            int(game.food.y < game.head.y),
            int(game.food.y > game.head.y),
        )

        return state

    def get_action(self, state):
        """Pick a random action sometimes, otherwise use the best known move."""
        if random.random() < self.epsilon:
            action_index = random.randint(0, 2)
        else:
            q_values = self.q_table[state]
            best_value = max(q_values)
            best_actions = [
                index for index, value in enumerate(q_values) if value == best_value
            ]
            action_index = random.choice(best_actions)

        return self.index_to_action(action_index)

    def train_step(self, state, action_index, reward, next_state, done):
        """Classic Q-learning update."""
        current_q = self.q_table[state][action_index]
        next_best_q = 0.0 if done else max(self.q_table[next_state])

        target_q = reward + (self.discount_rate * next_best_q)
        updated_q = current_q + self.learning_rate * (target_q - current_q)

        self.q_table[state][action_index] = updated_q

    def decay_epsilon(self):
        """Reduce exploration slowly after each game."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def action_to_index(self, action):
        return list(action).index(1)

    def index_to_action(self, index):
        action = [0, 0, 0]
        action[index] = 1
        return action

    def save(self, file_name="q_table.pkl"):
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "n_games": self.n_games,
        }

        with open(file_name, "wb") as file:
            pickle.dump(data, file)

    def load(self, file_name="q_table.pkl"):
        with open(file_name, "rb") as file:
            data = pickle.load(file)

        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0], data["q_table"])
        self.epsilon = data.get("epsilon", self.epsilon)
        self.n_games = data.get("n_games", 0)

    def _next_point(self, head, direction, block_size):
        if direction == Direction.RIGHT:
            return Point(head.x + block_size, head.y)
        if direction == Direction.LEFT:
            return Point(head.x - block_size, head.y)
        if direction == Direction.UP:
            return Point(head.x, head.y - block_size)
        return Point(head.x, head.y + block_size)

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
