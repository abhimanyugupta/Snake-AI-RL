import argparse
import os

import pygame

from agent import QLearningAgent
from snake_game import SnakeGameAI


class SliderControl:
    """Simple draggable slider used inside the pygame dashboard."""

    def __init__(self, label, min_value, max_value, value, x, y, width, formatter=None):
        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.formatter = formatter or (lambda current: f"{current:.2f}")
        self.track_rect = pygame.Rect(x, y + 20, width, 6)
        self.hit_rect = pygame.Rect(x, y + 10, width, 28)
        self.dragging = False

    @property
    def normalized(self):
        span = self.max_value - self.min_value
        if span == 0:
            return 0.0
        return (self.value - self.min_value) / span

    def set_normalized(self, ratio):
        ratio = max(0.0, min(1.0, ratio))
        self.value = self.min_value + ratio * (self.max_value - self.min_value)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hit_rect.collidepoint(event.pos):
                self.dragging = True
                self._update_from_mouse(event.pos[0])
                return True

        if event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_mouse(event.pos[0])
            return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            self._update_from_mouse(event.pos[0])
            return True

        return False

    def _update_from_mouse(self, mouse_x):
        ratio = (mouse_x - self.track_rect.x) / self.track_rect.width
        self.set_normalized(ratio)

    def draw_data(self, value_text=None):
        knob_x = self.track_rect.x + int(self.normalized * self.track_rect.width)
        return {
            "label": self.label,
            "value_text": value_text or self.formatter(self.value),
            "x": self.track_rect.x,
            "y": self.track_rect.y - 20,
            "track_x": self.track_rect.x,
            "track_y": self.track_rect.y,
            "track_w": self.track_rect.width,
            "track_h": self.track_rect.height,
            "ratio": self.normalized,
            "knob_x": knob_x,
            "knob_y": self.track_rect.y + self.track_rect.height // 2,
            "knob_radius": 8,
        }


class ToggleControl:
    def __init__(self, label, value, x, y, width, height):
        self.label = label
        self.value = value
        self.rect = pygame.Rect(x, y, width, height)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value
                return True
        return False

    def toggle(self):
        self.value = not self.value

    def draw_data(self):
        return {
            "label": self.label,
            "value": self.value,
            "x": self.rect.x,
            "y": self.rect.y,
            "w": self.rect.width,
            "h": self.rect.height,
        }


class TextInputControl:
    """Small text box for entering numeric values in the dashboard."""

    def __init__(self, label, value, x, y, width, height, max_length=6):
        self.label = label
        self.text = str(value)
        self.rect = pygame.Rect(x, y, width, height)
        self.max_length = max_length
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            return was_active or self.active

        if not self.active or event.type != pygame.KEYDOWN:
            return False

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
            self.active = False
            return True

        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            return True

        if event.unicode.isdigit() and len(self.text) < self.max_length:
            self.text += event.unicode
            return True

        return False

    def get_int(self, default=1, minimum=1, maximum=999999):
        raw_value = self.text.strip()
        if not raw_value:
            return max(minimum, min(maximum, default))

        value = int(raw_value)
        return max(minimum, min(maximum, value))

    def draw_data(self):
        return {
            "label": self.label,
            "text": self.text,
            "active": self.active,
            "x": self.rect.x,
            "y": self.rect.y,
            "w": self.rect.width,
            "h": self.rect.height,
            "hint": "Type a number",
        }


class TrainingDashboard:
    """Owns the interactive controls and the learning history graph."""

    def __init__(self, game, initial_speed, initial_delay_ms, initial_episode_goal):
        panel_x = max(game.board_w + 18, 18)
        slider_width = max(220, game.sidebar_width - 40)

        speed_ratio = self._speed_ratio_from_settings(initial_speed, initial_delay_ms)
        self.speed_slider = SliderControl(
            "Training speed", 0.0, 1.0, speed_ratio, panel_x, 220, slider_width
        )
        self.food_reward_slider = SliderControl(
            "Food reward", 1.0, 20.0, 10.0, panel_x, 264, slider_width
        )
        self.death_reward_slider = SliderControl(
            "Death penalty", -20.0, -1.0, -10.0, panel_x, 308, slider_width
        )
        self.step_reward_slider = SliderControl(
            "Step reward", -1.0, 1.0, 0.0, panel_x, 352, slider_width
        )

        self.show_arrows_toggle = ToggleControl("Arrows [A]", True, panel_x, 400, 150, 28)
        self.show_dangers_toggle = ToggleControl(
            "Danger [D]", True, panel_x + 162, 400, 150, 28
        )
        self.show_graph_toggle = ToggleControl("Graph [G]", True, panel_x, 434, 150, 28)
        self.pause_toggle = ToggleControl(
            "Pause [Space]", False, panel_x + 162, 434, 150, 28
        )
        self.turbo_toggle = ToggleControl("Turbo [T]", False, panel_x, 468, 150, 28)
        self.episode_input = TextInputControl(
            "Episode goal", initial_episode_goal, panel_x + 162, 468, 150, 30
        )
        self.keep_open_toggle = ToggleControl("Keep open [K]", True, panel_x, 506, 312, 28)

        self.sliders = [
            self.speed_slider,
            self.food_reward_slider,
            self.death_reward_slider,
            self.step_reward_slider,
        ]
        self.toggles = [
            self.show_arrows_toggle,
            self.show_dangers_toggle,
            self.show_graph_toggle,
            self.pause_toggle,
            self.turbo_toggle,
            self.keep_open_toggle,
        ]
        self.inputs = [self.episode_input]

        self.initial_episode_goal = initial_episode_goal
        self.last_reward = 0.0
        self.score_history = []
        self.average_history = []

    @property
    def current_fps(self):
        if self.turbo_toggle.value:
            return int(60 + (self.speed_slider.value * 300))
        return int(5 + (self.speed_slider.value * 115))

    @property
    def current_delay_ms(self):
        if self.turbo_toggle.value:
            return 0
        return int((1.0 - self.speed_slider.value) * 140)

    @property
    def render_every_n_steps(self):
        if not self.turbo_toggle.value:
            return 1
        return 2 + int(self.speed_slider.value * 10)

    @property
    def speed_mode_label(self):
        if self.turbo_toggle.value:
            return "Turbo"

        ratio = self.speed_slider.value
        if ratio < 0.34:
            return "Slow"
        if ratio < 0.67:
            return "Medium"
        return "Fast"

    @property
    def reward_config(self):
        return {
            "food": round(self.food_reward_slider.value, 2),
            "death": round(self.death_reward_slider.value, 2),
            "step": round(self.step_reward_slider.value, 2),
        }

    def get_episode_goal(self):
        return self.episode_input.get_int(default=self.initial_episode_goal)

    def should_draw_frame(self, step_number, force=False):
        if force or not self.turbo_toggle.value:
            return True
        return step_number <= 1 or (step_number % self.render_every_n_steps == 0)

    def _speed_ratio_from_settings(self, speed, delay_ms):
        speed_ratio = max(0.0, min(1.0, (speed - 5) / 115))
        delay_ratio = 1.0 - max(0.0, min(1.0, delay_ms / 140 if delay_ms else 0.0))
        return round((speed_ratio + delay_ratio) / 2, 2)

    def handle_events(self, events):
        for event in events:
            consumed_by_input = False
            for input_control in self.inputs:
                if input_control.handle_event(event):
                    consumed_by_input = True
                    break

            if consumed_by_input:
                continue

            if event.type == pygame.KEYDOWN:
                self._handle_shortcuts(event.key)

            for slider in self.sliders:
                if slider.handle_event(event):
                    break
            else:
                for toggle in self.toggles:
                    if toggle.handle_event(event):
                        break

    def _handle_shortcuts(self, key):
        if key == pygame.K_SPACE:
            self.pause_toggle.toggle()
        elif key == pygame.K_a:
            self.show_arrows_toggle.toggle()
        elif key == pygame.K_d:
            self.show_dangers_toggle.toggle()
        elif key == pygame.K_g:
            self.show_graph_toggle.toggle()
        elif key == pygame.K_t:
            self.turbo_toggle.toggle()
        elif key == pygame.K_k:
            self.keep_open_toggle.toggle()
        elif key == pygame.K_1:
            self.turbo_toggle.value = False
            self.speed_slider.set_normalized(0.15)
        elif key == pygame.K_2:
            self.turbo_toggle.value = False
            self.speed_slider.set_normalized(0.5)
        elif key == pygame.K_3:
            self.turbo_toggle.value = False
            self.speed_slider.set_normalized(0.9)
        elif key == pygame.K_4:
            self.turbo_toggle.value = True
            self.speed_slider.set_normalized(1.0)

    def record_score(self, score):
        self.score_history.append(score)
        recent_scores = self.score_history[-20:]
        moving_average = sum(recent_scores) / len(recent_scores)
        self.average_history.append(moving_average)

    def build_dashboard_data(
        self,
        agent,
        game,
        state,
        action_info,
        current_game_number,
        episode_goal,
        best_score,
    ):
        candidate_points = game.get_relative_points()
        deadly_moves = {
            key: game.is_collision(point) for key, point in candidate_points.items()
        }

        q_values = action_info["q_values"]
        state_bits = ", ".join(str(bit) for bit in state)
        danger_bits = " / ".join(str(bit) for bit in state[0:3])
        direction_bits = " / ".join(str(bit) for bit in state[3:7])
        food_bits = " / ".join(str(bit) for bit in state[7:11])
        draw_mode = (
            f"every {self.render_every_n_steps} steps"
            if self.turbo_toggle.value
            else f"{self.current_delay_ms} ms delay"
        )
        display_goal = max(episode_goal, current_game_number)

        return {
            "panel_title": "RL Training Dashboard",
            "metrics": [
                ("Run", f"{current_game_number}/{display_goal}"),
                ("Total learned", agent.n_games),
                ("Score", game.score),
                ("Best score", best_score),
                ("Epsilon", f"{agent.epsilon:.3f}"),
                (
                    "Decision",
                    f"{action_info['action_label']} ({action_info['decision_type']})",
                ),
                ("Last reward", f"{self.last_reward:+.2f}"),
            ],
            "sliders": [
                self.speed_slider.draw_data(
                    f"{self.speed_mode_label} | {self.current_fps} fps | {draw_mode}"
                ),
                self.food_reward_slider.draw_data(f"{self.food_reward_slider.value:+.1f}"),
                self.death_reward_slider.draw_data(f"{self.death_reward_slider.value:+.1f}"),
                self.step_reward_slider.draw_data(f"{self.step_reward_slider.value:+.2f}"),
            ],
            "toggles": [toggle.draw_data() for toggle in self.toggles],
            "inputs": [input_control.draw_data() for input_control in self.inputs],
            "show_arrows": self.show_arrows_toggle.value,
            "show_dangers": self.show_dangers_toggle.value,
            "show_graph": self.show_graph_toggle.value,
            "q_values": q_values,
            "action_labels": agent.ACTION_LABELS,
            "action_index": action_info["action_index"],
            "action_key": action_info["action_key"],
            "decision_type": action_info["decision_type"],
            "candidate_points": candidate_points,
            "deadly_moves": deadly_moves,
            "q_values_y": 552,
            "graph_y": 628,
            "graph_h": 92,
            "state_y": 738,
            "state_lines": [
                f"Tuple: {state_bits}",
                f"Danger S/R/L: {danger_bits} | Dir L/R/U/D: {direction_bits}",
                f"Food L/R/U/D: {food_bits}",
                f"Food view: {agent.explain_food_view(state)}",
            ],
            "help_lines": [
                "Model: Q-table dictionary, not a neural net.",
                "Click Episode goal, then type a number.",
                "1/2/3 = slow/med/fast, 4 or T = turbo.",
                "K keeps the dashboard open after training.",
            ],
            "graph_scores": self.score_history[-60:],
            "graph_averages": self.average_history[-60:],
        }


def hold_training_window_open(game):
    """Keep the final dashboard visible until the user closes it."""
    pygame.event.clear()  # flush stale events from training
    final_view = dict(game.dashboard_data)
    final_view["overlay_title"] = "Training finished"
    final_view["overlay_subtitle"] = "Press Enter, Q, Esc, or close the window."
    game.set_dashboard_data(final_view)

    while not game.quit_requested:
        events = pygame.event.get()
        game.handle_system_events(events)

        for event in events:
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_RETURN,
                pygame.K_KP_ENTER,
                pygame.K_q,
                pygame.K_ESCAPE,
            ):
                return

        game.draw()
        pygame.time.delay(30)


def train(episodes, render, speed, delay_ms, model_path, resume, save_every):
    agent = QLearningAgent()

    if resume and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded saved model from {model_path}")

    game = SnakeGameAI(w=640, h=700, window_h=900, render=render, speed=speed)
    dashboard = TrainingDashboard(
        game,
        initial_speed=speed,
        initial_delay_ms=delay_ms,
        initial_episode_goal=episodes,
    )
    best_score = 0
    session_start_games = agent.n_games
    training_completed = False

    try:
        while (agent.n_games - session_start_games) < dashboard.get_episode_goal():
            game.reset()
            dashboard.last_reward = 0.0
            score = game.score
            current_game_number = (agent.n_games - session_start_games) + 1
            state = agent.get_state(game)
            step_count = 0

            while True:
                episode_goal = dashboard.get_episode_goal()
                events = pygame.event.get() if render else []
                dashboard.handle_events(events)
                game.handle_system_events(events)

                if game.quit_requested:
                    break

                game.speed = dashboard.current_fps
                game.set_reward_config(dashboard.reward_config)

                if dashboard.pause_toggle.value:
                    preview = agent.get_policy_preview(state)
                    game.set_dashboard_data(
                        dashboard.build_dashboard_data(
                            agent=agent,
                            game=game,
                            state=state,
                            action_info=preview,
                            current_game_number=current_game_number,
                            episode_goal=episode_goal,
                            best_score=best_score,
                        )
                    )
                    if render:
                        game.draw()
                        pygame.time.delay(30)
                    continue

                action_info = agent.get_action_details(state)
                step_count += 1
                game.set_dashboard_data(
                    dashboard.build_dashboard_data(
                        agent=agent,
                        game=game,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                    )
                )

                if render and dashboard.should_draw_frame(step_count):
                    game.draw()
                    if dashboard.current_delay_ms > 0:
                        pygame.time.delay(dashboard.current_delay_ms)

                reward, game_over, score = game.play_step(
                    action_info["action"], events=[], draw_frame=False
                )
                dashboard.last_reward = reward

                next_state = agent.get_state(game)
                agent.train_step(
                    state=state,
                    action_index=action_info["action_index"],
                    reward=reward,
                    next_state=next_state,
                    done=game_over,
                )
                state = next_state

                if render and game_over:
                    final_view = dashboard.build_dashboard_data(
                        agent=agent,
                        game=game,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                    )
                    final_view["overlay_title"] = "Episode finished"
                    final_view["overlay_subtitle"] = f"Reward: {reward:+.2f}"
                    game.set_dashboard_data(final_view)
                    game.draw()
                    if dashboard.turbo_toggle.value:
                        pygame.time.delay(30)
                    else:
                        pygame.time.delay(max(100, dashboard.current_delay_ms))

                if game.quit_requested or game_over:
                    break

            if game.quit_requested:
                print("Training stopped because the game window was closed.")
                break

            agent.n_games += 1
            dashboard.record_score(score)
            agent.decay_epsilon()
            best_score = max(best_score, score)

            completed_this_run = agent.n_games - session_start_games
            print(
                f"Run {completed_this_run:>4}/{dashboard.get_episode_goal():<4} | "
                f"Total games: {agent.n_games:>4} | "
                f"Score: {score:>2} | "
                f"Best: {best_score:>2} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"States: {len(agent.q_table)}"
            )

            if agent.n_games % save_every == 0 or score == best_score:
                agent.save(model_path)

        training_completed = not game.quit_requested
    finally:
        agent.save(model_path)
        if render and training_completed and dashboard.keep_open_toggle.value:
            hold_training_window_open(game)
        game.close()

    print(f"Training finished. Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple Snake Q-learning agent with a live pygame dashboard."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of games to play during training.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=16,
        help="Initial speed. The on-screen slider can change it during training.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=60,
        help="Initial delay. The on-screen slider can change it during training.",
    )
    parser.add_argument(
        "--model-path",
        default="q_table.pkl",
        help="Where to save the learned Q-table.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load an existing Q-table before training.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Save the model every N finished games.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run training without the pygame window.",
    )

    args = parser.parse_args()
    render = not args.no_render
    speed = args.speed if render else 0
    delay_ms = args.delay_ms if render else 0

    train(
        episodes=args.episodes,
        render=render,
        speed=speed,
        delay_ms=delay_ms,
        model_path=args.model_path,
        resume=args.resume,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
