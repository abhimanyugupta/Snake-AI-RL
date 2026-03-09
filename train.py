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
        padding = 20
        # Title takes ~46px (font size 26 + 20)
        # Metrics take ~26*7 + 15 = 197px
        # We start controls after metrics in the left column
        start_y = padding + 46 + 197 + 15
        
        col_w = (game.sidebar_width - (padding * 2) - 20) // 2
        col_x = game.board_w + padding
        
        slider_width = col_w
        
        y = start_y
        speed_ratio = self._speed_ratio_from_settings(initial_speed, initial_delay_ms)
        self.speed_slider = SliderControl(
            "Training speed", 0.0, 1.0, speed_ratio, col_x, y, slider_width
        )
        y += 44
        self.food_reward_slider = SliderControl(
            "Food reward", 1.0, 20.0, 10.0, col_x, y, slider_width
        )
        y += 44
        self.death_reward_slider = SliderControl(
            "Death penalty", -20.0, -1.0, -10.0, col_x, y, slider_width
        )
        y += 44
        self.step_reward_slider = SliderControl(
            "Step reward", -1.0, 1.0, 0.0, col_x, y, slider_width
        )
        
        y += 50
        toggle_w = (col_w - 10) // 2
        self.show_arrows_toggle = ToggleControl("Arrows [A]", True, col_x, y, toggle_w, 28)
        self.show_dangers_toggle = ToggleControl("Danger [D]", True, col_x + toggle_w + 10, y, toggle_w, 28)
        
        y += 36
        self.show_graph_toggle = ToggleControl("Graph [G]", True, col_x, y, toggle_w, 28)
        self.pause_toggle = ToggleControl("Pause [Space]", False, col_x + toggle_w + 10, y, toggle_w, 28)
        
        y += 36
        self.turbo_toggle = ToggleControl("Turbo [T]", False, col_x, y, toggle_w, 28)
        self.episode_input = TextInputControl("Episode goal", initial_episode_goal, col_x + toggle_w + 10, y, toggle_w, 30)
        
        y += 38
        self.keep_open_toggle = ToggleControl("Keep open [K]", True, col_x, y, toggle_w, 28)
        self.headless_toggle = ToggleControl("No Render [H]", False, col_x + toggle_w + 10, y, toggle_w, 28)

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
            self.headless_toggle,
        ]
        self.inputs = [self.episode_input]

        self.initial_episode_goal = initial_episode_goal
        self.last_reward = 0.0
        self.score_history = []
        self.average_history = []

        # Interactive graph viewport state
        self.graph_view_end = None   # None = follow latest (auto-scroll)
        self.graph_view_size = 60    # Number of runs visible at once
        self.graph_drag_active = False
        self.graph_drag_start_x = 0
        self.graph_drag_start_end = 0
        self.graph_hover_index = None  # Index into full history being hovered
        self.graph_rect = None  # Set by the renderer so events know the bounds

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
        if self.headless_toggle.value:
            return False
        if force or not self.turbo_toggle.value:
            return True
        return step_number <= 1 or (step_number % self.render_every_n_steps == 0)

    def _speed_ratio_from_settings(self, speed, delay_ms):
        speed_ratio = max(0.0, min(1.0, (speed - 5) / 115))
        delay_ratio = 1.0 - max(0.0, min(1.0, delay_ms / 140 if delay_ms else 0.0))
        return round((speed_ratio + delay_ratio) / 2, 2)

    def sync_graph_rect(self, game):
        """Read back the graph rect that the renderer stored in the data dict."""
        data = game.dashboard_data
        if data and "_graph_rect" in data:
            self.graph_rect = data["_graph_rect"]

    def handle_events(self, events):
        for event in events:
            consumed_by_input = False
            for input_control in self.inputs:
                if input_control.handle_event(event):
                    consumed_by_input = True
                    break

            if consumed_by_input:
                continue

            # Graph interaction (zoom/pan/hover)
            if self._handle_graph_event(event):
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

    def _handle_graph_event(self, event):
        """Handle zoom, pan, and hover for the interactive graph."""
        gr = self.graph_rect
        if gr is None:
            return False

        total = len(self.score_history)
        if total < 2:
            return False

        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if gr.collidepoint(mx, my):
                # Multiplicative zoom: ~1.4x per scroll tick (fast for large datasets)
                if event.y > 0:  # scroll up = zoom in
                    new_size = max(10, int(self.graph_view_size / 1.4))
                else:  # scroll down = zoom out
                    new_size = min(total, int(self.graph_view_size * 1.4))
                self.graph_view_size = new_size
                # Clamp view_end
                if self.graph_view_end is not None:
                    self.graph_view_end = min(total, max(new_size, self.graph_view_end))
                return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if gr.collidepoint(event.pos):
                self.graph_drag_active = True
                self.graph_drag_start_x = event.pos[0]
                view_end = self.graph_view_end if self.graph_view_end is not None else total
                self.graph_drag_start_end = view_end
                return True

        if event.type == pygame.MOUSEMOTION:
            # Hover tooltip
            if gr.collidepoint(event.pos) and not self.graph_drag_active:
                plot_x = gr.x + 15
                plot_w = gr.width - 30
                if plot_w > 0:
                    view_end = self.graph_view_end if self.graph_view_end is not None else total
                    view_start = max(0, view_end - self.graph_view_size)
                    n_visible = view_end - view_start
                    rel_x = event.pos[0] - plot_x
                    idx = view_start + int(rel_x / plot_w * n_visible)
                    idx = max(view_start, min(view_end - 1, idx))
                    self.graph_hover_index = idx
            elif not self.graph_drag_active:
                self.graph_hover_index = None

            # Drag to pan
            if self.graph_drag_active:
                dx_pixels = event.pos[0] - self.graph_drag_start_x
                plot_w = gr.width - 30
                if plot_w > 0:
                    runs_per_pixel = self.graph_view_size / plot_w
                    delta_runs = int(-dx_pixels * runs_per_pixel)
                    new_end = self.graph_drag_start_end + delta_runs
                    new_end = max(self.graph_view_size, min(total, new_end))
                    self.graph_view_end = new_end
                return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.graph_drag_active:
            self.graph_drag_active = False
            # If dragged to the very end, re-enable auto-follow
            if self.graph_view_end is not None and self.graph_view_end >= total:
                self.graph_view_end = None
            return True

        return False

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
        elif key == pygame.K_h:
            self.headless_toggle.toggle()
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
        elif key == pygame.K_f:
            # Fit all: zoom out to show entire training history
            total = len(self.score_history)
            if total > 0:
                self.graph_view_size = total
                self.graph_view_end = None  # auto-follow

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
            "state_lines": [
                f"Tuple: {state_bits}",
                f"Danger (S/R/L): {danger_bits}",
                f"Direction (L/R/U/D): {direction_bits}",
                f"Food (L/R/U/D): {food_bits}",
                f"Food view: {agent.explain_food_view(state)}",
            ],
            "help_lines": [
                "Model: Q-table dict, not neural net.",
                "Click Editor goal to change target.",
                "1/2/3 = slow/med/fast, 4/T = turbo.",
                "K keeps window open after training.",
            ],
            "graph_scores": self.score_history,
            "graph_averages": self.average_history,
            "graph_view_end": self.graph_view_end,
            "graph_view_size": self.graph_view_size,
            "graph_hover_index": self.graph_hover_index,
        }


def hold_training_window_open(game, dashboard):
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

        # Process dashboard events so graph zoom/pan/hover works
        dashboard.sync_graph_rect(game)
        dashboard.handle_events(events)

        # Update graph viewport data in the dashboard_data
        game.dashboard_data["graph_view_end"] = dashboard.graph_view_end
        game.dashboard_data["graph_view_size"] = dashboard.graph_view_size
        game.dashboard_data["graph_hover_index"] = dashboard.graph_hover_index

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
        while True:
            # Check if we've reached the user-adjustable goal
            current_goal = dashboard.get_episode_goal()
            if (agent.n_games - session_start_games) >= current_goal:
                break
            game.reset()
            dashboard.last_reward = 0.0
            score = game.score
            current_game_number = (agent.n_games - session_start_games) + 1
            state = agent.get_state(game)
            step_count = 0

            while True:
                episode_goal = dashboard.get_episode_goal()
                events = pygame.event.get() if render else []
                dashboard.sync_graph_rect(game)
                dashboard.handle_events(events)
                game.handle_system_events(events)

                if game.quit_requested:
                    break

                game.speed = 0 if dashboard.headless_toggle.value else dashboard.current_fps
                game.set_reward_config(dashboard.reward_config)

                if dashboard.pause_toggle.value:
                    if dashboard.headless_toggle.value:
                        pygame.time.delay(5)  # minimal delay to keep events flowing
                        continue
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

                if render and game_over and not dashboard.headless_toggle.value:
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
            hold_training_window_open(game, dashboard)
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
