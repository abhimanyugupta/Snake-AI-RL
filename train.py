import argparse
import os

from agent import QLearningAgent
from snake_game import SnakeGameAI


def train(episodes, watch, speed, model_path, resume, save_every):
    agent = QLearningAgent()

    if resume and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded saved model from {model_path}")

    game = SnakeGameAI(render=watch, speed=speed)
    best_score = 0

    try:
        for episode in range(1, episodes + 1):
            game.reset()
            state = agent.get_state(game)

            while True:
                action = agent.get_action(state)
                reward, game_over, score = game.play_step(action)
                next_state = agent.get_state(game)

                agent.train_step(
                    state=state,
                    action_index=agent.action_to_index(action),
                    reward=reward,
                    next_state=next_state,
                    done=game_over,
                )

                state = next_state

                if game.quit_requested or game_over:
                    break

            if game.quit_requested:
                print("Training stopped because the game window was closed.")
                break

            agent.n_games += 1
            agent.decay_epsilon()
            best_score = max(best_score, score)

            print(
                f"Game {agent.n_games:>4} | "
                f"Score: {score:>2} | "
                f"Best: {best_score:>2} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

            if episode % save_every == 0 or score == best_score:
                agent.save(model_path)

    finally:
        agent.save(model_path)
        game.close()

    print(f"Training finished. Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a simple Snake Q-learning agent.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of games to play during training.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Show the pygame window while training.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=None,
        help="Frames per second. Leave empty for a sensible default.",
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
        help="Save the model every N games.",
    )

    args = parser.parse_args()
    speed = args.speed if args.speed is not None else (12 if args.watch else 0)

    train(
        episodes=args.episodes,
        watch=args.watch,
        speed=speed,
        model_path=args.model_path,
        resume=args.resume,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
