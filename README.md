# Snake AI with Reinforcement Learning

This is a beginner-friendly starter project for building a Snake AI.

It includes:
- A playable Snake game made with `pygame`
- A simple tabular Q-learning agent
- A training script that teaches the agent over many games

## Project Files

- `snake_game.py` - the game logic and manual playable version
- `agent.py` - the Q-learning agent
- `train.py` - the training loop
- `requirements.txt` - project dependency list
- `README.md` - setup and usage notes

## Setup

1. Create a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Play the Game Yourself

Run:

```powershell
python snake_game.py
```

Controls:
- Arrow keys or `W A S D` to move
- `R` or `Enter` to restart after game over
- `Q` or `Esc` to quit from the game-over screen

## Train the Agent

Run:

```powershell
python train.py --episodes 300
```

This will:
- Play 300 training games
- Save the learned table to `q_table.pkl`
- Print the score for each episode

If you want to watch the training:

```powershell
python train.py --episodes 100 --watch
```

## How the Agent Works

This starter uses tabular Q-learning instead of a neural network.

That means the agent:
- Looks at a small set of game facts, such as danger and food direction
- Chooses between 3 actions: go straight, turn right, or turn left
- Updates a Q-table based on rewards

This is a great first step because it is easier to understand than deep reinforcement learning.

## Good Next Steps

Once this project is working for you, you can extend it by:
- plotting training scores
- saving the best model separately
- adding a neural network with PyTorch
- improving the reward system
- increasing the state information the agent can see
