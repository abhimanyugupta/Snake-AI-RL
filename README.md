# Snake AI with Reinforcement Learning

This is a beginner-friendly starter project for learning reinforcement learning through Snake.

Important note:
- The current agent is **not** a neural network.
- It uses **tabular Q-learning**, which means it stores values in a Python dictionary called a Q-table.
- That makes it easier to understand the basics before moving to deep RL.

## Project Files

- `snake_game.py` - the Snake game, board rendering, overlays, and dashboard drawing
- `agent.py` - the tabular Q-learning agent and state/Q-value helpers
- `train.py` - the live training loop with sliders, toggles, and graph
- `requirements.txt` - project dependency list
- `README.md` - setup and usage notes

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Play the Game Yourself

```powershell
python snake_game.py
```

Controls:
- Arrow keys or `W A S D` to move
- `R` or `Enter` to restart after game over
- `Q` or `Esc` to quit from the game-over screen

## Train the Agent

```powershell
python train.py --episodes 300
```

The training dashboard now shows:
- the snake moving in real time
- score, game number, epsilon, reward, and known states
- the current action and whether it came from exploration or exploitation
- the current Q-values for straight, right, and left
- the 11-bit state representation the agent uses
- danger squares that would cause death
- colored action arrows on the board
- a live score / moving-average learning graph
- live reward sliders so you can change the reward function while training

## Dashboard Controls

Mouse controls:
- Drag the `Training speed` slider to move between slow and fast modes
- Drag the reward sliders to change `food`, `death`, and `step` rewards
- Click toggle buttons to turn overlays on or off

Keyboard shortcuts:
- `Space` - pause or resume training
- `A` - toggle action arrows
- `D` - toggle danger squares
- `G` - toggle the learning graph
- `1` - slow mode
- `2` - medium mode
- `3` - fast mode

## What the State Means

The agent does **not** see the whole board like a human.
It only sees 11 binary features:

1. danger straight
2. danger right
3. danger left
4. moving left
5. moving right
6. moving up
7. moving down
8. food left
9. food right
10. food up
11. food down

So when the dashboard says `Food L/R/U/D: 0 / 1 / 0 / 1`, it means the food is to the right and below the snake head.

## Why This Helps Learning

This project is meant to make the RL loop visible:
- **State**: what the agent knows right now
- **Action**: which move it picks
- **Reward**: what feedback it gets after that move
- **Q-value**: how good it currently believes each move is
- **Exploration vs exploitation**: whether it is trying something random or using what it already learned

## Useful Options

```powershell
python train.py --episodes 100 --resume
python train.py --episodes 100 --speed 20 --delay-ms 40
python train.py --episodes 100 --no-render
```

## Good Next Steps

Once you are comfortable with this version, useful next steps are:
- add a button to reset the Q-table from the UI
- save separate reward presets
- compare different epsilon schedules
- add plots for average reward per step
- replace the Q-table with a neural network using PyTorch
