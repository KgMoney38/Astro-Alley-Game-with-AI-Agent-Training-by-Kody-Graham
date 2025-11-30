# Astro Alley – Flappy Bird–Style RL Game

Astro Alley is a Flappy Bird–inspired arcade game where a rocket weaves through moving pipes.
Under the hood, it includes a PyTorch PPO (Proximal Policy Optimization) agent that can learn to fly the ship on its own.

Developer: Kody Graham

All images, video clips, and music are either original or from copyright-free sources.

# Features

- Arcade-style gameplay with tight hitboxes and smooth movement

- Human mode – play the game yourself

- AI autopilot mode – watch a trained PPO agent fly through the course

- Custom training environment (GameEnv) separate from the game visuals

- Live training dashboard with Matplotlib (episode length, return, eval performance)

# Requirements

You’ll need:

- Python 3.10+

- PyGame

- PyTorch
 (CPU build is fine)

- NumPy

- Matplotlib

Install (example):

pip install pygame torch numpy matplotlib

# Project Structure

Project Layout:

Astro-Alley/


- main.py              # Entry point – launches menu and game

- game.py              # Main game loop & high-level game logic

- barriers.py          # Pipe/obstacle definitions and constants

- player_icon.py       # Player sprite + movement/physics

- menu.py              # Main menu / customize menu UI

- autopilot_torch.py   # Runtime Torch policy (Actor–Critic) used by the game

- game_env.py          # Training environment for the agent (no graphics)

- pilot_training.py    # PPO training script (runs the live dashboard)

- assets/              # Images, fonts, sounds (if applicable)

- autopilot_policy.pt  # Saved trained weights (created by training script)

# How to Play the Game

Make sure your dependencies are installed.

From the project folder, run:

python main.py


Use the in-game menu to:

- Start a normal game (you control the ship).

- Enable AI / Autopilot mode (the trained agent controls the ship).

- Controls and options are shown in the game’s main menu and HUD.

# Training the AI Agent (PPO)

If you want to (re)train the agent yourself:

Verify PyTorch, NumPy, and Matplotlib are installed.

From the project folder, run:

python pilot_training.py


You’ll see a training dashboard window with:

- Training episode length

- Training return

- Eval length

- Eval average pipes passed

The script will periodically evaluate the current policy and save the best model to:

autopilot_policy.pt


The game’s AI mode (autopilot_torch.py) will load that file to fly the ship.

Note: Closing the training window or pressing Q (as indicated in the console header) will stop training and save the best model found so far.

# Notes

This project is primarily intended as a portfolio piece to demonstrate:

- Python application development

- Basic reinforcement learning setup

- Familiarity working with GitHub

- Implementing and training a PPO agent in PyTorch

- Feel free to explore the code, tweak hyperparameters, or change rewards to see how the agent’s behavior changes.

# Credits

Code & Game Design: Kody Graham

Art & Audio: Original or copyright-free resources