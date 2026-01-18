# Astro Alley – Flappy Bird–Style RL Game


REQUIREMENT: Python 3.13 (I have found 3.14 can fail to load pygame)
--


REQUIREMNET: PLEASE MAKE SURE TO RUN THIS IN POWER SHELL TO MAKE SURE YOU HAVE ALL DEPENDENCIES: pip install pygame torch numpy matplotlib moviepy 
--


Astro Alley is a Flappy Bird–inspired arcade game where the goal is to pilot a rocket through moving obstacles.
Two modes:
- User Mode (classic you play it style)
- Autopilot Mode (This uses my RL trained agent to play)

Note: You DO NOT have to retrain the agent before running the app from main.py. I let the training algorithm run for 37 hours and saved it to the repository. 
--

- You absolutely can run another training session and can quit anytime to save the best policy thus far. I just had time so I let it run. You can see that entire training session, sped up obviously, plus the entire project from code, to training, to the actual gameplay here: https://youtu.be/h-_DB1jLB14?si=vY6G8P_zvvaHYS9_

Developer: Kody Graham

All images, video clips, and music are either original or from copyright-free sources.

# Features

- Arcade-style gameplay with tight hitboxes and smooth movement

- User mode – play the game yourself

- AI autopilot mode – watch a trained agent navigate through the course

- Custom training environment (GameEnv) separate from the game visuals

- Live training dashboard with Matplotlib (episode length, return, eval performance)

# Requirements

You’ll need:

- Python 3.10-3.13 NOT 3.14 (pygame not supported)

- PyGame

- PyTorch

- NumPy

- Matplotlib

- See top for install instructions
  
# Project Structure

Project Layout:

Astro-Alley/


- main.py              # Entry point – launches menu and game

- game.py              #Main game loop & high-level game logic

- barriers.py          # Pipe/obstacle definitions and constants

- player_icon.py       # Player sprite + movement/physics

- menu.py              #Main menu / customize menu UI

- autopilot_torch.py   # Runtime Torch policy (Actor–Critic) used by the game

- game_env.py          # Training environment for the agent (no graphics)

- pilot_training.py    # PPO training script (runs the live dashboard)

- assets/              # Images, fonts, sounds (if applicable)

- autopilot_policy.pt  #Saved trained weights (created by training script)

# How to Play the Game

Make sure the dependencies are installed.

From the project folder, just run:

python main.py


Use the in-game menu to:

- Start a normal game (you control the ship).

- Enable AI/ Autopilot mode (the trained agent controls the ship).

- Controls and options are shown in the game’s main menu and HUD.

# Training the AI Agent (PPO)

If you want to (re)train the agent yourself:

Verify PyTorch, NumPy, and Matplotlib are installed.

Then from the project folder, run:

python pilot_training.py


You’ll see a multi grapj training dashboard window with:

- Training episode length

- Training return

- Eval length

- Eval average pipes passed

The script will periodically evaluate the current policy and save the best model to:

autopilot_policy.pt


The game’s AI mode (autopilot_torch.py) will load that file to fly the ship.

Note: Closing the training window or pressing Q (as indicated in the console header) will stop training and save the best model found so far.

# Notes

This project is primarily intended as a resume project to demonstrate:

- Python application development

- Basic reinforcement learning setup

- Familiarity working with GitHub

- Implementing and training a PPO agent in PyTorch

# Credits

Code & Game Design: Kody Graham

Art & Audio: Original or copyright-free resources
