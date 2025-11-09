#Kody Graham
#11/3/2025

#This file will handle my PyTorch policy for my AI mode

#Note for self: Done

from __future__ import annotations

import os
import sys

import numpy as np
import torch #PyTorch core library
import torch.nn as nn #PyTorch neural net modules

#Had to have to import my barriers class from a different directory than cd
BASE_DIR = os.path.dirname(__file__)
GAME_FILE_DIR= os.path.abspath(os.path.join(BASE_DIR, "..", "The Game Files"))
if GAME_FILE_DIR not in sys.path:
    sys.path.insert(0,GAME_FILE_DIR)
from barriers import SCREEN_HEIGHT, PIPE_GAP, PIPE_WIDTH, PIPE_SPEED, PIPE_MIN_TOP, PIPE_MAX_TOP

#Model will define both policy and value
class ActorCritic(nn.Module):
    def __init__(self,obs_dim: int=5,hidden: int=128, act_dim: int= 2):
        super().__init__() #Initialize the nn.Module base class
        self.body= nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden), nn.Tanh(),) #MLP that feeds both the policy and value heads
        self.pi= nn.Linear(hidden,act_dim) #Policy head
        self.v= nn.Linear(hidden,1) #Value head

        #Same init as training
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.pi.weight, gain=.01) #Small init for policy
        nn.init.constant_(self.pi.bias, 0.0)
        nn.init.orthogonal_(self.v.weight, gain=1.0) #For value head
        nn.init.constant_(self.v.bias, 0.0)

    #Computes the shared hidden and returns the policy logits and value estimates
    def forward(self,x: torch.Tensor):
        h = self.body(x)
        return self.pi(h), self.v(h)

#Policy wrapper for game will be what my game calls to decide when to jump
class TorchPolicy:
    def __init__(self, ckpt_path: str | None, device:str = "cpu"):
        self.device = torch.device(device)# or ("cuda" if torch.cuda.is_available() else "cpu")) #Try to use CUDA because using the graphics card will be more efficient
        self.model = ActorCritic(obs_dim=5, hidden=128, act_dim=2).to(self.device)
        self.last_obs = np.zeros(5, dtype=np.float32)
        loaded= False

        #Check weights loaded right
        if ckpt_path and os.path.isfile(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=self.device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.model.load_state_dict(state, strict=True)
                loaded = True
                print(f"Loaded policy from {ckpt_path}")
            except Exception as e:
                print(f"Failed to load policy from {ckpt_path}: {e}")

        if not loaded:
            print("WARNING: No policy found using random weights")

        self.model.eval()

    @staticmethod
    def player_rect(player):
        if hasattr(player, "get_rect"):
            return player.get_rect()
        if hasattr(player, "rect"): #Fall back
            return player.rect
        raise AttributeError("Player has no rect attribute")

    def next_pipe(self, player,pipes):
        prect= self.player_rect(player)  #Player rectangle
        px = float(prect.centerx) #player center

        best = None
        best_dx = float("inf")
        for p in pipes:
            if not hasattr(p, "rects"):
                continue
            top_rect, bot_rect = p.rects()
            cx = float(p.x) + .5 * float(p.width)
            dx = cx-px

            #Choose nearest pipe not behind
            if dx >= -1e-6 and dx < best_dx:
                best= p
                best_dx = dx
        return best

    #Observation builder same five dimension as training, HAS TO MATCH TRAINING!
    def make_obs(self, player, pipes, screen_height: int) -> np.ndarray:

        H = float(screen_height or SCREEN_HEIGHT)
        prect = self.player_rect(player) #Again my player rect just from game object
        py = float(prect.centery) #Center y

        # Read vertical velocity
        vy= float(getattr(player, "vel_y", 0.0))
        vy_norm = max(-1.5, min(1.5, vy/ 10.0)) #Normalize

        #Find next pipe
        p= self.next_pipe(player,pipes)

        #Reset edge if no pipe
        if p is None:
            gap_h= float(PIPE_GAP)
            obs = np.array([py/H, vy_norm, 0.0, 0.0, gap_h/ H], dtype=np.float32)
            self.last_obs = obs
            return obs

        #Derive the gap center/height
        top_rect, bottom_rect = p.rects()
        gap_top= float(top_rect.bottom)
        gap_bot = float(bottom_rect.top)
        gap_h = gap_bot - gap_top

        #Fall back to last obs
        if gap_h <= 1.0:
            return self.last_obs

        gap_center= .5 * (gap_top + gap_bot)

        #Pipe/player center x
        cx= float(p.x) + .5* float(p.width)
        px= float(prect.centerx)
        dx= cx-px

        t2g = dx / max(1.0, float(PIPE_SPEED))
        t2g_norm = max(-1.0, min(1.0, t2g/60.0))

        #Vertical offset to gap center, normalize gap size
        gap_center_offset = (gap_center - py)/H
        gap_h_norm= gap_h / H

        obs= np.array([py/H, vy_norm, t2g_norm, gap_center_offset, gap_h_norm], dtype=np.float32)

        self.last_obs = obs #Save latest obs for backup
        return obs

    #The actual policy decision, return true if ai decides to jump this frame
    def decide(self, player, pipes, screen_height: int) -> bool:

        obs = self.make_obs(player, pipes, screen_height) #Build from game state
        x= torch.from_numpy(obs).unsqueeze(0).to(self.device) #Convert to a device tensor with my batch dimensions
        with torch.no_grad(): #Stop tracking for faster infer
            logits, _ = self.model(x)
            action = int(torch.argmax(logits, dim=1).item()) #Pick action with highest logit, 0= no 1= jump

        #0 = no jump, 1 jump
        return action == 1

