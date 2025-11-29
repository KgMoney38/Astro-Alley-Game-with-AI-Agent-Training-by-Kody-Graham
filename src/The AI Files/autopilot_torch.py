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
    def __init__(self,obs_dim: int=7,hidden: int=256, act_dim: int= 2):
        super().__init__() #Initialize the nn.Module base class
        self.body= nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),
                                 nn.Linear(hidden,hidden), nn.Tanh(),
                                 nn.Linear(hidden,hidden), nn.Tanh(),) #MLP that feeds both the policy and value heads
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
    def __init__(self, ckpt_path: str | None, device:str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)# or ("cuda" if torch.cuda.is_available() else "cpu")) #Try to use CUDA because using the graphics card will be more efficient
        torch.set_grad_enabled(False)
        self.model = ActorCritic(obs_dim=7, hidden=256, act_dim=2).to(self.device)
        loaded= False
        self.cooldown = 0
        self.min_jump_interval = 3

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
            raise FileNotFoundError(f"{ckpt_path} not found")
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
        player_left = float(prect.left) #player center

        best = None
        best_dx = float("inf")
        for p in pipes:
            if not hasattr(p, "rects"):
                continue
            pipe_right = float(p.x) + float(p.width)
            dx = pipe_right-player_left

            #Choose nearest pipe not behind
            if dx >= -1e-6 and dx < best_dx:
                best= p
                best_dx = dx
        return best

    def next_two_pipes(self, pipes, px):
        future = [p for p in pipes if (p.x + p.width) >= px]
        future.sort(key=lambda p: p.x)
        p1 = future[0] if len(future) > 0 else None
        p2 = future[1] if len(future) > 1 else None
        return p1, p2

    def _too_low_in_gap(self, player, pipes, margin_px: float = 10.0) -> bool:
        """
        Return True if the player is horizontally inside the current pipe
        and the bottom of the ship is within 'margin_px' of the top of the
        lower pipe. This is the case where you usually die by scraping the
        top of the bottom pipe.
        """
        p = self.next_pipe(player, pipes)
        if p is None:
            return False

        prect = self.player_rect(player)

        pipe_left = float(p.x)
        pipe_right = float(p.x + p.width)

        #Only care when we're actually over or inside pipe horizontally
        if not (pipe_left <= float(prect.centerx) <= pipe_right):
            return False

        gap_top = float(p.top_height)
        gap_bottom = gap_top + float(p.gap)

        ship_bottom = float(prect.bottom)

        #If this gets too small, we want to jump
        dist_to_lower_pipe = gap_bottom - ship_bottom

        return dist_to_lower_pipe <= margin_px

    #Observation builder same five dimension as training, HAS TO MATCH TRAINING!
    def make_obs(self, player, pipes, screen_height: int, pipe_speed = 4.0) -> np.ndarray:
        px= player.get_rect().centerx
        py = player.get_rect().centery
        vy = getattr(player, "vel_y", getattr(player, "vy", 0.0))

        py_norm = float(py/ float(screen_height))

        vy_norm = float(np.tanh(vy/8.0))

        p1, p2 = self.next_two_pipes(pipes, px)

        t2g1 = 0.0
        t2g2 = 0.0

        center1 = 0.0
        center2 = 0.0

        gap1=0.0

        def pipe_features(p):
            cx = p.x + p.width *.5
            cy = p.top_height + p.gap * .5
            dx=cx-px
            half_gap = max(1.0, p.gap * .5)

            t2g= float(np.tanh(dx/260.0))
            center = float(np.clip((py-cy) / half_gap, -1.0, 1.0))
            return t2g, center, float(p.gap/ screen_height)

        if p1 is not None:
            t2g1, center1, gap1 = pipe_features(p1)

        if p2 is not None:
            t2g2, center2, _ = pipe_features(p2)

        return np.array([py_norm, vy_norm, t2g1, center1, gap1, t2g2, center2], dtype=np.float32)


    #The actual policy decision, return true if ai decides to jump this frame
    def decide(self, player, pipes, screen_height: int) -> bool:

        if self.cooldown > 0:
            self.cooldown -= 1
        obs = self.make_obs(player, pipes, screen_height) #Build from game state
        x= torch.from_numpy(obs).unsqueeze(0).to(self.device) #Convert to a device tensor with my batch dimensions
        with torch.inference_mode(): #Stop tracking for faster infer
            logits, _ = self.model(x)
            action = int(torch.argmax(logits, dim=1).item()) #Pick action with highest logit, 0= no 1= jump
        if self._too_low_in_gap(player, pipes, margin_px=10.0):
            action = 1
            self.cooldown = self.min_jump_interval  #reset cooldown from this safety jump

        else:
            if action == 1 and self.cooldown == 0:
                self.cooldown = self.min_jump_interval
            elif action == 1:
                action=0

        #0 = no jump, 1 jump
        return bool(action)

