#Kody Graham
#11/3/2025

#This file will handle my PyTorch policy for my AI mode

from __future__ import annotations

import os
import sys
import warnings
import time
import numpy as np
import torch #PyTorch core library
import torch.nn as nn #PyTorch neural net modules

#Because i wanted my AI related .py files clearly separated in a different directory
BARRIERS_DIR= os.path.join(os.path.dirname(__file__), "The Game Files")
if BARRIERS_DIR not in sys.path:
    sys.path.insert(0, BARRIERS_DIR)

#Model will define both policy and value
class ActorCritic(nn.Module):
    def __init__(self,hidden: int=64,obs_dim: int=5, act_dim: int= 2):
        super().__init__() #Initialize the nn.Module base class
        self.body= nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden), nn.Tanh(),) #MLP that feeds both the policy and value heads
        self.pi= nn.Linear(hidden,act_dim) #Policy head
        self.v= nn.Linear(hidden,1) #Value head

    #Computes the shared hidden and returns the policy logits and value estimates
    def forward(self,x: torch.Tensor):
        h = self.body(x)
        return self.pi(h), self.v(h)

#Policy wrapper for game will be what my game calls to decide when to jump
class TorchPolicy:
    def __init__(self, ckpt_path: str | None, device:str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")) #Try to use CUDA because using the graphics card will be more efficient
        self.hidden = 64
        self.model = ActorCritic(self.hidden).to(self.device)
        self.last_jump_ms = 0

        if ckpt_path and os.path.isfile(ckpt_path): #Check checkpoint path
            self.load(ckpt_path)
        else:
            warnings.warn(f"[TorchPolicy] No checkpoint found at {ckpt_path} -> use random weights", RuntimeWarning)

        self.model.eval()

    #Loading helper
    def load(self,path:str) -> None: #Load the state_dict from disk
        sd = torch.load(path, map_location="cpu") #Load checkpoint on the CPU
        if not isinstance(sd,dict): #Validate state_dict
            raise RuntimeError(f"Checkpoint is not a dict")

        #Try to infer hidden size from the checkpint
        inferred_hidden = None
        if"body.0.weight" in sd:
            inferred_hidden = sd["body.0.weight"].shape[0]

        #If different then default update hidden width
        if inferred_hidden is not None and inferred_hidden != self.hidden:
            self.hidden = inferred_hidden
            self.model = ActorCritic(hidden=self.hidden).to(self.device)

        #Try to strict load to make sure the architecture match
        try:
            self.model.load_state_dict(sd, strict=True)
            print(f"[TorchPolicy] loaded checkpoint from {path} (hidden={self.hidden})")
        except Exception as e:
            warnings.warn(f"[TorchPolicy] Failed to load checkpoint from {path}: {e}")

    #Tensor helper to convert numpy observations to a Torch tensor
    def to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        x= torch.from_numpy(obs.astype(np.float32))
        if x.ndim==1:
            x = x.unsqueeze(0) #Add a dimension to my batch
        return x.to(self.device)

    #Observation builder same five dimension as training, HAS TO MATCH TRAINING!
    def make_obs(self, player, pipes, screen_h: int, screen_w: int = 1000) -> np.ndarray:
        pr = player.get_rect()
        px = float(pr.centerx) #Player center x
        py = float(pr.centery) #Center y

        #Read vertical velocity
        vy = float(getattr(player, "vel_y", getattr(player, "vy",0)))
        vy_norm = np.clip(vy/800, -1.5,1.5) #Normalize it

        #Find next pipe
        next_pipe = None
        nearest_dx = 1e9

        for p in pipes:
            dx = float(p.x + getattr(p, "width", 0)-px)
            if dx >= -10 and dx < nearest_dx:
                nearest_dx = dx
                next_pipe = p

        #Fallback, use a fake but reasonable pipe if there are none
        if next_pipe is None:
            class Fake:
                pass
            next_pipe = Fake()
            next_pipe.x = px+240
            next_pipe.width = 120
            fake_top= type("R", (), {"bottom":screen_h * .35})
            fake_bottom = type("R", (), {"top":screen_h * .65})
            def rects():
                return fake_top, fake_bottom
            next_pipe.rects = rects

        #Derive the gap center/height
        top_rect, bottom_rect = next_pipe.rects()
        gap_top= float(top_rect.bottom)
        gap_bot = float(bottom_rect.top)
        gap_h = max(1, gap_bot - gap_top)
        gap_center_y= gap_top +.5*gap_h

        #Time to gap normalized
        try:
            from barriers import PIPE_SPEED
            pipe_speed = float(PIPE_SPEED)
        except Exception:
            pipe_speed = 4

        #dx= player horizontal distance
        dx_to_gap_front = float(next_pipe.x -px)
        frames_to_gap = dx_to_gap_front / max(1e-6, pipe_speed)
        t2g_norm = float(np.clip(frames_to_gap/60,-1,1))

        #Vertical offset to gap center, normalize gap size
        gap_center_offset = float((gap_center_y - py)/ float(screen_h))
        gap_h_norm= float(min(1,gap_h / float(screen_h)))

        obs= np.array([py/float(screen_h), vy_norm, t2g_norm, gap_center_offset, gap_h_norm], dtype=np.float32)

        return obs

    #The actual policy decision, return true if ai decides to jump this frame
    def decide(self, player, pipes, screen_h: int, screen_w: int = 1000, min_jump_ms: int =120) -> bool:

        now= int(time.time() * 1000)
        obs = self.make_obs(player, pipes, screen_h, screen_w) #Build from game state
        x= self.to_tensor(obs) #Convert to a device tensor with my batch dimensions

        with torch.no_grad(): #Stop tracking for faster infer
            logits, _ = self.model(x)
            action = int(torch.argmax(logits, dim=1).item()) #Pick action with highest logit, 0= no 1= jump

        if action ==1 and (now- self.last_jump_ms) >= min_jump_ms:
            self.last_jump_ms = now
            return True #Signal to the game that we should jump
        return False

