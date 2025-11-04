#Kody Graham
#11/3/2025

#This is where i will set up the environment for training my agent
#Note for Self: reward near gap, maybe use time to next gap for the timing like i set up in game unless an easier way presents its self, loop

from __future__ import annotations #To allow forward referenced types
import math, random
import numpy as np
from dataclasses import dataclass


@dataclass
class _Pipe: #_ because i used Pipe in my barriers class, keep them distinct
    x: float #Pipe left edge
    width: int
    gap_y: float #y center of pipe gap
    gap_h: int #total height of gap

class GameEnv:
    def __init__(
            self,
            width: int = 1000,
            height: int =720,
            pipe_speed: float = 4,
            pipe_width: int = 120,
            gap_range: tuple[int,int] = (230,320),
            pipe_dx: int = 320,
            gravity: float = 0.88,
            jump_impulse: float=-12.5,
            terminate_on_impact: bool = True,
            max_steps: int = 5000
    ):
        self.width = width
        self.height = height
        self.pipe_speed = float(pipe_speed)
        self.pipe_width = int(pipe_width)
        self.gap_range = gap_range
        self.pipe_dx = pipe_dx
        self.gravity = float(gravity)
        self.jump_impulse = float(jump_impulse)
        self.term_on_wall = bool(terminate_on_impact)
        self.max_steps = int(max_steps)

        #My agent state
        self.px = int(self.width *.10) #Fixed pilot x like in the real game
        self.py = None
        self.vy = None #Vertical velocity

        self.pipes: list[_Pipe]=[] #List of current active pipes
        self.steps = 0 #Steps since reset
        self.rng = random.Random(42) #RNG default will override later

        #Spawn anchor for when to add pipes
        self._spawn_anchor_x=None

    def reset(self):

    def step(self):

    def rand_gap(self):

    def next_pipe(self):

    def collision_and_pass(self):


