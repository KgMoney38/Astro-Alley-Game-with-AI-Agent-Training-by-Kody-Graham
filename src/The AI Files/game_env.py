#Kody Graham
#11/3/2025

#This is where i will set up the environment for training my agent
#Note for Self: reward near gap, maybe use time to next gap for the timing like i set up in game unless an easier way presents its self, loop

#Note for self: Done

from __future__ import annotations #To allow forward referenced types
import math, random
import os
import sys

import numpy as np
from dataclasses import dataclass

#Had to have to import my barriers class from a different directory than cd
BASE_DIR = os.path.dirname(__file__)
GAME_FILE_DIR= os.path.abspath(os.path.join(BASE_DIR, "..", "The Game Files"))
if GAME_FILE_DIR not in sys.path:
    sys.path.insert(0,GAME_FILE_DIR)
from barriers import SCREEN_HEIGHT, PIPE_GAP, PIPE_WIDTH, PIPE_SPEED, PIPE_MIN_TOP, PIPE_MAX_TOP

#Aligned with player_icon.py
PLAYER_W=99
PLAYER_H=33
PLAYER_HALF_H = PLAYER_H * .5

#Simple pipe representation for training env
@dataclass
class _Pipe: # "_" because i used Pipe in my barriers class, keep them distinct
    x: float #Pipe left edge
    width: int
    gap_y: float #y center of pipe gap
    gap_h: int #total height of gap

#Class that handles the environment training will run on
class GameEnv:
    def __init__(
            self,
            width: int = 1000,
            height: int = SCREEN_HEIGHT,
            pipe_speed: int= PIPE_SPEED,
            pipe_width: int =PIPE_WIDTH,
            gap: int = max(40, PIPE_GAP-20),
            pipe_dx: int = 360,
            gravity: float = 0.5,
            jump_impulse: float=-10.0,
            terminate_on_impact: bool = True,
            max_steps: int = 5000,


            #Rewards
            pass_reward: float= 10.0,
            crash_penalty: float = -10.0,
            step_reward: float = .010,
            jump_penalty: float = .000,
            shaping_scale: float = .06,
            shaping_max_dx: float = 260.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.pipe_speed = float(pipe_speed)
        self.pipe_width = int(pipe_width)
        self.gap = int(gap)
        self.pipe_dx = int(pipe_dx)
        self.gravity = float(gravity)
        self.jump_impulse = float(jump_impulse)
        self.terminate_on_impact = bool(terminate_on_impact)
        self.max_steps = int(max_steps)

        self.pass_reward = float(pass_reward)
        self.crash_penalty = float(crash_penalty)
        self.step_reward = float(step_reward)
        self.jump_penalty = float(jump_penalty)
        self.shaping_scale = float(shaping_scale)
        self.shaping_max_dx = float(shaping_max_dx)

        #My agent state
        self.px = int(self.width *.10) #Fixed pilot x like in the real game

        self.py = None
        self.vy = None #Vertical velocity

        self.pipes: list[_Pipe]=[] #List of current active pipes
        self.steps: int=0 #Steps since reset
        self.rng = random.Random(42) #RNG default will override later

    def reset(self, seed: int | None=None):
        if seed is not None: #To help determine a final seed value later
            self.rng = random.Random(int(seed))

        self.py = self.height * .5 #Start y position center everytime
        self.vy = 0.0
        self.steps = 0 #Step counter
        self.pipes.clear()

        cy, gh = self.rand_gap()
        first_x = float(self.width) #Set random gap for first pipe

        #Pipe object
        self.pipes.append(_Pipe(x=first_x,width=self.pipe_width,gap_y=cy,gap_h=gh))

        obs = self.get_obs()
        return obs, {}

    #Determine if autopilot should increase - vertical velocity or wait
    def step(self, action: int):
        # Action 0=no jump, 1=jump
        self.steps += 1

        #Apply autopilot action
        if int(action) == 1:
            self.vy = self.jump_impulse

        #G
        self.vy += self.gravity
        self.vy = max(self.jump_impulse, min(10.0,self.vy))
        self.py += self.vy #Integrate velocity for new y pos

        #Move pipes
        for pipe in self.pipes:
            pipe.x -= self.pipe_speed

        #Spawn new pipes
        if self.pipes:
            rightmost_x = max(pipe.x for pipe in self.pipes)
            if rightmost_x<= self.width - self.pipe_dx:
                cy,gh = self.rand_gap()
                spawn_x = float(self.width)
                self.pipes.append(_Pipe(x=spawn_x,width=self.pipe_width,gap_y=cy,gap_h=gh))
        else:
            cy,gh = self.rand_gap()
            spawn_x = float(self.width)
            self.pipes.append(_Pipe(x= spawn_x, width= self.pipe_width,gap_y=cy,gap_h=gh))

        #Delete off screen pipes
        self.pipes = [ pipe for pipe in self.pipes if pipe.x +pipe.width > 0 ]

        #Compute the reward and termination
        reward = 0.0 #No reward to start
        terminated = False
        truncated = False #Time limit check for max_steps in an episode

        #Collision Detection
        #Wall
        top = self.py - PLAYER_HALF_H
        bottom = self.py + PLAYER_HALF_H

        hit_top = top < 0.0
        hit_bottom = bottom > float(self.height)

        if( hit_top or hit_bottom) and self.terminate_on_impact:
            terminated = True
            reward += self.crash_penalty

        #Pipe
        collided = False
        passed= False

        if not terminated:
            collided,passed = self.collision_and_pass()
            if collided and self.terminate_on_impact:
                terminated = True
                reward += self.crash_penalty

        if passed:
            reward += self.pass_reward

        #Dense shape
        #Encourage being centered in gap when pipe close
        next_p= self.next_pipe()
        if next_p is not None:
            cx = next_p.x +.5 * next_p.width
            dx = cx- self.px

            if 0.0 <= dx <= self.shaping_max_dx:
                cy = next_p.gap_y
                half_gap = .5 * max(1.0,float(next_p.gap_h))
                dy_norm = abs(self.py - cy)/ half_gap
                closeness = 1.0 - min(1.0, dy_norm)
                closeness_sq= closeness * closeness
                w= math.exp(-dx/self.shaping_max_dx)
                reward += self.shaping_scale * w * closeness_sq
                edge_thresh = .6
                if dy_norm> edge_thresh:
                    penalty = (dy_norm-edge_thresh)**2
                    reward -= .06*w* penalty

                if next_p is not None and 0.0 <= dx <=self.shaping_max_dx:
                    cy = next_p.gap_y
                    half_gap = .5 * max(1.0,float(next_p.gap_h))
                    align_raw = (cy- self.py) *(-self.vy)
                    align_norm = max(-1.0, min(1.0, align_raw/ (10*half_gap)))
                    reward += .02 * w * align_norm

        #Small survival reward and small penalty to discourage spam jumping
        reward += self.step_reward

        if int(action) == 1:
            reward -= self.jump_penalty

        #Time limit
        if self.steps >= self.max_steps:
            truncated = True

        obs = self.get_obs() #Build next observation
        info= {"passed": bool(passed), "collided": bool(collided),}
        return obs, float(reward), bool(terminated), bool(truncated), info

    #Helpers
    def rand_gap(self) -> tuple[float,int]:

        gh= int(self.gap) #Slightly tighter than real game for better training

        #Keep the gap all the way in the screen
        max_top =max(50, self.height-gh-50)
        top_hi = min(PIPE_MAX_TOP, max_top)
        top = self.rng.randint(PIPE_MIN_TOP, top_hi)
        cy = top+ gh/2.0

        return cy, gh

    def next_pipe(self) -> _Pipe | None:
        #Pipe center from the smallest positive dx from my ship
        best= None
        best_dx= math.inf #Smallest positive dx
        player_left =self.px- PLAYER_W * .5

        for pipe in self.pipes:
            pipe_right = pipe.x + pipe.width
            dx = pipe_right - player_left

            #Need the closest non passed pipe
            if dx >= -1e-6 and dx < best_dx:
                best_dx = dx
                best = pipe

        return best

    #Pipes will just be considered as 2 rectangles and the gap will obviously be in the middle
    def collision_and_pass(self) -> tuple[bool,bool]:
        collided = False
        passed = False

        player_w, player_h= PLAYER_W, PLAYER_H #Same as my actually player icon for consistent training and real game parameters
                                    #For tighter fit i increased size player size here for training but left it the same in player_icon
        px1 = self.px - player_w*.5
        py1 = self.py - player_h*.5

        for pipe in self.pipes:

            gap_top = pipe.gap_y - pipe.gap_h*.5
            gap_bottom = pipe.gap_y + pipe.gap_h*.5

            top_rect= (pipe.x,0.0,pipe.width, gap_top)
            bottom_rect= (pipe.x, gap_bottom, pipe.width, self.height - gap_bottom)

            #AABB vs top and bottom
            if self.axis_aligned_bound_box(px1,py1, player_w, player_h,*top_rect) or \
               self.axis_aligned_bound_box(px1,py1,player_w, player_h,*bottom_rect):
                collided = True
                
            #Pass center check
            pipe_right = pipe.x + pipe.width
            if pipe_right < self.px <= pipe_right + self.pipe_speed:
                passed = True
        
        return collided, passed

    #Test between rectangles A and B
    @staticmethod
    def axis_aligned_bound_box(ax,ay, aw, ah, bx,by, bw, bh) -> bool:
        return (ax<bx +bw) and (ax+aw> bx) and (ay < by+bh) and (ay+ ah> by)
    
    def get_obs(self)-> np.ndarray:
        H= float(self.height)
        py= float(self.py)
        vy= float(self.vy) #Velocity
        vy_norm = max(-1.5, min(1.5, vy/ 10.0))

        pipe= self.next_pipe()
        if pipe is None:
            return np.array([py/H, vy_norm,0.0, 0.0, self.gap/H], dtype=np.float32)
        
        cx= pipe.x + .5 * pipe.width
        dx = cx - self.px
        t2g = dx/ max(1.0,self.pipe_speed)
        t2g_norm= max(-1.0,min(1.0, t2g/60.0))

        gap_center = pipe.gap_y
        gap_h = float(pipe.gap_h)
        #max_top = max(50, int(self.height - gh - 50))
        #top = self.rng.randint(PIPE_MIN_TOP, min(PIPE_MAX_TOP, max_top))
        gap_center_offset = (gap_center-py) / H
        gap_h_norm= gap_h/ H

        #[0] vertical pos normalized, [1]normalized velocity, [2] normalized time to gap center, [3] offset to gap center, [4] normal gap height
        return np.array([py/H, vy_norm, t2g_norm, gap_center_offset, gap_h_norm], dtype=np.float32,)
        
        



