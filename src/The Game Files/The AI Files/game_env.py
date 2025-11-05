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
        self.terminate_on_impact = bool(terminate_on_impact)
        self.max_steps = int(max_steps)

        #My agent state
        self.px = int(self.width *.10) #Fixed pilot x like in the real game
        self.py = None
        self.vy = None #Vertical velocity

        self.pipes: list[_Pipe]=[] #List of current active pipes
        self.steps = 0 #Steps since reset
        self.rng = random.Random(42) #RNG default will override later

        #Spawn anchor for when to add pipes
        self.spawn_anchor_x=None

    def reset(self, seed:int|None=None):
        if seed is not None: #To help determine a final seed value later
            self.rng = random.Random(int(seed))

        self.py = self.height * .5 #Start y position center everytime
        self.vy = 0
        self.steps = 0 #Step counter

        self.pipes.clear()
        first_gap = self.rand_gap() #Set random gap for first pipe
        first_x = self.width +120

        #Pipe object
        self.pipes.append(_Pipe(x=first_x,width=self.pipe_width,gap_y=first_gap[0],gap_h=first_gap[1]))

        self.spawn_anchor_x=first_x #Initialize spawn anchor to pipe x

    #Determine if autopilot should increase - vertical velocity or wait
    def step(self, action: int):
        # Action 0=no jump, 1=jump
        self.steps += 1

        #Apply autopilot action
        if int(action) == 1:
            self.vy = self.jump_impulse
        self.vy += self.gravity

        self.vy = float(max(-25, min(25,self.vy))) #Limit the velocity
        self.py += self.vy #Integrate velocity for new y pos

        #Move pipes
        for pipe in self.pipes:
            pipe.x -= self.pipe_speed

        #Spawn new pipes
        if self.pipes:
            last = self.pipes[-1]
            if last.x<= self.spawn_anchor_x - self.pipe_dx:
                gy,gh = self.rand_gap()
                nx = last.x + self.pipe_dx
                self.pipes.append(_Pipe(x=nx,width=self.pipe_width,gap_y=gy,gap_h=gh))
                self.spawn_anchor_x=nx
        else:
            gy,gh = self.rand_gap()
            self.pipes.append(_Pipe(x=self.width +120, width= self.pipe_width,gap_y=gy,gap_h=gh))
            self.spawn_anchor_x= self.pipes[-1].x

        #Delete off screen pipes
        self.pipes = [ pipe for pipe in self.pipes if pipe.x +pipe.width > 0 ]

        #Compute the reward and termination
        reward = 0.0 #No reward to start
        terminated = False
        truncated = False #Time limit check for max_steps in an episode

        #Collision Detection
        #Wall
        hit_top = self.py<0
        hit_bottom = self.py> self.height
        if (hit_top or hit_bottom) and self.terminate_on_impact:
            terminated = True
            reward-=1.0

        #Pipe
        collided, passed = self.collision_and_pass()
        if collided:
            reward -=1.0
            terminated = True
        if passed:
            reward += 1.0

        #Dense shape
        #Encourage being centered in gap when pipe close
        p= self.next_pipe()
        if p is not None:
            cx = p.x +.5 * p.width
            dx = cx- self.px
            cy = p.gap_y
            half_gap = .5 * max(1.0,float(p.gap_h))

            #Only shape when pipe close
            if 0 <= dx <= 280:
                dy_norm = abs(self.py - cx)/half_gap
                reward += .03* (1- min(1,dy_norm))

        #Small survival reward and small penalty to discourage spam jumping
        reward += .005
        if int(action) == 1:
            reward -= .002

        #Time limit
        if self.steps >= self.max_steps:
            truncated = True

        obs = self.get_obs() #Build next observation
        return obs, float(reward), bool(terminated), bool(truncated), {}

    #Helpers
    def rand_gap(self) -> tuple[float,int]:
        gh = self.rng.randint(self.gap_range[0],self.gap_range[1]) #Sample gap height

        #Keep the gap all the way in the screen
        top_margin=40
        bottom_margin= 40
        gmin= top_margin +gh/2
        gmax = self.height - bottom_margin - gh/2
        cy = self.rng.uniform(gmin,gmax)
        return cy,gh

    def next_pipe(self):
        #Pipe center from the smallest positive dx from my ship
        best= None
        best_dx= math.inf #Smallest positive dx

        for pipe in self.pipes:
            cx = pipe.x + .5 * pipe.width
            dx = cx-self.px

            #Need the closest non passed pipe
            if dx >= -1e-6 and dx < best_dx:
                best_dx = dx
                best = pipe

        return best

    #Pipes will just be considered as 2 rectangles and the gap will obviously be in the middle
    def collision_and_pass(self):
        collided = False
        passed = False
        player_w, player_h= 100, 50 #Same as my actually player icon for consistent training and real game parameters
        px1 = self.px - player_w*.5
        px2 = self.px + player_w*.5
        py1 = self.py - player_h*.5
        py2 = self.py + player_h*.5

        for pipe in self.pipes:
            top_rect= (pipe.x,0,pipe.width, pipe.gap_y-pipe.gap_h*.5)
            bottom_rect= (pipe.x, pipe.gap_y+ pipe.gap_h*.5, pipe.width, self.height - (pipe.gap_y + pipe.gap_h*.5))

            #AABB vs top and bottom
            if self.axis_aligned_bound_box(px1,py1, player_w, player_h*top_rect) or \ 
               self.axis_aligned_bound_box(px1,py1,player_w, player_h*bottom_rect):
                collided = True
                
                #Pass center check
                cx = pipe.x +.5 * pipe.width
                if cx < self.px <= cx + self.pipe_speed:
                    passed = True
        
        return collided, passed

    #Test between rectangles A and B
    @staticmethod
    def axis_aligned_bound_box(ax,ay, aw, ah, bx,by, bw,bh) -> bool:
        return (ax<bx +bw) and (ax+aw> bx) and (ay < by+bh) and (ay+ ah> by)
    
    def get_obs(self)-> np.ndarray:
        py= float(self.py)
        vy= float(self.vy) #Velocity
        next_pipe= self.next_pipe()
        
        if next_pipe is None:
            return np.array([py/self.height, max(-1.5, min(1.5, vy/ 800)),0,0, 120/self.height], dtype=np.float32)
        
        cx= next_pipe.x+ .5 * next_pipe.width
        dx = cx - self.px
        t2g = dx/ max(1,self.pipe_speed)
        norm_t2g= max(-1,min(1, t2g/60))
        
        cy = next_pipe.gap_y
        gh = float(next_pipe.gap_height)

        #[0] vertical pos normalized, [1]normalized velocity, [2] normalized time to gap center, [3] offset to gap center, [4] normal gap height
        return np.array([py/self.height, max(-1.5, min(1.5,vy/800)), norm_t2g, (cy-py)/ self.height, gh/ self.height], dtype=np.float32)
        
        



