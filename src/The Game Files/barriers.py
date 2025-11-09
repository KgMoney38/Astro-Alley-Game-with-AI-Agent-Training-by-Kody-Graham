#Kody Graham, 8/24/2025
#Generate and control the pipes that act as obstacles
#Note for self: Done

import os
import random
from typing import Tuple, Dict, LiteralString
import pygame

#TEMP Variables: Will move to settings later but want a testable version quick so adding them here for now
SCREEN_HEIGHT = 800
PIPE_WIDTH = 100
PIPE_GAP = 160
PIPE_SPEED = 4
PIPE_MIN_TOP = 100
PIPE_MAX_TOP = 400

def asset_path(*parts: str) -> LiteralString | str | bytes:
    #Resolve path to asset relative to the png
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", *parts)

class Pipe:
    #Scale two pipes one top and one bottom and make them move left
    #Class type cache so it only loads the png one time
    _surfaces: Dict[str, pygame.Surface] = {}
    HITBOX_INSET_X= 8
    TOP_TRIM_BOTTOM = 8
    BOTTOM_TRIM_TOP = 10

    def __init__(self, x: int, top_height: int, screen_height: int = SCREEN_HEIGHT, gap: int = PIPE_GAP, width: int = PIPE_WIDTH, speed: int = PIPE_SPEED, image_name: str = "obstacle.png") -> None:

        if image_name not in Pipe._surfaces:
            Pipe._surfaces[image_name] = pygame.image.load(asset_path(image_name)).convert_alpha()
        base= Pipe._surfaces[image_name]

        self.x: float = float(x)
        self.top_height: int = int(top_height)
        self.screen_height: int = int(screen_height)
        self.gap: int = int(gap)
        self.width: int = int(width)
        self.speed: int = int(speed)
        self.image_name: str = image_name
        self.passed: bool = False #Will be true after player passes to iterate score

        #Compute the height of the bottom from the top + the gap
        bottom_y = self.top_height + self.gap
        self.bottom_height: int = max(0,self.screen_height - bottom_y)

        #Flip for top
        self.top_image: pygame.Surface = pygame.transform.smoothscale(pygame.transform.flip(base, False, True), (self.width, self.top_height),)

        #Bottom png is normal
        self.bottom_image: pygame.Surface = pygame.transform.smoothscale(base, (self.width, self.bottom_height))

    #Factory Method
    @classmethod
    def spawn(cls, screen_width: int, screen_height: int = SCREEN_HEIGHT, image_name: str = "obstacle.png",) -> "Pipe":
        max_top = max(50, screen_height - PIPE_GAP - 50)
        top = random.randint(PIPE_MIN_TOP, min(PIPE_MAX_TOP, max_top))
        return cls(x=screen_width, top_height=top, screen_height=screen_height, image_name=image_name,)

    #Game loop API
    def update(self) -> None:
        self.x -= self.speed

    def off_screen(self) -> bool:
        return self.x + self.width < 0

    def rects(self) -> Tuple[pygame.Rect, pygame.Rect]:

        trim_x=4

        ix = self.HITBOX_INSET_X

        top_width = max(1, self.width-2*ix)
        top_height = max(1, self.top_height- self.TOP_TRIM_BOTTOM)
        top_rect = pygame.Rect(int(self.x) + trim_x, 0, self.width, self.top_height-10)

        bottom_y = self.top_height + self.gap
        bottom_x = int(self.x)+ ix
        bottom_height = max(1, self.bottom_height - self.BOTTOM_TRIM_TOP)
        bottom_rect = pygame.Rect(bottom_x, bottom_y +self.BOTTOM_TRIM_TOP, top_width, bottom_height,)


        return top_rect, bottom_rect

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.top_image, (int(self.x), 0))
        surface.blit(self.bottom_image, (int(self.x), self.top_height + self.gap))

        #Debug the tops of the boxes to avoid center collision
        #top_rect, bottom_rect = self.rects()
        #pygame.draw.rect(surface, pygame.Color("red"), top_rect,2)
        #pygame.draw.rect(surface, pygame.Color("blue"), bottom_rect,2)
