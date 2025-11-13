#Kody Graham, 8/24/2025
#Generate and control the pipes that act as obstacles
#Note for self: Done

import os
import random
from typing import Tuple, Dict
import pygame

#TEMP Variables: Will move to settings later but want a testable version quick so adding them here for now
SCREEN_HEIGHT = 800
PIPE_WIDTH = 100
PIPE_GAP = 160
PIPE_SPEED = 4
PIPE_MIN_TOP = 100
PIPE_MAX_TOP = 400

#To debug my pipes might make it a mode you can on and off later
DEBUG_PIPE_OVERLAY = True

def asset_path(*parts: str) -> str:
    #Resolve path to asset relative to the png
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", *parts)

class Pipe:
    #Scale two pipes one top and one bottom and make them move left
    #Class type cache so it only loads the png one time
    _surfaces: Dict[str, pygame.Surface] = {}

    def __init__(self, x: int, top_height: int, screen_height: int = SCREEN_HEIGHT, gap: int = PIPE_GAP, width: int = PIPE_WIDTH, speed: int = PIPE_SPEED, image_name: str = "obstacle.png") -> None:

        if image_name not in Pipe._surfaces:
            Pipe._surfaces[image_name] = pygame.image.load(asset_path(image_name)).convert_alpha()
        base= Pipe._surfaces[image_name]

        self.x = float(x)
        self.top_height = int(top_height)
        self.screen_height = int(screen_height)
        self.gap = int(gap)
        self.width = int(width)
        self.speed = int(speed)
        self.image_name = image_name
        self.passed = False #Will be true after player passes to iterate score

        #Compute the height of the bottom from the top + the gap
        bottom_y = self.top_height + self.gap
        self.bottom_height = max(0,self.screen_height - bottom_y)

        #Flip for top
        self.top_image = pygame.transform.smoothscale(pygame.transform.flip(base, False, True), (self.width, self.top_height))

        #Bottom png is normal
        self.bottom_image = pygame.transform.smoothscale(base, (self.width, self.bottom_height))

        #Masks
        self.top_mask = pygame.mask.from_surface(self.top_image, 1)
        self.bottom_mask = pygame.mask.from_surface(self.bottom_image, 1)

        if DEBUG_PIPE_OVERLAY:
            self.top_mask_surface= self.top_mask.to_surface(setcolor=(255,0,0, 120), unsetcolor=(0,0,0,0))
            self.top_mask_surface.set_colorkey((0,0,0))
            self.bot_mask_surface= self.bottom_mask.to_surface(setcolor=(0,0,255, 120), unsetcolor=(0,0,0,0))
            self.bot_mask_surface.set_colorkey((0,0,0))

    #Factory Method
    @classmethod
    def spawn(cls, screen_width: int, screen_height: int = SCREEN_HEIGHT, image_name: str = "obstacle.png") -> "Pipe":
        max_top = max(50, screen_height - PIPE_GAP - 50)
        top = random.randint(PIPE_MIN_TOP, min(PIPE_MAX_TOP, max_top))
        return cls(x=screen_width, top_height=top, screen_height=screen_height, image_name=image_name)

    #Game loop API
    def update(self) -> None:
        self.x -= self.speed

    def off_screen(self) -> bool:
        return self.x + self.width < 0

    def rects(self) -> tuple[pygame.Rect, pygame.Rect]:

        top_rect = pygame.Rect(int(self.x), 0, int(self.width), int(self.top_height))
        bottom_y = self.top_height + self.gap
        bottom_rect = pygame.Rect(int(self.x), int(bottom_y), int(self.width), int(self.bottom_height))

        return top_rect, bottom_rect

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.top_image, (int(self.x), 0))
        surface.blit(self.bottom_image, (int(self.x), self.top_height + self.gap))

        #Debug the tops of the boxes to avoid center collision
        #top_rect, bottom_rect = self.rects()
        #pygame.draw.rect(surface, pygame.Color("red"), top_rect,2)
        #pygame.draw.rect(surface, pygame.Color("blue"), bottom_rect,2)

    #Developer Mode: option so i can debug my masks and safe area of the gap
    def debug_draw(self, surface: pygame.Surface) -> None:
        if not DEBUG_PIPE_OVERLAY:
            return

        #Top mask
        surface.blit(self.top_mask_surface, (int(self.x), 0))

        #Bottom mask
        by = self.top_height+ self.gap
        surface.blit(self.bot_mask_surface, (int(self.x), int(by)))

        #Gap safe area overlay
        gap_top = self.top_height
        gap_bottom = self.top_height + self.gap
        safe= pygame.Surface((self.width, max(1, gap_bottom -gap_top)), pygame.SRCALPHA)
        safe.fill((0,255,0,60))
        surface.blit(safe, (int(self.x), int(gap_top)))
        pygame.draw.rect(surface, (188,0,255), (int(self.x), int(gap_top), self.width, self.gap), 2)
