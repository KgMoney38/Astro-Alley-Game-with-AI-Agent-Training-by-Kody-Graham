#Kody Graham, 8/24/2025
#Class that will control my player icon in the game, it will handle the gravity effect and the jump.

import os
from typing import Tuple, Optional
import pygame

#Resolve the path so that the image will always load correctly
def asset_path(*parts: str) -> str:
    """Resolve assets so the file will load from any working directory"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", *parts)

class Player:
    def __init__(self, x: int, y: int, image_name: str = "ship.png", size: Tuple[int,int] = (100, 50), ) -> None:

        #Load the image and then scale it to a normal size
        image_full_path = asset_path(image_name)
        self.base_image = pygame.image.load(image_full_path).convert_alpha()
        if size:
            self.base_image = pygame.transform.smoothscale(self.base_image, size)

        #This will be the image that is actually displayed
        self.image: pygame.Surface = self.base_image.copy()
        self.rect: pygame.Rect = self.image.get_rect(center = (x, y))

        #Set up the physics for the game
        self.vel_y: float = 0.0
        self.gravity: float = 0.5
        self.jump_strength: float = -10 #Negitive is up because of where the axis starts
        self.max_fall_speed: float = 10 #Limit the fall speed

        #Rotation
        self.max_tilt_up: float = 25.0
        self.max_tilt_down: float = -60.0
        self.tilt: float = 0.0

        #Flame
        self.flame_duration_ms = 120
        self.flame_ms_left: int=0

        flame_path = asset_path("flame.png")
        self.flame_base: pygame.Surface = pygame.image.load(flame_path).convert_alpha()

        #Scale flame
        fw = int(self.base_image.get_width())
        fh = int(self.base_image.get_height())
        self._flame_base = pygame.transform.smoothscale(self.flame_base, (fw, fh))

    def ignite(self) -> None:
        """Call when space pressed."""
        self.flame_ms_left = self.flame_duration_ms

    def jump(self) -> None:
        self.vel_y = self.jump_strength

    def update(self, dt_ms: int) -> None:
        #Apply the gravity effect
        self.vel_y += self.gravity
        if self.vel_y > self.max_fall_speed:
            self.vel_y = self.max_fall_speed

        #Move
        self.rect.y += int(self.vel_y)

        #Calculate the tilt based on the fall speed
        v = max(-10.0, min(10.0, self.vel_y))
        span_in = 20.0 #Range -10 to 10
        t = (v+ 10.0) / span_in
        self.tilt = self.max_tilt_up + t * (self.max_tilt_down - self.max_tilt_up)

        #Rotate based on the center
        self.image = pygame.transform.rotozoom(self.base_image, -self.tilt, 1.0)
        self.rect = self.image.get_rect(center = self.rect.center)

        #Flame timer
        if self.flame_ms_left > 0:
            self.flame_ms_left = max(0, self.flame_ms_left - dt_ms)

    def get_rect(self) -> pygame.Rect:
        #Shrink player collision box without changing image size
        return self.rect.inflate(-self.rect.width * .5, -self.rect.height * .6)

    def draw(self, surface: pygame.Surface) -> None:
        #Draw flame
        if self.flame_ms_left > 0:
            self._draw_flame(surface)
        #Draw ship
        surface.blit(self.image, self.rect)

    def reset(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        if x is not None or y is not None:
            cx = x if x is not None else self.rect.centerx
            cy = y if y is not None else self.rect.centery
            self.rect.center = (cx, cy)

        self.vel_y = 0.0
        self.tilt = 0.0
        self.image = self.base_image.copy()
        self.rect = self.image.get_rect(center = self.rect.center)
        self.flame_ms_left = 0

    #Flame Helper
    def _draw_flame(self, surface: pygame.Surface) -> None:
        #Rotate flame with ship
        flame = pygame.transform.rotozoom(self.flame_base, -self.tilt,.035)

        #Anchor near back
        back_local = pygame.math.Vector2(-self.base_image.get_width()*.5,0)

        offset = back_local.rotate(-self.tilt)
        anchor = pygame.math.Vector2(self.rect.center) + offset

        push_back= pygame.math.Vector2(-1,0).rotate(-self.tilt)*6
        center = anchor + push_back

        #Slow flame
        center.y = center.y-29
        center.x= center.x-10

        flame_rect= flame.get_rect(center = (int(center.x), int(center.y)))

        surface.blit(flame, flame_rect)