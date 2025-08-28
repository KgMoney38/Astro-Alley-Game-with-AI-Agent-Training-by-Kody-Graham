#Kody Graham
#8/24/2025

import pygame
from typing import List

from player_icon import Player
from barriers import Pipe, SCREEN_HEIGHT #Reused from barriers class

#Fast settings to get to testable tonight
SCREEN_WIDTH = 1000
FPS = 60
SPAWN_MS = 1500
BIRD_START_X = 100
STRIP_PATH = "assets/strip_background.png"
scroll_speed= 3
background = (0,0,0)
text = (255,255,255)

class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Space Ship")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 50)

        #Load the player and the pipes
        self.player = Player(x=BIRD_START_X, y=SCREEN_HEIGHT//2, image_name="ship.png", size = (80,80))
        self.pipes: List[Pipe] = [Pipe.spawn(SCREEN_WIDTH, SCREEN_HEIGHT)]
        self.score = 0
        self.game_over = False
        self._spawn_ms = 0
        self.running = True

        #Make my background move continuously
        self.bg_image= pygame.image.load(STRIP_PATH).convert()
        #Keep track of the scroll position
        self.bg_x =0

    #Reset
    def reset(self) -> None:
        self.player = Player(x=BIRD_START_X, y=SCREEN_HEIGHT//2, image_name="ship.png", size = (80,80))
        self.pipes = [Pipe.spawn(SCREEN_WIDTH, SCREEN_HEIGHT)]
        self.score = 0
        self.game_over = False
        self.bg_x = 0
        self._spawn_ms = 0

    def run(self) -> None:
        while self.running:
            dt_ms = self.clock.tick(FPS)
            self._handle_events()
            if not self.game_over:
                self._update(dt_ms)
            self._draw()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.game_over:
                        self.reset()
                    else:
                        #Jump
                        self.player.jump()
                elif event.key == pygame.K_RETURN:
                    self.reset()

    def _update(self, dt_ms: int) -> None:

        #Increment the timer
        self.player.update()
        self._spawn_ms += dt_ms

        #New pipes spawn based on timer
        if self._spawn_ms >= SPAWN_MS:
            self.pipes.append(Pipe.spawn(SCREEN_WIDTH, SCREEN_HEIGHT))
            self._spawn_ms = 0

        #Move the pipes and keep the score
        for pipe in self.pipes:
            pipe.update()
            if not pipe.passed and (pipe.x + pipe.width < BIRD_START_X):
                pipe.passed = True
                self.score += 1

        #Delete when off the screen
        self.pipes = [pipe for pipe in self.pipes if not pipe.off_screen()]

        #Track the collisions
        bird_rect = self.player.get_rect()
        for pipe in self.pipes:
            top_rect, bot_rect = pipe.rects()
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bot_rect):
                self.game_over = True
                break

        #Check top/bottom collision
        if bird_rect.bottom > SCREEN_HEIGHT or bird_rect.top < 0:
            self.game_over = True

        #Scroll background image
        self.bg_x -= scroll_speed
        if self.bg_x <= -self.bg_image.get_width():
            self.bg_x = 0

    def _draw(self) -> None:
        #Scroll background
        self.screen.blit(self.bg_image, (self.bg_x, 0))
        self.screen.blit(self.bg_image, (self.bg_x + self.bg_image.get_width(), 0))

        #Pipes and the player
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.player.draw(self.screen)

        #score
        score_surface = self.font.render(str(self.score), True, text)
        self.screen.blit(score_surface, (SCREEN_WIDTH// 2 - score_surface.get_width() // 2, 20))

        if self.game_over:
            msg = "GAME OVER- ->Space<- to Restart"
            msg_surface = self.font.render(msg, True, text)
            self.screen.blit(msg_surface, (SCREEN_WIDTH // 2 - msg_surface.get_width() // 2, SCREEN_HEIGHT // 2 - msg_surface.get_height() // 2),)

        pygame.display.flip()
