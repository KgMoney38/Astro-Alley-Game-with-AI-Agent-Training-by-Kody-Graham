#Kody Graham
#8/24/2025

import pygame
from typing import List

import self

from player_icon import Player
from barriers import Pipe, SCREEN_HEIGHT #Reused from barriers class

#Fast settings to get to testable tonight
SCREEN_WIDTH = 1000
FPS = 60
SPAWN_MS = 1500
BIRD_START_X = 100
STRIP_PATH = "assets/strip_background.png"
num_frames = 50
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
        self.player = Player(x=BIRD_START_X, y=SCREEN_HEIGHT//2, image_name="ship.png", size = (50,50))
        self.pipes: List[Pipe] = [Pipe.spawn(SCREEN_WIDTH//2), Pipe.spawn(SCREEN_WIDTH, SCREEN_HEIGHT)]
        self.score = 0
        self.game_over = False
        self._spawn_ms = 0
        self.running = True

        #Make my background move with every jump
        strip = pygame.image.load(STRIP_PATH).convert()

        #Slice my strip into num_frames frames, each exactly the same size as my screen
        self.bg_frames: List[pygame.Surface] = []
        for i in range(num_frames):
            rect = pygame.Rect(i*SCREEN_WIDTH,0, SCREEN_WIDTH, SCREEN_HEIGHT)
            self.bg_frames.append(pygame.Surface(rect).copy())
        self.bg_index = 0

        #Reset
        def reset(self) -> None:
            self.player = Player(x=BIRD_START_X, y=SCREEN_HEIGHT//2, image_name="ship.png", size = (50,50))
            self.pipes = [Pipe.spawn(SCREEN_WIDTH, SCREEN_HEIGHT)]
            self.score = 0
            self.game_over = False
            self.bg_index = 0
            self._spawm_ms = 0

        def run(self) -> None:
            while self.running:
                dt_ms = self.clock.tick(FPS)
                self._handle_events()
                if not self.game_over:
                    self.update(dt_ms)
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
                            #Move background
                            self.bg_index = (self.bg_index + 1) % num_frames
                    elif event.key == pygame.K_RETURN:
                        self.reset()

        def _update(self, dt_ms: int) -> None:
            self.player.update()

            #New pipes spawn based on timer
            if self.spawn_ms >= SPAWN_MS:
                self.pipes.append(Pipe.spawn(SCREEN_WIDTH, SCREEN_HEIGHT))
                self._spawn_ms = 0

            #Move the pipes and keep the score
            for p in self.pipes:
                p.update()
                if not p.passed and (p.x + p.width < BIRD_START_X):
                    p.passed = True
                    self.score += 1

            #Delete when off the screen
            while self.pipes and self.pipes[0].off_screen():
                self.pipes.pop(0)

            #Track the collisions
            bird_rect = self.player.get_rect()
            for pipe in self.pipes:
                top_rect, bot_rect = p.rects()
                if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bot_rect):
                    self.game_over = True
                    break

        def _draw(self) -> None:
            #Draw the frame for the background and only advance on jump
            self.screen.fill(background)
            self.screen.blit(self.bg_frames[self.bg_index], (0,0))
            text = (255, 255, 255)

            #Pipes and the player
            for p in self.pipes:
                p.draw(self.screen)
            self.player.draw(self.screen)

            #score
            score_surface = self.front.render(str(self.score), True, text)
            self.screen.blit(score_surface, (SCREEN_WIDTH// 2 - score_surface.get_width() // 2, 20))

            if self.game_over:
                msg = "GAME OVER- ->Space<- to Restart"
                text = self.font.render(msg, True, text)
                self.screen.blit(text, SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2)

            pygame.display.flip()
