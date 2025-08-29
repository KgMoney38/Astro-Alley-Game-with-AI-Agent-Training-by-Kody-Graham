#Kody Graham
#8/24/2025

import pygame
from typing import List

import os, json #For my lifetime highscore

from player_icon import Player
from barriers import Pipe, SCREEN_HEIGHT #Reused from barriers class

#Fast settings to get to testable tonight
#Any ALL_CAPS var can be moved to settings .py and passed in
SCREEN_WIDTH = 1000
FPS = 60
SPAWN_MS = 1500
BIRD_START_X = 100
STRIP_PATH = "assets/strip_background.png"
scroll_speed= 4
background = (0,0,0)
text = (255,255,255)

#For highest score which will be lifetime not just runtime
DATA_DIR= os.path.join(os.path.dirname(__file__), "data")
HIGH_SCORE_FILE = os.path.join(DATA_DIR, "highscore.json")


def load_high_score() -> int:
    try:
        with open(HIGH_SCORE_FILE, "r", encoding="utf-8") as f:
            return int(json.load(f).get("highscore",0))
    except Exception:
        return 0

def save_high_score(score: int) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(HIGH_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump({"highscore":int(score)}, f)

class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Space Ship")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 50)

        #Load the player and the obstacles
        self.player = Player(x=BIRD_START_X, y=SCREEN_HEIGHT//2, image_name="ship.png", size = (120,60))
        self.pipes: List[Pipe] = [Pipe.spawn(SCREEN_WIDTH, SCREEN_HEIGHT)]
        self.score = 0
        self.high_score = 0
        self.all_time_high_score = load_high_score()
        self.rounds_played = 0
        self.game_over = False
        self._spawn_ms = 0
        self.running = True

        #Make my background move continuously
        self.bg_image= pygame.image.load(STRIP_PATH).convert()
        #Keep track of the scroll position
        self.bg_x =0

    #Reset
    def reset(self) -> None:
        self.player = Player(x=BIRD_START_X, y=SCREEN_HEIGHT//2, image_name="ship.png", size = (120,60))
        self.pipes = [Pipe.spawn(SCREEN_WIDTH, SCREEN_HEIGHT)]
        self.score = 0
        self.game_over = False
        self.bg_x = 0
        self._spawn_ms = 0
        self.player.flame_ms_left = 0

    #Called when round ends
    def _on_game_over(self) -> None:
        if not self.game_over:
            self.game_over = True
            self.rounds_played +=1
            if self.score > self.high_score:
                self.high_score = self.score
            if self.score > self.all_time_high_score:
                self.all_time_high_score = self.score
                save_high_score(self.all_time_high_score)

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
                        self.player.ignite()
                elif event.key == pygame.K_RETURN:
                    self.reset()

    def _update(self, dt_ms: int) -> None:

        #Increment the timer
        self.player.update(dt_ms)
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
                self._on_game_over()
                break

        #Check top/bottom collision
        if bird_rect.bottom > SCREEN_HEIGHT or bird_rect.top < 0:
            self._on_game_over()

        #Scroll background image
        self.bg_x -= scroll_speed
        if self.bg_x <= -self.bg_image.get_width():
            self.bg_x = 0

    def _draw(self) -> None:
        #Scroll background
        self.screen.blit(self.bg_image, (self.bg_x, 0))
        self.screen.blit(self.bg_image, (self.bg_x + self.bg_image.get_width(), 0))

        #Obstacles and the player
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.player.draw(self.screen)

        #score and high score
        score_surface = self.font.render(str(self.score), True, text)
        self.screen.blit(score_surface, (SCREEN_WIDTH// 2 - score_surface.get_width() // 2, 20))

        #High
        if self.rounds_played > 0:
            best_surface = self.font.render(f"Highest Score: {self.high_score}", True, text)
            bx = (SCREEN_WIDTH // 2) - best_surface.get_width() // 2
            by = SCREEN_HEIGHT - best_surface.get_height()-16
            self.screen.blit(best_surface,(bx,by))

        #Game over!
        if self.game_over:
            msg = "GAME OVER- ->Space<- to Restart"
            msg_surface = self.font.render(msg, True, text)
            cx= SCREEN_WIDTH // 2
            cy = SCREEN_HEIGHT // 2

            self.screen.blit(msg_surface, (cx - msg_surface.get_width() // 2, cy - msg_surface.get_height() // 2))

            #All time high
            ath_surface= self.font.render(f"All-Time High Score: {self.all_time_high_score}", True, text)
            self.screen.blit(ath_surface,(cx- ath_surface.get_width() // 2, cy - ath_surface.get_height() // 2 + msg_surface.get_height() +12))

        pygame.display.flip()
