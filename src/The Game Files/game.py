#Kody Graham
#8/24/2025

import pygame
from typing import List
from menu import MainMenu, CustomizeMenu
import os, json #For my lifetime highscore
import hmac,hashlib,platform #To protect highscore integrity at least a little
from player_icon import Player
from barriers import Pipe, SCREEN_HEIGHT, PIPE_SPEED #Reused from barriers class

#Sound effects
pygame.mixer.pre_init(44100, -16, 2, 512)

#Fast settings to get to testable tonight
#Any ALL_CAPS var can be moved to settings .py and passed in
SCREEN_WIDTH = 1000
FPS = 60
SPAWN_MS = 1500
BIRD_START_X = 100
STRIP_PATH = "../The Game Files/assets/strip_background.png"
scroll_speed= 4
background = (0,0,0)
text = (255,255,255)

#For highest score which will be lifetime not just runtime
DATA_DIR= os.path.join(os.path.dirname(__file__), "data")
HIGH_SCORE_FILE = os.path.join(DATA_DIR, "highscore.json")

#Protect All-Time High Score!
SECRET_KEY= "SECRET-DEMO-KEY-v1:: 9d8d7ea4b6b940aa8b4e2f0a1d5f37"
SECRET_KEY=os.environ.get("GAME_SECRET", SECRET_KEY)
BIND_TO_DEVICE= False #So it is available through GitHub to anyone who pulls the repo

def _sign(score: int, salt: str) -> str:
    device = platform.node() if BIND_TO_DEVICE else ""
    msg = f"{score}|{salt}|{device}".encode("utf-8")
    return hmac.new(SECRET_KEY.encode("utf-8"), msg, hashlib.sha256).hexdigest()

Here = os.path.dirname(__file__)
ASSETS_AUDIO_DIR = os.path.join(Here, "assets", "sounds")

def sound_path(*parts: str) -> str:
    return os.path.join(ASSETS_AUDIO_DIR, *parts)

def load_high_score() -> int:
    try:
        with open(HIGH_SCORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        score = int(data.get("score", 0))
        salt = data.get("salt", "")
        sig = data.get("sig", "")

        #Verify
        if sig and hmac.compare_digest(sig, _sign(score, salt)):
            return score
        #MISSING OR INVALID!
        return 0

    except Exception:
        return 0

def save_high_score(score: int) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    salt = os.urandom(16).hex()
    sig= _sign(int(score), salt)
    payload = {"score": int(score), "salt": salt, "sig": sig}
    with open(HIGH_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)

class Game:
    def __init__(self) -> None:
        self.bg_x = None
        self.pipes = None
        self.player = None
        self.bg_image = None
        pygame.init()
        pygame.display.set_caption("Astro Alley")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 50)

        #For main menu
        self.state= "menu"
        self.menu = MainMenu(self)

        here= os.path.dirname(__file__)
        assets= os.path.join(here, "assets")
        self.selected_ship = os. path.join(assets, "ship.png")
        self.selected_bg = os.path.join(assets, "strip_background.png")
        self.selected_ob = os. path.join(assets, "obstacle.png")
        self.customize = CustomizeMenu(self)

        #Score and flags
        self.score = 0
        self.high_score = 0
        self.all_time_high_score = load_high_score()
        self.rounds_played = 0
        self.game_over = False
        self._spawn_ms = 0
        self.running = True

        #Sound effects
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size =-16,channels=2, buffer=512)
            pygame.mixer.set_num_channels(8)

            opensource_music_file = sound_path("backgroundmusic.mp3")
            if not os.path.isfile(opensource_music_file):
                print("Music file not found", opensource_music_file)
            else:
                pygame.mixer.music.load(opensource_music_file)
                pygame.mixer.music.set_volume((.15))
                pygame.mixer.music.play(-1,fade_ms=1500)
            self._music_paused = False

            self.snd_jump = pygame.mixer.Sound(sound_path("jump.wav"))
            self.snd_crash = pygame.mixer.Sound(sound_path("crash.wav"))
            self.snd_jump.set_volume(.04)
            self.snd_crash.set_volume(0.03)

        except Exception as e:
            print("Audio disabled:", e)
            self.snd_jump= self.snd_crash = None
            self._music_paused = True

        self.reset()

    #For Main Menu
    def open_menu(self):
        self.state = "menu"

    def start_user_game(self):
        self.state = "play"

    def start_ai_game(self):
        self.state = "play"

    def open_customize(self):
        self.state = "customize"# change later

    def quit(self):
        self.running = False

    #Reset
    def reset(self) -> None:
       ship_name = os.path.basename(self.selected_ship)
       self.player = Player(x=BIRD_START_X, y= int(SCREEN_HEIGHT-SCREEN_HEIGHT/2), image_name=ship_name, size=(120,60))

       self.bg_image = pygame.image.load(self.selected_bg).convert()
       self.bg_x = 0

       ob_name = os.path.basename(self.selected_ob)

       spacing_px = int(PIPE_SPEED *(SPAWN_MS / 1000.0) * FPS)
       initial_x = SCREEN_WIDTH/2 + spacing_px

       self.pipes: List[Pipe] = [Pipe.spawn(int(initial_x), SCREEN_HEIGHT, image_name=ob_name)]

       self.score = 0
       self.game_over = False
       self._spawn_ms = 0
       self.player.flame_ms_left = 0


    def set_music_volume(self, delta:float) -> None:

        if pygame.mixer.get_init():
            v = pygame.mixer.music.get_volume()
            pygame.mixer.music.set_volume(max(0.0, min(1.0,v+delta)))

    def toggle_music(self)-> None:
        if not pygame.mixer.get_init():
            return
        if getattr(self, "_music_paused", False):
            pygame.mixer.music.unpause()
            self._music_paused = False
        else:
            pygame.mixer.music.pause()
            self._music_paused = True

    #Called when round ends
    def _on_game_over(self) -> None:
        if not self.game_over:
            self.game_over = True
            self.rounds_played +=1
            if self.snd_crash:
                self.snd_crash.play()
            if self.score > self.high_score:
                self.high_score = self.score
            if self.score > self.all_time_high_score:
                self.all_time_high_score = self.score
                save_high_score(self.all_time_high_score)

    def run(self) -> None:
        while self.running:
            dt_ms = self.clock.tick(FPS)

            if self.state == "menu":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    else:
                        self.menu.handle_event(event)
                self.menu.draw(self.screen)
                pygame.display.flip()
                continue

            if self.state == "customize":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    else:
                        self.customize.handle_event(event)
                self.customize.draw(self.screen)
                pygame.display.flip()
                continue

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
                    self.open_menu()
                elif event.key == pygame.K_m:
                    self.toggle_music()
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.set_music_volume(-.05)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.set_music_volume(+.05)

                elif event.key == pygame.K_SPACE:
                    if self.game_over:
                        self.reset()
                    else:
                        #Jump
                        self.player.jump()
                        self.player.ignite()

                        if self.snd_jump:
                            self.snd_jump.play()

                elif event.key == pygame.K_RETURN:
                    self.reset()

    def _update(self, dt_ms: int) -> None:

        #Increment the timer
        self.player.update(dt_ms)
        self._spawn_ms += dt_ms

        #New pipes spawn based on timer
        if self._spawn_ms >= SPAWN_MS:
            ob_name = os.path.basename(self.selected_ob)
            self.pipes.append(Pipe.spawn(SCREEN_WIDTH, SCREEN_HEIGHT, image_name=ob_name))
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
            msg = "GAME OVER ->Space<- to Restart"
            msg_surface = self.font.render(msg, True, text)
            cx= SCREEN_WIDTH // 2
            cy = SCREEN_HEIGHT // 2

            self.screen.blit(msg_surface, (cx - msg_surface.get_width() // 2, cy - msg_surface.get_height() // 2))

            #All time high
            ath_surface= self.font.render(f"All-Time High Score: {self.all_time_high_score}", True, text)
            self.screen.blit(ath_surface,(cx- ath_surface.get_width() // 2, cy - ath_surface.get_height() // 2 + msg_surface.get_height() +12))

        pygame.display.flip()
