#Kody Graham
#8/24/2025
import threading

import pygame
from typing import List, Optional

from menu import MainMenu, CustomizeMenu
import os, json #For my lifetime highscore
import hmac,hashlib,platform #To protect highscore integrity at least a little
from player_icon import Player
from barriers import Pipe, SCREEN_HEIGHT, PIPE_SPEED #Reused from barriers class

#For my AI Autopilot
from AI import Autopilot, start_training_async
import textwrap
import queue

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

audioDir = os.path.dirname(__file__)
ASSETS_AUDIO_DIR = os.path.join(audioDir, "assets", "sounds")

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
        self.screen = None
        self.canvas = None
        self.bg_x = None
        self.pipes = None
        self.player = None
        self.bg_image = None
        pygame.init()
        pygame.display.set_caption("Astro Alley")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 50)

        #Graph for training model
        GRAPH_H = 333
        self.graph_h =GRAPH_H

        #Fixed virtual resolution
        self.virtual_size = (SCREEN_WIDTH, SCREEN_HEIGHT+ GRAPH_H)

        self.custom_window_size(fullscreen=True)
        self.canvas= pygame.Surface(self.virtual_size).convert()
        self.training_data= {"gen": [], "best": [],"avg": []}
        self.training_thread = None
        self.model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
        self.neat_is_trained = False
        self.training_q = queue.Queue()

        #AI indicator
        self.hud_font = pygame.font.SysFont(None, 33)
        self.countdown_font = pygame.font.SysFont(None, 99)
        self.countdown_label_font = pygame.font.SysFont(None, 42)

        #For main menu
        self.state= "menu"
        self.menu = MainMenu(self)

        here= os.path.dirname(__file__)
        assets= os.path.join(here, "assets")
        self.selected_ship = os. path.join(assets, "ship.png")
        self.selected_bg = os.path.join(assets, "strip_background.png")
        self.selected_ob = os. path.join(assets, "obstacle.png")
        self.customize = CustomizeMenu(self)

        self.ai_restart_timer= None
        self.ai_enabled = False
        self.autopilot: Optional[Autopilot] = None
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
        self.ai_enabled = False
        self.autopilot = None
        self.ai_restart_timer = None
        self.state = "play"
        self.reset()

    def start_ai_game(self):
        self.ai_enabled = True
        #Clear old model so we have a fresh training demo each time
        if os.path.exists(self.model_path):
            try: os.remove(self.model_path)
            except Exception as e: print("Error removing the old model:", e)

        #Reset the graph
        self.training_data= {"gen": [], "best": [],"avg": []}
        self.training_q = queue.Queue()

        #check for neat_config.txt
        cfg_path= os.path.join(os.path.dirname(__file__), "neat-config.txt")
        if not os.path.isfile(cfg_path):
            print("Config file not found:", cfg_path)

        self.training_thread = start_training_async(cfg_path, self.model_path,n_generations=33,progress_q=self.training_q)
        self.neat_is_training = True

        #No model load when first winner is saved
        self.autopilot = None
        self.state = "play"
        self.reset()

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
       self.ai_restart_timer=None



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

            #Auto restart for ai mode
            if getattr(self, "ai_enabled", False):
                self.ai_restart_timer= pygame.time.get_ticks() +3000

            else:
                self.ai_restart_timer = None

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
            self._update(dt_ms)
            self._draw()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                #Will work in both of my game modes
                if event.key == pygame.K_ESCAPE:
                    self.open_menu()
                    #Normal window
                    if self.state == "menu" and getattr(self, "is_fullscreen", False):
                        self.is_fullscreen = False
                        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                        continue
                    else:
                        self.open_menu()
                        continue

                elif event.key == pygame.K_m:
                    self.toggle_music()
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.set_music_volume(-.05)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.set_music_volume(+.05)

                elif event.key == pygame.K_SPACE:
                    if self.game_over:
                        #Allow space to restart even in my AI mode
                        self.reset()
                    elif not getattr(self, "ai_enabled", False):
                        #Only allow user jump when AI is off
                        self.player.jump()
                        self.player.ignite()
                        if self.snd_jump:
                            self.snd_jump.play()
                    else:
                        #AI mode ignore the space
                        pass

                elif event.key == pygame.K_RETURN:
                    self.reset()

    def custom_window_size(self, fullscreen: bool= False, margin:int =10)-> None:
        if fullscreen:
            #Will use pygame to adjust scale later
            self.is_fullscreen = True
            self.screen = pygame.display.set_mode(0,0), pygame.FULLSCREEN
            self.fullscreen_exit_msg_until= pygame.time.get_ticks()+5000
            return
        else:
            #Leaving my windowed path
            self.is_fullscreen = False
            #Get screen size
            try:
                sizes= pygame.display.get_desktop_sizes()
                desktop_w, desktop_h = max(sizes,key=lambda s: s[0]*s[1])
            except Exception:
                info = pygame.display.Info()
                desktop_w, desktop_h = info.current_w, info.current_h

            #Target window size with margin
            out_w= max(640, desktop_w-margin*2)
            out_h= max(480, desktop_h-margin*2)

            #Position window this took a while remember it
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{margin}, {margin}"

            self.screen= pygame.display.set_mode((out_w, out_h), pygame.RESIZABLE)
            #self.canvas= pygame.Surface(self.virtual_size).convert()

    def _update(self, dt_ms: int) -> None:

        #Pull any training progress updates
        try:
            while True:
                upd= self.training_q.get_nowait()
                self.training_data["gen"].append(upd["gen"])
                self.training_data["best"].append(upd["best"])
                self.training_data["avg"].append(upd["avg"])
        except queue.Empty:
            pass

        #Auto restart for AI Mode
        if self.ai_enabled and self.game_over and self.ai_restart_timer is not None:
            if pygame.time.get_ticks() >= self.ai_restart_timer:
                self.reset()
                return
            else:
                return

        if self.game_over:
            return

        #Use my AI to decide to jump
        if self.ai_enabled and not self.game_over and self.autopilot is not None:
            try:
                if self.autopilot.decide(self.player, self.pipes, SCREEN_HEIGHT):
                    self.player.jump()
                    self.player.ignite()
            except Exception as e:
                #Failsafe
                print(f"[AUTOPILOT ERROR!] {e}")
                self.ai_enabled = False

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

        s= self.canvas
        s.fill((0,0,0))

        #Scroll background
        s.blit(self.bg_image, (self.bg_x, 0))
        s.blit(self.bg_image, (self.bg_x + self.bg_image.get_width(), 0))

        #Frame center for overlay
        cx = self.virtual_size[0]//2
        cy = SCREEN_HEIGHT//2

        #Obstacles and the player
        for pipe in self.pipes:
            pipe.draw(s)
        self.player.draw(s)

        #score and high score
        score_surface = self.font.render(str(self.score), True, text)
        s.blit(score_surface, (SCREEN_WIDTH// 2 - score_surface.get_width() // 2, 20))

        #High
        if self.rounds_played > 0:
            best_surface = self.font.render(f"Highest Score: {self.high_score}", True, text)
            bx = (SCREEN_WIDTH // 2) - best_surface.get_width() // 2
            by = SCREEN_HEIGHT - best_surface.get_height()-16
            s.blit(best_surface,(bx,by))

        #AI Badge
        if self.ai_enabled:
            mode= "NEAT"
            label = self.hud_font.render(f"AI: {mode}", True, text)
            pad =6
            bg_rect= pygame.Rect(12-pad,12-pad,label.get_width()+pad*2,label.get_height()+ pad*2)
            pygame.draw.rect(s,(0,0,0),bg_rect, border_radius=6)
            s.blit(label, (12,12))



        #Game over!
        msg = "GAME OVER ->Space<- to Restart"
        msg_surface = self.font.render(msg, True, text)

        if self.game_over:
            s.blit(msg_surface, (cx - msg_surface.get_width() // 2, cy - msg_surface.get_height() // 2))

            #AI auto restart
            if self.ai_enabled and self.ai_restart_timer:
                ms_left = max(0, self.ai_restart_timer - pygame.time.get_ticks()-300)
                secs = (ms_left//1000)+1

                if pygame.time.get_ticks()+1200 >= self.ai_restart_timer:
                    self.reset()
                    return



                line1 = self.countdown_label_font.render(f"AI RESTARTING IN...", True, (255,0,0))
                line2 = self.countdown_font.render(f"{secs}", True, (255,0,0))

                y1 =cy - 140
                y2 = y1+ line1.get_height()+10
                s.blit(line1, (cx-line1.get_width()//2,y1))
                s.blit(line2, (cx-line2.get_width()//2,y2))

        if self.ai_enabled:
            #My AI training graph
            panel_y=SCREEN_HEIGHT
            panel_h= self.graph_h
            panel_w = s.get_width()
            pygame.draw.rect(s, (10,10,20), (0,panel_y,panel_w, panel_h))
            axis_color=(80,80,100)
            pygame.draw.line(s, axis_color, (50,panel_y+20),(50, panel_y+panel_h-30), 1)
            pygame.draw.line(s,axis_color,(50,panel_y+panel_h-30),(panel_w-20, panel_y+panel_h-30),1)

            #Labels
            label= self.hud_font.render("NEAT Training (Best/Average)", True, (200,200,233))
            s.blit(label, (40,panel_y+4))

            gens= self.training_data["gen"]
            best= self.training_data["best"]
            avg= self.training_data["avg"]

            if gens:
                x0 =60
                x1 = panel_w-30
                y0 = panel_y+panel_h-33
                y1 = panel_y+25
                width= x1-x0
                height = y0-y1

            max_fit = max(max(best) if best else 1, max(avg) if avg else 1,1)
            def pt(i,arr):
                #Map gen index to x and fitness to y
                xi= x0+ (i/max(1, len(gens)-1))*width
                yi = y0- (arr[i]/max_fit)*height
                return int(xi),int(yi)

            #Draw the lines
            def draw_series(arr, color):
                if len(arr) >=2:
                    for i in range(1,len(arr)):
                        pygame.draw.line(s, color, pt(i-1, arr), pt(i,arr), 2)

            draw_series(best, (240,100,120))
            draw_series(avg, (120,130,220))

            #Legend for graph
            legend_best= self.hud_font.render("Best", True, (180,240,180))
            legend_avg= self.hud_font.render("Avg", True, (180,220,240))
            s.blit(legend_best, (panel_w-140, panel_y+6))
            s.blit(legend_avg, (panel_w-80, panel_y+6))

        win_width, win_height = self.screen.get_size()
        view_width, view_height = self.virtual_size
        scale= min(win_width/view_width, win_height/view_height)

        dst_width, dst_height = int(view_width*scale), int(view_height*scale)
        offset_x= (win_height - dst_width)//2
        offset_y= (win_height - dst_height)//2

        frame = pygame.transform.smoothscale(self.canvas, (dst_width, dst_height))
        self.screen.fill((0,0,0))
        self.screen.blit(frame, (offset_x, offset_y))

        if getattr(self,"is_fullscreen",False) and pygame.time.get_ticks() < getattr(self,"fullscreen_exit_msg_until",0):
            hint=self.hud_font.render('Press "ESC" to Exit Fullscreen', True, (230,230,230))
            hx=(win_width - hint.get_width())//2
            hy= 100
            pygame.draw.rect(self.screen, (0,0,0), (hx-12,hx-12, hy-8, hint.get_width() +24, hint.get_height() +16), border_radius=8)
            self.screen.blit(hint, (hx,hy))

        #Alltime high
        if self.game_over:
            ath_surface= self.font.render(f"All-Time High Score: {self.all_time_high_score}", True, text)
            s.blit(ath_surface,(cx- ath_surface.get_width() // 2, cy - ath_surface.get_height() // 2 + msg_surface.get_height() +12))

        pygame.display.flip()
