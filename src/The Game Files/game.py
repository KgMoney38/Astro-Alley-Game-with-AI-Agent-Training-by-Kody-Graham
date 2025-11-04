#Kody Graham
#8/24/2025
#Class controls almost all functionality of my game, controls the entire game loop
import sys
from collections import deque

#Note for self: Done for now

import pygame
from typing import List, Optional
import os


os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
from launch_video import play_video_cover, play_video_sequence_cover

from menu import MainMenu, CustomizeMenu
import os, json #For my lifetime highscore
import hmac,hashlib,platform #To protect highscore integrity at least a little
from player_icon import Player
from barriers import Pipe, SCREEN_HEIGHT, PIPE_SPEED #Reused from barriers class

#For my AI Autopilot
AI_DIR= os.path.dirname(os.path.realpath(__file__), "The AI Files")
if AI_DIR not in sys.path:
    sys.path.insert(0,AI_DIR)
from autopilot_torch import TorchPolicy

#Sound effects
pygame.mixer.pre_init(44100, -16, 2, 512)

#Fast settings to get to testable tonight
#Any ALL_CAPS var can be moved to settings .py and passed in
FULLSCREEN_FLAGS= pygame.FULLSCREEN|pygame.DOUBLEBUF
WINDOWED_FLAGS= pygame.RESIZABLE|pygame.DOUBLEBUF
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

HINT_SHOW_MS = 500
HINT_FADE_MS = 4000
HINT_ALPHA_START=200
HINT_BOX_BASE_ALPHA= 255

#Sign method to protect my highscore
def _sign(score: int, salt: str) -> str:
    device = platform.node() if BIND_TO_DEVICE else ""
    msg = f"{score}|{salt}|{device}".encode("utf-8")
    return hmac.new(SECRET_KEY.encode("utf-8"), msg, hashlib.sha256).hexdigest()

audioDir = os.path.dirname(__file__)
ASSETS_AUDIO_DIR = os.path.join(audioDir, "assets", "sounds")

def sound_path(*parts: str) -> str:
    return os.path.join(ASSETS_AUDIO_DIR, *parts)

#Load my all time high score
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

#Save my all time high score
def save_high_score(score: int) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    salt = os.urandom(16).hex()
    sig= _sign(int(score), salt)
    payload = {"score": int(score), "salt": salt, "sig": sig}
    with open(HIGH_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)

#My game logic and implementation class
class Game:
    def __init__(self) -> None:
        self.music_paused = None
        self.fullscreen_exit_fade_until = None
        self.neat_is_training = None
        self.is_fullscreen = True
        self.fullscreen_exit_msg_until = None
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

        #Play music
        self._music_paused = False
        self.start_music(volume=.15)

        #Launch Screen
        self.custom_window_size(fullscreen=True)

        #Verify my videos that I added to make my game look more professional are available
        try:
            assets_dir= os.path.join(os.path.dirname(__file__), "assets", "launch_video")
            mp4_launch= os.path.join(assets_dir, "launch_video.mp4")
            mp4_welcome= os.path.join(assets_dir, "welcome_video.mp4")

            #Play the sounds in order
            mp4s= [p for p in (mp4_launch,mp4_welcome) if os.path.isfile(p)]
            if mp4s:
                play_video_sequence_cover(self.screen, mp4s, cap_fps=60, between_fade_ms=160,final_fade_ms=220,sfx_volumes=[.05]*len(mp4s))
        except Exception as e:
            print("Intro skipped", e)

        #Graph for training model
        graph_h = 333
        self.graph_h =graph_h

        #Fixed virtual resolution
        self.virtual_size = (SCREEN_WIDTH, SCREEN_HEIGHT+ graph_h)

        self.canvas= pygame.Surface(self.virtual_size).convert()

        #New graph, going to use a vertical error buffer
        self.vert_err = deque(maxlen=1200)

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
        self.policy_path = os.path.join(AI_DIR, "flappy_policy.pt")
        self.autopilot=None
        self.customize = CustomizeMenu(self)

        #For AI
        self.ai_restart_timer= None
        self.ai_enabled = False

        #Score and flags
        self.score = 0
        self.high_score = 0
        self.all_time_high_score = load_high_score()
        self.rounds_played = 0
        self.game_over = False
        self._spawn_ms = 0
        self.running = True

        self.snd_jump = pygame.mixer.Sound(sound_path("jump.wav"))
        self.snd_crash = pygame.mixer.Sound(sound_path("crash.wav"))
        self.snd_jump.set_volume(.04)
        self.snd_crash.set_volume(0.03)
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

    #Start ai version of game using a PyTorch policy i will train
    def start_ai_game(self):
        self.ai_enabled = True

        #Reset old graph
        self.vert_err.clear()

        #Build Torch policy
        try:
            self.autopilot = TorchPolicy(ckpt_path=self.policy_path)
        except Exception as e:
            print("AI Failed to load Torch policy", e)
            self.autopilot = TorchPolicy(ckpt_path=None)

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

       #Clear my graph on new run
       self.vert_err.clear()



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

    #Closest pipe whose edge hasnt passed ship
    def next_pipe(self):
        future = [p for p in self.pipes if (p.x + p.width) >= BIRD_START_X]
        return min(future, key=lambda p: p.x) if future else None

    #Game loop
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

    #Function for most of my key listeners
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                #Will work in both of my game modes
                if event.key == pygame.K_ESCAPE:
                    if self.state == "menu":
                        # Toggle windowed if on Main Menu
                        self.set_fullscreen(not self.is_fullscreen)
                    else:
                        # If in game go back to main menu and maintain window state
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

    #Function to set the full screen mode properly
    def set_fullscreen(self, on: bool) -> None:
        self.is_fullscreen = on

        if on:
            pygame.display.quit()
            pygame.display.init()
            self.screen = pygame.display.set_mode((0, 0), pygame.NOFRAME| pygame.DOUBLEBUF)

        else:
            self.screen = pygame.display.set_mode((1280,720), pygame.RESIZABLE |pygame.DOUBLEBUF)

        #Timer for press esc hint
        now= pygame.time.get_ticks()
        self.fullscreen_exit_msg_until = now+ HINT_SHOW_MS+2000
        self.fullscreen_exit_fade_until= now + HINT_FADE_MS

    #Function to scale my game when the window size is adjusted
    def custom_window_size(self, fullscreen: bool= True, margin:int =10)-> None:
        if fullscreen:
            #Will use pygame to adjust scale later
            self.is_fullscreen = True
            self.set_fullscreen(fullscreen)
            return
        else:
            #Leaving my windowed path
            self.is_fullscreen = False
            self.set_fullscreen(self.is_fullscreen)
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

            #Position window
            os.environ["SDL_VIDEO_WINDOW_POS"] = "{margin}, {margin}"

            self.screen= pygame.display.set_mode((out_w, out_h), pygame.RESIZABLE)
            #self.canvas= pygame.Surface(self.virtual_size).convert()

    #Update the game as it is running
    def _update(self, dt_ms: int) -> None:

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

        #Vertical error for my graph
        p=self.next_pipe()
        if p is not None:
            top_rect, bot_rect = p.rects()
            gap_cy=.5* (top_rect.bottom + bot_rect.top)
            ship_cy = self.player.get_rect().centery
            self.vert_err.append(float(gap_cy-ship_cy)) #0 means perfectly centered


        #Scroll background image
        self.bg_x -= scroll_speed
        if self.bg_x <= -self.bg_image.get_width():
            self.bg_x = 0

    #Draw the game screen
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
            mode= "Torch"
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
            # Alltime high
            ath_surface = self.font.render(f"All-Time High Score: {self.all_time_high_score}", True, text)
            s.blit(ath_surface, (cx - ath_surface.get_width() // 2, cy - ath_surface.get_height() // 2 + msg_surface.get_height() + 12))

            #AI auto restart
            if self.ai_enabled and self.ai_restart_timer:
                ms_left = max(0, self.ai_restart_timer - pygame.time.get_ticks()-300)
                secs = (ms_left//1000)+1

                if pygame.time.get_ticks()+1200 >= self.ai_restart_timer:
                    self.reset()
                    return



                line1 = self.countdown_label_font.render("AI RESTARTING IN...", True, (255,0,0))
                line2 = self.countdown_font.render(f"{secs}", True, (255,0,0))

                y1 =cy - 140
                y2 = y1+ line1.get_height()+10
                s.blit(line1, (cx-line1.get_width()//2,y1))
                s.blit(line2, (cx-line2.get_width()//2,y2))

        if self.ai_enabled:
            #My center line error graph
            panel_y=SCREEN_HEIGHT
            panel_h= self.graph_h
            panel_w = s.get_width()

            #panel background
            pygame.draw.rect(s, (10,10,20), (0,panel_y,panel_w, panel_h))

            #Labels
            label= self.hud_font.render("Vertical Error: 0 = Centered", True, (255,0,0))
            s.blit(label, (40,panel_y+4))

            #Plot
            x0 =60
            x1 = panel_w-30
            y0 = panel_y+panel_h-33
            y1 = panel_y+25
            width= x1-x0
            height = y0-y1

            data = list(self.vert_err)

            #Draw my grid
            grid_color=(0,0,255)
            gx=8
            gy=4
            for i in range(1,gx):
                xi= x0 + int(width*i/gx)
                pygame.draw.line(s, grid_color, (xi, y1), (xi,y0), 1)
            for i in range(1,gy):
                yi=y1+ int(height*i/gy)
                pygame.draw.line(s, grid_color, (x0, yi), (x1,yi), 1)

            if len(data) >= 2:
                lo = min(min(data),.0)
                hi = max(max(data),.0)
                rng= hi - lo if hi>lo else 1

                #Baseline=0
                zy= y0 - int(((0-lo)/rng)*height)
                pygame.draw.line(s, (80, 80, 100), (x0, zy), (x1,zy), 1)

                #line
                pts=[]
                n= len(data)
                for i, v in enumerate(data):
                    px= x0 + int(i/(n-1)*width)
                    py= y0 - int(((v-lo)/rng)*height)
                    pts.append((px,py))
                pygame.draw.lines(s,(120,180,255), False, pts, 2)

                #Quick stats
                cur = data[-1]
                k =min(600, len(data))
                mae= sum(abs(x) for x in data[-k:])/float(k)
                stat= self.hud_font.render(f"now: {cur:+.1f}px | MAE (recent): {mae:.1f}px", True, (200,200,233))
                s.blit(stat, (int(x1-SCREEN_WIDTH*.38), y1-22))

        #Scaling parameters for the game screen
        win_width, win_height = self.screen.get_size()
        content_h = SCREEN_HEIGHT + (self.graph_h if self.ai_enabled else 0)

        src_rect = pygame.Rect(0, 0, SCREEN_WIDTH, content_h)
        src_surface = self.canvas.subsurface(src_rect)

        #Different scaling when AI is enabled
        if self.ai_enabled:
            scale= min(win_width/SCREEN_WIDTH, win_height/content_h)
            dst_width, dst_height = int(SCREEN_WIDTH*scale*1.78), int(content_h*scale)
            offset_x= (win_width-dst_width)//2
            offset_y=0
        else:
            scale = max(win_width / SCREEN_WIDTH, win_height / content_h)
            dst_width, dst_height = int(SCREEN_WIDTH * scale), int(content_h * scale * .75)
            offset_x = (win_width - dst_width) // 2
            offset_y = (win_height - dst_height) // 2

        frame = pygame.transform.smoothscale(src_surface, (dst_width, dst_height))
        self.screen.fill((0,0,0))
        self.screen.blit(frame, (offset_x, offset_y))

        if self.is_fullscreen:
            self.draw_fullscreen_hint(self.screen)

        pygame.display.flip()

    #Self Explanatory
    def draw_fullscreen_hint(self, surface: pygame.Surface)->None:
        if not getattr(self,"is_fullscreen",False):
            return

        now= pygame.time.get_ticks()
        show_until = getattr(self,"fullscreen_exit_msg_until",0) or 0
        fade_until = getattr(self,"fullscreen_exit_fade_until",0) or 0

        if now>= fade_until:
            return

        if now< show_until:
            alpha=HINT_ALPHA_START
        else:
            p = (fade_until-now)/float(HINT_FADE_MS)

            p= 1-(1-p) *(1-p)

            alpha = max(0,min(255, int(HINT_ALPHA_START*p)))

        hint = self.hud_font.render('Press "ESC" to Exit Fullscreen', True, (230, 230, 230))
        hx = (self.screen.get_width() - hint.get_width()) // 2
        hy = 120
        pad_x, pad_y =12,8
        box_w = hint.get_width()+2*pad_x
        box_h = hint.get_height()+2*pad_y

        container = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        container.fill((0,0,0, HINT_BOX_BASE_ALPHA))
        container.blit(hint, (pad_x,pad_y))
        container.set_alpha(alpha)

        surface.blit(container, (hx-pad_x,hy-pad_y))

    #Self Explanatory
    def start_music(self, volume):
        # Music
        self._music_paused = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                pygame.mixer.set_num_channels(8)

            if not pygame.mixer.music.get_busy():
                opensource_music_file = sound_path("backgroundmusic.mp3")
                if os.path.isfile(opensource_music_file):
                    pygame.mixer.music.load(opensource_music_file)
                    pygame.mixer.music.set_volume(volume)
                    pygame.mixer.music.play(-1, fade_ms=1200)
            self.music_paused = False
        except Exception as e:
            print("Audio Disabled: ", e)
            self._music_paused = True
