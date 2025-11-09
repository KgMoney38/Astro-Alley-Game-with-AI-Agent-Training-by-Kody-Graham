#Kody Graham
#08/29/2025
#This file will contain the main menu for my game

#Note for self: Done

import pygame
import os
from launch_video import play_video_cover #For my customize launch video


def asset_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", *parts)

#Scale background to fit and center it
def cover_blit(surface: pygame.Surface, image: pygame.Surface) -> None:
    sw,sh = surface.get_size()
    iw,ih = image.get_size()
    scale = max(sw/iw, sh/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    scaled = pygame.transform.smoothscale(image, (nw, nh))
    x=(sw - nw) // 2
    y=(sh - nh) // 2
    surface.blit(scaled, (x,y))

#Class for all the buttons on the main and customization menus
class Button:
    def __init__(self, text, rect, font, on_click, bg=(40,40,48), fg=(255,255,255)):
        self.text = text
        self.rect = pygame.Rect(rect)
        self.font = font
        self.on_click = on_click
        self.bg, self.fg = bg, fg
        self.hovered = False

    #Just a listener for the hover over and click of my buttons
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()
    
    #Draw my buttons
    def draw(self, surface):
        color = (80,80,90) if self.hovered else (self.bg)
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        label = self.font.render(self.text, True, self.fg)
        surface.blit(label, label.get_rect(center=self.rect.center))

#Class to toggle my music on off icon
class IconToggle:
    def __init__(self, center, image_on_path, image_off_path, get_state, on_toggle, size=48):

        self.get_state = get_state
        self.on_toggle = on_toggle

        on_img = pygame.image.load(image_on_path).convert_alpha()
        off_img = pygame.image.load(image_off_path).convert_alpha()
        self.img_on = pygame.transform.smoothscale(on_img, (size, size))
        self.img_off = pygame.transform.smoothscale(off_img, (size, size))
        self.rect = self.img_on.get_rect(center=center)
    
    #Listener for my music on off button
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button ==1:
            if self.rect.collidepoint(event.pos):
                self.on_toggle()
    
    #Draw it
    def draw(self, surface):
        img = self.img_on if self.get_state() else self.img_off
        surface.blit(img, self.rect)

#Class for the start screen of my game
class MainMenu:
    def __init__(self, game):
        self.game = game
        w, h = game.screen.get_size()
        cx = w // 2
        f = game.font
        
        #Main menu background path
        bg_path = asset_path("menu_bg.jpg")

        self.raw_bg= pygame.image.load(bg_path).convert() if os.path.exists(bg_path) else None
        self.overlay = None

        #Buttons
        self.buttons = []
        self.y_offset=[]

        #Add my menu options to the main menu
        def add(label, cb, y_offset):
            r = pygame.Rect(0, 0, 360, 64)
            self.buttons.append(Button(label, r, f, cb))
            self.y_offset.append(y_offset)

        add("Start Game - User", game.start_user_game, -90)
        add("Start - Auto Pilot(AI)", game.start_ai_game, -10)
        add("Mission Control", game.open_customize, 70)
        add("Exit", game.quit, 150)


    def handle_event(self , event):
        #Esc listener
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game.set_fullscreen(not self.game.is_fullscreen)
                return

        for b in self.buttons:
            b.handle_event(event)

    #Draw background
    def draw(self, surface):

        if self.raw_bg:
            cover_blit(surface, self.raw_bg)
            if(self.overlay is None) or (self.overlay.get_size() != surface.get_size()):
                self.overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                self.overlay.fill((0,0,0,110))
            surface.blit(self.overlay, (0, 0))

        else:
            surface.fill((0, 0, 0))

        center_x = surface.get_width()//2
        center_y = surface.get_height()//2
        
        for b, dir_y in zip(self.buttons, self.y_offset):
            b.rect.centerx=center_x
            b.rect.centery=center_y +dir_y
        
        #Old label for main menu
        #title = self.game.font.render("Astro Alley", True, (255,0,0))
        #surface.blit(title, title.get_rect(midtop=(surface.get_width()//2, 60)))
        
        for b in self.buttons:
            b.draw(surface)
        
        #Display the press esc hint
        self.game.draw_fullscreen_hint(surface)

#Class for my customization menu
class CustomizeMenu:
    def __init__(self, game):
        self.game = game
        self.font = game.font
        self.copilot_font= pygame.font.SysFont(None, 33)
        self.copilot_rect = pygame.Rect(0, 0, 0, 0)
        self.ship_files = ["ship.png", "ship1.png", "ship2.png"]
        self.bg_files = ["strip_background.png", "strip_background1.png", "strip_background2.png"]
        self.ob_files = ["obstacle.png", "obstacle1.png", "obstacle2.png"]

        bg_path = asset_path("customize_bg.png")
        self.customize_bg = pygame.image.load(bg_path).convert() if os.path.exists(bg_path) else None
        self.customize_overlay = None

        #Cropped ships
        self.ships = [(name, self._load(asset_path(name),(200,100), alpha=True)) for name in self.ship_files if os.path.exists(asset_path(name))]

        #Backgrounds cropped
        self.bgs = [(name, self._bg_preview(asset_path(name), crop_w=900, crop_h=800, thumb=(300,160))) for name in self.bg_files if os.path.exists(asset_path(name))]

        #Obstacles
        self.obs = [(name, self._load(asset_path(name), (95, 220), alpha=True)) for name in self.ob_files if os.path.exists(asset_path(name))]

        self.ship_idx = self._index_from_name(self.ships, os.path.basename(game.selected_ship))
        self.bg_idx = self._index_from_name(self.bgs, os.path.basename(game.selected_bg))
        self.ob_idx = self._index_from_name(self.obs, os.path.basename(game.selected_ob))

        w, h = game.screen.get_size()
        self.ship_rects = self._row_rects_centered(count = len(self.ships), box_w=210, box_h=110, y=130, gap=28, screen_w=w)
        self.bg_rects = self._row_rects_centered(count=len(self.bgs), box_w=300, box_h=170, y=300, gap=28, screen_w=w)
        self.ob_rects = self._row_rects_centered(count=len(self.obs), box_w=110, box_h=250, y=528, gap=55, screen_w=w)

        #Music toggle
        self.music_toggle = IconToggle(center=(98,670), image_on_path=asset_path("music_on.png"), image_off_path=asset_path("music_off.png"),get_state=lambda: not game._music_paused, on_toggle= game.toggle_music, size=58)

        #Basic Buttons
        self.btn_back = Button("Back", pygame.Rect(30, h-70, 140, 50), self.font, game.open_menu)
        self.btn_start = Button("Launch!", pygame.Rect(w-210, h-70, 180, 50), self.font, self._save_and_start)

        self.last_side=game.screen.get_size()

    #Label for whether AI is enabled or not
    def draw_copilot_label(self, surface: pygame.Surface) -> None:
        enabled= bool(getattr(self.game,"ai_enabled", False))
        base= self.font.render("AI Copilot: ", True, (255,255,255))
        status_txt = "Enabled" if enabled else "Disabled"
        status_color= (0,0,255) if enabled else (255,0,0)
        status = self.font.render(status_txt, True, status_color)

        pad_x, pad_y= 14,8
        box_w = base.get_width() +status.get_width() +pad_x *2
        box_h = max(base.get_height(), status.get_height())+ pad_y*2
        x = (surface.get_width()-box_w)//2
        y= surface.get_height()- box_h-12

        self.copilot_rect.update(x,y, box_w, box_h)

        show = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        pygame.draw.rect(show, (0,0,0,150), show.get_rect(), border_radius=12)
        show.blit(base,(pad_x,pad_y))
        show.blit(status,(pad_x+base.get_width(),pad_y))
        surface.blit(show,(x,y))


    def _index_from_name(self, items, name):
        for i, (nm,_) in enumerate(items):
            if nm == name:
                return i
        return 0

    #Took some work but this crops the bg image to whatever screen size the system is running on
    def _bg_preview(self, path: str, crop_w: int, crop_h: int, thumb: tuple[int, int]) -> pygame.Surface:

        img = pygame.image.load(path).convert()

        #Minimum height for crop
        if img.get_height() != crop_h:
            scale = crop_h / img.get_height()
            img = pygame.transform.smoothscale(img, (max(1, int (img.get_width()*scale)), crop_h))

        #If width than crop, tile horizontally first
        if img.get_width() < crop_w:
            tiled = pygame.Surface((crop_w, img.get_height())).convert()
            x=0
            while x < crop_w:
                tiled.blit(img, (x,0))
                x+= img.get_width()
            img = tiled

        x0 = max(0, (img.get_width() - crop_w) // 2)
        y0 = max(0, (img.get_height() - crop_h) // 2)
        crop = img.subsurface(pygame.Rect(x0, y0, crop_w, crop_h)).copy()

        return pygame.transform.smoothscale(crop, thumb)

    #Loads the images
    def _load(self, path, size, alpha=True):
        img = pygame.image.load(path).convert_alpha() if alpha else pygame.image.load(path).convert()
        return pygame.transform.smoothscale(img, size)

    #Determine sizes of rectangles and return it
    def _row_rects_centered(self, *, count: int, box_w: int, box_h: int, y: int, gap: int, screen_w: int):
        total_w = count * box_w + (count-1) * gap if count > 0 else 0
        x= (screen_w - total_w) // 2
        rects = []
        for _ in range(count):
            rects.append(pygame.Rect(x, y, box_w, box_h))
            x += box_w + gap
        return rects

    #Self explanatory
    def _save_and_start(self):

        #Play my customize video
        try:
            vid_dir= asset_path("launch_video")
            mp4 = os.path.join(vid_dir,"customize_video.mp4")
            if os.path.isfile(mp4):
                play_video_cover(self.game.screen, mp4,cap_fps=60,fade_out_ms=220)
        except Exception as e:
            print("Customize launch skipped",e)

        #Existing save and start
        if self.ships: self.game.selected_ship = asset_path(self.ships[self.ship_idx][0])
        if self.bgs: self.game.selected_bg = asset_path(self.bgs[self.bg_idx][0])
        if self.obs: self.game.selected_ob = asset_path(self.obs[self.ob_idx][0])

        if getattr(self.game, "ai_enabled", False):
            self.game.start_ai_game()
        else:
            self.game.start_user_game()

        self.game.state = "play"

    #Redraw my layout if window size is adjusted
    def relayout(self,surface:pygame.Surface) -> None:
        if surface.get_size() != self.last_side:
            w,h = surface.get_size()
            self.ship_rects = self._row_rects_centered(count= len(self.ships), box_w=210, box_h=110, y=130, gap=28, screen_w=w)
            self.bg_rects = self._row_rects_centered(count= len(self.bgs), box_w=300, box_h=170, y=300, gap=28, screen_w=w)
            self.ob_rects = self._row_rects_centered(count= len(self.obs), box_w=110, box_h=250, y=528, gap=55, screen_w=w)

            self.btn_back.rect.topleft=(30,h-70)
            self.btn_start.rect.topright=(w-30,h-70)
            self.last_side= (w,h)

    #Basic listener
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            for i, r in enumerate(self.ship_rects):
                if i < len(self.ships) and r.collidepoint(pos): self.ship_idx = i
            for i, b in enumerate(self.bg_rects):
                if i < len(self.bgs) and b.collidepoint(pos): self.bg_idx = i
            for i, o in enumerate(self.ob_rects):
                if i < len(self.obs) and o.collidepoint(pos): self.ob_idx = i

            if self.copilot_rect.collidepoint(pos):
                self.game.ai_enabled = not getattr(self.game, "ai_enabled", False)

        self.music_toggle.handle_event(event)
        self.btn_back.handle_event(event)
        self.btn_start.handle_event(event)

    #Draw the rows and border
    def _draw_row(self, surface, items, rects, sel_idx):
        for i, r in enumerate(rects):
            if i >= len(items):
                break
            _, img = items[i]
            surface.blit(img, img.get_rect(center=r.center))
            self._outline(surface, r, (0,0,255) if i == sel_idx else (50,50,50), 3, 10)

    #Draw rest of my customization menu
    def draw(self, surface:pygame.Surface) -> None:

        #Draw background
        if self.customize_bg:
            cover_blit(surface, self.customize_bg)
            if(self.customize_overlay is None) or (self.customize_overlay.get_size() != surface.get_size()):
                self.customize_overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                self.customize_overlay.fill((0,0,0,120))
            surface.blit(self.customize_overlay, (0,0))
        else:
            surface.fill(pygame.Color("black"))


        #Protect my layout after any resizing
        self.relayout(surface)

        def title(txt:str, y: int, color=(255,255,255), underline= True, center: bool = True):
            label = self.font.render(txt, True, color)
            if center:
                rect = label.get_rect(midtop=(surface.get_width()//2, y))
            else:
                rect = label.get_rect(topleft=(50, y))
            surface.blit(label, rect)

            if underline:
                pygame.draw.line(surface, color, (rect.left, rect.bottom + 4), (rect.right, rect.bottom + 4), 3)

        title("Welcome to Mission Control: Prepare for Launch!", 30, (220, 60, 60),underline=False,center=True)
        title("Ships", 85, (255,255,255), underline=True, center=True)
        title("Routes", 256, underline= True, center=True)
        title("Obstacles", 484, underline= True, center=True)
        title("Music", 590, underline= True, center=False)

        self._draw_row(surface, self.ships, self.ship_rects, self.ship_idx)
        self._draw_row(surface, self.bgs, self.bg_rects, self.bg_idx)
        self._draw_row(surface, self.obs, self.ob_rects, self.ob_idx)

        self.music_toggle.draw(surface)
        self.btn_back.draw(surface)
        self.btn_start.draw(surface)

        self.draw_copilot_label(surface)


    def _outline(self, surface, rect, color=(255,215,0), w=3, radius=8):
        pygame.draw.rect(surface, color, rect, width= w, border_radius=radius)
