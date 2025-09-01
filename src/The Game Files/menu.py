#Kody Graham
#08/29/2025
#This file will contain the main menu for my game

import pygame
import os


def asset_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", *parts)

class Button:
    def __init__(self, text, rect, font, on_click, bg=(40,40,48), fg=(255,255,255)):
        self.text = text
        self.rect = pygame.Rect(rect)
        self.font = font
        self.on_click = on_click
        self.bg, self.fg = bg, fg
        self.hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()

    def draw(self, surface):
        color = (80,80,90) if self.hovered else (self.bg)
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        label = self.font.render(self.text, True, self.fg)
        surface.blit(label, label.get_rect(center=self.rect.center))

class IconToggle:
    def __init__(self, center, image_on_path, image_off_path, get_state, on_toggle, size=48):

        self.get_state = get_state
        self.on_toggle = on_toggle

        on_img = pygame.image.load(image_on_path).convert_alpha()
        off_img = pygame.image.load(image_off_path).convert_alpha()
        self.img_on = pygame.transform.smoothscale(on_img, (size, size))
        self.img_off = pygame.transform.smoothscale(off_img, (size, size))
        self.rect = self.img_on.get_rect(center=center)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button ==1:
            if self.rect.collidepoint(event.pos):
                self.on_toggle()

    def draw(self, surface):
        img = self.img_on if self.get_state() else self.img_off
        surface.blit(img, self.rect)

class MainMenu:
    def __init__(self, game):
        self.game = game
        w, h = game.screen.get_size()
        cx = w // 2
        y= h // 2- 90
        f = game.font

        bg_file = "menu_bg.jpg"
        bg_path = asset_path("menu_bg.jpg")

        if os.path.exists(bg_path):
            raw_bg = pygame.image.load(bg_path).convert()
            sw, sh = raw_bg.get_size()
            scale = max(w / sw, h/sh)
            nw, nh = int(sw * scale), int(sh * scale)
            scaled = pygame.transform.smoothscale(raw_bg, (nw, nh))
            x = (nw-w) //2
            ycrop = (nh-h) // 2
            self.bg = scaled.subsurface(pygame.Rect(x,ycrop,w,h)).copy()
            self.overlay = pygame.Surface((w,h), pygame.SRCALPHA)
            self.overlay.fill((0,0,0,110))
        else:
            self.bg = None
            self.overlay = None

        #Buttons
        self.buttons = []
        def add(label, cb):
            r = pygame.Rect(0, 0, 360, 64); r.center = (cx,y)
            self.buttons.append(Button(label, r, f, cb))
        add("Start Game - User", game.start_user_game); y += 80
        add("Start - Auto Pilot(AI)", game.start_ai_game); y += 80
        add("Mission Control", game.open_customize);
        y+=80
        add("Exit", game.quit)

    def handle_event(self , event):
        for b in self.buttons:
            b.handle_event(event)

    def draw(self, surface):
        if self.bg is not None:
            surface.blit(self.bg, (0, 0))
            if self.overlay is not None:
                surface.blit(self.overlay, (0, 0))
        else:
            surface.fill((10, 12, 22))

        title = self.game.font.render("Astro Alley", True, (255,0,0))
        surface.blit(title, title.get_rect(midtop=(surface.get_width()//2, 60)))
        for b in self.buttons:
            b.draw(surface)

class CustomizeMenu:
    def __init__(self, game):
        self.game = game
        self.font = game.font

        self.ship_files = ["ship.png", "ship1.png", "ship2.png"]
        self.bg_files = ["strip_background.png", "strip_background1.png", "strip_background2.png"]
        self.ob_files = ["obstacle.png", "obstacle1.png", "obstacle2.png"]

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

        self.btn_back = Button("Back", pygame.Rect(30, h-70, 140, 50), self.font, game.open_menu)
        self.btn_start = Button("Launch!", pygame.Rect(w-210, h-70, 180, 50), self.font, self._save_and_start)

    def _index_from_name(self, items, name):
        for i, (nm,_) in enumerate(items):
            if nm == name:
                return i
        return 0

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

    def _load(self, path, size, alpha=True):
        img = pygame.image.load(path).convert_alpha() if alpha else pygame.image.load(path).convert()
        return pygame.transform.smoothscale(img, size)

    def _row_rects_centered(self, *, count: int, box_w: int, box_h: int, y: int, gap: int, screen_w: int):
        total_w = count * box_w + (count-1) * gap if count > 0 else 0
        x= (screen_w - total_w) // 2
        rects = []
        for _ in range(count):
            rects.append(pygame.Rect(x, y, box_w, box_h))
            x += box_w + gap
        return rects

    def _save_and_start(self):
        if self.ships: self.game.selected_ship = asset_path(self.ships[self.ship_idx][0])
        if self.bgs: self.game.selected_bg = asset_path(self.bgs[self.bg_idx][0])
        if self.obs: self.game.selected_ob = asset_path(self.obs[self.ob_idx][0])
        self.game.reset()
        self.game.state = "play"

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            for i, r in enumerate(self.ship_rects):
                if i < len(self.ships) and r.collidepoint(pos): self.ship_idx = i
            for i, b in enumerate(self.bg_rects):
                if i < len(self.bgs) and b.collidepoint(pos): self.bg_idx = i
            for i, o in enumerate(self.ob_rects):
                if i < len(self.obs) and o.collidepoint(pos): self.ob_idx = i

        self.music_toggle.handle_event(event)
        self.btn_back.handle_event(event)
        self.btn_start.handle_event(event)

    def _draw_row(self, surface, items, rects, sel_idk):
        for i, r in enumerate(rects):
            if i >= len(items): break
            _, img = items[i]
            surface.blit(img, img.get_rect(center=r.center))
            self._outline(surface, r, (0,0,255) if i == sel_idk else (100,100,120), 3, 10)

    def draw(self, surface:pygame.Surface) -> None:
        #Base background
        surface.fill(pygame.Color("black"))

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


    def _outline(self, surface, rect, color=(255,215,0), w=3, radius=8):
        pygame.draw.rect(surface, color, rect, width= w, border_radius=radius)
