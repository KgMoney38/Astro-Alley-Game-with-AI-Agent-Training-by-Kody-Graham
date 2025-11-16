#Kody Graham
#08/29/2025
#This file will contain the main menu for my game

#Note for self: Done


import pygame, os
from launch_video import play_video_cover

#Design-space resolution so the layout scales cleanly on any screen
DESIGN_W, DESIGN_H = 1280, 720
AUTOSCROLL_MS  = 1000
HOVER_DWELL_MS = 1000
SHIFT_UP = 100

UNSEL_BORDER_COLOR = (40, 40, 40)

#Layout dictionary for my fixed 1280x720 design canvas
LAYOUT = {
    "title_y": 14,
    "ships_title_y": 58,
    "routes_title_y": 178,
    "obstacles_title_y": 387,

    "ships": {"y": 111, "box": (150,50), "gap": 30},
    "routes": {"y": 231, "box": (280,140), "gap": 40},
    "obstacles": {"y": 450, "box": (70,175), "gap": 50},

    "arrow_pad": 26,

    "music_label": (60, DESIGN_H - 203),
    "debug_label": (DESIGN_W-180, DESIGN_H-203),

    "back_btn": (30, DESIGN_H-70, 160,52),
    "launch_btn": (DESIGN_W-210, DESIGN_H-70, 180,52) ,

    "music_toggle": (105, DESIGN_H-130),
    "debug_toggle": (DESIGN_W-120, DESIGN_H-130),

    "ai_margin_bottom": 15,

    "border_ships": {"pad": 6, "offset": (0,0), "unselected_thickness": 4, "unselected_radius": 12},
    "border_routes": {"pad": 8, "offset": (0,0), "unselected_thickness": 4, "unselected_radius": 14},
    "border_obstacles": {"pad": 18, "offset": (0,0), "unselected_thickness": 4, "unselected_radius": 16},
}

#Helper to build full paths into my assets folder
def asset_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", *parts)

#Scale background to fit and center it
def cover_blit(surface: pygame.Surface, image: pygame.Surface) -> None:
    sw, sh = surface.get_size()
    iw, ih = image.get_size()
    scale = max(sw/iw, sh/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    surface.blit(pygame.transform.smoothscale(image, (nw, nh)),
                 ((sw-nw)//2, (sh-nh)//2))

#Letterbox my 1280x720 canvas into whatever screen size I'm running
def letterbox_rect(screen: pygame.Surface) -> pygame.Rect:
    sw, sh = screen.get_size()
    s = min(sw / DESIGN_W, sh / DESIGN_H)
    cw, ch = int(DESIGN_W * s), int(DESIGN_H * s)
    return pygame.Rect((sw - cw)//2, (sh - ch)//2, cw, ch)

# (path_key, size) -> Surface
#Simple cache for scaled images so I do not rescale every frame
_scaled_cache = {}

def get_scaled_image(surf: pygame.Surface, size: tuple[int,int], key: str | None = None) -> pygame.Surface:
    if key:
        k = (key, size)
        cached = _scaled_cache.get(k)
        if cached: return cached
    scaled = pygame.transform.smoothscale(surf, size)
    if key: _scaled_cache[k] = scaled
    return scaled

#Class for all the buttons on the main and customization menus
class Button:
    def __init__(self, label, rect_design, font, on_click,
                 bg=(40,40,48), fg=(255,255,255)):
        #Store everything in design-space coordinates so it all scales together
        self.label = label
        self.rect_d = pygame.Rect(rect_design)
        self.font = font
        self.on_click = on_click
        self.bg, self.fg = bg, fg
        self._hover = False

    #Just a listener for the hover over and click of my buttons
    def handle_event_d(self, event, pos_d):
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect_d.collidepoint(pos_d)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect_d.collidepoint(pos_d):
                self.on_click()

    #Draw my buttons
    def draw_d(self, canvas: pygame.Surface):
        color = (80,80,90) if self._hover else self.bg
        pygame.draw.rect(canvas, color, self.rect_d, border_radius=12)
        label = self.font.render(self.label, True, self.fg)
        canvas.blit(label, label.get_rect(center=self.rect_d.center))

#Class to toggle my music on off icon
class IconToggle:
    def __init__(self, center_d, path_on, path_off, get_state, on_toggle, size_px=52):
        self.cx, self.cy = center_d
        self.size = size_px
        self.get_state = get_state
        self.on_toggle = on_toggle
        self.img_on  = pygame.image.load(path_on).convert_alpha()
        self.img_off = pygame.image.load(path_off).convert_alpha()
        self.rect_d = pygame.Rect(0,0,size_px,size_px)
        self.rect_d.center = center_d

    #Listener for my music on off button
    def handle_event_d(self, event, pos_d):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect_d.collidepoint(pos_d):
                self.on_toggle()

    #Draw it
    def draw_d(self, canvas: pygame.Surface):
        img = self.img_on if self.get_state() else self.img_off
        scaled = pygame.transform.smoothscale(img, (self.size, self.size))
        r = scaled.get_rect(center=(self.cx, self.cy))
        self.rect_d = r
        canvas.blit(scaled, r)

#Class for the start screen of my game
class MainMenu:
    def __init__(self, game):
        self.game = game
        #Render everything to a fixed 1280x720 canvas first, then scale it to the real screen
        self.canvas = pygame.Surface((DESIGN_W, DESIGN_H)).convert_alpha()
        #Main menu background path
        bg_path = asset_path("menu_bg.jpg")
        self.raw_bg = pygame.image.load(bg_path).convert() if os.path.exists(bg_path) else None
        self.overlay = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        self.overlay.fill((0,0,0,110))

        f = game.font
        cx = DESIGN_W//2 - 180
        #Add my menu options to the main menu
        self.buttons = [
            Button("Start Game - User",      (cx, 300, 360, 64), f, game.start_user_game),
            Button("Start - Auto Pilot(AI)", (cx, 380, 360, 64), f, game.start_ai_game),
            Button("Mission Control",        (cx, 460, 360, 64), f, game.open_customize),
            Button("Exit",                   (cx, 540, 360, 64), f, game.quit),
        ]

    #Convert real mouse position into my design-space coordinates
    def _screen_to_design(self, screen_pos):
        rect = letterbox_rect(self.game.screen)
        if not rect.collidepoint(screen_pos): return None
        sx = (screen_pos[0] - rect.x) / rect.w
        sy = (screen_pos[1] - rect.y) / rect.h
        return (sx * DESIGN_W, sy * DESIGN_H)

    #Feed design-space mouse coords into the button listeners
    def handle_event(self, event):
        pos_d = None
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            pos_d = self._screen_to_design(pygame.mouse.get_pos())
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.game.set_fullscreen(not self.game.is_fullscreen); return
        if pos_d:
            for b in self.buttons:
                b.handle_event_d(event, pos_d)

    def draw(self, screen):
        c = self.canvas
        c.fill((0,0,0))
        if self.raw_bg: cover_blit(c, self.raw_bg); c.blit(self.overlay, (0,0))
        for b in self.buttons: b.draw_d(c)
        rect = letterbox_rect(screen)
        frame = pygame.transform.smoothscale(c, (rect.w, rect.h))
        screen.fill((0,0,0)); screen.blit(frame, rect.topleft)
        self.game.draw_fullscreen_hint(screen)

#Class for my customization menu
class CustomizeMenu:
    def __init__(self, game):
        self.game = game
        self.font = game.font
        #Use a separate 1280x 720 canvas for the customize screen
        self.canvas = pygame.Surface((DESIGN_W, DESIGN_H)).convert_alpha()

        #Background image with a dark overlay
        path = asset_path("customize_bg.png")
        self.customize_bg = pygame.image.load(path).convert() if os.path.exists(path) else None
        self.overlay = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        self.overlay.fill((0,0,0,120))

        #Selected border image that gets cropped down to just the visible frame
        self.border_png_raw = None
        self.border_png_cropped = None
        border_path = asset_path("border.png")
        if os.path.exists(border_path):
            base = pygame.image.load(border_path).convert_alpha()
            self.border_png_raw = base
            mask = pygame.mask.from_surface(base)
            rects = mask.get_bounding_rects()
            if rects:
                bbox = rects[0].copy()
                for r in rects[1:]:
                    bbox.union_ip(r)
                self.border_png_cropped = base.subsurface(bbox).copy()
            else:
                self.border_png_cropped = base

        #Unselected border image so side tiles have a different frame
        self.border_unsel_png_raw = None
        self.border_unsel_png_cropped = None
        border_unsel_path = asset_path("border_unselected.png")
        if os.path.exists(border_unsel_path):
            base_u = pygame.image.load(border_unsel_path).convert_alpha()
            self.border_unsel_png_raw = base_u
            mask_u = pygame.mask.from_surface(base_u)
            rects_u = mask_u.get_bounding_rects()
            if rects_u:
                bbox_u = rects_u[0].copy()
                for r in rects_u[1:]:
                    bbox_u.union_ip(r)
                self.border_unsel_png_cropped = base_u.subsurface(bbox_u).copy()
            else:
                self.border_unsel_png_cropped = base_u

        #Collect all of my ship, route, and obstacle image names
        self.ship_files = [f"ship{i if i else ''}.png" for i in range(0,10)]
        self.bg_files   = [f"strip_background{i if i else ''}.png" for i in range(0,10)]
        self.ob_files   = [f"obstacle{i if i else ''}.png" for i in range(0,10)]

        #Build thumbnails for each row so I am not drawing full resolution files every frame
        self.ships = [(n, self._load(asset_path(n), LAYOUT["ships"]["box"], True))
                      for n in self.ship_files if os.path.exists(asset_path(n))]
        self.bgs   = [(n, self._bg_preview(asset_path(n), 900, 800, LAYOUT["routes"]["box"]))
                      for n in self.bg_files   if os.path.exists(asset_path(n))]
        self.obs   = [(n, self._load(asset_path(n), LAYOUT["obstacles"]["box"], True))
                      for n in self.ob_files   if os.path.exists(asset_path(n))]

        #Figure out which ship/bg/obstacle is currently selected when I open the menu
        self.ship_idx = self._index_from(os.path.basename(game.selected_ship), self.ships)
        self.bg_idx   = self._index_from(os.path.basename(game.selected_bg),   self.bgs)
        self.ob_idx   = self._index_from(os.path.basename(game.selected_ob),   self.obs)

        #Create three centered rows of rects for ships, routes, and obstacles
        self.ship_rects = self._row_centered(3, *LAYOUT["ships"]["box"], LAYOUT["ships"]["y"], LAYOUT["ships"]["gap"])
        self.bg_rects   = self._row_centered(3, *LAYOUT["routes"]["box"], LAYOUT["routes"]["y"], LAYOUT["routes"]["gap"])
        self.ob_rects   = self._row_centered(3, *LAYOUT["obstacles"]["box"], LAYOUT["obstacles"]["y"], LAYOUT["obstacles"]["gap"])

        #Left and right arrow hitboxes for each row
        self.arrow_left  = self._try_img("arrow_left.png",  (72,36))
        self.arrow_right = self._try_img("arrow_right.png", (72,36))
        self.ship_left_r, self.ship_right_r = self._arrow_rects(self.ship_rects)
        self.bg_left_r,   self.bg_right_r   = self._arrow_rects(self.bg_rects)
        self.ob_left_r,   self.ob_right_r   = self._arrow_rects(self.ob_rects)

        #Music and debug toggles reuse the same IconToggle helper
        self.music_toggle = IconToggle(LAYOUT["music_toggle"], asset_path("music_on.png"), asset_path("music_off.png"),
                                       get_state=lambda: not game._music_paused, on_toggle=game.toggle_music, size_px=52)
        self.debug_toggle = IconToggle(LAYOUT["debug_toggle"], asset_path("debug_on.png"), asset_path("debug_off.png"),
                                       get_state=lambda: bool(self.game.dev_mode),
                                       on_toggle=lambda: setattr(self.game, "dev_mode", not bool(self.game.dev_mode)), size_px=52)

        #Back and Launch buttons for leaving this screen or starting the game
        self.btn_back  = Button("Back",    LAYOUT["back_btn"],   self.font, game.open_menu)
        self.btn_start = Button("Launch!", LAYOUT["launch_btn"], self.font, self._save_and_start)

        self.copilot_rect = pygame.Rect(0,0,0,0)

        #Music track arrows
        mx, my = LAYOUT["music_toggle"]
        self.music_arrow_left = self._try_img("arrow_left.png", (38,25))
        self.music_arrow_right = self._try_img("arrow_right.png", (38,25))

        if self.music_arrow_left:
            self.music_left_rect = self.music_arrow_left.get_rect()
            self.music_left_rect.center = (mx - 50, my)
        else:
            self.music_left_rect = pygame.Rect(0,0,0,0)

        if self.music_arrow_right:
            self.music_right_rect = self.music_arrow_right.get_rect()
            self.music_right_rect.center = (mx + 50, my)
        else:
            self.music_right_rect = pygame.Rect(0,0,0,0)

        #Music index indicator
        self.music_slot_rects = []
        slots = 5

        slot_size = 15
        slot_gap=10

        #x,y
        bar_y =my+33
        bar_width = slots * slot_size + (slots-1)*slot_gap

        bar_x= mx- bar_width //2

        for i in range(slots):
            r = pygame.Rect(bar_x+ i * (slot_size+slot_gap), bar_y, slot_size, slot_size)
            self.music_slot_rects.append(r)

        #State for my hover-based autoscroll logic
        now = pygame.time.get_ticks()
        self.hover_dir_ship = 0; self.next_tick_ship = now; self.hover_start_ship = None
        self.hover_dir_bg   = 0; self.next_tick_bg   = now; self.hover_start_bg   = None
        self.hover_dir_ob   = 0; self.next_tick_ob   = now; self.hover_start_ob   = None

    #Play my customize launch video, save the choices, then start the correct game mode
    def _save_and_start(self):
        try:
            mp4 = os.path.join(asset_path("launch_video"), "customize_video.mp4")
            if os.path.isfile(mp4):
                play_video_cover(self.game.screen, mp4, cap_fps=60, fade_out_ms=220)
        except Exception as e:
            print("Customize launch skipped:", e)

        if self.ships: self.game.selected_ship = asset_path(self.ships[self.ship_idx][0])
        if self.bgs:   self.game.selected_bg   = asset_path(self.bgs[self.bg_idx][0])
        if self.obs:   self.game.selected_ob   = asset_path(self.obs[self.ob_idx][0])

        if getattr(self.game, "ai_enabled", False):
            self.game.start_ai_game()
        else:
            self.game.start_user_game()
        self.game.state = "play"

    #Turn screen coordinates into design-space so the layout stays consistent in fullscreen or windowed
    def _screen_to_design(self, screen_pos):
        rect = letterbox_rect(self.game.screen)
        if not rect.collidepoint(screen_pos): return None
        sx = (screen_pos[0] - rect.x) / rect.w
        sy = (screen_pos[1] - rect.y) / rect.h
        return (sx * DESIGN_W, sy * DESIGN_H)

    def _try_img(self, name, size):
        p = asset_path(name)
        return self._load(p, size, True) if os.path.exists(p) else None

    def _index_from(self, basename, items):
        for i,(nm,_) in enumerate(items):
            if nm == basename: return i
        return 0

    #Build a row of boxes centered horizontally using the layout values
    def _row_centered(self, count, w, h, y, gap):
        total = count*w + (count-1)*gap
        x = (DESIGN_W - total)//2
        return [pygame.Rect(x + i*(w+gap), y, w, h) for i in range(count)]

    #Position the left/right arrows just outside the first and last box
    def _arrow_rects(self, rects):
        pad = LAYOUT["arrow_pad"]
        cy = rects[0].centery
        left  = pygame.Rect(rects[0].left  - pad - 72, cy - 18, 72, 36)
        right = pygame.Rect(rects[-1].right + pad,     cy - 18, 72, 36)
        return left, right

    #Took some work but this crops the bg image to whatever screen size the system is running on
    def _bg_preview(self, path, crop_w, crop_h, thumb):
        img = pygame.image.load(path).convert()
        if img.get_height() != crop_h:
            s = crop_h / img.get_height()
            img = pygame.transform.smoothscale(img, (max(1,int(img.get_width()*s)), crop_h))
        if img.get_width() < crop_w:
            tiled = pygame.Surface((crop_w, img.get_height())).convert()
            x = 0
            while x < crop_w:
                tiled.blit(img, (x,0))
                x += img.get_width()
            img = tiled
        x0 = max(0, (img.get_width() - crop_w)//2)
        y0 = max(0, (img.get_height()- crop_h)//2)
        crop = img.subsurface(pygame.Rect(x0,y0,crop_w,crop_h)).copy()
        return pygame.transform.smoothscale(crop, thumb)

    def _load(self, path, size, alpha=True):
        img = pygame.image.load(path).convert_alpha() if alpha else pygame.image.load(path).convert()
        return pygame.transform.smoothscale(img, size)

    #Handle hover autoscroll, AI toggle, arrows, and the basic buttons/toggles
    def handle_event(self, event):
        pos_d = None
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            pos_d = self._screen_to_design(pygame.mouse.get_pos())

        if event.type == pygame.MOUSEMOTION and pos_d:
            now = pygame.time.get_ticks()

            #Inner helper that figures out which side tile is being hovered and when to start scrolling
            def upd(rects, hover_dir, hover_start):
                new_dir = -1 if rects[0].collidepoint(pos_d) else (1 if rects[2].collidepoint(pos_d) else 0)
                if hover_dir == 0 and new_dir != 0:
                    hover_start = now
                elif new_dir == 0:
                    hover_start = None
                return new_dir, hover_start

            self.hover_dir_ship, self.hover_start_ship = upd(self.ship_rects, self.hover_dir_ship, self.hover_start_ship)
            self.hover_dir_bg,   self.hover_start_bg   = upd(self.bg_rects,   self.hover_dir_bg,   self.hover_start_bg)
            self.hover_dir_ob,   self.hover_start_ob   = upd(self.ob_rects,   self.hover_dir_ob,   self.hover_start_ob)

        if pos_d and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.copilot_rect.collidepoint(pos_d):
                self.game.ai_enabled = not bool(getattr(self.game, "ai_enabled", False))
                return

            if self.music_left_rect.collidepoint(pos_d):
                if hasattr(self.game, "cycle_music"):
                    self.game.cycle_music(-1)
                return

            if self.music_right_rect.collidepoint(pos_d):
                if hasattr(self.game, "cycle_music"):
                    self.game.cycle_music(1)
                return

        if pos_d and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if   self.ship_left_r.collidepoint(pos_d): self.ship_idx = (self.ship_idx - 1) % max(1, len(self.ships))
            elif self.ship_right_r.collidepoint(pos_d): self.ship_idx = (self.ship_idx + 1) % max(1, len(self.ships))
            elif self.bg_left_r.collidepoint(pos_d):   self.bg_idx   = (self.bg_idx   - 1) % max(1, len(self.bgs))
            elif self.bg_right_r.collidepoint(pos_d):  self.bg_idx   = (self.bg_idx   + 1) % max(1, len(self.bgs))
            elif self.ob_left_r.collidepoint(pos_d):   self.ob_idx   = (self.ob_idx   - 1) % max(1, len(self.obs))
            elif self.ob_right_r.collidepoint(pos_d):  self.ob_idx   = (self.ob_idx   + 1) % max(1, len(self.obs))

        if pos_d:
            self.music_toggle.handle_event_d(event, pos_d)
            self.debug_toggle.handle_event_d(event, pos_d)
            self.btn_back.handle_event_d(event, pos_d)
            self.btn_start.handle_event_d(event, pos_d)

    #Draw the customize layout, then letterbox the canvas to the real screen
    def draw(self, screen: pygame.Surface) -> None:
        self._apply_autoscroll()

        c = self.canvas
        c.fill((0,0,0))
        if self.customize_bg: cover_blit(c, self.customize_bg); c.blit(self.overlay,(0,0))

        #Helper to draw section titles with an optional underline
        def title(txt, y, color=(255,255,255), underline=True):
            label = self.font.render(txt, True, color)
            c.blit(label, (DESIGN_W//2 - label.get_width()//2, y))
            if underline:
                y2 = y + label.get_height() + 4
                pygame.draw.line(c, color,
                                 (DESIGN_W//2 - label.get_width()//2, y2),
                                 (DESIGN_W//2 + label.get_width()//2, y2), 3)

        title("Welcome to Mission Control: Prepare for Launch!", LAYOUT["title_y"], (220,60,60), underline=False)
        title("Ships",     LAYOUT["ships_title_y"])
        title("Routes",    LAYOUT["routes_title_y"])
        title("Obstacles", LAYOUT["obstacles_title_y"])

        self._draw_row_with_png_border(c, items=self.ships, rects=self.ship_rects,  sel_idx=self.ship_idx,  cfg=LAYOUT["border_ships"])
        self._draw_row_with_png_border(c, items=self.bgs,   rects=self.bg_rects,    sel_idx=self.bg_idx,    cfg=LAYOUT["border_routes"])
        self._draw_row_with_png_border(c, items=self.obs,   rects=self.ob_rects,    sel_idx=self.ob_idx,    cfg=LAYOUT["border_obstacles"])

        if self.arrow_left and self.arrow_right:
            c.blit(self.arrow_left,  self.ship_left_r)
            c.blit(self.arrow_right, self.ship_right_r)
            c.blit(self.arrow_left,  self.bg_left_r)
            c.blit(self.arrow_right, self.bg_right_r)
            c.blit(self.arrow_left,  self.ob_left_r)
            c.blit(self.arrow_right, self.ob_right_r)

        music_lbl = self.font.render("Music", True, (255,255,255))
        c.blit(music_lbl, LAYOUT["music_label"])
        pygame.draw.line(c, (255,255,255),
                         (LAYOUT["music_label"][0], LAYOUT["music_label"][1] + music_lbl.get_height() + 4),
                         (LAYOUT["music_label"][0] + music_lbl.get_width(), LAYOUT["music_label"][1] + music_lbl.get_height() + 4), 3)

        dbg_lbl = self.font.render("Debug", True, (255,255,255))
        c.blit(dbg_lbl, LAYOUT["debug_label"])
        pygame.draw.line(c, (255,255,255),
                         (LAYOUT["debug_label"][0], LAYOUT["debug_label"][1] + dbg_lbl.get_height() + 4),
                         (LAYOUT["debug_label"][0] + dbg_lbl.get_width(), LAYOUT["debug_label"][1] + dbg_lbl.get_height() + 4), 3)

        self.music_toggle.draw_d(c)

        if self.music_arrow_left:
            c.blit(self.music_arrow_left, self.music_left_rect)
        if self.music_arrow_right:
            c.blit(self.music_arrow_right, self.music_right_rect)

        self.draw_music_index_bar(c)

        self.debug_toggle.draw_d(c)
        self.btn_back.draw_d(c)
        self.btn_start.draw_d(c)

        self._draw_copilot_badge(c)

        rect = letterbox_rect(screen)
        frame = pygame.transform.smoothscale(c, (rect.w, rect.h))
        screen.fill((0,0,0)); screen.blit(frame, rect.topleft)

    #Draw the three rows with either the selected PNG border or the unselected one
    def _draw_row_with_png_border(self, canvas, *, items, rects, sel_idx, cfg):
        if not items: return
        n = len(items)

        pad = int(cfg.get("pad", 6))
        dx, dy = cfg.get("offset", (0, 0))
        un_th  = int(cfg.get("unselected_thickness", 3))
        un_rad = int(cfg.get("unselected_radius", 12))

        for rect, off in zip(rects, (-1, 0, 1)):
            idx = (sel_idx + off) % n
            _, img = items[idx]
            img_rect = img.get_rect(center=rect.center)
            canvas.blit(img, img_rect)

            frame_rect = rect.inflate(pad * 2, pad * 2).move(dx if off == 0 else 0, dy if off == 0 else 0)

            if off == 0:
                # Selected use border.png
                if self.border_png_cropped:
                    scaled = get_scaled_image(self.border_png_cropped,
                                              (frame_rect.width, frame_rect.height),
                                              key="border_cropped")
                    canvas.blit(scaled, frame_rect.topleft)
                elif self.border_png_raw:
                    scaled = get_scaled_image(self.border_png_raw,
                                              (frame_rect.width, frame_rect.height),
                                              key="border_raw")
                    canvas.blit(scaled, frame_rect.topleft)
                else:
                    pygame.draw.rect(canvas, (255,0,0), frame_rect, width=max(4, un_th+1), border_radius=max(12, un_rad))
            else:
                # UNselected → use border_unselected.png if present, otherwise fallback line
                if self.border_unsel_png_cropped:
                    scaled_u = get_scaled_image(self.border_unsel_png_cropped,
                                                (frame_rect.width, frame_rect.height),
                                                key="border_unsel_cropped")
                    canvas.blit(scaled_u, frame_rect.topleft)
                elif self.border_unsel_png_raw:
                    scaled_u = get_scaled_image(self.border_unsel_png_raw,
                                                (frame_rect.width, frame_rect.height),
                                                key="border_unsel_raw")
                    canvas.blit(scaled_u, frame_rect.topleft)
                else:
                    pygame.draw.rect(canvas, UNSEL_BORDER_COLOR, frame_rect, width=un_th, border_radius=un_rad)

    #Clickable badge that shows if the AI copilot is enabled or disabled
    def _draw_copilot_badge(self, canvas):
        enabled = bool(getattr(self.game, "ai_enabled", False))
        base = self.font.render("AI Copilot: ", True, (255,255,255))
        status = self.font.render("Enabled" if enabled else "Disabled",
                                  True, (0,0,255) if enabled else (255,0,0))
        pad_x, pad_y = 14, 8
        box_w = base.get_width() + status.get_width() + pad_x*2
        box_h = max(base.get_height(), status.get_height()) + pad_y*2
        x = DESIGN_W//2 - box_w//2
        y = DESIGN_H - LAYOUT["ai_margin_bottom"] - box_h
        self.copilot_rect = pygame.Rect(x, y, box_w, box_h)
        panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (0,0,0,150), panel.get_rect(), border_radius=12)
        panel.blit(base, (pad_x, pad_y))
        panel.blit(status, (pad_x + base.get_width(), pad_y))
        canvas.blit(panel, (x, y))

    def draw_music_index_bar(self, canvas: pygame.Surface) -> None:
        if not getattr(self, "music_slot_rects", None):
            return

        tracks = getattr(self.game, "music_tracks", [])
        if not tracks:
            return
        active = getattr(self.game, "current_music_index", 2)
        if len(tracks) > 0:
            active %= len(tracks)
        else:
            active = 0

        for i, rect in enumerate(self.music_slot_rects):
            if i >= len(tracks):
                border_color = (90, 10, 10)
                pygame.draw.rect(canvas, border_color, rect, width=2)
                continue

            is_active = (i== active)

            if not is_active:
                border_color = (180, 40, 40)
                pygame.draw.rect(canvas, border_color, rect, width=2)

            else:
                #Glow
                glow_color = (180, 20, 20)
                inner_color = (255, 0, 0)

                glow_rect= rect.inflate(14, 14)
                glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(glow_surface, (*glow_color,80), glow_surface.get_rect())
                canvas.blit(glow_surface, glow_rect.topleft)

                pygame.draw.rect(canvas, glow_color, rect.inflate(6,6), width=3)
                pygame.draw.rect(canvas, inner_color, rect, width=2)

    #Hover on the side tiles for a bit, then step through that row automatically
    def _apply_autoscroll(self):
        now = pygame.time.get_ticks()

        def maybe_advance(dir_val, start_ms, next_ms, idx, length):
            if dir_val == 0 or length == 0:
                return next_ms, idx
            if start_ms is None or (now - start_ms) < HOVER_DWELL_MS:
                return now + AUTOSCROLL_MS, idx
            if now >= next_ms:
                idx = (idx + dir_val) % length
                next_ms = now + AUTOSCROLL_MS
            return next_ms, idx

        if self.ships:
            self.next_tick_ship, self.ship_idx = maybe_advance(self.hover_dir_ship, self.hover_start_ship,
                                                               self.next_tick_ship, self.ship_idx, len(self.ships))
        if self.bgs:
            self.next_tick_bg, self.bg_idx = maybe_advance(self.hover_dir_bg, self.hover_start_bg,
                                                           self.next_tick_bg, self.bg_idx, len(self.bgs))
        if self.obs:
            self.next_tick_ob, self.ob_idx = maybe_advance(self.hover_dir_ob, self.hover_start_ob,
                                                           self.next_tick_ob, self.ob_idx, len(self.obs))
