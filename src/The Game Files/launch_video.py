#Kody Graham
#10/28/2025
#Class to ensure my videos will play perfectly without having to worry about crowding my game.py

#Note for self: Done

import os
import pygame
from typing import TYPE_CHECKING, Optional, Sequence

#Moviepy import for both versions
if TYPE_CHECKING:
    from moviepy.editor import VideoFileClip as _VideoFileClipType

try:
    #For moviepy v2
    from moviepy import VideoFileClip
except (ImportError, AttributeError):
    #Fall back for v1
    from moviepy.editor import VideoFileClip

#Self Explanatory
def fade_to_black(screen: pygame.Surface, last_frame: pygame.Surface | None, pos: tuple[int, int], ms: int = 220) -> None:
    if ms <= 0:
        return

    overlay = pygame.Surface(screen.get_size())
    overlay.fill((0, 0, 0))
    t0 = pygame.time.get_ticks()
    clock = pygame.time.Clock()

    while True:
        t = pygame.time.get_ticks() - t0
        if t >= ms:
            break

        if last_frame is not None:
            screen.blit(last_frame, pos)

        overlay.set_alpha(int(255 * (t / max(1, ms))))
        screen.blit(overlay, (0, 0))
        pygame.display.flip()
        clock.tick_busy_loop(120)

#SFX Helpers all self explanatory
def _find_launch_sfx(mp4_path: str, base_name: str = "launch_sound") -> Optional[str]:

    video_dir = os.path.dirname(mp4_path)
    assets_dir = os.path.dirname(video_dir)
    extentions = ("wav","ogg", "mp3","m4a")

    for ext in extentions:
        p = os.path.join(video_dir, f"{base_name}.{ext}")
        if os.path.isfile(p):
            return p

    for ext in extentions:
        p = os.path.join(assets_dir, "sounds", f"{base_name}.{ext}")
        if os.path.isfile(p):
            return p

    return None

#Play the requested sound effect
def _try_play_sfx(sfx_path: str, volume: float = 1.0) -> None:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(16)
        snd = pygame.mixer.Sound(sfx_path)
        snd.set_volume(max(0.0, min(1.0, volume)))
        ch = pygame.mixer.find_channel(True)
        (ch.play(snd) if ch else snd.play())
    except Exception as e:
        print(f"[LAUNCH] SFX error ({os.path.basename(sfx_path)}): {e}")

#Play the requested video exactly sized to user screen
def play_video_cover(
        screen: pygame.Surface,
        mp4_path: str,
        *,
        duration_limit: float | None = None,
        skip_keys: Sequence[int] = (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN),
        cap_fps: int = 60,
        fade_out_ms: int = 220,
        launch_sfx_name: str | None = None,
        launch_sfx_volume: float = 0.05,
) -> None:

    if not os.path.isfile(mp4_path):
        return

    clock = pygame.time.Clock()
    start_ms = pygame.time.get_ticks()
    clip = None

    last_surf: pygame.Surface | None = None
    last_pos: tuple[int, int] = (0, 0)

    try:
        clip = VideoFileClip(mp4_path, audio=False)

        #Start the SFX when the video starts
        if launch_sfx_name is None:
            launch_sfx_name = sfx_name_from_video(mp4_path)
        sfx_path = _find_launch_sfx(mp4_path, base_name=launch_sfx_name)
        if sfx_path:
            _try_play_sfx(sfx_path, volume=launch_sfx_volume)

        src_fps = int(round(clip.fps or 30))
        fps = max(15, min(cap_fps, src_fps))
        cutoff_ms = None
        if duration_limit is not None:
            grace_ms = max(50, int(1000 / fps))
            cutoff_ms = start_ms + int(duration_limit * 1000) + grace_ms

        frames = clip.iter_frames(fps=fps, dtype="uint8")

        for frame in frames:
            #Quick skip
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    try:
                        clip.close()
                    finally:
                        raise SystemExit
                if event.type == pygame.KEYDOWN and event.key in skip_keys:
                    fade_to_black(screen, last_surf, last_pos, fade_out_ms)
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    fade_to_black(screen, last_surf, last_pos, fade_out_ms)
                    return

            if cutoff_ms is not None and pygame.time.get_ticks() >= cutoff_ms:
                break

            #Fill any screen. Note for self: Double check on Tara's laptop later
            sw, sh = screen.get_size()
            out_w, out_h = sw, sh
            off_x, off_y = 0, 0

            #Per frame scale
            vh, vw = frame.shape[0], frame.shape[1]
            src = pygame.image.frombuffer(frame.tobytes(), (vw, vh), "RGB")
            surf = pygame.transform.smoothscale(src, (out_w, out_h))

            last_surf = surf
            last_pos = (off_x, off_y)
            screen.blit(surf, (off_x, off_y))
            pygame.display.flip()
            clock.tick(fps)

    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass

        fade_to_black(screen, last_surf, last_pos, fade_out_ms)

#Easy function to strip video from mypath name and add sound. Makes it easy for me to keep track of which sound goes with which video
def sfx_name_from_video(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]# For Example launch_video
    if stem.endswith("_video"):
        stem = stem[:-6]  #strip video
    return f"{stem}_sound" #add sound

#Play my first two videos back to back as an intro
def play_video_sequence_cover(
        screen: pygame.Surface,
        mp4_paths: Sequence[str],
        *,
        cap_fps: int = 60,
        between_fade_ms: int = 160,
        final_fade_ms: int = 220,
        duration_limits: Sequence[float | None] | None = None,
        sfx_volumes: Sequence[float] | None = None,
        sfx_names: Sequence[str | None] | None = None,
        skip_keys: Sequence[int] = (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN),
) -> None:

    if not mp4_paths:
        return

    n = len(mp4_paths)
    duration_limits = list(duration_limits) if duration_limits is not None else [None] * n
    sfx_volumes= list(sfx_volumes) if sfx_volumes is not None else [0.1] * n
    sfx_names =list(sfx_names) if sfx_names is not None else [None] * n

    for i, path in enumerate(mp4_paths):
        fade_ms = between_fade_ms if i < n - 1 else final_fade_ms

        #Options for each one
        limit = duration_limits[i]
        vol   = sfx_volumes[i]
        sfx   = sfx_names[i] or sfx_name_from_video(path)

        #Reuse same video player
        play_video_cover(screen, path, duration_limit=limit, skip_keys=skip_keys ,cap_fps=cap_fps, fade_out_ms=fade_ms, launch_sfx_name=sfx, launch_sfx_volume=vol,)

