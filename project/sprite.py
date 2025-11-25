import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple

from PyQt5 import QtCore, QtGui

SPRITE_WIDTH = 64
SPRITE_HEIGHT = 64


def _asset_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "assets", name)


def _load_pixmap(path: str) -> QtGui.QPixmap:
    p = QtGui.QPixmap(path)
    if p.isNull():
        raise FileNotFoundError(f"Missing sprite asset: {path}")
    return p


def _slice_row(sheet: QtGui.QPixmap, frame_w: int, frame_h: int, *, count: int) -> List[QtGui.QPixmap]:
    frames = []
    for i in range(count):
        frames.append(sheet.copy(i * frame_w, 0, frame_w, frame_h))
    return frames


def _slice_grid(
    sheet: QtGui.QPixmap,
    frame_w: int,
    frame_h: int,
    *,
    cols: int,
    rows: int,
    count: int,
) -> List[QtGui.QPixmap]:
    frames = []
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= count:
                return frames
            frames.append(sheet.copy(c * frame_w, r * frame_h, frame_w, frame_h))
            i += 1
    return frames


def _scaled(frames: List[QtGui.QPixmap], w: int, h: int) -> List[QtGui.QPixmap]:
    out = []
    for p in frames:
        out.append(p.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    return out


def _flip_h(p: QtGui.QPixmap) -> QtGui.QPixmap:
    return p.transformed(QtGui.QTransform().scale(-1, 1))


def _add_outline(p: QtGui.QPixmap, outline_px: int = 2) -> QtGui.QPixmap:
    """
    Add a simple white outline around a pixmap by stamping a mask around it.
    Minimal + readable, not fancy.
    """
    if outline_px <= 0:
        return p

    img = p.toImage().convertToFormat(QtGui.QImage.Format_ARGB32_Premultiplied)
    w, h = img.width(), img.height()

    base = QtGui.QPixmap(w, h)
    base.fill(QtCore.Qt.transparent)

    # build a mask from alpha
    mask = img.createAlphaMask()
    white = QtGui.QPixmap.fromImage(mask)
    white_img = white.toImage().convertToFormat(QtGui.QImage.Format_ARGB32_Premultiplied)

    # tint to white
    for y in range(white_img.height()):
        for x in range(white_img.width()):
            a = QtGui.qAlpha(white_img.pixel(x, y))
            if a:
                white_img.setPixel(x, y, QtGui.qRgba(255, 255, 255, 255))
    white = QtGui.QPixmap.fromImage(white_img)

    offsets: List[Tuple[int, int]] = []
    r2 = outline_px * outline_px
    for dy in range(-outline_px, outline_px + 1):
        for dx in range(-outline_px, outline_px + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= r2:
                offsets.append((dx, dy))

    painter = QtGui.QPainter(base)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

    for (dx, dy) in offsets:
        painter.drawPixmap(dx, dy, white)

    painter.drawPixmap(0, 0, p)
    painter.end()

    return base


class AnimState(Enum):
    IDLE = auto()
    RUN = auto()
    JUMP_UP = auto()
    JUMP = auto()
    JUMP_DOWN = auto()
    PUNCH = auto()


class StickmanSprite:
    """
    Stickman sprite renderer with a small finite state machine (FSM).

    Sheets expected in ./assets:
      - Thin.png     (idle row, 6 frames)
      - Run.png      (run row, 9 frames)
      - Jump.png     (grid, 2x2, 4 frames)
      - JumpUp.png   (row, 1 frame)
      - JumpDown.png (row, 1 frame)
      - Punch.png    (grid, 4x3, 12 frames)  <-- added
    """

    def __init__(self, *, width: int = SPRITE_WIDTH, height: int = SPRITE_HEIGHT, outline_px: int = 2):
        self.w = int(width)
        self.h = int(height)
        self.outline_px = int(outline_px)

        # simple FPS per state
        self._fps: Dict[AnimState, float] = {
            AnimState.IDLE: 10.0,
            AnimState.RUN: 16.0,
            AnimState.JUMP: 10.0,
            AnimState.JUMP_UP: 1.0,
            AnimState.JUMP_DOWN: 1.0,
            AnimState.PUNCH: 18.0,
        }

        self._state = AnimState.IDLE
        self._frame_idx = 0
        self._accum_ms = 0

        self._clock = QtCore.QElapsedTimer()
        self._clock.start()
        self._last_ms = int(self._clock.elapsed())

        self._frames_right: Dict[AnimState, List[QtGui.QPixmap]] = {}
        self._frames_left: Dict[AnimState, List[QtGui.QPixmap]] = {}

        self._load_frames()

    def _load_frames(self) -> None:
        def load(name: str) -> QtGui.QPixmap:
            return _load_pixmap(_asset_path(name))

        idle = _slice_row(load("Thin.png"), 64, 64, count=6)
        run = _slice_row(load("Run.png"), 64, 64, count=9)
        jump = _slice_grid(load("Jump.png"), 64, 64, cols=2, rows=2, count=4)
        jump_up = _slice_row(load("JumpUp.png"), 64, 64, count=1)
        jump_down = _slice_row(load("JumpDown.png"), 64, 64, count=1)
        punch = _slice_grid(load("Punch.png"), 64, 64, cols=4, rows=3, count=12)

        def prep(frames: List[QtGui.QPixmap]) -> List[QtGui.QPixmap]:
            scaled = _scaled(frames, self.w, self.h)
            return [_add_outline(p, self.outline_px) for p in scaled]

        self._frames_right = {
            AnimState.IDLE: prep(idle),
            AnimState.RUN: prep(run),
            AnimState.JUMP: prep(jump),
            AnimState.JUMP_UP: prep(jump_up),
            AnimState.JUMP_DOWN: prep(jump_down),
            AnimState.PUNCH: prep(punch),
        }
        self._frames_left = {k: [_flip_h(p) for p in v] for k, v in self._frames_right.items()}

    def _choose_state(self, *, moving: bool, on_ground: bool, vy: float, punching: bool) -> AnimState:
        if punching:
            return AnimState.PUNCH
        if on_ground:
            return AnimState.RUN if moving else AnimState.IDLE
        if abs(vy) < 2.0:
            return AnimState.JUMP
        if vy < -2.0:
            return AnimState.JUMP_UP
        if vy > 2.0:
            return AnimState.JUMP_DOWN
        return AnimState.JUMP

    def _set_state(self, new_state: AnimState) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        self._frame_idx = 0
        self._accum_ms = 0

    def _tick_animation(self) -> None:
        now_ms = int(self._clock.elapsed())
        dt_ms = max(0, min(100, now_ms - self._last_ms))
        self._last_ms = now_ms

        frames = self._frames_right.get(self._state, [])
        if len(frames) <= 1:
            self._frame_idx = 0
            return

        fps = float(self._fps.get(self._state, 10.0))
        frame_ms = max(1, int(1000.0 / fps))

        self._accum_ms += dt_ms
        while self._accum_ms >= frame_ms:
            self._accum_ms -= frame_ms
            if self._state == AnimState.PUNCH:
                # play once (stop on last frame)
                self._frame_idx = min(self._frame_idx + 1, len(frames) - 1)
            else:
                self._frame_idx = (self._frame_idx + 1) % len(frames)

    def draw(
        self,
        painter: QtGui.QPainter,
        x: int,
        y: int,
        *,
        facing: int = 1,
        moving: bool = False,
        on_ground: bool = True,
        vy: float = 0.0,
        punching: bool = False,
    ) -> None:
        state = self._choose_state(moving=moving, on_ground=on_ground, vy=vy, punching=punching)
        self._set_state(state)
        self._tick_animation()

        facing = 1 if facing >= 0 else -1
        frame_map = self._frames_left if facing < 0 else self._frames_right
        frames = frame_map.get(self._state, [])
        if not frames:
            return

        idx = self._frame_idx % len(frames)
        painter.drawPixmap(int(x), int(y), frames[idx])
