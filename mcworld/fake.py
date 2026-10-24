"""Подставной мир для --simulate: без сервера, тот же набор методов."""

import math
import time

from flygame.constants import BODY_MAX_HEALTH
from flygame.state import Body, Sighting, Terrain

from .constants import PLAYER_SIZE


class FakeWorld:
    """Самопроверка без сервера: игрок ходит вокруг мухи и подходит вплотную."""

    def __init__(self):
        self.start = time.perf_counter()
        self._last = self.start
        self.body = Body()
        self._player = (0.0, 70.0, 18.0)   # мировые координаты преследователя

    use_motion = False

    def command(self, text):
        return ""

    multi_type_ok = True

    def spawn_body(self, invulnerable=False):
        return Body()

    def check_multi_type(self):
        return True

    def claim_body(self):
        pass

    def quiet(self):
        pass

    def pick_flip_name(self):
        return False

    def set_flipped(self, on):
        pass

    def body_position(self):
        return [self.body.x, self.body.y, self.body.z]

    def locate_body(self):
        return Body()

    def move(self, body, mode=None):
        self.body = body

    def close(self):
        pass

    def player_invisible(self):  # noqa: D102
        return False

    def probe_terrain(self):
        return Terrain()

    def probe_odor(self):
        return []       # запаха в самопроверке нет

    def set_lookahead(self, travel_yaw, speed, walking=False):
        pass            # рельефа в самопроверке нет

    def starve(self, amount):
        return True     # голодание в самопроверке не проверяется

    def check_passable(self):
        return []       # рельефа в самопроверке нет

    def try_feed(self, blocks=True):
        return None     # еды в самопроверке нет

    mismatched = 0
    bad_answers = ()

    def health(self):
        return BODY_MAX_HEALTH          # в самопроверке никто не бьёт

    def scan(self):
        """Игрок идёт на муху и преследует её, пока та убегает."""
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if now - self.start > 5.0:
            dx = self.body.x - self._player[0]
            dz = self.body.z - self._player[2]
            length = math.hypot(dx, dz) or 1.0
            if length > 1.2:
                step = min(3.0 * dt, length - 1.2)
                self._player = (self._player[0] + dx / length * step,
                                self.body.y,
                                self._player[2] + dz / length * step)
        return [Sighting("player", self._player, PLAYER_SIZE)]
