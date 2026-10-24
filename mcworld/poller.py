"""Опрос мира в фоновом потоке: сущности, рельеф, здоровье, еда.Главный цикл берёт отсюда последний снимок и не ждёт сервер."""

import threading
import time
import traceback

from flygame.constants import BODY_MAX_HEALTH
from flygame.state import Scan, Terrain

from .constants import POLL_INTERVAL_S, STILL_TERRAIN_S, TERRAIN_EVERY

# Сглаживание замера шага опроса: доля нового цикла в среднем.
PERIOD_SMOOTH = 0.2


#голод
FEED_EVERY = 15           # проверять БЛОКИ еды раз в N опросов (1.5 с)
FEED_ITEM_EVERY = 5       # проверять ПРЕДМЕТЫ раз в N опросов (0.5 с)

#голодание

STARVE_EXPECT_S = 2.5     # сколько ждём обещанную просадку, прежде чем забыть о ней.
HEALTH_RETRIES = 2        # столько подряд пустых чтений подряд = гибель

#урон
HEALTH_EVERY = 2          # читать здоровье раз в N опросов (5 раз/с)
GROUND_EVERY = 2          # как часто проверять землю, пока муха падает


#фоновый опрос

class WorldPoller(threading.Thread):
    recheck_terrain = False   # ставится после рывка, см. run()
    polls = 0                 # сколько опросов прошло - для счётчика
    feed_now = False          # ноги на еде - проверить сейчас, не ждать
    _expect_drop = 0.0        # обещанная просадка здоровья от голодания
    _expect_until = 0.0

    def expect_damage(self, amount):
        """Предупредить: сейчас здоровье просядет на столько, и это не удар."""
        with self.lock:
            self._expect_drop += float(amount)
            self._expect_until = time.perf_counter() + STARVE_EXPECT_S

    def __init__(self, world, interval_s=POLL_INTERVAL_S, debug=False):
        super().__init__(daemon=True)
        self.world = world
        self.interval = interval_s
        self.debug = debug
        self.lock = threading.Lock()
        self.scan = Scan()
        self.errors = 0
        self.sync_body = False

        self.feeding = False
        self._meals = []
        self.watch_health = False
        self.watch_ground = False
        self.full_health = BODY_MAX_HEALTH
        self.bad_readings = 0
        self.bad_values = []      # что именно приходило, для разбора
        self.lost_body = 0        # сколько раз тело переставало находиться
        self._missing = 0
        self._hits = []
        self._last_health = None
        self._dead = False
        self._terrain = Terrain()
        self._body_pos = None
        self._running = True

    def run(self):
        counter = 0
        last_start = None
        while self._running:
            started = time.perf_counter()
            if last_start is not None and hasattr(self.world, "poll_period"):
                self.world.poll_period += (started - last_start
                                           - self.world.poll_period) * PERIOD_SMOOTH
            last_start = started
            try:
                sightings = self.world.scan()
                counter += 1
                self.polls = counter

                if self._terrain_due(counter):
                    self.recheck_terrain = False
                    self._terrain = self.world.probe_terrain()
                if self.watch_health and counter % HEALTH_EVERY == 0:
                    self._check_health()
                if self.feeding and (self.feed_now or counter % FEED_ITEM_EVERY == 0):
                    self.feed_now = False
                    meal = self.world.try_feed(
                        blocks=(counter % FEED_EVERY == 0))
                    if meal:
                        with self.lock:
                            self._meals.append((counter, meal))
                        # Укус меняет блоки под и рядом: рельеф - сразу.
                        self.recheck_terrain = True
                if self.sync_body:
                    self._body_pos = self.world.body_position()
                with self.lock:
                    self.scan = Scan(index=counter, sightings=sightings,
                                     terrain=self._terrain)
                if self.debug:
                    seen = ", ".join(f"{s.name}@{s.position[0]:.0f},"
                                     f"{s.position[2]:.0f}" for s in sightings)
                    print(f"\n[опрос {counter}] видит: {seen or 'никого'} | "
                          f"замурована {self._terrain.stuck} "
                          f"впереди {self._terrain.free_ahead} "
                          f"высота {self._terrain.above_ground}")
            except Exception as exc:
                self.errors += 1
                if self.errors <= 3:
                    where = ""
                    frames = [f for f in traceback.extract_tb(exc.__traceback__)
                              if f.filename.endswith("minecraft_bridge.py")]
                    if frames:
                        where = f" (строка {frames[-1].lineno}, {frames[-1].name})"
                    print(f"\n[опрос] ошибка: {type(exc).__name__}: {exc}{where}")
            time.sleep(max(0.0, self.interval - (time.perf_counter() - started)))

    def latest(self):
        with self.lock:
            return self.scan

    def _terrain_due(self, counter):
        """Щупать ли рельеф в этом опросе."""
        if self.recheck_terrain:
            return True
        due = getattr(self.world, "terrain_due", None)
        verdict = due(STILL_TERRAIN_S) if due is not None else "plan"
        if verdict == "now":
            return True
        if self.watch_ground:
            return counter % GROUND_EVERY == 1
        return counter % TERRAIN_EVERY == 1 and verdict != "skip"

    def server_body_position(self):
        return self._body_pos

    def _check_health(self):
        """Уследить за здоровьем. Величина урона - просадка между опросами."""
        now_hp = self.world.health()
        if now_hp is None:
            self._missing += 1
            if self._missing == 1:
                self.lost_body += 1
            if self._missing >= HEALTH_RETRIES:
                with self.lock:
                    self._dead = True
            return
        self._missing = 0
        if now_hp > self.full_health + 1e-6:
            self.bad_readings += 1
            if len(self.bad_values) < 12:
                self.bad_values.append(
                    (round(now_hp, 2),
                     (getattr(self.world, "last_answer", "") or "")[:70]))
            return
        was = self._last_health
        self._last_health = now_hp
        if was is not None and now_hp < was - 1e-6:
            drop = was - now_hp
            with self.lock:
                if self._expect_until and time.perf_counter() > self._expect_until:
                    self._expect_drop = 0.0
                taken = min(drop, self._expect_drop)
                self._expect_drop -= taken
            drop -= taken
            if drop > 1e-6:
                with self.lock:
                    self._hits.append((drop, now_hp))

    def hits(self):
        """Удары с прошлого вызова: список (урон, здоровье после)."""
        with self.lock:
            taken, self._hits = self._hits, []
        return taken

    def health_now(self):
        return self._last_health

    def died(self):
        with self.lock:
            return self._dead

    def revive(self):
        """После пересоздания тела: забыть смерть и прежнее здоровье."""
        with self.lock:
            self._dead = False
            self._hits.clear()
        self._last_health = None
        self._missing = 0

    def meals(self):
        """Что съедено с прошлого вызова: список (номер опроса, (что, id))."""
        with self.lock:
            eaten, self._meals = self._meals, []
        return eaten

    def stop(self):
        self._running = False
