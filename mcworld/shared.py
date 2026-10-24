"""Общий опрос гонки: что одинаково для всех мух, спрашивается один раз."""

import math
import re
import threading
import time

from flygame.constants import FOOD_ITEMS
from flygame.parsing import parse_name, parse_numbers, parse_sightings

from .constants import (
    BODY_TAG, IGNORED_TYPES, INVISIBILITY_EVERY, MAX_DISTANCE, NEARBY_LIMIT,
    ODOR_LIMIT, ODOR_RANGE, POLL_INTERVAL_S, PROJECTILE_DISTANCE, PROJECTILE_LIMIT,
    PROJECTILE_TYPES)

CREATURES_MAX = 32
PROJECTILES_MAX = 12
ODOR_MAX = 24
ODOR_EVERY_SNAPSHOTS = 5        # еда - раз в полсекунды, как у мухи

MARGIN = 2.0
ODOR_MARGIN = 4.0               # запах берётся из снимка до секунды давности
FRESH_S = 0.35                  # снимок старше этого мухе не отдаётся
ODOR_FRESH_S = 1.0
SAME_SPOT = 0.01                # своё тело в ответе: столько от места телепорта
CLUSTER_GAP = 10.0              # мухи ближе этого - одна кучка, один запрос существ
CROWD_HOLD_S = 5.0              # кучка в людном месте - столько на своих запросах

TP_RE = re.compile(r"^tp @e\[tag=flybrain_active_(\w+),limit=1\] "
                   r"(-?[\d.]+) (-?[\d.]+) (-?[\d.]+)")
SUMMON_RE = re.compile(r"summon minecraft:\w+ (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) "
                       r".*\"flybrain_(\w+)\"")
KILL_RE = re.compile(r"^kill @e\[tag=flybrain_(\w+)\]$")


def _up(radius):
    """Радиус для селектора: вверх до сотых, ровно то число, что прочтёт сервер."""
    return math.ceil(radius * 100.0) / 100.0


def creature_query(selector, exclude_tag):
    """Свой запрос существ мухи. Один текст для мухи и для шлюза."""
    excluded = ",".join(f"type=!{t}" for t in IGNORED_TYPES)
    return (f"execute at {selector} as @e[tag=!{exclude_tag},{excluded},"
            f"limit={NEARBY_LIMIT},sort=nearest,distance=..{MAX_DISTANCE}] "
            f"run data get entity @s Pos")


def clusters(bodies, gap):
    """Кучки мух: связаны, если цепочкой соседей ближе gap. Списки имён."""
    groups = []
    for name in sorted(bodies):
        linked = [g for g in groups
                  if any(math.dist(bodies[name], bodies[m]) <= gap for m in g)]
        merged = sorted([name] + [m for g in linked for m in g])
        groups = [g for g in groups if all(g is not l for l in linked)] + [merged]
    return groups


class SharedSight:
    """Снимки мира для всех мух гонки. Живёт в процессе шлюза.

    command(text) - выполнить команду через шлюз (одна очередь со всеми).
    observe(text, answer) шлюз зовёт после КАЖДОЙ выполненной команды, в том
    же потоке и по порядку - так известно, где тела мух в момент запроса.
    """

    def __init__(self, command, interval_s=POLL_INTERVAL_S, see_flies=True):
        self.command = command
        self.interval = interval_s
        self.see_flies = see_flies
        self._crowded = {}          # кучка (frozenset имён) -> до какого времени
        self.creature_queries = 0   # сколько запросов существ задано всего
        self.lock = threading.Condition()
        self.bodies = {}            # имя мухи -> (x, y, z) по последней команде
        self.snapshot = None
        self.seq = 0
        self._running = False
        self._capture = None        # текст запроса, на котором снять тела
        self._captured = None
        self._effects_path = None
        self._invisible = False
        self._odor = None
        self.multi_type_ok = False
        self.errors = []

    #тела мух: по командам, которые идут через шлюз

    def observe(self, text, answer):
        """Где тело мухи - по её же телепорту, появлению или kill."""
        moved = TP_RE.match(text)
        with self.lock:
            if moved:
                if "no entity" in (answer or "").lower():
                    self.bodies.pop(moved.group(1), None)   # тела уже нет
                else:
                    self.bodies[moved.group(1)] = tuple(
                        float(moved.group(i)) for i in (2, 3, 4))
            elif text.startswith("summon "):
                spawned = SUMMON_RE.match(text)
                if spawned:
                    self.bodies[spawned.group(4)] = tuple(
                        float(spawned.group(i)) for i in (1, 2, 3))
            elif text.startswith("kill "):
                killed = KILL_RE.match(text)
                if killed:
                    self.bodies.pop(killed.group(1), None)
            if self._capture is not None and text == self._capture:
                self._captured = dict(self.bodies)

    def _ask(self, text):
        """Запрос снимка и тела мух в тот момент, когда сервер его выполнил."""
        with self.lock:
            self._capture, self._captured = text, None
        answer = self.command(text)
        with self.lock:
            bodies, self._capture, self._captured = self._captured, None, None
        return answer, bodies or {}

    #снимки

    def start(self):
        self._running = True
        self._check_multi_type()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False
        with self.lock:
            self.lock.notify_all()

    def wait_newer(self, seq, timeout):
        """Свежий снимок новее seq; ждёт до timeout. None - не дождались."""
        deadline = time.perf_counter() + timeout
        with self.lock:
            while (self.snapshot is None or self.snapshot["seq"] <= seq
                   or time.perf_counter() - self.snapshot["time"] > FRESH_S):
                left = deadline - time.perf_counter()
                if left <= 0 or not self._running:
                    return None
                self.lock.wait(left)
            return self.snapshot

    def package(self, snapshot):
        """Снимок для отправки мухе: с возрастом его и запаха на этот момент."""
        now = time.perf_counter()
        odor = snapshot["odor"]
        return dict(snapshot, age=now - snapshot["time"],
                    odor_age=None if odor is None else now - odor["time"])

    def _check_multi_type(self):
        """Принимает ли сервер перечисление типов в селекторе (как у мухи)."""
        if len(PROJECTILE_TYPES) < 2:
            return
        kinds = ",".join(f"type={t}" for t in PROJECTILE_TYPES)
        answer = (self.command(f"execute if entity @e[{kinds},limit=1]") or "").lower()
        self.multi_type_ok = "test passed" in answer or "test failed" in answer

    def _run(self):
        while self._running:
            started = time.perf_counter()
            try:
                self._take()
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(max(0.0, self.interval - (time.perf_counter() - started)))

    def _take(self):
        with self.lock:
            bodies = dict(self.bodies)
        if not bodies:
            return
        raw = [sum(p[i] for p in bodies.values()) / len(bodies) for i in range(3)]
        center = tuple(float(f"{v:.3f}") for v in raw)      # как прочтёт сервер
        spread = max(math.dist(center, p) for p in bodies.values())
        where = f"execute positioned {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"
        number = self.seq + 1

        if number % INVISIBILITY_EVERY == 1:
            self._invisible = self._player_invisible(where)
        player = None
        if not self._invisible:
            answer = self.command(f"{where} run data get entity @p Pos")
            position = parse_numbers(answer)
            if position:
                player = (parse_name(answer) or "player", tuple(position))

        creatures = self._creatures(bodies)

        wanted = PROJECTILE_TYPES if self.multi_type_ok else PROJECTILE_TYPES[:1]
        kinds = ",".join(f"type={t}" for t in wanted)
        reach = _up(spread + MARGIN + PROJECTILE_DISTANCE)
        answer, seen_bodies = self._ask(
            f"{where} as @e[{kinds},limit={PROJECTILES_MAX},sort=nearest,"
            f"distance=..{reach:.2f},nbt={{inGround:0b}}] run data get entity @s Pos")
        found = parse_sightings(answer)
        projectiles = {"center": center, "reach": reach, "items": found,
                       "full": len(found) < PROJECTILES_MAX, "bodies": seen_bodies,
                       "multi_type_ok": self.multi_type_ok}

        if number % ODOR_EVERY_SNAPSHOTS == 1:
            reach = _up(spread + ODOR_MARGIN + ODOR_RANGE)
            items = {}
            for item_id in FOOD_ITEMS:
                answer = self.command(
                    f"{where} as @e[type=item,nbt={{Item:{{id:\"{item_id}\"}}}},"
                    f"distance=..{reach:.2f},limit={ODOR_MAX},sort=nearest] "
                    f"run data get entity @s Pos")
                found = [pos for _n, pos in parse_sightings(answer)]
                items[item_id] = (found, len(found) < ODOR_MAX)
            self._odor = {"time": time.perf_counter(), "center": center,
                          "reach": reach, "items": items}

        snapshot = {"seq": number, "time": time.perf_counter(), "player": player,
                    "invisible": self._invisible, "creatures": creatures,
                    "see_flies": self.see_flies,
                    "projectiles": projectiles, "odor": self._odor}
        with self.lock:
            self.seq = number
            self.snapshot = snapshot
            self.lock.notify_all()

    def _creatures(self, bodies):
        """Существа для каждой мухи: {имя: часть снимка}. Часть - общая на кучку или своя у мухи (см. начало модуля)."""
        now = time.perf_counter()
        parts = {}
        for group in clusters(bodies, CLUSTER_GAP):
            key = frozenset(group)
            if len(group) > 1 and self._crowded.get(key, 0.0) <= now:
                part = self._cluster_part({n: bodies[n] for n in group})
                if not part["full"]:
                    self._crowded[key] = now + CROWD_HOLD_S
                for name in group:
                    parts[name] = part
            else:
                for name in group:
                    parts[name] = self._single_part(name)
        return parts

    def _cluster_part(self, bodies):
        """Один запрос на кучку мух - вокруг её центра, с запасом."""
        raw = [sum(p[i] for p in bodies.values()) / len(bodies) for i in range(3)]
        center = tuple(float(f"{v:.3f}") for v in raw)      # как прочтёт сервер
        spread = max(math.dist(center, p) for p in bodies.values())
        reach = _up(spread + MARGIN + MAX_DISTANCE)
        excluded = ",".join(f"type=!{t}" for t in IGNORED_TYPES)

        tag = "" if self.see_flies else f"tag=!{BODY_TAG},"
        self.creature_queries += 1
        answer, seen_bodies = self._ask(
            f"execute positioned {center[0]:.3f} {center[1]:.3f} {center[2]:.3f} "
            f"as @e[{tag}{excluded},limit={CREATURES_MAX},"
            f"sort=nearest,distance=..{reach:.2f}] run data get entity @s Pos")
        found = parse_sightings(answer)
        return {"kind": "cluster", "center": center, "reach": reach, "items": found,
                "full": len(found) < CREATURES_MAX, "bodies": seen_bodies}

    def _single_part(self, name):
        """Свой запрос мухи - тот же текст, что она задала бы сама."""
        own = f"{BODY_TAG}_{name}" if self.see_flies else BODY_TAG
        self.creature_queries += 1
        answer, seen_bodies = self._ask(
            creature_query(f"@e[tag={BODY_TAG}_active_{name},limit=1]", own))
        return {"kind": "single", "items": parse_sightings(answer),
                "bodies": seen_bodies}

    def _player_invisible(self, where):
        if self._effects_path is False:
            return False
        paths = [self._effects_path] if self._effects_path else \
            ["active_effects", "ActiveEffects"]
        for path in paths:
            answer = self.command(f"{where} run data get entity @p {path}") or ""
            if "has the following entity data" in answer:
                self._effects_path = path
                return "invisibility" in answer.lower()
        return False


#сторона мухи: отбор своих ближайших

def nearest(items, origin, radius, limit):
    """Ближайшие к origin в радиусе (включительно), не больше limit.

    items - [(имя, позиция)]. Сортировка устойчивая: при равных расстояниях
    порядок как в ответе сервера.
    """
    found = [(math.dist(origin, pos), name, pos) for name, pos in items]
    found = [f for f in found if f[0] <= radius]
    found.sort(key=lambda f: f[0])
    return [(name, pos) for _, name, pos in found[:limit]]


def covers(part, origin, radius):
    """Шар мухи (origin, radius) целиком внутри шара общего запроса."""
    return math.dist(part["center"], origin) + radius <= part["reach"]


def own_entry(items, at):
    """Номер своего тела в ответе: единственная запись у места телепорта."""
    close = [i for i, (_n, pos) in enumerate(items) if math.dist(pos, at) <= SAME_SPOT]
    return close[0] if len(close) == 1 else None


def view_for(snapshot, own, multi_type_ok, see_flies=True):
    """Что муха own получила бы своими запросами - по общему снимку."""
    part = snapshot["creatures"].get(own)
    if part is None:
        return None
    at = part["bodies"].get(own)
    if at is None:
        return None
    creatures = None
    if snapshot.get("see_flies", True) != see_flies:
        pass
    elif part["kind"] == "single":
        creatures = list(part["items"])
    elif part["full"] and covers(part, at, MAX_DISTANCE):
        pool = list(part["items"])
        if see_flies:
            mine = own_entry(pool, at)
            if mine is None:
                return None
            del pool[mine]
        creatures = nearest(pool, at, MAX_DISTANCE, NEARBY_LIMIT)

    part = snapshot["projectiles"]
    at = part["bodies"].get(own)
    projectiles = None
    if (at is not None and part["full"] and part["multi_type_ok"] == multi_type_ok
            and covers(part, at, PROJECTILE_DISTANCE)):
        projectiles = nearest(part["items"], at, PROJECTILE_DISTANCE, PROJECTILE_LIMIT)

    return {"player": snapshot["player"], "invisible": snapshot["invisible"],
            "creatures": creatures, "projectiles": projectiles}


def odor_for(snapshot, item_id, origin, now=None):
    """Куски одного вида для запаха мухи в origin, или None."""
    odor = snapshot.get("odor")
    if odor is None or origin is None or item_id not in odor["items"]:
        return None
    now = time.perf_counter() if now is None else now
    if snapshot["odor_age"] + (now - snapshot["received"]) > ODOR_FRESH_S:
        return None
    found, full = odor["items"][item_id]
    if not full or not covers(odor, origin, ODOR_RANGE):
        return None
    return [pos for _n, pos in nearest([("", p) for p in found], origin,
                                       ODOR_RANGE, ODOR_LIMIT)]
