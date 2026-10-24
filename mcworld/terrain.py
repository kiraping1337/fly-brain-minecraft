"""Рельеф вокруг тела и досмотр проходимых блоков."""

import math
import time

from flygame.constants import WALK_FAR, WALK_NEAR, WALK_ROOM_MARGIN
from flygame.geometry import heading_vector
from flygame.state import Terrain

from . import shape
from .constants import (
    AHEAD_BITS, EYE_BITS, LAVA_BIT, PASSABLE_BLOCKS, PASSABLE_TIERS,
    PASS_CACHE_MAX, PASS_CACHE_S, PASS_HOLDER, POLL_INTERVAL_S, PROBE_PLAYER,
    SOLID_BLOCKS, STEP_BIT, STILL_MOVE, STILL_SPEED, STILL_TURN_DEG,
    TERRAIN_EVERY, TERRAIN_LAG_S, UNDER_LAVA_BIT,
    UNDER_WATER_BIT, WALK_BIT, WALK_L_BIT, WALK_R_BIT, WATER_BIT)
from . import gateway, ident

# Дальние точки ходьбы занимают биты луча полёта: пешком луч не шлётся.
WALK_FAR_BITS = AHEAD_BITS[1:1 + len(WALK_FAR)]
FACT_WAIT_S = 60.0        # сколько ждать, пока другая муха выяснит общий факт
PASS_CHUNK_CHARS = 900


class TerrainPart:
    """Часть MinecraftWorld: рельеф вокруг тела и досмотр проходимых блоков."""

    def set_lookahead(self, travel_yaw, speed, walking=False):
        """Куда смотреть зондом препятствия и на сколько блоков вперёд."""
        forward_x, forward_z = heading_vector(travel_yaw)

        period = max(POLL_INTERVAL_S, getattr(self, "poll_period", POLL_INTERVAL_S))
        interval = TERRAIN_EVERY * period + TERRAIN_LAG_S
        reach = max(1.5, abs(speed) * interval + 1.0)

        steps = [1.5]
        while steps[-1] + 1.0 <= reach and len(steps) < len(AHEAD_BITS):
            steps.append(steps[-1] + 1.0)
        self._ahead_ray = [(forward_x * d, forward_z * d) for d in steps]
        self._walking = bool(walking)
        self._moving = abs(speed) > STILL_SPEED

        left_x, left_z = forward_z, -forward_x
        self._near = (travel_yaw, forward_x * WALK_NEAR, forward_z * WALK_NEAR,
                      left_x * shape.BODY_HALF, left_z * shape.BODY_HALF,
                      tuple((forward_x * d, forward_z * d) for d in WALK_FAR))

    def _ahead_probes(self):
        """Точки луча по курсу полёта: [(бит, сдвиг), ...]."""
        return [(bit, f"positioned ~{dx:.2f} ~ ~{dz:.2f} ")
                for bit, (dx, dz) in zip(AHEAD_BITS, self._ahead_ray)]

    def probe_terrain(self):
        """Что вокруг тела."""
        self._ensure_scoreboard()
        counter = f"{PROBE_PLAYER} {ident.PROBE_OBJECTIVE}"
        self.command(f"scoreboard players set {counter} 0")
        walking = self._walking
        probe_yaw, ndx, ndz, blx, blz, far = self._near

        base = self.body_at

        def spot(dx, dy, dz):
            if base is None:
                return f"positioned ~{dx:.2f} ~{dy:.2f} ~{dz:.2f} ", None
            x, y, z = base[0] + dx, base[1] + dy, base[2] + dz
            return (f"positioned {x:.2f} {y:.2f} {z:.2f} ",
                    (math.floor(x), math.floor(y), math.floor(z)))

        cells = {}
        if walking:
            points = [(WALK_BIT, (ndx, 0.0, ndz)),
                      (STEP_BIT, (ndx, 1.0, ndz)),
                      (WALK_L_BIT, (ndx + blx, 0.0, ndz + blz)),
                      (WALK_R_BIT, (ndx - blx, 0.0, ndz - blz))]
            points += [(bit, (fdx, 0.0, fdz))
                       for bit, (fdx, fdz) in zip(WALK_FAR_BITS, far)]
            under_near, _ = spot(ndx, -1.0, ndz)
        else:
            points = [(bit, (dx, 0.0, dz))   # луч по курсу ПОЛЁТА, не взгляда
                      for bit, (dx, dz) in zip(AHEAD_BITS, self._ahead_ray)]
        ahead = []
        for bit, offset in points:
            where, cell = spot(*offset)
            ahead.append((bit, where))
            cells[bit] = cell
        eye_probes = []
        if base is None:
            where, cell = spot(0.0, shape.EYE_HEIGHT, 0.0)
            eye_probes.append((EYE_BITS[0], where, cell))
        else:
            ey = base[1] + shape.EYE_HEIGHT
            corners = sorted({(math.floor(base[0] + dx), math.floor(ey),
                               math.floor(base[2] + dz))
                              for dx in (-shape.EYE_BOX_HALF, shape.EYE_BOX_HALF)
                              for dz in (-shape.EYE_BOX_HALF, shape.EYE_BOX_HALF)})
            for bit, (cx, cy, cz) in zip(EYE_BITS, corners):
                where = f"positioned {cx + 0.5:.2f} {cy + 0.5:.2f} {cz + 0.5:.2f} "
                eye_probes.append((bit, where, (cx, cy, cz)))
        for bit, where, cell in eye_probes:
            cells[bit] = cell

        blocking = ([a for a in ahead if a[0] != STEP_BIT] if walking else ahead)
        groups = [[(b, w, cells.get(b)) for b, w, _ in eye_probes],
                  [(b, w, cells.get(b)) for b, w in blocking]]
        singles = [(2, spot(0.0, 1.0, 0.0)[0]),          # над головой
                   (8, spot(0.0, -2.5, 0.0)[0]),         # запас высоты
                   (16, spot(0.0, -1.0, 0.0)[0])]        # прямо под брюхом
        singles += [(b, w) for b, w in ahead if walking and b == STEP_BIT]
        known = self._known_passable
        preset, chained = 0, []
        for group in groups:
            members = []
            for bit, where, cell in group:
                if known(cell):
                    preset |= bit
                else:
                    members.append((bit, where))
            if len(members) > 1:
                chained.append(members)
                self._air_chain(counter, members)
            else:
                singles += members
        for bit, where in singles:
            self._air_chain(counter, [(bit, where)])
        under_body = spot(0.0, -1.0, 0.0)[0]
        for bit, liquid in ((UNDER_WATER_BIT, "minecraft:water"),
                            (UNDER_LAVA_BIT, "minecraft:lava")):
            self.command(f"execute at {ident.BODY_SELECTOR} {under_body}"
                         f"if block ~ ~ ~ {liquid} "
                         f"run scoreboard players add {counter} {bit}")
        if walking:

            for bit, liquid in ((WATER_BIT, "minecraft:water"),
                                (LAVA_BIT, "minecraft:lava")):
                self.command(f"execute at {ident.BODY_SELECTOR} {under_near}"
                             f"if block ~ ~ ~ {liquid} "
                             f"run scoreboard players add {counter} {bit}")

        bits = self.read_score(counter)
        if bits is None:
            return self._last_terrain
        failed = [m for m in chained if not bits & m[0][0]]
        if failed:
            for members in failed:
                for bit, where in members:
                    self._air_chain(counter, [(bit, where)])
            bits = self.read_score(counter)
            if bits is None:
                return self._last_terrain
        bits |= preset

        doubtful = [(bit, where, cells.get(bit))
                    for bit, where in [(b, w) for b, w, _ in eye_probes] + blocking
                    if not bits & bit]
        for bit in self._passable_many(doubtful):
            bits |= bit

        here = all(bits & bit for bit, _, _ in eye_probes)
        overhead = bool(bits & 2)
        walk_room = None
        if walking:
            free_ahead = bool(bits & WALK_BIT)
            run = 0.0
            for bit, dist in zip((WALK_BIT,) + WALK_FAR_BITS, (WALK_NEAR,) + WALK_FAR):
                if not bits & bit:
                    break
                run = dist
            walk_room = max(0.0, run - shape.EYE_BOX_HALF - WALK_ROOM_MARGIN) if run else 0.0
            step_up = bool(bits & STEP_BIT) and overhead
            wet_ahead = bool(bits & (WATER_BIT | LAVA_BIT))
            edge_left = not (bits & WALK_L_BIT)
            edge_right = not (bits & WALK_R_BIT)
        else:
            edge_left = edge_right = False
            free_ahead = all(bits & bit for bit, _ in ahead)
            step_up = wet_ahead = False
        self._last_terrain = Terrain(
            stuck=not here,
            free_ahead=free_ahead,
            above_ground=bool(bits & 8),
            clear_below=bool(bits & 16),
            step_up=step_up,
            wet_ahead=wet_ahead,
            edge_left=edge_left,
            edge_right=edge_right,
            wet_below=bool(bits & (UNDER_WATER_BIT | UNDER_LAVA_BIT)),
            probe_yaw=probe_yaw,
            at=base,
            walk_room=walk_room,
        )
        self._probe_key = (base, probe_yaw, walking, self._moving, time.perf_counter())
        return self._last_terrain

    def _air_chain(self, counter, members):
        """Одна команда: все точки - воздух? Да - прибавить сумму их битов."""
        checks = "".join(f"at {ident.BODY_SELECTOR} {where}if block ~ ~ ~ minecraft:air "
                         for _bit, where in members)
        total = sum(bit for bit, _where in members)
        self.command(f"execute {checks}run scoreboard players add {counter} {total}")

    def _known_passable(self, cell):
        """Досмотр недавно признал клетку проходимой (см. _passable_many)."""
        hit = self._pass_cache.get(cell) if cell is not None else None
        return hit is not None and hit[0] and time.perf_counter() - hit[1] < PASS_CACHE_S

    def terrain_due(self, still_s):
        """Нужен ли щуп рельефа: "now" - сейчас же, "skip" - можно не щупать,
        "plan" - по расписанию."""
        key = getattr(self, "_probe_key", None)
        if key is None:
            return "now"
        base, yaw, walking, moving, when = key
        if walking != self._walking or (self._moving and not moving):
            return "now"
        if self._moving or base is None or self.body_at is None:
            return "plan"
        moved = math.dist(base, self.body_at)
        turned = abs((self._near[0] - yaw + 180.0) % 360.0 - 180.0)
        if (moved < STILL_MOVE and turned < STILL_TURN_DEG
                and time.perf_counter() - when < still_s):
            return "skip"
        return "plan"

    def check_passable(self):
        """Какие блоки сервер знает - в гонке один раз на всех мух."""
        link = self._connection
        if gateway.GATEWAY is None or not hasattr(link, "fact"):
            return self._check_passable_here()
        with self._talking:
            status, value = link.fact("passable", FACT_WAIT_S)
        if status == "ok":
            self._passable, self._solid, unknown, self._pass_chunks = value
            return unknown
        unknown = self._check_passable_here()
        with self._talking:
            link.fact_set("passable", (self._passable, self._solid, unknown,
                                       self._pass_chunks))
        return unknown

    def _check_passable_here(self):
        """Какие блоки из списка проходимых сервер вообще знает."""
        counter = f"{PROBE_PLAYER} {ident.PROBE_OBJECTIVE}"
        self._ensure_scoreboard()

        def parses(entry):
            self.command(f"scoreboard players set {counter} -1")
            self.command(f"execute at {ident.BODY_SELECTOR} store result score "
                         f"{counter} if block ~ ~ ~ {entry}")
            answer = self.read_score(counter)
            return answer is not None and answer >= 0

        if not parses("minecraft:air"):
            return None
        unknown = [entry for entry in PASSABLE_BLOCKS if not parses(entry)]
        self._passable = tuple(e for e in PASSABLE_BLOCKS if e not in unknown)
        self._solid = tuple(e for e in SOLID_BLOCKS if parses(e))
        self._pass_chunks = self._verified_chunks(counter)
        return unknown

    def _verified_chunks(self, counter):
        """Список проходимого - цепочками "unless block", проверенными на сервере."""
        chunks, current = [], ""
        for entry in self._passable:
            link = f"unless block ~ ~ ~ {entry} "
            if current and len(current) + len(link) > PASS_CHUNK_CHARS:
                chunks.append(current)
                current = ""
            current += link
        if current:
            chunks.append(current)
        if not chunks:
            return None
        for chunk in chunks:
            self.command(f"scoreboard players set {counter} -1")
            self.command(f"execute at {ident.BODY_SELECTOR} store success score "
                         f"{counter} {chunk.rstrip()}")
            answer = self.read_score(counter)
            if answer is None or answer < 0:
                return None
        return tuple(chunks)

    def _passable_many(self, items):
        """Какие из сомнительных точек проходимы. items: [(бит, где, клетка)]."""
        now = time.perf_counter()
        cache = self._pass_cache
        passable, todo = set(), []
        for bit, where, cell in items:
            hit = cache.get(cell) if cell is not None else None
            if hit is not None and now - hit[1] < PASS_CACHE_S:
                if hit[0]:
                    passable.add(bit)
            else:
                todo.append((bit, where, cell))
        if todo and getattr(self, "_pass_chunks", None):
            passable |= self._passable_chained(todo, now)
        elif todo:
            passable |= self._passable_staged(todo, now)
        if len(cache) > PASS_CACHE_MAX:
            for cell in [c for c, (_, t) in cache.items() if now - t >= PASS_CACHE_S]:
                cache.pop(cell, None)
        return passable

    def _passable_chained(self, todo, now):
        """Досмотр цепочками: на точку - по команде на кусок списка."""
        chunks = self._pass_chunks
        counter = f"{PASS_HOLDER} {ident.PROBE_OBJECTIVE}"
        self.command(f"scoreboard players set {counter} 0")
        k = len(chunks)
        for i, (bit, where, cell) in enumerate(todo):
            for j, chunk in enumerate(chunks):
                self.command(f"execute at {ident.BODY_SELECTOR} {where}{chunk}"
                             f"run scoreboard players add {counter} {1 << (k * i + j)}")
        bits = self.read_score(counter)
        found = set()
        if bits is None:
            return found
        cache = self._pass_cache
        for i, (bit, where, cell) in enumerate(todo):
            outside = all((bits >> (k * i + j)) & 1 for j in range(k))
            if not outside:
                found.add(bit)
            if cell is not None:
                cache[cell] = (not outside, now)
        return found

    def _passable_staged(self, todo, now):
        """Прежний досмотр: ступенями, по команде на вид. Запасной путь."""
        cache = self._pass_cache
        passable = set()
        stages = ([(True, PASSABLE_TIERS[0]), (False, self._solid)]
                  + [(True, tier) for tier in PASSABLE_TIERS[1:]])
        for passable_stage, tier in stages:
            entries = [e for e in tier
                       if (e in self._passable if passable_stage else True)]
            if not todo or not entries:
                continue
            holders = [f"#p{i} {ident.PROBE_OBJECTIVE}" for i in range(len(todo))]
            for holder in holders:
                self.command(f"scoreboard players set {holder} 0")
            for (bit, where, cell), holder in zip(todo, holders):
                for entry in entries:
                    self.command(f"execute at {ident.BODY_SELECTOR} {where}"
                                 f"if block ~ ~ ~ {entry} "
                                 f"run scoreboard players set {holder} 1")
            still = []
            for (bit, where, cell), holder in zip(todo, holders):
                answer = self.read_score(holder)
                if answer is None:
                    continue
                if answer > 0 and passable_stage:
                    passable.add(bit)
                    if cell is not None:
                        cache[cell] = (True, now)
                elif answer > 0:
                    if cell is not None:
                        cache[cell] = (False, now)
                else:
                    still.append((bit, where, cell))
            todo = still
        for bit, where, cell in todo:
            if cell is not None:
                cache[cell] = (False, now)
        return passable

    def _is_passable(self, where=""):
        """Стоит ли в этом месте блок, сквозь который можно пройти."""
        counter = f"{PASS_HOLDER} {ident.PROBE_OBJECTIVE}"
        self.command(f"scoreboard players set {counter} 0")
        for block in self._passable:
            self.command(f"execute at {ident.BODY_SELECTOR} {where}"
                         f"if block ~ ~ ~ {block} "
                         f"run scoreboard players add {counter} 1")
        return (self.read_score(counter) or 0) > 0
