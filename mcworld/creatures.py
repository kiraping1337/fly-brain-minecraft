"""Существа: опрос, устойчивые имена, проверка прямой видимости."""

import math
import time

from flygame.parsing import parse_name, parse_numbers, parse_sightings, size_of
from flygame.state import Sighting

from .constants import (
    BODY_TAG, INVISIBILITY_EVERY,
    PLAYER_SIZE, PROBE_PLAYER, PROJECTILE_DISTANCE, PROJECTILE_LIMIT,
    PROJECTILE_TYPES, SIGHT_EVERY, SIGHT_HOLDER, SIGHT_LIFT, SIGHT_MAX,
    TRACK_GATE, TRACK_GATE_FAST)
from . import gateway, ident, shared

# Сколько муха ждёт общий снимок, прежде чем спросить сама
SHARED_WAIT_S = 0.5
SHARED_WAIT_SHORT_S = 0.12
SHARED_MISSES_SHORT = 3


class CreaturesPart:
    """Часть MinecraftWorld: существа: опрос, устойчивые имена, проверка прямой видимости."""

    #восприятие

    def player_invisible(self):  # noqa: D102
        if self._poll % INVISIBILITY_EVERY != 1:
            return self._invisible
        if self._effects_path is False:
            return False
        paths = [self._effects_path] if self._effects_path else \
            ["active_effects", "ActiveEffects"]
        for path in paths:
            answer = self.command(f"data get entity @p {path}") or ""
            if "has the following entity data" in answer:
                self._effects_path = path
                self._invisible = "invisibility" in answer.lower()
                return self._invisible
        self._invisible = False
        return False

    def check_multi_type(self):
        """Принимает ли сервер несколько неотрицаемых type= в одном селекторе."""
        if len(PROJECTILE_TYPES) < 2:
            return self.multi_type_ok
        self._ensure_scoreboard()
        counter = f"{PROBE_PLAYER} {ident.PROBE_OBJECTIVE}"
        self.command(f"scoreboard players set {counter} -1")
        kinds = ",".join(f"type={t}" for t in PROJECTILE_TYPES)
        self.command(f"execute store result score {counter} "
                     f"if entity @e[{kinds},limit=1]")
        answer = self.read_score(counter)
        self.multi_type_ok = answer is not None and answer >= 0
        return self.multi_type_ok

    def shared_view(self):
        """Свой отбор из общего снимка гонки или None - спросить самой."""
        link = self._shared_link
        if link is False:
            return None
        stats = self.shared_stats
        try:
            if link is None:
                if gateway.GATEWAY is None:
                    self._shared_link = False
                    return None
                link = gateway.GatewayConnection(gateway.GATEWAY, ident.NAME).connect()
                self._shared_link = link
            misses = getattr(self, "_shared_misses", 0)
            wait = SHARED_WAIT_S if misses < SHARED_MISSES_SHORT else SHARED_WAIT_SHORT_S
            status, snapshot = link.shared(self._shared_seq, wait)
        except Exception:
            stats["missed"] += 1
            self._shared_misses = getattr(self, "_shared_misses", 0) + 1
            return None
        if status == "off":
            self._shared_link = False
            link.disconnect()
            return None
        if snapshot is None:
            stats["missed"] += 1
            self._shared_misses = misses + 1
            return None
        self._shared_misses = 0
        snapshot["received"] = time.perf_counter()
        self._shared_seq = snapshot["seq"]
        self._shared_last = snapshot
        stats["taken"] += 1
        view = shared.view_for(snapshot, ident.NAME, self.multi_type_ok,
                               self.see_flies)
        if view is None:
            stats["whole"] += 1
        else:
            stats["creatures"] += view["creatures"] is None
            stats["projectiles"] += view["projectiles"] is None
        return view

    def scan(self):
        """Ближайший игрок, несколько ближайших существ и летящие снаряды."""
        self._poll += 1
        view = self.shared_view()
        if view is not None:
            self._invisible = view["invisible"]
        else:
            self._invisible = self.player_invisible()

        sightings, seen, placed = [], set(), set()

        def add(name, position, size, is_player=False):
            spot = tuple(round(v, 1) for v in position)
            if spot in placed:
                return                  # тот же объект пришёл дважды
            placed.add(spot)
            key = name
            if key in seen:             # трёх зомби зовут одинаково
                key = f"{name}#{len(sightings)}"
            seen.add(key)
            sightings.append(Sighting(key, tuple(position), size, is_player))

        prefix = f"execute at {ident.BODY_SELECTOR} run data get entity"

        if not self._invisible:
            if view is not None:
                if view["player"] is not None:
                    name, position = view["player"]
                    add(name, position, PLAYER_SIZE, is_player=True)
            else:
                answer = self.command(f"{prefix} @p Pos")
                position = parse_numbers(answer)
                if position:
                    add(parse_name(answer) or "player", position, PLAYER_SIZE,
                        is_player=True)

        creatures = view["creatures"] if view is not None else None
        if creatures is None:
            creatures = parse_sightings(self.command(shared.creature_query(
                ident.BODY_SELECTOR, ident.OWN_TAG if self.see_flies else BODY_TAG)))
        for name, position in creatures:
            add(name, position, size_of(name))
        seeable = len(sightings)

        projectiles = view["projectiles"] if view is not None else None
        if projectiles is None:
            wanted = PROJECTILE_TYPES if self.multi_type_ok else PROJECTILE_TYPES[:1]
            kinds = ",".join(f"type={t}" for t in wanted)
            projectiles = parse_sightings(self.command(
                f"execute at {ident.BODY_SELECTOR} as @e[{kinds},"
                f"limit={PROJECTILE_LIMIT},sort=nearest,"
                f"distance=..{PROJECTILE_DISTANCE},nbt={{inGround:0b}}] "
                f"run data get entity @s Pos"))
        for name, position in projectiles:
            add(name, position, size_of(name))

        return self._only_visible(self._stable_keys(sightings, seeable), seeable)

    def _stable_keys(self, sightings, seeable):
        """Имя держится за существом, а не за местом в списке."""
        tracks = getattr(self, "_tracks", {})
        base = lambda key: key.split("#", 1)[0]
        pairs = []
        for i, s in enumerate(sightings):
            gate = TRACK_GATE_FAST if i >= seeable else TRACK_GATE
            for key, (kind, where) in tracks.items():
                if kind == base(s.name):
                    d = math.dist(s.position, where)
                    if d <= gate:
                        pairs.append((d, i, key))
        pairs.sort()
        taken_i, taken_key, keys = set(), set(), {}
        for d, i, key in pairs:
            if i not in taken_i and key not in taken_key:
                keys[i] = key
                taken_i.add(i)
                taken_key.add(key)
        used = set(keys.values())
        fresh = {}
        for i, s in enumerate(sightings):
            if i in keys:
                continue
            kind = base(s.name)
            name, n = kind, 2
            while name in used:
                name = f"{kind}#{n}"
                n += 1
            keys[i] = name
            used.add(name)
        out = []
        for i, s in enumerate(sightings):
            out.append(Sighting(keys[i], s.position, s.size, s.is_player))
            fresh[keys[i]] = (base(s.name), s.position)
        self._tracks = fresh
        return out

    def _only_visible(self, sightings, seeable):
        """Выбросить тех, кого муха видеть не может: между ними блок."""
        self._sight_tick = getattr(self, "_sight_tick", 0) + 1
        cache = getattr(self, "_sight_cache", {})
        people = sightings[:seeable]
        if (people and self.body_at is not None
                and self._sight_tick % SIGHT_EVERY == 1):
            fx, fy, fz = self.body_at
            fy += 0.3
            counter = f"{SIGHT_HOLDER} {ident.PROBE_OBJECTIVE}"
            self._ensure_scoreboard()
            self.command(f"scoreboard players set {counter} 0")
            checked = people[:SIGHT_MAX]
            for i, s in enumerate(checked):
                mx, my, mz = s.position
                my += s.size / 2.0
                checks = ""
                for f in (1.0 / 3.0, 2.0 / 3.0):
                    px = fx + (mx - fx) * f
                    py = fy + (my - fy) * f + SIGHT_LIFT
                    pz = fz + (mz - fz) * f
                    checks += (f"positioned {px:.2f} {py:.2f} {pz:.2f} "
                               f"if block ~ ~ ~ minecraft:air ")
                self.command(f"execute {checks}"
                             f"run scoreboard players add {counter} {1 << i}")
            bits = self.read_score(counter)
            if bits is not None:
                cache = {s.name: bool((bits >> i) & 1) for i, s in enumerate(checked)}
            self._sight_cache = cache
        visible = [s for s in people if cache.get(s.name, True)]
        return visible + sightings[seeable:]
