"""Еда: запах предметов, укусы блоков, раскладка кусков."""

import math

import numpy as np

from flygame.parsing import parse_sightings

from .constants import (
    BERRY_BUSH, BERRY_YIELD, CRUMB_SPEED, CRUMB_TAG, FOOD_BLOCKS, FOOD_DROPS,
    FOOD_HOLDER, FOOD_SPOTS, ODOR_LIMIT, ODOR_RANGE, REACH)
from . import ident, shared


class FoodPart:
    """Часть MinecraftWorld: еда: запах предметов, укусы блоков, раскладка кусков."""

    def probe_odor(self):
        """Где лежат пахнущие куски фруктов. Список (x, y, z)."""
        found = []
        snapshot = self._shared_last
        if snapshot is not None:
            self.shared_stats["odor_calls"] += 1
        asked = False
        for item_id in self.food_items:
            if snapshot is not None:
                got = shared.odor_for(snapshot, item_id, self.body_at)
                if got is not None:
                    found.extend(got)
                    continue
                asked = True
            answer = self.command(
                f"execute at {ident.BODY_SELECTOR} as "
                f"@e[type=item,nbt={{Item:{{id:\"{item_id}\"}}}},"
                f"distance=..{ODOR_RANGE:.0f},limit={ODOR_LIMIT},sort=nearest] "
                f"run data get entity @s Pos")
            for _name, position in parse_sightings(answer):
                found.append(position)
        if asked:
            self.shared_stats["odor"] += 1
        return found

    #еда

    def _stored(self, clause, where=""):
        """execute ... store result у тела, ответ приходит числом."""
        self._ensure_scoreboard()
        counter = f"{FOOD_HOLDER} {ident.PROBE_OBJECTIVE}"
        self.command(f"execute at {ident.BODY_SELECTOR} {where}"
                     f"store result score {counter} {clause}")
        return self.read_score(counter) or 0

    def try_feed(self, blocks=True):
        """Один укус, если муха касается еды. (что, id, пояснение) или None."""
        if self._stored(f"if entity @e[type=item,distance=..{REACH}]"):
            for item_id in self.food_items:
                target = (f"@e[type=item,distance=..{REACH},limit=1,sort=nearest,"
                          f"nbt={{Item:{{id:\"{item_id}\"}}}}]")
                count = self._stored(f"run data get entity {target} Item.count")
                if count <= 0:
                    continue
                if count > 1:
                    self.command(f"execute at {ident.BODY_SELECTOR} run data modify "
                                 f"entity {target} Item.count set value {count - 1}")
                else:
                    self.command(f"execute at {ident.BODY_SELECTOR} run kill {target}")
                return "item", item_id, ""

        if not blocks or self.food_only:
            return None

        counter = f"{FOOD_HOLDER} {ident.PROBE_OBJECTIVE}"
        self.command(f"scoreboard players set {counter} 0")
        checks = (f"at {ident.BODY_SELECTOR} "
                  + "".join(f"unless block ~ ~ ~ {BERRY_BUSH}[age={age}] "
                            for age in (3, 2)))
        for dx, dy, dz, _label in FOOD_SPOTS:
            checks += (f"at {ident.BODY_SELECTOR} positioned ~{dx} ~{dy} ~{dz} "
                       + "".join(f"unless block ~ ~ ~ {block} " for block in FOOD_BLOCKS))
        self.command(f"execute {checks}run scoreboard players set {counter} 1")
        if self.read_score(counter) == 1:
            return None

        for age in (3, 2):
            picked = self._stored(
                f"if block ~ ~ ~ {BERRY_BUSH}[age={age}] "
                f"run setblock ~ ~ ~ {BERRY_BUSH}[age=1]")
            if picked:
                low, high = BERRY_YIELD[age]
                count = int(np.random.randint(low, high + 1))
                self._scatter_crumbs((0, 0, 0), "minecraft:sweet_berries", count)
                return "block", BERRY_BUSH, f"куст, ягод {count}"

        counter = f"{FOOD_HOLDER} {ident.PROBE_OBJECTIVE}"
        self.command(f"scoreboard players set {counter} 0")
        table, bit = [], 1
        for dx, dy, dz, label in FOOD_SPOTS:
            for block in FOOD_BLOCKS:
                self.command(f"execute at {ident.BODY_SELECTOR} positioned ~{dx} ~{dy} ~{dz} "
                             f"if block ~ ~ ~ {block} "
                             f"run scoreboard players add {counter} {bit}")
                table.append((bit, (dx, dy, dz), label, block))
                bit <<= 1
        bits = self.read_score(counter) or 0

        for bit, spot, label, block in table:
            if not bits & bit:
                continue
            dx, dy, dz = spot
            broken = self._stored(
                f"if block ~ ~ ~ {block} run setblock ~ ~ ~ minecraft:air destroy",
                where=f"positioned ~{dx} ~{dy} ~{dz} ")
            if broken:
                pieces = self._gather_drops(spot, FOOD_DROPS[block])
                return "block", block, f"{label}, кусков {pieces}"
        return None

    def _gather_drops(self, spot, drop_id):
        """Выпавшее из укушенного блока - по одному куску, ближе к мухе."""
        dx, dy, dz = spot
        at_block = f"positioned ~{dx} ~{dy} ~{dz} "
        stack = (f"@e[type=item,tag=!{CRUMB_TAG},distance=..1.5,limit=1,"
                 f"sort=nearest,nbt={{Item:{{id:\"{drop_id}\"}}}}]")
        count = self._stored(f"run data get entity {stack} Item.count",
                             where=at_block)
        if count <= 0:
            return 0
        self.command(f"execute at {ident.BODY_SELECTOR} {at_block}run kill {stack}")
        return self._scatter_crumbs(spot, drop_id, count)

    def _scatter_crumbs(self, spot, drop_id, count):
        """Положить count кусков по одному, на полпути между местом и мухой."""
        dx, dy, dz = spot
        middle = f"positioned ~{dx / 2:g} ~{dy / 2:g} ~{dz / 2:g}"
        for k in range(count):
            angle = 2.0 * math.pi * (k + 0.5) / count
            vx, vz = CRUMB_SPEED * math.cos(angle), CRUMB_SPEED * math.sin(angle)
            self.command(
                f"execute at {ident.BODY_SELECTOR} {middle} run summon item "
                f"~ ~0.3 ~ {{Item:{{id:\"{drop_id}\",count:1}},"
                f"Motion:[{vx:.3f}d,0.18d,{vz:.3f}d],PickupDelay:32767,"
                f"Tags:[\"{CRUMB_TAG}\"]}}")
        return count
