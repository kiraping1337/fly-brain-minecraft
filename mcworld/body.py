"""Тело: появление, поиск, здоровье, переворот, перемещение, голодание."""

import re

from flygame.parsing import parse_numbers
from flygame.state import Body

from . import ident, shape
from .constants import FLIP_NAMES, FOOD_HOLDER

HEALTH_RE = re.compile(r"has the following entity data: (-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)f\s*$")


class BodyPart:
    """Часть MinecraftWorld: тело: появление, поиск, здоровье, переворот, перемещение, голодание."""

    #тело

    def spawn_body(self, invulnerable=False):
        """Создать тело. По умолчанию смертное - иначе урон не проходит."""
        self.command(f"kill @e[tag={ident.OWN_TAG}]")
        shield = "Invulnerable:1b," if invulnerable else ""
        nbt = (f'{{{shield}{shape.BODY_NBT}NoAI:1b,NoGravity:1b,Silent:1b,'
               f'PersistenceRequired:1b,Tags:[{ident.SPAWN_TAGS}]}}')
        if self.spawn_at is None:
            self.command(f"execute at @p run summon {shape.BODY_ENTITY} ^ ^1 ^3 {nbt}")
        else:
            x, y, z, yaw = self.spawn_at
            self.command(f"summon {shape.BODY_ENTITY} {x:.2f} {y:.2f} {z:.2f} "
                         f"{nbt[:-1]},Rotation:[{yaw:.1f}f,0f]}}")
        return self.locate_body()

    #здоровье

    def health(self):
        """Здоровье тела или None, если тела больше нет."""
        answer = self.command(f"data get entity {ident.BODY_SELECTOR} Health") or ""
        found = HEALTH_RE.search(answer)
        if not found:
            return None
        hp = float(found.group(1))
        return hp if hp > 0 else None

    def claim_body(self):
        """Выбрать ровно одно тело и пометить его личным тегом."""
        self.command(f"tag @e[tag={ident.ACTIVE_TAG}] remove {ident.ACTIVE_TAG}")
        self.command(f"tag @e[tag={ident.OWN_TAG},limit=1] add {ident.ACTIVE_TAG}")

    def body_position(self):
        return parse_numbers(self.command(f"data get entity {ident.BODY_SELECTOR} Pos"))

    def locate_body(self):
        self.claim_body()
        pos = parse_numbers(self.command(f"data get entity {ident.BODY_SELECTOR} Pos"))
        rotation = parse_numbers(
            self.command(f"data get entity {ident.BODY_SELECTOR} Rotation"))
        if not pos:
            return None
        return Body(x=pos[0], y=pos[1], z=pos[2],
                    yaw=rotation[0] if rotation else 0.0)

    #переворот на время оглушения

    def _named(self, name):
        """Подходит ли тело под селектор name=<name>. Ответ числом."""
        self._ensure_scoreboard()
        counter = f"{FOOD_HOLDER} {ident.PROBE_OBJECTIVE}"
        self.command(f"scoreboard players set {counter} 0")
        self.command(f"execute store result score {counter} "
                     f"if entity @e[tag={ident.ACTIVE_TAG},limit=1,name={name}]")
        return bool(self.read_score(counter))

    def pick_flip_name(self):
        """Подобрать рабочий формат CustomName и запомнить его."""
        self._flip_form = None
        name = FLIP_NAMES[0]
        for form in (f'"{name}"', f"'\"{name}\"'"):
            self.command(f"data modify entity {ident.BODY_SELECTOR} CustomName "
                         f"set value {form}")
            works = self._named(name)
            self.command(f"data remove entity {ident.BODY_SELECTOR} CustomName")
            if works:
                self._flip_form = form
                return True
        return False

    def set_flipped(self, on):
        """Повесить или снять имя-пасхалку, от которого тело рисуется вверх ногами."""
        if not getattr(self, "_flip_form", None):
            return
        if on:
            self.command(f"data modify entity {ident.BODY_SELECTOR} CustomName "
                         f"set value {self._flip_form}")
        else:
            self.command(f"data remove entity {ident.BODY_SELECTOR} CustomName")

    def move(self, body, mode=None):
        if mode is None:
            mode = "motion" if self.use_motion else "tp"
        if mode == "motion":
            self.command(
                f"data merge entity {ident.BODY_SELECTOR} "
                f"{{Motion:[{body.vx / 20.0:.4f}d,0.0d,{body.vz / 20.0:.4f}d]}}")
            self.command(f"data merge entity {ident.BODY_SELECTOR} "
                         f"{{Rotation:[{body.yaw:.1f}f,0.0f]}}")
        else:

            place = (round(body.x, 2), round(body.y, 2), round(body.z, 2),
                     round(body.yaw, 1))
            if place == self._last_place:
                return
            self._last_place = place
            self.body_at = place[:3]
            self.command(f"tp {ident.BODY_SELECTOR} {body.x:.3f} {body.y:.3f} {body.z:.3f} "
                         f"{body.yaw:.1f} 0")

    def starve(self, amount):
        """Снять здоровье голоданием. True, если сервер команду принял."""
        answer = self.command(
            f"damage {ident.BODY_SELECTOR} {amount:.1f} minecraft:starve") or ""
        return not any(word in answer for word in
                       ("Unknown", "Incorrect", "Expected", "HERE"))
