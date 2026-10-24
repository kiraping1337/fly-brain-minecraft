"""Прогоняет мост по БОЕВОЙ ветке, подсунув вместо RCON заглушку.

Запуск:  python tools/smoke_bridge.py            # игрок гоняет муху
         python tools/smoke_bridge.py --food     # еда под мухой и сбоку
         python tools/smoke_bridge.py --damage   # удары, гибель, возрождение
         python tools/smoke_bridge.py --water    # кругом вода
         python tools/smoke_bridge.py --flood    # еда, после укуса - вода
         python tools/smoke_bridge.py --wall     # каменная коробка: застревание
         python tools/smoke_bridge.py --steps    # террасы: удушье краем глаз
         ... --lag 6                             # ответ сервера 6 мс на команду

Сценарий --damage: игрок подходит и бьёт муху раз в несколько секунд -
слабо, потом сильно, - а под конец добивает. Проверяется весь путь урона:
чтение здоровья, толчок, оглушение, рана, гибель и пересоздание тела.

Сценарий --food: игрок далеко и не мешает. Под мухой арбуз, где бы она ни
села, а рядом с ней (по +X) тыква - пока их не укусят. Проверяется весь путь
еды: посадка, укус снизу и сбоку, раскладка кусков, поедание.

Сценарий --water: верхний слой - вода. Муха должна пытаться сесть, ни разу
не сесть и не остаться на воде.

Сценарий --flood: как --food, но после первого укуса верхний слой
становится водой. Муха стоит на месте укуса и должна сразу взлететь, а
потом не садиться под воду, хоть рядом и пахнут упавшие куски.

Во всех режимах прогон падает, если щупы от позиции тела не выполнились ни
разу; в режимах с существами - ещё и если ни разу не проверялась видимость.
Сервер не нужен. Ненулевой код возврата означает, что мост сломан.
"""

import io
import math
import re
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos"))

SENT: list[str] = []
GROUND_Y = 70.0

WORLD = {"body": [0.0, 70.0, 0.0], "player": (0.0, 16.0), "moved": None,
         "scores": {}, "ground": GROUND_Y,
         # тихий мир: игрок далеко, существ нет - муха может успокоиться
         "calm": False,
         # вода вместо верхнего слоя; flood - станет водой после первого укуса
         "water": False, "flood": False,
         # сценарий с едой
         "food": False, "melon": False, "side": None, "items": [], "bites": 0,
         "eaten": 0, "eaten_ids": [],
         # сценарий с уроном
         "damage": False, "health": 10.0, "hurt_until": 0.0, "hits": [],
         "deaths": 0, "alive": True, "watching": False, "flip": False,
         "flips": 0,
         # порча чтения здоровья
         "spoil_high": 0, "spoil_empty": 0, "spoiled": 0, "notes": 0,
         # Несколько мух (гонка, tools/smoke_race.py)
         "multi": False, "bodies": {}, "types": {}, "current": ""}

WORLD.update({"relief": None, "choke": False, "choked": [], "choke_at": -1.0,
              "healed": 0, "lag": 0.0})

STORED_RE = re.compile(r"store (?:result|success) score (\S+) \S+ (.*)$")
ABSOLUTE_RE = re.compile(r"positioned (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) ")
RELATIVE_RE = re.compile(r"positioned ~(-?[\d.]*) ~(-?[\d.]*) ~(-?[\d.]*) ")
IF_BLOCK_RE = re.compile(r"if block ~ ~ ~ (\S+)")

CHAIN_RE = re.compile(r"((?:at @e\[[^\]]*\] )?(?:positioned \S+ \S+ \S+ )?)"
                      r"(if|unless) block ~ ~ ~ (\S+)")
RUN_SCORE_RE = re.compile(r"run scoreboard players (add|set) (\S+) (\S+) (-?\d+)$")
STORE_TARGET_RE = re.compile(r"store (?:result|success) score (\S+) (\S+) ")
MULTI_ACTIVE_RE = re.compile(r"tag=flybrain_active_(\w+)")
MULTI_OWN_RE = re.compile(r"tag=flybrain_(?!active)(\w+)")
MULTI_SUMMON_RE = re.compile(r"summon (minecraft:\w+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) "
                             r".*\"flybrain_(\w+)\"")
MULTI_EXCLUDE_RE = re.compile(r"tag=!flybrain_(\w+)")

def score_key(holder, objective):
    """Ключ счёта: держатель, а у нескольких мух - держатель и счёт."""
    return f"{holder} {objective}" if WORLD["multi"] else holder


def stored_key(text):
    found = STORE_TARGET_RE.search(text)
    return score_key(found.group(1), found.group(2))


# Теги блоков, которые мост спрашивает и которые встречаются в этом мире.
TAGS = {"#minecraft:dirt": {"minecraft:dirt", "minecraft:grass_block"},
        # для проверок с произвольными блоками (tools/check_probes.py)
        "#minecraft:flowers": {"minecraft:poppy", "minecraft:dandelion"},
        "#minecraft:small_flowers": {"minecraft:poppy", "minecraft:dandelion"},
        "#minecraft:leaves": {"minecraft:oak_leaves"},
        "#minecraft:base_stone_overworld": {"minecraft:stone"}}


#мир из блоков

def probe_point(text, body):
    """Где проверяется блок: абсолютная точка, сдвиг от тела или само тело."""
    found = ABSOLUTE_RE.search(text)
    if found:
        return tuple(float(found.group(i)) for i in (1, 2, 3))
    found = RELATIVE_RE.search(text)
    if found:
        return tuple(body[i] + float(found.group(i + 1) or 0.0) for i in range(3))
    return tuple(body)


def cell_of(point):
    return tuple(math.floor(v) for v in point)


def top_layer():
    return math.floor(WORLD["ground"]) - 1


def column_top(x, z):
    """Верхний слой колонки: по рельефу, если он задан, иначе ровное поле."""
    relief = WORLD["relief"]
    return relief(x, z) if relief else top_layer()


def terraces(x, z):
    """Террасы --steps: плиты 5x5 на трёх уровнях. Соседние плиты расходятся
    на блок (ступенька, на неё встают) или на два (стенка)."""
    tx, tz = math.floor(x / 5), math.floor(z / 5)
    if tx in (-1, 0) and tz in (-1, 0):
        return 69                       # старт на ровном
    return 69 + ((tx * 73856093) ^ (tz * 19349663)) % 3


def choke_check(body):
    """Удушье, как у сервера: квадрат шириной 0.8 от тела на высоте глаз."""
    from mcworld import shape
    now = time.perf_counter()
    if now - WORLD["choke_at"] < 0.5:
        return
    ey = body[1] + shape.EYE_HEIGHT
    half = shape.EYE_BOX_HALF
    cells = {(math.floor(body[0] + dx), math.floor(ey), math.floor(body[2] + dz))
             for dx in (-half, half) for dz in (-half, half)}
    if all(block_at((c[0] + 0.5, c[1] + 0.5, c[2] + 0.5), body) == "minecraft:air"
           for c in cells):
        return
    WORLD["choke_at"] = now
    WORLD["choked"].append((now, tuple(body), STATE["now"]))
    if WORLD.get("choke_log") is not None:
        solid = [c for c in sorted(cells)
                 if block_at((c[0] + 0.5, c[1] + 0.5, c[2] + 0.5), body) != "minecraft:air"]
        WORLD["choke_log"].append({
            "t": round(now, 2), "state": STATE["now"], "body": [round(v, 2) for v in body],
            "own_top": column_top(math.floor(body[0]), math.floor(body[2])),
            "solid": [(c, column_top(c[0], c[2])) for c in solid],
            "trail": [(round(t - now, 2), [round(v, 2) for v in b]) for t, b in WORLD["trail"]]})
    WORLD["health"] -= 1.0
    if WORLD["health"] <= 2.0:
        WORLD["health"] = 10.0
        WORLD["healed"] += 1


def block_at(point, body, fixtures=False):
    """Какой блок в этой точке."""
    x, y, z = cell_of(point)
    if (x, y, z) in WORLD.get("blocks", {}):
        return WORLD["blocks"][(x, y, z)]
    top = column_top(x, z)
    bx, bz = math.floor(body[0]), math.floor(body[2])
    if WORLD["melon"] and (x, y, z) == (bx, top, bz):
        return "minecraft:melon"
    if (fixtures and WORLD["side"]
            and (x, y, z) == (math.floor(body[0] + 1.0), top + 1, bz)):
        return WORLD["side"]
    if y > top:
        return "minecraft:air"
    if y == top:
        return "minecraft:water" if WORLD["water"] else "minecraft:grass_block"
    return "minecraft:dirt"


def block_is(block, test):
    test = test.split("[")[0]
    if test.startswith("#"):
        return block in TAGS.get(test, ())
    return block == test


def check_block(text, body):
    links = CHAIN_RE.findall(text)
    if len(links) > 1 or (links and links[0][1] == "unless"):
        result, where_now = True, ""
        for where, kind, test in links:
            where_now = where or where_now
            hit = check_block(where_now + f"if block ~ ~ ~ {test}", body)
            result = result and (hit if kind == "if" else not hit)
        return result
    test = IF_BLOCK_RE.search(text).group(1)
    fixtures = RELATIVE_RE.search(text) is not None and not ABSOLUTE_RE.search(text)
    return block_is(block_at(probe_point(text, body), body, fixtures), test)


#предметы

def origin_of(text, body):
    """Откуда считается селектор: от тела, со сдвигом из positioned."""
    return probe_point(text, body)


def matching_items(text, origin):
    """Предметы, которые выбрал бы селектор @e[type=item,...] из команды."""
    radius = re.search(r"type=item,[^\]]*distance=\.\.([\d.]+)", text)
    wanted = re.search(r'Item:\{id:"([^"]+)"', text)
    excluded = re.search(r"tag=!([\w]+)", text)
    limit = re.search(r"type=item,[^\]]*limit=(\d+)", text)
    found = []
    for item in WORLD["items"]:
        distance = math.dist(item["pos"], origin)
        if radius and distance > float(radius.group(1)):
            continue
        if wanted and item["id"] != wanted.group(1):
            continue
        if excluded and excluded.group(1) in item["tags"]:
            continue
        found.append((distance, item))
    found = [item for _, item in sorted(found, key=lambda x: x[0])]
    return found[:int(limit.group(1))] if limit else found


def bite(text, body):
    """Укус блока: setblock air destroy. Блок пропадает, падает стопка."""
    if not check_block(text, body):
        return 0
    point = probe_point(text, body)
    block = block_at(point, body, fixtures=True)
    WORLD["bites"] += 1
    if block == "minecraft:melon":
        WORLD["melon"] = False
        WORLD["ground"] -= 1.0          # под мухой теперь яма
        drop = ("minecraft:melon_slice", 3)
        if WORLD["flood"]:
            WORLD["water"] = True       # --flood: земля стала водой
    else:
        WORLD["side"] = None
        drop = ("minecraft:pumpkin", 1)
    # Предмет лежит НА верхнем слое, а не в воздухе над ним.
    WORLD["items"].append({"id": drop[0], "count": drop[1], "tags": [],
                           "pos": (point[0], WORLD["ground"], point[2])})
    return 1


def select_entities(text, origin):
    """Ответ на "execute ... as @e[...] run data get entity @s Pos"."""
    selector = re.search(r"as @e\[([^\]]*)\]", text).group(1)
    radius = re.search(r"distance=\.\.([\d.]+)", selector)
    limit = re.search(r"limit=(\d+)", selector)
    if "type=minecraft:arrow" in selector:
        pool = list(WORLD.get("arrows", []))
    else:
        excluded = re.search(r"tag=!flybrain(_\w+)?\b", selector)
        pool = list(WORLD.get("mobs", []))
        for name, pos in sorted(WORLD["bodies"].items()):
            if excluded and (excluded.group(1) is None
                             or excluded.group(1) == f"_{name}"):
                continue
            pool.append((WORLD["types"][name].split(":")[1].capitalize(), pos))
    found = [(math.dist(origin, pos), name, pos) for name, pos in pool]
    if radius:
        found = [f for f in found if f[0] <= float(radius.group(1))]
    found.sort(key=lambda f: f[0])
    if limit:
        found = found[:int(limit.group(1))]
    return "".join(sightings_text(name, [pos]) for _, name, pos in found)


def stored_value(text, clause, body):
    """Что сервер записал бы в счёт по execute ... store."""
    if clause.startswith("if block"):
        if "setblock" in clause:
            return bite(text, body)
        return int(check_block(text, body))
    if clause.startswith("unless block"):
        return int(check_block(text, body))
    items = matching_items(clause, origin_of(text, body))
    if clause.startswith("if entity"):
        return len(items)
    if "data get entity" in clause:
        return items[0]["count"] if items else 0
    return 0


def sightings_text(name, points):
    """Ответ сервера по нескольким сущностям - склеен в одну строку."""
    return "".join(f"{name} has the following entity data: "
                   f"[{x:.3f}d, {y:.3f}d, {z:.3f}d]" for x, y, z in points)


class FakeConnection:
    """Отвечает на команды в том же формате, что настоящий сервер."""

    def __init__(self, *args, **kwargs):
        self.start = time.perf_counter()
        if WORLD["moved"] is None:
            WORLD["moved"] = self.start

    @property
    def body(self):
        if WORLD["multi"] and WORLD["current"] in WORLD["bodies"]:
            return WORLD["bodies"][WORLD["current"]]
        return WORLD["body"]

    @body.setter
    def body(self, value):
        if WORLD["multi"] and WORLD["current"] in WORLD["bodies"]:
            WORLD["bodies"][WORLD["current"]] = value
        else:
            WORLD["body"] = value

    @property
    def _player(self):
        return WORLD["player"]

    @_player.setter
    def _player(self, value):
        WORLD["player"] = value

    def connect(self):
        return self

    def disconnect(self):
        return None

    def command(self, text):
        SENT.append(text)
        if WORLD["lag"]:
            time.sleep(WORLD["lag"])
        if WORLD["multi"]:
            found = MULTI_ACTIVE_RE.search(text) or MULTI_OWN_RE.search(text)
            if found:
                WORLD["current"] = found.group(1)
            spawned = MULTI_SUMMON_RE.search(text)
            if spawned:
                name = spawned.group(5)
                WORLD["bodies"][name] = [float(spawned.group(i)) for i in (2, 3, 4)]
                WORLD["types"][name] = spawned.group(1)
                return ""

        moved = re.match(r"tp @e\[tag=flybrain\S* (-?[\d.]+) (-?[\d.]+) (-?[\d.]+)", text)
        if moved:
            self.body = [float(moved.group(i)) for i in (1, 2, 3)]
            if WORLD["choke"]:
                then, last = (WORLD.get("trail") or [(time.perf_counter(), self.body)])[-1]
                hop = math.hypot(self.body[0] - last[0], self.body[2] - last[2])
                if STATE["now"] == "пешком" and hop < 1.0:
                    WORLD["walked"] = WORLD.get("walked", 0.0) + hop
                    WORLD["walk_s"] = WORLD.get("walk_s", 0.0) + min(
                        time.perf_counter() - then, 0.5)
                WORLD["trail"] = (WORLD.get("trail") or [])[-11:] + [
                    (time.perf_counter(), list(self.body))]
                choke_check(self.body)

        if "scoreboard players get @p fly" in text:
            WORLD["scores"][stored_key(text)] = (
                WORLD["notes"] if "flyplay" in text else 0)
            return ""
        if text.startswith("data get entity") and text.endswith(" Health"):
            WORLD["watching"] = True
            if WORLD["spoil_high"] > 0:
                WORLD["spoil_high"] -= 1
                WORLD["spoiled"] += 1
                return "Bee has the following entity data: 11.2f"
            if WORLD["spoil_empty"] > 0:
                WORLD["spoil_empty"] -= 1
                WORLD["spoiled"] += 1
                return ""
            if not WORLD["alive"]:
                return "No entity was found"
            return f"Bee has the following entity data: {WORLD['health']:.1f}f"
        if "Health 100" in text:

            value = int(round(WORLD["health"] * 100)) if WORLD["alive"] else 0
            WORLD["scores"][stored_key(text)] = value
            return ""
        if "CustomName set value" in text:
            WORLD["flip"] = "'" not in text.split("set value ")[1]
            return ""
        if "data remove entity" in text and "CustomName" in text:
            WORLD["flip"] = False
            return ""
        if "name=Dinnerbone" in text and "store result" in text:
            if WORLD["flip"]:
                WORLD["flips"] += 1
            WORLD["scores"][stored_key(text)] = (
                1 if WORLD["flip"] else 0)
            return ""
        if text.startswith("kill @e[tag=flybrain]"):
            WORLD["alive"], WORLD["health"] = True, 10.0
            return ""

        stored = STORED_RE.search(text)
        if stored:
            clause = stored.group(2)
            selector = re.search(r"@e\[([^\]]*)\]", clause)
            if selector and len(re.findall(r"type=(?!!)", selector.group(1))) > 1:
                return "Expected end of input"
            WORLD["scores"][stored_key(text)] = stored_value(text, clause, self.body)
            return ""

        if "run kill @e[type=item" in text:
            items = matching_items(text, origin_of(text, self.body))
            if items:
                WORLD["items"].remove(items[0])
                if "positioned" not in text:
                    WORLD["eaten"] += 1
                    WORLD["eaten_ids"].append(items[0]["id"])
            return ""
        if "Item.count set value" in text:
            items = matching_items(text, origin_of(text, self.body))
            if items:
                items[0]["count"] = int(text.rsplit(" ", 1)[1])
                WORLD["eaten"] += 1
                WORLD["eaten_ids"].append(items[0]["id"])
            return ""
        if "run summon item" in text:
            origin = origin_of(text, self.body)
            motion = re.search(r"Motion:\[(-?[\d.]+)d,(-?[\d.]+)d,(-?[\d.]+)d\]", text)
            vx, vz = (float(motion.group(1)), float(motion.group(3))) if motion else (0, 0)
            WORLD["items"].append({
                "id": re.search(r'id:"([^"]+)"', text).group(1),
                "count": int(re.search(r"count:(\d+)", text).group(1)),
                "tags": re.findall(r'Tags:\["([^"]+)"\]', text),
                # разлёт: ломтик улетает примерно на 8 тиков своей скорости
                "pos": (origin[0] + vx * 8, WORLD["ground"], origin[2] + vz * 8),
            })
            return "Summoned new Melon Slice"

        if "if block ~ ~ ~ " in text or "unless block ~ ~ ~ " in text:
            if "#minecraft:replaceable" in text:
                return "Unknown block tag '#minecraft:replaceable'"
            passed = check_block(text, self.body)
            action = RUN_SCORE_RE.search(text)
            if action:
                if passed:
                    how, value = action.group(1), int(action.group(4))
                    holder = score_key(action.group(2), action.group(3))
                    before = WORLD["scores"].get(holder, 0) if how == "add" else 0
                    WORLD["scores"][holder] = before + value
                return ""
            return "Test passed" if passed else ""

        #скорборд
        if text.startswith("scoreboard players set"):
            parts = text.split()
            WORLD["scores"][score_key(parts[3], parts[4])] = int(parts[5])
            return ""
        if text.startswith("scoreboard players get"):
            parts = text.split()
            holder = parts[3]
            value = WORLD["scores"].get(score_key(parts[3], parts[4]), 0)
            return f"{holder} has {value} [{parts[4]}]"
        if text.startswith("gamerule "):
            parts = text.split()
            if not re.fullmatch(r"[a-z_]+", parts[1]):
                return ("Unknown or incomplete command, see below for error"
                        f"...{text}<--[HERE]")
            rules = WORLD.setdefault("rules", {})
            if len(parts) > 2:
                rules[parts[1]] = parts[2]
                return f"Gamerule {parts[1]} is now set to: {parts[2]}"
            return f"Gamerule {parts[1]} is currently set to: {rules.get(parts[1], 'true')}"
        if text.startswith(("gamerule", "tag", "kill", "summon", "tp",
                            "data merge", "execute at")) and "data get" not in text:
            return ""

        #сущности
        if "Rotation" in text:
            return "flybrain has the following entity data: [0.0f, 0.0f]"
        if "active_effects" in text or "ActiveEffects" in text:
            return "Found no elements matching active_effects"
        if "Pos" in text:
            if "as @e[type=item" in text:
                items = matching_items(text, origin_of(text, self.body))
                return sightings_text("Item", [i["pos"] for i in items])
            if WORLD["multi"] and "as @e[" in text:
                return select_entities(text, origin_of(text, self.body))
            if "type=minecraft:arrow" in text:
                return ""
            if "@p" in text and WORLD["calm"]:
                return "tester has the following entity data: [5000.0d, 70.0d, 5000.0d]"
            if "tag=!" in text and WORLD["calm"]:
                return ""
            if "@p" in text:
                now = time.perf_counter()
                step = 2.5 * max(0.0, now - WORLD["moved"])
                WORLD["moved"] = now
                chasing = int((now - self.start) / 12.0) % 2 == 0
                dx = self.body[0] - self._player[0]
                dz = self.body[2] - self._player[1]
                length = math.hypot(dx, dz) or 1.0
                if not chasing:
                    dx, dz, length = -dx, -dz, length
                    move = step
                    self._player = (self._player[0] + dx / length * move,
                                    self._player[1] + dz / length * move)
                elif length > 1.5:
                    move = min(step, length - 1.5)
                    self._player = (self._player[0] + dx / length * move,
                                    self._player[1] + dz / length * move)
                return (f"tester has the following entity data: "
                        f"[{self._player[0]:.3f}d, 70.0d, {self._player[1]:.3f}d]")
            if "tag=!" in text:
                return "Bat has the following entity data: [6.0d, 71.0d, 4.0d]"
            return (f"flybrain has the following entity data: "
                    f"[{self.body[0]:.3f}d, {self.body[1]:.3f}d, {self.body[2]:.3f}d]")
        if text.startswith("execute if entity"):
            if len(re.findall(r"type=(?!!)", text)) > 1:
                return "Expected end of input"
            return "Test passed. Count: 1"
        return ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def install_stub():
    module = types.ModuleType("mcrcon")
    module.MCRcon = FakeConnection
    sys.modules["mcrcon"] = module


def beat_the_fly():
    """Игрок бьёт муху: слабо, сильно, потом насмерть."""
    while not WORLD["watching"]:
        time.sleep(0.1)
    plan = [(3.0, 1.0), (9.0, 4.0), (15.0, 1.0), (22.0, 100.0)]
    start = time.perf_counter()
    for when, amount in plan:
        while time.perf_counter() - start < when:
            time.sleep(0.05)
        if not WORLD["alive"]:
            continue
        WORLD["health"] = max(0.0, WORLD["health"] - amount)
        WORLD["hurt_until"] = time.perf_counter() + 0.5
        WORLD["hits"].append(amount)
        if WORLD["health"] <= 0:
            WORLD["alive"] = False
            WORLD["deaths"] += 1


class Tee(io.TextIOBase):
    """Печатать как обычно и заодно запоминать: по выводу проверяем события."""
    def __init__(self, real):
        self.real, self.parts = real, []

    def write(self, text):
        self.parts.append(text)
        if text.startswith("\r[") or text.startswith("["):
            for word, state in (("идёт", "пешком"), ("полёт", "в полёте"),
                                ("сидит", "сидит")):
                if word in text:
                    STATE["now"] = state
                    break
            else:
                STATE["now"] = "прочее"
        if "ПОСАДКА #" in text:
            LANDED.append((WORLD["body"][1] - WORLD["ground"], WORLD["water"]))
        return self.real.write(text)

    def flush(self):
        self.real.flush()

    @property
    def text(self):
        return "".join(self.parts)


MODES = ("--food", "--damage", "--water", "--flood", "--wall", "--steps")
STATE = {"now": "прочее"}    # что делала муха по последней строке состояния
LANDED: list[tuple] = []     # (высота тела над верхним слоем, вода ли он)


def sys_seconds(argv, default=180.0):
    """--seconds из командной строки смоука (порог --steps считается от него)."""
    at = [i for i, a in enumerate(argv[:-1]) if a == "--seconds"]
    return float(argv[at[-1] + 1]) if at else default


def main():
    install_stub()
    started = time.perf_counter()
    import demos.minecraft_bridge as bridge

    argv = [a for a in sys.argv[1:] if a not in MODES]
    mode = {m: m in sys.argv[1:] for m in MODES}
    if "--lag" in argv:
        at = argv.index("--lag")
        WORLD["lag"] = float(argv[at + 1]) / 1000.0
        del argv[at:at + 2]
    if mode["--damage"]:
        WORLD["damage"] = True
        argv = ["--seconds", "40", "--respawn"] + argv
        threading.Thread(target=beat_the_fly, daemon=True).start()
    if mode["--food"] or mode["--flood"]:
        WORLD["food"] = WORLD["melon"] = WORLD["calm"] = True
        WORLD["side"] = "minecraft:pumpkin"
        WORLD["flood"] = mode["--flood"]
        argv = ["--seconds", "60", "--hunger", "0.8"] + argv
    if mode["--water"]:
        WORLD["water"] = WORLD["calm"] = True
        argv = ["--seconds", "60"] + argv
    if mode["--wall"]:
        WORLD["blocks"] = {(x, y, z): "minecraft:stone"
                           for x in range(-5, 6) for z in range(-5, 6)
                           if max(abs(x), abs(z)) == 5 for y in range(69, 80)}
        WORLD["player"] = (2.0, 2.0)
        argv = ["--seconds", "40", "--leash", "0"] + argv
    if mode["--steps"]:

        WORLD["relief"] = terraces
        WORLD["choke"] = WORLD["calm"] = True
        WORLD["choke_log"] = []
        if not WORLD["lag"]:
            WORLD["lag"] = 0.006
        argv = ["--seconds", "180", "--leash", "0", "--hunger", "1.0",
                "--starve", "0"] + argv
    sys.argv = ["minecraft_bridge.py", "--spawn", "--seconds", "4",
                "--seed", "42"] + argv

    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        bridge.main()
    finally:
        sys.stdout = tee.real
    log = tee.text

    elapsed = max(time.perf_counter() - started, 1e-6)
    print("\n" + "=" * 60)
    print(f"Команд отправлено: {len(SENT)} за {elapsed:.0f} с "
          f"({len(SENT) / elapsed:.0f}/с)")
    kinds: dict[str, int] = {}
    for command in SENT:
        head = re.split(r"\s", command.strip())[0] or "?"
        if head == "execute":
            head = "execute/if" if "if block" in command else "execute/data"
        kinds[head] = kinds.get(head, 0) + 1
    for head, count in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {head:16s} {count:5d}  ({count / elapsed:.0f}/с)")

    errors = []
    for required in ("gamerule send_command_feedback false",
                     "gamerule log_admin_commands false"):
        if not any(c.startswith(required) for c in SENT):
            errors.append(f"не отправлена команда {required!r}")
    body_probes = sum(1 for c in SENT
                      if c.startswith("execute at") and ABSOLUTE_RE.search(c)
                      and "if block" in c)
    sight_checks = sum(1 for c in SENT if "#sight" in c and "if block" in c)
    print(f"\nЩупов от позиции тела: {body_probes}, проверок видимости: {sight_checks}")
    if body_probes == 0:
        errors.append("щупы от позиции тела не выполнились ни разу")
    if not WORLD["calm"] and sight_checks == 0:
        errors.append("проверка видимости не выполнилась ни разу")

    landings = log.count("ПОСАДКА #")
    over_water = log.count("НАД ВОДОЙ")
    on_water = log.count("НА ВОДЕ")
    print(f"Посадок: {landings}, отказов сесть на воду: {over_water}, "
          f"взлётов с воды: {on_water}")
    if LANDED:
        print("  высота тела над верхним слоем при посадке: " + ", ".join(
            f"{h:+.2f}{' (вода)' if wet else ''}" for h, wet in LANDED))

    if mode["--damage"]:
        print(f"Урон: ударов нанесено {len(WORLD['hits'])} "
              f"{WORLD['hits']}, смертей {WORLD['deaths']}")
        if len(WORLD["hits"]) < 4 or not WORLD["deaths"]:
            errors.append("не прошли все удары или муха не погибла")
    if mode["--food"] or mode["--flood"]:
        crumbs = sum(1 for i in WORLD["items"] if "flybrain_crumb" in i["tags"])
        print(f"Еда: укусов {WORLD['bites']}, съедено {WORLD['eaten']}, "
              f"ломтиков осталось {crumbs}")
        eaten = ", ".join(f"{i.split(':')[1]} {WORLD['eaten_ids'].count(i)}"
                          for i in dict.fromkeys(WORLD["eaten_ids"]))
        print(f"Съедено по видам: {eaten or 'ничего'}")
    if mode["--food"]:
        if WORLD["bites"] < 2 or "minecraft:pumpkin" not in WORLD["eaten_ids"]:
            errors.append("муха не укусила оба блока или не съела тыкву")
    if mode["--flood"]:
        bitten = log.find("УКУС")
        if bitten < 0:
            errors.append("муха ни разу не укусила - вода так и не появилась")
        elif "НА ВОДЕ" not in log[bitten:]:
            errors.append("после укуса земля стала водой, а муха с неё не взлетела")
    sunk = [h for h, wet in LANDED if wet and h < 0.0]
    if sunk:
        errors.append(f"села под воду {len(sunk)} раз (тело ниже поверхности на "
                      + ", ".join(f"{-h:.2f}" for h in sunk) + " бл)")
    if mode["--wall"]:
        rescues = log.count("ЗАСТРЯЛА в блоке")
        print(f"Застреваний в стене: {rescues}")
        if rescues > 20:
            errors.append(f"застреваний {rescues} за 40 с - муха влетает в ту же стену")
    if mode["--steps"]:
        by_state: dict[str, int] = {}
        for _when, _where, state in WORLD["choked"]:
            by_state[state] = by_state.get(state, 0) + 1
        steps = re.findall(r"шагов: (\d+)", log)
        print(f"Удушье краем глаз (как на сервере): {len(WORLD['choked'])} "
              f"ударов - " + (", ".join(f"{k} {v}" for k, v in by_state.items()) or "нет"))
        print(f"  мост: 'УДАР в блоке' {log.count('в блоке: не оглушает')}, "
              f"'ЗАСТРЯЛА' {log.count('ЗАСТРЯЛА в блоке')}; "
              f"шагов {steps[-1] if steps else '?'}; лечений заглушки {WORLD['healed']}")
        walk_s = WORLD.get("walk_s", 0.0)
        print(f"  пешком пройдено {WORLD.get('walked', 0.0):.0f} бл за {walk_s:.0f} с "
              f"({WORLD.get('walked', 0.0) / max(walk_s, 1e-6):.2f} бл/с)")
        walked = int(steps[-1]) if steps else 0
        on_foot = by_state.get("пешком", 0)
        if walked < 150:
            errors.append(f"муха прошла {walked} шагов - ходьба по террасам не проверена")
        elif on_foot >= 5 * max(1.0, float(sys_seconds(sys.argv)) / 180.0):
            errors.append(f"на ходу задохнулась {on_foot} раз за {walked} шагов - "
                          f"входит краем глаз в ступеньки")
        dump = ROOT / "output" / "steps_chokes.json"
        dump.parent.mkdir(exist_ok=True)
        import json
        dump.write_text(json.dumps(WORLD["choke_log"], ensure_ascii=False, indent=1),
                        encoding="utf-8")
    if mode["--water"]:
        if landings:
            errors.append(f"муха села {landings} раз, а садиться некуда - кругом вода")
        if not over_water:
            errors.append("муха ни разу не пыталась сесть - отказ над водой не проверен")

    if errors:
        sys.exit("ОШИБКА: " + "; ".join(errors))
    print("\nбоевая ветка проходит целиком")


if __name__ == "__main__":
    main()
