"""Мухи с разными именами не трогают друг друга: проверка по командам.

Для каждого имени прогоняется сценарий мира на подставном сервере из
smoke_bridge: появление тела в заданной точке, захват, рельеф, существа,
запах, здоровье, укусы, переворот, голодание, перемещение, табло и надпись.
Все отправленные команды проверяются:

  - нет ни одного чужого имени (тега, счёта, надписи другой мухи);
  - нет команды, которая задевает ВСЕХ мух: селектора по общему тегу
    flybrain (он допустим только в списке тегов при появлении тела);
  - все проверки через счёт идут в свой счёт flybrain_ИМЯ, а не в общий;
  - чужая муха не исключена из опроса существ - иначе мухи не видят
    друг друга;
  - с --no-see-flies, наоборот, опрос существ исключает всех мух (общий тег
    flybrain) - и это единственное место, где общий тег допустим вне списка
    тегов при появлении: исключение из выборки никого не задевает.

Сервер не нужен. Запуск:
    python tools/check_isolation.py
"""

import copy
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import smoke_bridge as stub  # noqa: E402

stub.install_stub()
PRISTINE = copy.deepcopy(stub.WORLD)
CLOCK = [1000.0]
time.perf_counter = lambda: CLOCK[0]
time.time = lambda: CLOCK[0]
time.sleep = lambda s: CLOCK.__setitem__(0, CLOCK[0] + s)

import demos.minecraft_bridge as bridge  # noqa: E402
from flygame.state import Body  # noqa: E402
from mcworld import display  # noqa: E402

FLIES = (("a", "allay", (10.0, 70.0, 0.0, 90.0)),
         ("b", "bat", (10.0, 70.0, 4.0, 90.0)),
         ("c", "chicken", (10.0, 70.0, 8.0, 90.0)))
COMMON = re.compile(r"\bflybrain\b(?![_\w])")          # общий тег или счёт
TAGS_LIST = re.compile(r"Tags:\[[^\]]*\]")


def run(name, body_name, spawn_at, see_flies=True):
    stub.WORLD.clear()
    stub.WORLD.update(copy.deepcopy(PRISTINE))
    stub.WORLD.update(food=True, melon=True, side="minecraft:pumpkin", calm=False)
    stub.SENT.clear()
    bridge.apply_body(body_name)
    bridge.apply_name(name)
    w = bridge.MinecraftWorld()
    w.see_flies = see_flies
    w.spawn_at = spawn_at
    w.quiet()
    w.check_passable()
    w.spawn_body()
    w.locate_body()
    w.body_at = spawn_at[:3]
    for walking in (False, True):
        w.set_lookahead(30.0, 3.0, walking)
        w.probe_terrain()
    for _ in range(3):
        w.scan()
    w.probe_odor()
    w.health()
    w.try_feed()
    w.try_feed(False)
    w.set_flipped(True)
    w.set_flipped(False)
    w.starve(1.0)
    fly = Body()
    fly.x, fly.y, fly.z = spawn_at[:3]
    w.move(fly)
    display.board_show(w)
    display.board_set(w, 0.5)
    display.label_create(w, spawn_at[:3], 0.5)
    display.label_update(w, 0.6)
    display.label_follow(w, spawn_at[:3])
    display.label_remove(w)
    display.board_hide(w)
    w.close()
    return list(stub.SENT)


def problems(name, sent, others, see_flies=True):
    found = []
    own_tag = f"flybrain_{name}"
    hidden = own_tag if see_flies else "flybrain"     # кого исключает опрос существ
    for text in sent:
        bare = TAGS_LIST.sub("", text)
        if not see_flies:
            bare = bare.replace("tag=!flybrain,", "")
        for other in others:
            if re.search(rf"_{other}\b", text):
                found.append(f"чужое имя {other}: {text[:100]}")
        if COMMON.search(bare):
            found.append(f"задевает всех мух: {text[:100]}")
        if "scoreboard" in text or "store result score" in text or \
                "store success score" in text:
            for objective in re.findall(r"#\w+ (\S+)", text):
                if objective.startswith("flybrain") and objective != own_tag:
                    found.append(f"не свой счёт {objective}: {text[:100]}")
        if "tag=!" in text and "@e[" in text and "type=item" not in text:
            excluded = re.findall(r"tag=!([\w]+)", text)
            if excluded and excluded != [hidden]:
                found.append(f"из опроса исключены не те: {text[:100]}")
    if not any(f"tag=!{hidden}," in t for t in sent):
        found.append(f"опрос существ не исключает {hidden} (или не прошёл)")
    if not any(f'Tags:["flybrain","{own_tag}"]' in t for t in sent):
        found.append("тело создано без общего и личного тега")
    return found


def main():
    names = [n for n, _, _ in FLIES]
    bad = 0
    for (name, body_name, spawn_at), see in [(f, True) for f in FLIES] + [
            (FLIES[0], False)]:
        sent = run(name, body_name, spawn_at, see)
        found = problems(name, sent, [n for n in names if n != name], see)
        summon = next((t for t in sent if "summon minecraft:" in t), "")
        print(f"муха {name} ({body_name}){'' if see else ', без соседей'}: "
              f"команд {len(sent)}, "
              + ("чисто" if not found else f"ПРОБЛЕМ {len(found)}"))
        print(f"    появление: {summon[:110]}")
        for line in found[:8]:
            print("    " + line)
        bad += bool(found)
    print("ИЗОЛЯЦИЯ В ПОРЯДКЕ" if not bad else "ЕСТЬ ПЕРЕСЕЧЕНИЯ")
    sys.exit(bool(bad))


if __name__ == "__main__":
    main()
