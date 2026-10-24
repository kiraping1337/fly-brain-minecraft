"""Сценарии поиска еды с настоящим мозгом, без сервера. Фиксированные зёрна.

Запуск:
    python tools/field_scenarios.py                    # все сценарии
    python tools/field_scenarios.py memory --single    # одно место в памяти
    python tools/field_scenarios.py lost --no-search
    python tools/field_scenarios.py near --no-bilateral --trials 6
    python tools/field_scenarios.py trail --spacing 3 --start-hunger 0.35
    python tools/field_scenarios.py trail --no-meal-search --no-memory
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("mcrcon", types.SimpleNamespace(MCRcon=object))

import demos.minecraft_bridge as bridge  # noqa: E402
import flygame.constants as consts  # noqa: E402
import flygame.memory as fly_memory  # noqa: E402
import flygame.odor as fly_odor  # noqa: E402
from flygame.calibration import calibrate as real_calibrate  # noqa: E402
from flygame.mind import Mind  # noqa: E402
from flygame.motion import steer  # noqa: E402
from flygame.state import Body, Terrain  # noqa: E402
from mcworld.constants import ODOR_LIMIT, ODOR_RANGE, REACH  # noqa: E402
from mcworld.poller import FEED_ITEM_EVERY  # noqa: E402

GROUND = 64.0          # высота пола поля: тело стоит на ней
FOUND_RADIUS = 3.0     # блоков: съела ближе этого к цели - нашла цель
MELON = "minecraft:melon_slice"
STATES = ("flying", "landing", "perched", "walking")


#ключи --no-*: подмена в КАЖДОМ модуле, где имя живёт

def patch(name, value):
    """Подменить имя во всех модулях проекта, где лежит тот же объект."""
    holders = [module for module in list(sys.modules.values())
               if (getattr(module, "__name__", "") or "").startswith(
                   ("demos", "flybrain", "flygame")) and hasattr(module, name)]
    if not holders:
        return 0
    holders.sort(key=lambda module: not module.__name__.startswith("flygame"))
    original = getattr(holders[0], name)
    hits = 0
    for module in holders:
        if getattr(module, name, None) is original:
            setattr(module, name, value)
            hits += 1
    return hits


def apply_switches(args):
    applied = []
    if args.single:
        applied.append(("PLACES_MAX", patch("PLACES_MAX", 1)))
    if args.no_search:
        applied.append(("searching", patch("searching", lambda body: False)))
    if args.no_memory:
        applied.append(("path_integral_pull",
                        patch("path_integral_pull", lambda body: 0.0)))
    if args.no_bilateral:
        applied.append(("bilateral_odor",
                        patch("bilateral_odor", lambda body, sources: 0.0)))
    if args.no_meal_search:
        applied.append(("MEAL_SEARCH_S", patch("MEAL_SEARCH_S", 0.0)))
    elif args.meal_search is not None:
        applied.append((f"MEAL_SEARCH_S={args.meal_search:g}",
                        patch("MEAL_SEARCH_S", args.meal_search)))
    for name, hits in applied:
        if hits == 0:
            sys.exit(f"Ключ не сработал: {name} не найден ни в одном модуле")
    return [name for name, _ in applied]


#мозг и калибровки: те же функции, что в main моста

def calibrate(circuit_path, bridge_args):
    """Та же калибровка, что у моста: flygame.calibration.calibrate.

    Раньше здесь была своя копия кода main. С шага 3б калибровка - одна
    функция, и сценарии гоняют ровно её; числа в calib сохраняются в JSON
    и сверяются с точностью до 9 знаков.
    """
    bridge_args.circuit = circuit_path
    cal = real_calibrate(bridge_args)
    calib = {"gain": cal.brain.gain, "noise_hz": cal.base_noise,
             "baseline_hz": cal.decoder.baseline, "odor_rest": cal.odor_rest,
             "idle": cal.idle_arousal, "perch": cal.perch_thrust,
             "perch_hungry": cal.perch_hungry, "takeoff": cal.takeoff_thrust,
             "hungry_extra": cal.hungry_extra,
             "lateral_baseline": cal.decoder.lateral_baseline}
    return cal, calib


#сценарии

def ring(rng, distance):
    """Точка на заданном расстоянии в случайном направлении."""
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return distance * math.sin(angle), distance * math.cos(angle)


def pile(x, z, count, spread, rng):
    return [[x + rng.normal(0.0, spread), GROUND, z + rng.normal(0.0, spread), MELON]
            for _ in range(count)]


def walk_memory(body, frm, to, step=0.5):
    """Перенести муху из frm в to, ведя вектор памяти тем же правилом моста."""
    dx, dz = to[0] - frm[0], to[1] - frm[1]
    n = max(1, int(math.hypot(dx, dz) / step))
    for _ in range(n):
        fly_memory.update_path_integral(body, dx / n, dz / n, 0.0)


SCENARIOS = {}
TRAIL = {"spacing": 3.0, "length": 30.0}
START_HUNGER = [None]       # --start-hunger: голод на старте для всех сценариев


def scenario(name, seconds, hunger, about):
    def wrap(build):
        SCENARIOS[name] = {"build": build, "seconds": seconds,
                           "hunger": hunger, "about": about}
        return build
    return wrap


@scenario("near", 60.0, 0.8, "куча арбуза в 10 блоках, в пределах чутья")
def build_near(body, rng):
    x, z = ring(rng, 10.0)
    return pile(x, z, 6, 0.4, rng), (x, z)


@scenario("last", 60.0, 0.9,
          "доесть остатки: стоит на ломтике, ещё 5 раскатились на 1.5-4 блока")
def build_last(body, rng):
    items = [[body.x, GROUND, body.z, MELON]]
    for _ in range(5):
        x, z = ring(rng, rng.uniform(1.5, 4.0))
        items.append([body.x + x, GROUND, body.z + z, MELON])
    return items, (body.x, body.z)


@scenario("trap", 90.0, 0.9,
          "ловушка ровного запаха: помнит кучу в 26 блоках, запах от неё слабый")
def build_trap(body, rng):
    x, z = ring(rng, 26.0)
    body.x, body.z = x, z                 # поела у кучи...
    fly_memory.remember_meal(body)
    walk_memory(body, (x, z), (0.0, 0.0))  # ...и ушла на старт
    body.x = body.z = 0.0
    return pile(x, z, 6, 0.4, rng), (x, z)


@scenario("memory", 90.0, 0.9,
          "два места: здесь пусто, второе в 34 блоках, за пределом чутья")
def build_memory(body, rng):
    bx, bz = ring(rng, 34.0)
    body.x, body.z = bx, bz               # ела на дальнем месте...
    fly_memory.remember_meal(body)
    walk_memory(body, (bx, bz), (0.0, 0.0))
    body.x = body.z = 0.0
    fly_memory.remember_meal(body)            # ...потом здесь, и здесь всё съела
    return pile(bx, bz, 6, 0.4, rng), (bx, bz)


@scenario("lost", 60.0, 0.9,
          "без запаха и без памяти; еда в 40 блоках, за пределом чутья")
def build_lost(body, rng):
    x, z = ring(rng, 40.0)
    return pile(x, z, 6, 0.4, rng), (x, z)


@scenario("trail", 150.0, 0.35,
          "дорожка: ломтики в линию до финиша, муха стоит у начала")
def build_trail(body, rng):
    angle = rng.uniform(0.0, 2.0 * math.pi)
    ux, uz = math.sin(angle), math.cos(angle)
    count = int(TRAIL["length"] / TRAIL["spacing"])
    items = [[ux * TRAIL["spacing"] * k, GROUND, uz * TRAIL["spacing"] * k, MELON]
             for k in range(1, count + 1)]
    last = items[-1]
    return items, (last[0], last[2])


#мир и цикл

def terrain_here(body):
    """Ровное поле: снимок свежий на каждом такте и снят по курсу движения."""
    height = body.y - GROUND
    return Terrain(above_ground=height > 2.5,
                          clear_below=height > 0.02,
                          probe_yaw=body.travel_yaw,
                          at=body.position)


def odor_list(body, items):
    """Как probe_odor: ближайшие куски каждого вида в радиусе чутья."""
    found = []
    for kind in consts.FOOD_ITEMS:
        near = [(math.dist(body.position, it[:3]), tuple(it[:3]))
                for it in items if it[3] == kind]
        near = sorted(p for p in near if p[0] <= ODOR_RANGE)
        found += [p for _, p in near[:ODOR_LIMIT]]
    return found


def bite(body, items):
    """Как try_feed для предметов: ближайший кусок в пределах REACH."""
    for kind in consts.FOOD_ITEMS:
        near = [it for it in items if it[3] == kind
                and math.dist(body.position, it[:3]) <= REACH]
        if near:
            piece = min(near, key=lambda it: math.dist(body.position, it[:3]))
            items.remove(piece)
            return kind
    return None


def run_trial(name, seed, cal, calib, bridge_args, seconds):
    """Один прогон. decoder - откалиброванный образец; он не трогается."""
    spec = SCENARIOS[name]
    brain, odor = cal.brain, cal.odor
    decoder = copy.deepcopy(cal.decoder)
    mind = Mind(brain, decoder, cal.encoder, cal.wind, odor)
    dt = 1.0 / consts.CONTROL_HZ
    np.random.seed(seed)                       # снос вектора памяти
    brain.rng = np.random.default_rng(seed)    # шум мозга
    layout = np.random.default_rng(seed + 9000)

    body = Body()
    body.x, body.y, body.z = 0.0, GROUND, 0.0
    body.yaw = body.travel_yaw = float(layout.uniform(0.0, 360.0))
    body.odor_rest = calib["odor_rest"]
    body.hunger = spec["hunger"] if START_HUNGER[0] is None else START_HUNGER[0]
    body.visual_mask = brain.circuit.visual_mask
    items, target = spec["build"](body, layout)
    reach = math.hypot(*target) or 1.0
    along = 0.0
    arrived = None
    home = (0.0, GROUND, 0.0)

    brain.reset()
    brain.spontaneous_hz = calib["noise_hz"]

    state, calm_since, eating_until = "perched", 0.0, 0.0
    sources, pending, feed_now = [], None, False
    meals, cells, far = [], set(), 0.0
    flying_ticks = 0
    print_hash = hashlib.sha1()
    ticks = int(seconds / dt)
    for tick in range(ticks):
        now = tick * dt
        pace = consts.FLIGHT_HUNGER if state in ("flying", "landing") else 1.0
        body.hunger = min(1.0, body.hunger + pace * dt
                          / (bridge_args.hunger_minutes * 60.0))
        if pending is not None:
            eating_until = now + consts.EAT_HOLD_S
            body.hunger = max(0.0, body.hunger - consts.FOOD_ITEMS.get(pending, 0.1))
            body.meals += 1
            body.meal_search = consts.MEAL_SEARCH_S
            fly_memory.remember_meal(body)
            meals.append((round(now, 2), round(body.x, 2), round(body.z, 2)))
            pending = None

        urge = body.hunger ** bridge_args.hunger_curve
        brain.spontaneous_hz = calib["noise_hz"] + calib["hungry_extra"] * urge

        if tick % consts.ODOR_EVERY == 0:
            sources = odor_list(body, items)
        concentration = fly_odor.odor_concentration(body, sources)
        rates, _ = mind.sense([], 1.0)
        rates = mind.smell(rates, concentration)
        body.odor_side = fly_odor.bilateral_odor(body, sources)
        body.touching = fly_odor.touching_food(body, sources)
        feed_now = body.touching and body.hunger >= consts.HUNGRY

        motor = mind.step(rates, dt * 1000.0)
        level = mind.odor_level()
        if body.odor_level is None:
            body.odor_level = level
        if body.odor_slow == 0.0:
            body.odor_slow = level
        body.odor_level += (level - body.odor_level) * (
            1.0 - math.exp(-dt / consts.ODOR_SMOOTH_S))
        body.odor_slow += (level - body.odor_slow) * (
            1.0 - math.exp(-dt / consts.ODOR_SLOW_S))
        mind.adapt(motor, dt)

        terrain = terrain_here(body)
        tau = (consts.TAU_AROUSAL_RISE if motor.thrust > body.arousal
               else consts.TAU_AROUSAL_FALL)
        body.arousal += (motor.thrust - body.arousal) * min(1.0, dt / tau)
        airborne = (motor.climb > bridge_args.takeoff_climb
                    or body.arousal > calib["takeoff"])
        if state == "flying":
            land_at = calib["perch"] + (calib["perch_hungry"] - calib["perch"]) * urge
            over_food = (body.hunger >= consts.HUNGRY
                         and motor.retreat <= consts.ODOR_DIVE_RETREAT
                         and body.odor_level > body.odor_rest * consts.LAND_ON_ODOR)
            if over_food:
                state = "landing"
            elif body.arousal >= land_at:
                calm_since = now
            elif now - calm_since > consts.PERCH_AFTER_S:
                state = "landing"
        elif airborne:
            state, calm_since = "flying", now
        elif state in ("perched", "walking"):
            if body.arousal >= calib["perch"]:
                calm_since = now
                if state == "perched":
                    state = "walking"
            elif now - calm_since > consts.PERCH_AFTER_S:
                state = "perched"
        elif state == "landing" and not terrain.clear_below:
            state = "perched"

        holding = ((now < eating_until
                    or (body.touching and body.hunger >= consts.HUNGRY))
                   and state in ("perched", "walking"))
        steer(body, motor, dt, home, bridge_args, terrain, state,
                     hold=holding)
        body.y = max(body.y, GROUND)
        if (state in ("perched", "walking") and body.hunger >= consts.HUNGRY
                and (feed_now or tick % (FEED_ITEM_EVERY * 2) == 0)):
            pending = bite(body, items)

        flying_ticks += state in ("flying", "landing")
        along = max(along, (body.x * target[0] + body.z * target[1]) / reach)
        if arrived is None and math.hypot(body.x - target[0], body.z - target[1]) <= 1.5:
            arrived = round(now, 1)
        cells.add((math.floor(body.x), math.floor(body.z)))
        far = max(far, math.hypot(body.x, body.z))
        print_hash.update(struct.pack(
            "<6dB", body.x, body.y, body.z, body.yaw, body.hunger,
            body.arousal, STATES.index(state)))

    reached = [m for m in meals
               if math.hypot(m[1] - target[0], m[2] - target[1]) <= FOUND_RADIUS]
    return {"seed": seed, "found": bool(reached),
            "found_s": reached[0][0] if reached else None,
            "meals": len(meals), "far": round(far, 1), "cells": len(cells),
            "along": round(along, 1), "arrived": arrived,
            "flying": round(flying_ticks / ticks, 3),
            "hunger_end": round(body.hunger, 3), "saccades": body.saccades,
            "fingerprint": print_hash.hexdigest()[:16]}


#вывод и сравнение

def report(name, rows):
    found = [r for r in rows if r["found"]]
    times = [r["found_s"] for r in found]
    print(f"  {name:7s} нашла {len(found)}/{len(rows)}"
          + (f" (в среднем за {np.mean(times):.0f} с)" if times else "")
          + f", кусков {np.mean([r['meals'] for r in rows]):.1f}"
          f", ушла до {np.mean([r['far'] for r in rows]):.0f} бл"
          f", к цели до {np.mean([r['along'] for r in rows]):.0f} бл"
          f", клеток {np.mean([r['cells'] for r in rows]):.0f}"
          f", в воздухе {100 * np.mean([r['flying'] for r in rows]):.0f}%")


def compare(saved, current):
    """Совпадают ли калибровки и отпечатки. Ключ прогона - сценарий,
    ключи --no-*, длительность и зерно."""
    bad = 0
    for field, value in current["calibration"].items():
        old = saved["calibration"].get(field)
        if old != value:
            bad += 1
            print(f"  КАЛИБРОВКА {field}: было {old}, стало {value}")
    old_runs = {run["key"]: run for run in saved["runs"]}
    checked = 0
    for run in current["runs"]:
        old = old_runs.get(run["key"])
        if old is None:
            continue
        checked += 1
        if old["fingerprint"] != run["fingerprint"]:
            bad += 1
            print(f"  РАСХОДИТСЯ {run['key']}: было {old['fingerprint']}, "
                  f"стало {run['fingerprint']}")
    print(f"\nСверено прогонов: {checked}, расхождений: {bad}")
    if checked == 0:
        print("  Сверять нечего: в сохранённом файле нет прогонов с теми же "
              "сценариями, ключами, длительностью и зёрнами")
    return bad == 0 and checked > 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*",
                   help="какие сценарии: " + ", ".join(SCENARIOS) + "; без имён - все")
    p.add_argument("--trials", type=int, default=4, help="прогонов на сценарий")
    p.add_argument("--seconds", type=float, default=None,
                   help="длительность прогона; по умолчанию своя у сценария")
    p.add_argument("--seed0", type=int, default=100, help="первое зерно")
    p.add_argument("--quick", action="store_true",
                   help="для сверки при рефакторинге: 1 прогон по 20 с")
    p.add_argument("--single", action="store_true",
                   help="помнить одно кормное место (PLACES_MAX = 1)")
    p.add_argument("--no-search", action="store_true",
                   help="без расширяющегося поиска")
    p.add_argument("--no-memory", action="store_true",
                   help="память о местах не тянет")
    p.add_argument("--no-bilateral", action="store_true",
                   help="без сравнения двух антенн")
    p.add_argument("--no-meal-search", action="store_true",
                   help="без локального поиска после укуса")
    p.add_argument("--meal-search", type=float, default=None, metavar="С",
                   help="сколько секунд длится локальный поиск после укуса")
    p.add_argument("--start-hunger", type=float, default=None,
                   help="голод на старте вместо своего у сценария")
    p.add_argument("--hunger-minutes", type=float, default=None,
                   help="как у моста: за сколько минут сидя голод растёт с нуля "
                        "до единицы; по умолчанию - мостовое")
    p.add_argument("--spacing", type=float, default=TRAIL["spacing"],
                   help="trail: шаг между ломтиками, блоков")
    p.add_argument("--length", type=float, default=TRAIL["length"],
                   help="trail: длина дорожки до финиша, блоков")
    p.add_argument("--circuit", default=str(ROOT / "data" / "circuit.npz"))
    p.add_argument("--save", default=None, help="записать результат в JSON")
    p.add_argument("--compare", default=None,
                   help="сверить отпечатки с прошлым JSON; код выхода 1 при "
                        "расхождении")
    return p.parse_args()


def main():
    args = parse_args()
    unknown = [n for n in args.names if n not in SCENARIOS]
    if unknown:
        sys.exit(f"Нет таких сценариев: {', '.join(unknown)}. "
                 f"Есть: {', '.join(SCENARIOS)}")
    names = args.names or list(SCENARIOS)
    trials = 1 if args.quick else args.trials
    switches = apply_switches(args)
    TRAIL.update(spacing=args.spacing, length=args.length)
    START_HUNGER[0] = args.start_hunger
    if args.start_hunger is not None:
        switches.append(f"голод{args.start_hunger:g}")
    if "trail" in names:
        switches.append(f"шаг{args.spacing:g}x{args.length:g}")

    argv, sys.argv = sys.argv, ["minecraft_bridge.py"]
    try:
        bridge_args = bridge.parse_args()
    finally:
        sys.argv = argv
    if args.hunger_minutes is not None:
        bridge_args.hunger_minutes = args.hunger_minutes
        switches.append(f"голод-мин{args.hunger_minutes:g}")

    started = time.perf_counter()
    print("Калибровка...", end="", flush=True)
    cal, calib = calibrate(args.circuit, bridge_args)
    print(f"\rКалибровка: GAIN {calib['gain']:.4f}, шум {calib['noise_hz']:.1f} Гц, "
          f"покой {calib['idle']:.2f}, посадка ниже {calib['perch']:.2f}, "
          f"голодной +{calib['hungry_extra']:.2f} Гц "
          f"({time.perf_counter() - started:.0f} с)")
    if switches:
        print("Выключено: " + ", ".join(switches))

    runs = []
    for name in names:
        spec = SCENARIOS[name]
        seconds = 20.0 if args.quick else (args.seconds or spec["seconds"])
        print(f"\n{name}: {spec['about']}, {seconds:.0f} с, голод {spec['hunger']}")
        rows = []
        for trial in range(trials):
            seed = args.seed0 + trial
            t0 = time.perf_counter()
            row = run_trial(name, seed, cal, calib,
                            bridge_args, seconds)
            row["key"] = "|".join([name, ",".join(switches) or "-",
                                   f"{seconds:g}", str(seed)])
            rows.append(row)
            print(f"  зерно {seed}: "
                  + (f"нашла за {row['found_s']:.0f} с" if row["found"]
                     else "не нашла")
                  + f", кусков {row['meals']}, ушла до {row['far']} бл, "
                  f"к цели {row['along']} бл"
                  + (f" (у цели на {row['arrived']:.0f} с)" if row["arrived"] is not None
                     else "") + ", "
                  f"клеток {row['cells']}, отпечаток {row['fingerprint']} "
                  f"({time.perf_counter() - t0:.0f} с)")
        runs += rows
        report(name, rows)

    result = {"calibration": {k: round(float(v), 9) for k, v in calib.items()},
              "runs": runs}
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(f"\nЗаписано: {args.save}")
    if args.compare:
        saved = json.loads(Path(args.compare).read_text())
        if not compare(saved, result):
            sys.exit(1)


if __name__ == "__main__":
    main()
