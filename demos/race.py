"""Гонка мух: несколько мух в одном мире, каждая со своим мозгом.

Каждая муха - отдельный процесс demos/minecraft_bridge.py со своим именем,
телом и зерном шума: так мозги считаются на разных ядрах. С сервером
говорит только этот процесс - шлюз (mcworld/gateway.py): у RCON один буфер
ответов на все соединения, и мухи со своими соединениями забирали бы ответы
друг друга. Мухи видят друг друга как обычных существ.

То, что для всех мух одинаково - игрок, существа, летящие стрелы, куски
еды, - шлюз спрашивает один раз на всех (общий опрос, mcworld/shared.py),
а каждая муха отбирает из ответа своих ближайших. --no-shared - каждая
муха спрашивает сама, как до шага 2б. --no-see-flies - мухи не видят друг
друга (мобы и игрок остаются). --own-food - у каждой мухи свой вид еды
(по порядку --fly: арбуз, яблоко, сладкие ягоды, светящиеся ягоды, тыква):
она чует и ест только его, так что на общей дорожке не съедает куски
соперниц; насыщает свой кусок у всех одинаково. Одиночную муху это не трогает.

Запуск:
    python demos/race.py --fly a:bee --fly b:chicken --fly c:bat
    python demos/race.py --fly a:allay:10,70,0,90 --fly b:bat:10,70,4,90 -- --hunger 0.6

--fly ИМЯ:ТЕЛО[:X,Y,Z,YAW] - муха; без координат мухи встают в ряд в 4
блоках перед игроком, через --gap блоков. YAW - как в игре по F3.
Всё после "--" передаётся каждому мосту как есть (--hunger, --ignore,
--leash и т. д.).

Вывод мух пишется в output/race_ИМЯ.log; здесь печатаются события
(посадки, еда, удары, гибель) с именем мухи, раз в 10 с - нагрузка на
сервер и последняя строка состояния каждой мухи. Ctrl+C останавливает всех:
мухи доделывают своё (табло, надпись) через шлюз, потом шлюз закрывается.
"""

import argparse
import math
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flygame.parsing import parse_numbers  # noqa: E402
from mcworld.constants import RCON_HOST, RCON_PASSWORD, RCON_PORT  # noqa: E402
from mcworld.display import BOARD_OBJECTIVE  # noqa: E402
from mcworld.gateway import KEY_ENV, GatewayServer  # noqa: E402
from mcworld.ident import NAME_RE  # noqa: E402
from mcworld.shape import BODIES  # noqa: E402
from mcworld.shared import SharedSight  # noqa: E402
from mcworld.rcon import open_rcon  # noqa: E402
from flygame.constants import FOOD_NAMES  # noqa: E402

# Своя еда (--own-food): по порядку --fly.
RACE_FOODS = ("minecraft:melon_slice", "minecraft:apple", "minecraft:sweet_berries",
              "minecraft:glow_berries", "minecraft:pumpkin")

BRIDGE = ROOT / "demos" / "minecraft_bridge.py"
REPORT_EVERY_S = 10.0
STOP_WAIT_S = 20.0        # сколько ждать, пока мухи доделают своё после Ctrl+C

# Строки запуска моста
QUIET = ("Схема:", "возбуждающих", "типы:", "Калибровка", "Механосенсорика",
         "Фоновый шум", "Фон на выходе", "Возбуждение в покое", "Голод: у голодной",
         "Подключаюсь", "Вывод команд", "Связь есть", "Тел с тегом", "Оглушённая",
         "Проходимые блоки", "Снаряды:", "Голод пишется", "в OBS:", "Подбираю")
STATUS_RE = re.compile(r"\[\s*\d+\.\d+s\]")


def parse_args():
    if "--" in sys.argv:
        cut = sys.argv.index("--")
        own, extra = sys.argv[1:cut], sys.argv[cut + 1:]
    else:
        own, extra = sys.argv[1:], []
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fly", action="append", required=True, metavar="ИМЯ:ТЕЛО[:X,Y,Z,YAW]",
                   help="муха гонки; повторить для каждой")
    p.add_argument("--gap", type=float, default=3.0,
                   help="расстояние между мухами в ряду перед игроком, блоков")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="остановить гонку через N секунд (0 = до Ctrl+C)")
    p.add_argument("--no-shared", action="store_true",
                   help="без общего опроса: каждая муха сама спрашивает игрока, "
                        "существ, стрелы и еду; для сравнения нагрузки")
    p.add_argument("--own-food", action="store_true",
                   help="у каждой мухи свой вид еды; на дорожку класть все виды")
    p.add_argument("--no-see-flies", action="store_true",
                   help="мухи не видят друг друга: ни глазами, ни потоком воздуха")
    p.add_argument("--no-priority", action="store_true",
                   help="телепорты в общей очереди с опросами, по порядку "
                        "прихода; для сравнения, насколько очередь дёргает мух")
    args = p.parse_args(own)
    args.no_see_flies = args.no_see_flies or "--no-see-flies" in extra
    if args.no_see_flies and "--no-see-flies" not in extra:
        extra = extra + ["--no-see-flies"]
    args.extra = extra
    flies = []
    for spec in args.fly:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            p.error(f"--fly {spec}: нужно ИМЯ:ТЕЛО или ИМЯ:ТЕЛО:X,Y,Z,YAW")
        name, body = parts[0], parts[1]
        if not NAME_RE.fullmatch(name):
            p.error(f"--fly {spec}: имя - a-z, 0-9 и _, до 12 знаков")
        if body not in BODIES:
            p.error(f"--fly {spec}: тело одно из {', '.join(sorted(BODIES))}")
        at = None
        if len(parts) == 3:
            try:
                at = tuple(float(v) for v in parts[2].split(","))
            except ValueError:
                at = ()
            if len(at) != 4:
                p.error(f"--fly {spec}: координаты - четыре числа X,Y,Z,YAW")
        flies.append({"name": name, "body": body, "at": at})
    names = [f["name"] for f in flies]
    if len(set(names)) != len(names):
        p.error("имена мух повторяются")
    if args.own_food:
        if len(flies) > len(RACE_FOODS):
            p.error(f"--own-food: видов еды {len(RACE_FOODS)}, а мух {len(flies)}")
        for fly, food in zip(flies, RACE_FOODS):
            fly["food"] = food
    args.flies = flies
    return args


def place_in_row(gate, flies, gap):
    """Мухам без координат - места в ряд в 4 блоках перед игроком."""
    if all(f["at"] for f in flies):
        return
    pos = parse_numbers(gate.command("data get entity @p Pos") or "")
    rot = parse_numbers(gate.command("data get entity @p Rotation") or "")
    if not pos or not rot:
        sys.exit("Игрока в мире нет, а у мух не заданы координаты: зайдите в игру "
                 "или задайте --fly ИМЯ:ТЕЛО:X,Y,Z,YAW")
    yaw = rot[0]
    rad = math.radians(yaw)
    ahead_x, ahead_z = -math.sin(rad), math.cos(rad)
    right_x, right_z = -math.cos(rad), -math.sin(rad)
    free = [f for f in flies if not f["at"]]
    for i, fly in enumerate(free):
        side = (i - (len(free) - 1) / 2.0) * gap
        fly["at"] = (pos[0] + ahead_x * 4.0 + right_x * side, pos[1] + 1.0,
                     pos[2] + ahead_z * 4.0 + right_z * side, yaw)


def launch(fly, address, key, args):
    log_path = ROOT / "output" / f"race_{fly['name']}.log"
    log_path.parent.mkdir(exist_ok=True)
    x, y, z, yaw = fly["at"]
    command = [sys.executable, str(BRIDGE), "--name", fly["name"], "--body", fly["body"],
               "--spawn", "--spawn-at", f"{x:.2f}", f"{y:.2f}", f"{z:.2f}", f"{yaw:.1f}",
               "--gateway", f"{address[0]}:{address[1]}"]
    if args.seconds:
        command += ["--seconds", f"{args.seconds:g}"]
    if fly.get("food"):
        command += ["--food-only", fly["food"]]
    command += args.extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env[KEY_ENV] = key
    log = open(log_path, "wb")
    process = subprocess.Popen(command, cwd=str(ROOT), stdout=log,
                               stderr=subprocess.STDOUT, env=env)
    return {"fly": fly, "process": process, "log": log, "path": log_path,
            "offset": 0, "tail": "", "status": "", "printed_exit": False}


def follow(runner):
    """Новые строки лога: события печатаем, строку состояния запоминаем."""
    with open(runner["path"], "rb") as f:
        f.seek(runner["offset"])
        data = f.read()
    runner["offset"] += len(data)
    text = runner["tail"] + data.decode("utf-8", errors="replace")
    pieces = text.split("\n")
    runner["tail"] = pieces.pop()             # недописанная строка
    for piece in pieces + [runner["tail"]]:
        for part in piece.split("\r"):
            if STATUS_RE.match(part):
                runner["status"] = part.rstrip()
    for piece in pieces:
        for part in piece.split("\r"):
            line = part.strip()
            if not line or STATUS_RE.match(part) or line.startswith(QUIET):
                continue
            print(f"{runner['fly']['name']:>6} | {line}")


TICK_QUERY = {"on": True}


def server_tick(gate):
    """Сколько миллисекунд сервер тратит на тик - /tick query, одной строкой."""
    if not TICK_QUERY["on"]:
        return ""
    try:
        answer = gate.command("tick query") or ""
    except Exception:
        answer = ""
    if not answer or "nknown" in answer or "ncorrect" in answer:
        TICK_QUERY["on"] = False
        return ""
    return " ".join(answer.split())[:140]


def report(gate, runners, started, since, sight=None):
    """Нагрузка на сервер с прошлой сводки и последняя строка каждой мухи."""
    per_fly, total, busy = gate.snapshot()
    waits = gate.take_waits()
    tick = server_tick(gate)
    now = time.perf_counter()
    span = max(now - since[0], 1e-6)
    rate = (total - since[1]) / span
    load = 100.0 * (busy - since[2]) / span
    snapshots = sight.seq if sight is not None else 0
    queries = sight.creature_queries if sight is not None else 0
    shots = (snapshots - since[3]) / span
    per_shot = (queries - since[4]) / max(snapshots - since[3], 1)
    since[:] = [now, total, busy, snapshots, queries]
    print(f"\n[{now - started:6.1f}s] ШЛЮЗ: {rate:.0f} команд/с, сервер занят "
          f"{load:.0f}% времени; всего команд: "
          + ", ".join(f"{name} {count}" for name, (count, _) in sorted(per_fly.items())))
    if sight is not None:
        print(f"          общий опрос: {shots:.1f} снимков/с, запросов существ "
              f"{per_shot:.1f} на снимок"
              + (f", ошибок {len(sight.errors)}: {sight.errors[-1]}" if sight.errors else ""))
    stalls = gate.take_stalls()
    if stalls:
        print(f"          ЗАМИРАНИЯ СЕРВЕРА ({len(stalls)}): " + ", ".join(
            f"{clock} {spent:.2f} с" for clock, spent, _ in stalls[:8])
            + (" ..." if len(stalls) > 8 else ""))
    print("          ожидание в шлюзе: " + "; ".join(
        f"{kind} {avg * 1000:.1f} мс, макс {top * 1000:.0f}"
        for kind, (avg, top, count) in waits.items() if count)
        + (f"; сервер: {tick}" if tick else ""))
    for r in runners:
        state = "работает" if r["process"].poll() is None else f"вышла ({r['process'].returncode})"
        print(f"{r['fly']['name']:>6} | {state:10s} {r['status'][:150]}")
    print()


def main():
    args = parse_args()
    print(f"Шлюз: подключаюсь к {RCON_HOST}:{RCON_PORT} ...")

    rcon = open_rcon(RCON_HOST, RCON_PASSWORD, RCON_PORT)
    rcon.connect()
    key = secrets.token_hex(16)
    gate = GatewayServer(rcon, key, priority=not args.no_priority)
    sight = None
    if not args.no_shared:
        sight = SharedSight(lambda text: gate.command(text, "общий"),
                            see_flies=not args.no_see_flies)
        gate.observers.append(sight.observe)
        gate.shared = sight
        sight.start()
    place_in_row(gate, args.flies, args.gap)

    gate.command(f"scoreboard players reset * {BOARD_OBJECTIVE}")
    print(f"Шлюз слушает {gate.address[0]}:{gate.address[1]}")
    for fly in args.flies:
        x, y, z, yaw = fly["at"]
        print(f"  муха {fly['name']}: {fly['body']}, старт {x:.1f} {y:.1f} {z:.1f}, "
              f"поворот {yaw:.0f}, лог output/race_{fly['name']}.log"
              + (f", ест только {FOOD_NAMES[fly['food']]} ({fly['food']})"
                 if fly.get("food") else ""))
    if args.extra:
        print(f"  всем мухам: {' '.join(args.extra)}")
    print("  телепорты " + ("в общей очереди (--no-priority)" if args.no_priority
                            else "вне очереди"))
    print("  мухи " + ("НЕ видят друг друга (--no-see-flies)" if args.no_see_flies
                       else "видят друг друга"))
    print("  игрок, существа, стрелы и еда - " +
          ("каждая муха спрашивает сама (--no-shared)" if sight is None else
           "общий опрос на всех мух"
           + (", стрелы трёх видов" if sight.multi_type_ok else ", стрелы одного вида")))
    print("Мухи калибруются (около минуты), потом появятся в мире. Ctrl+C - стоп.\n")

    runners = [launch(fly, gate.address, key, args) for fly in args.flies]
    started = time.perf_counter()
    since = [started, 0, 0.0, 0, 0]
    next_report = started + REPORT_EVERY_S
    stopping_at = None
    try:
        while True:
            try:
                time.sleep(0.5)
                for r in runners:
                    follow(r)
                    if r["process"].poll() is not None and not r["printed_exit"]:
                        r["printed_exit"] = True
                        print(f"{r['fly']['name']:>6} | -- процесс завершился, "
                              f"код {r['process'].returncode}")
                if all(r["process"].poll() is not None for r in runners):
                    break
                if stopping_at is not None and time.perf_counter() > stopping_at:
                    print("Не все мухи остановились сами - завершаю принудительно.")
                    break
                if time.perf_counter() >= next_report:
                    next_report += REPORT_EVERY_S
                    report(gate, runners, started, since, sight)
            except KeyboardInterrupt:

                if stopping_at is not None:
                    break
                print("\nОстанавливаю мух...")
                stopping_at = time.perf_counter() + STOP_WAIT_S
    finally:
        for r in runners:
            if r["process"].poll() is None:
                r["process"].terminate()
        for r in runners:
            try:
                r["process"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                r["process"].kill()
            follow(r)
            r["log"].close()
        try:
            gate.command(f"scoreboard players reset * {BOARD_OBJECTIVE}")
            gate.command("scoreboard objectives setdisplay sidebar")
        except Exception:
            pass
        report(gate, runners, started, since, sight)
        if sight is not None:
            sight.stop()
        if gate.errors:
            print(f"Ошибок в шлюзе: {len(gate.errors)}, первая: {gate.errors[0]}")
        gate.close()
        try:
            rcon.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
