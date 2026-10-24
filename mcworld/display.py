"""Что показывается: табло голода и надпись над мухой в игре,
файл для OBS, строка состояния и отчёт о командах в консоли.
"""

import json
import os
import shutil

from flygame.constants import ODOR_GRADIENT_DEAD
from flygame.odor import odor_dive_allowed

from .constants import POLL_INTERVAL_S
from .link import COMMAND_STATS
from . import ident


#таблица голода в игре
BOARD_OBJECTIVE = "flyhunger"   # общее табло; заголовок и строка мухи - в ident
SOLO_HOLDER = "hunger"
BOARD_USERS = "#flies"


def board_show(world):
    """Завести таблицу и вывести её сбоку"""
    title = json.dumps(ident.BOARD_TITLE)
    world.command(f"scoreboard objectives add {BOARD_OBJECTIVE} dummy {title}")
    world.command(f"scoreboard objectives modify {BOARD_OBJECTIVE} displayname {title}")
    world.command(f"scoreboard objectives setdisplay sidebar {BOARD_OBJECTIVE}")
    if ident.NAME:
        world.command(f"scoreboard players reset {SOLO_HOLDER} {BOARD_OBJECTIVE}")
        world.command(f"scoreboard players add {BOARD_USERS} {BOARD_OBJECTIVE} 1")
    else:
        world.command(f"scoreboard players reset * {BOARD_OBJECTIVE}")
    world.command(f"scoreboard players set {ident.BOARD_HOLDER} {BOARD_OBJECTIVE} 0")

    world.command(f"scoreboard players display name {ident.BOARD_HOLDER} "
                  f"{BOARD_OBJECTIVE} {json.dumps(ident.BOARD_LINE)}")


def board_set(world, hunger):
    """Записать голод в процентах."""
    world.command(f"scoreboard players set {ident.BOARD_HOLDER} {BOARD_OBJECTIVE} "
                  f"{int(round(max(0.0, min(1.0, hunger)) * 100))}")

HUNGER_FORMAT = "Голод: {percent}%"


def write_hunger_file(path, text):
    """Записать так, чтобы OBS никогда не прочёл полупустой файл."""
    spare = path.with_name(path.name + ".tmp")
    try:
        spare.write_text(text, encoding="utf-8")
        os.replace(spare, path)
        return True
    except OSError:
        return False


def board_hide(world):
    """Убрать таблицу с экрана - иначе висело бы застывшее число."""
    if ident.NAME:
        world.command(f"scoreboard players reset {ident.BOARD_HOLDER} "
                      f"{BOARD_OBJECTIVE}")
        world.command(f"scoreboard players remove {BOARD_USERS} {BOARD_OBJECTIVE} 1")
        world.command(f"execute if score {BOARD_USERS} {BOARD_OBJECTIVE} matches ..0 "
                      f"run scoreboard objectives setdisplay sidebar")
        return
    world.command("scoreboard objectives setdisplay sidebar")


#надпись над мухой
LABEL_LIFT = 0.9          # блоков над телом


def label_text(hunger):
    """Текст надписи: голод в процентах, цвет - по уровню голода."""
    percent = int(round(max(0.0, min(1.0, hunger)) * 100))
    color = "green" if percent < 40 else "yellow" if percent < 75 else "red"
    return f'{{text:{json.dumps(f"Голод {percent}%")},color:"{color}"}}'


def label_create(world, at, hunger):
    world.command(f"kill @e[type=text_display,tag={ident.LABEL_TAG}]")   # остатки прошлого запуска
    x, y, z = at
    world.command(f"summon text_display {x:.2f} {y + LABEL_LIFT:.2f} {z:.2f} "
                  f"{{Tags:[\"{ident.LABEL_TAG}\"],billboard:\"center\","
                  f"teleport_duration:3,shadow:1b,text:{label_text(hunger)}}}")


def label_update(world, hunger):
    world.command(f"data merge entity @e[type=text_display,tag={ident.LABEL_TAG},limit=1] "
                  f"{{text:{label_text(hunger)}}}")


def label_follow(world, at):
    x, y, z = at
    world.command(f"tp @e[type=text_display,tag={ident.LABEL_TAG},limit=1] "
                  f"{x:.2f} {y + LABEL_LIFT:.2f} {z:.2f}")


def label_remove(world):
    world.command(f"kill @e[type=text_display,tag={ident.LABEL_TAG}]")
STATS_EVERY_S = 10.0      # как часто печатать сводку по командам


def safe_flip(world, on):
    """Переворот - украшение. Оборванная связь не должна ронять сессию."""
    try:
        world.set_flipped(on)
    except Exception as exc:
        print(f"\n[переворот] не вышло: {type(exc).__name__}: {exc}")


STATE_WORDS = {"flying": "в полёте", "landing": "снижается", "perched": "сидит",
               "walking": "идёт", "eating": "ест", "stunned": "оглушена"}


def report_commands(since, now, polls):
    """Строка сводки по командам за промежуток: всего, опрос, ответ, на что."""
    at, total0, busy0, kinds0, polls0 = since
    kinds, total, busy = COMMAND_STATS.snapshot()
    span = max(now - at, 1e-6)
    sent = total - total0
    per_kind = {k: (kinds.get(k, 0) - kinds0.get(k, 0)) / span for k in kinds}
    top = sorted(per_kind.items(), key=lambda kv: -kv[1])
    parts = ", ".join(f"{k} {v:.0f}" for k, v in top if v >= 0.5)
    reply = (busy - busy0) / sent * 1000.0 if sent else 0.0
    line = (f"КОМАНДЫ: {sent / span:.0f}/с, опрос {(polls - polls0) / span:.1f} Гц "
            f"из {1.0 / POLL_INTERVAL_S:.0f}, ответ {reply:.1f} мс - {parts}")
    return line, (now, total, busy, kinds, polls)


def report_loop(loop, span, control_hz):
    """Строка про такт управления за промежуток; счётчики обнуляются."""
    ticks = loop["ticks"]
    brain = loop["brain_s"] / ticks * 1000.0 if ticks else 0.0
    moves = loop["moves"]
    move = loop["move_s"] / moves * 1000.0 if moves else 0.0
    line = (f"ТАКТ: {ticks / max(span, 1e-6):.1f} Гц из {control_hz:.0f}, "
            f"длиннейший {loop['worst_ms']:.0f} мс; мозг {brain:.1f} мс на такт; "
            f"телепорт ждёт {move:.1f} мс, макс {loop['move_max'] * 1000.0:.0f}")
    loop.update(ticks=0, worst_ms=0.0, brain_s=0.0, move_s=0.0, moves=0, move_max=0.0)
    return line


def report_shared(stats, before):
    """Строка про общий опрос гонки за промежуток, или "" вне гонки."""
    d = {k: v - before.get(k, 0) for k, v in stats.items()}
    before.clear()
    before.update(stats)
    if not any(d.values()):
        return ""
    return (f"ОБЩИЙ ОПРОС: снимков {d['taken']}, не дождалась {d['missed']}; "
            f"сама: всё {d['whole']}, существа {d['creatures']}, "
            f"стрелы {d['projectiles']}, запах {d['odor']} из {d['odor_calls']}")


def show_status(elapsed_s, body, threats, rates, motor, wind_rates=None,
                state="flying", hunger=0.0, wounded=0.0, ignore="none"):  # noqa: D103
    blind = "" if ignore == "none" else f" [не видит: {ignore}]"
    if body.meal_search > 0.0:
        blind = " ищет рядом" + blind
    if threats:
        nearest = min(threats, key=lambda t: t.distance)
        target = (f"{nearest.key[:10]:10s} {nearest.distance:5.1f} бл "
                  f"{nearest.azimuth_deg:+4.0f}°")
    else:
        target = "никого рядом" + " " * 8
    bar_len = int(motor.thrust * 10)
    bar = "#" * bar_len + "." * (10 - bar_len)
    air = f" ветер {wind_rates.max():5.1f}" if wind_rates is not None else ""
    smell = ""
    if body.odor_level is not None and body.odor_level > body.odor_rest:
        change = body.odor_level - body.odor_slow
        arrow = ("^" if change > ODOR_GRADIENT_DEAD else
                 "v" if change < -ODOR_GRADIENT_DEAD else "-")
        smell = f" запах {body.odor_level:4.1f}{arrow}"
        if odor_dive_allowed(body, motor):
            smell += " вниз"
    seen_hz = rates[body.visual_mask].max() if body.visual_mask is not None else rates.max()
    line = (f"[{elapsed_s:6.1f}s] {target}  LC {seen_hz:5.1f} Гц{air}  "
            f"DN L{motor.left_hz:5.1f} R{motor.right_hz:5.1f}  "
            f"уход {motor.escape_deg:+4.0f}° возб {body.arousal:.2f} "
            f"[{bar}] {body.speed:4.1f} бл/с голод {hunger:.2f}{smell} "
            f"{'рана %.2f ' % wounded if wounded > 0.02 else ''}"
            f"{STATE_WORDS[state]:9s}{blind}")

    room = shutil.get_terminal_size((160, 24)).columns - 1
    print("\r" + line[:room].ljust(room), end="")
