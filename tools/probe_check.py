"""Показывает дословно, что сервер отвечает на проверку блоков.

Мост определяет наличие земли по ответу на команду execute if block. Если
сервер отвечает не теми словами, которые мост ищет, или молчит совсем, то
земли для мухи просто не существует и она проваливается. Этот скрипт
проверяет ровно это и ничего больше.

Запуск:  python tools/probe_check.py

"""

import sys

from mcrcon import MCRcon

HOST = "127.0.0.1"
PORT = 25575
PASSWORD = "flyproject"

PROBES = [
    ("на своём уровне", "if block ~ ~ ~ minecraft:air"),
    ("над головой", "positioned ~ ~1 ~ if block ~ ~ ~ minecraft:air"),
    ("под ногами", "positioned ~ ~-1 ~ if block ~ ~ ~ minecraft:air"),
    ("на 2.5 ниже", "positioned ~ ~-2.5 ~ if block ~ ~ ~ minecraft:air"),
    ("по курсу", "positioned ^ ^ ^1.5 if block ~ ~ ~ minecraft:air"),
]


def run(mc, feedback):
    mc.command(f"gamerule sendCommandFeedback {'true' if feedback else 'false'}")
    print(f"\n=== sendCommandFeedback = {feedback} ===")
    for label, clause in PROBES:
        command = f"execute at @p {clause}"
        answer = mc.command(command)
        verdict = "непонятно"
        low = (answer or "").lower()
        if "passed" in low:
            verdict = "ПУСТО"
        elif "failed" in low:
            verdict = "БЛОК"
        print(f"  {label:16s} -> {verdict:9s} ответ: {answer!r}")


def main():
    try:
        with MCRcon(HOST, PASSWORD, port=PORT) as mc:
            position = mc.command("data get entity @p Pos")
            print(f"Позиция игрока: {position!r}")

            run(mc, True)
            run(mc, False)

            print("\nЧто должно быть: 'под ногами' даёт БЛОК, когда вы стоите "
                  "на земле,\nа 'над головой' даёт ПУСТО. Если хоть где-то "
                  "'непонятно' - мост\nне видит землю, и дело именно в этом.")
            mc.command("gamerule sendCommandFeedback false")
    except Exception as exc:
        sys.exit(f"Не подключился: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
