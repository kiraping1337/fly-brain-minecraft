"""Проверка гонки без сервера: шлюз и три мухи на заглушке.

Часть 1 - шлюз сам по себе, на эхо-сервере. Несколько клиентов шлют
команды одновременно; каждый должен получить ответ именно на свою команду
(ради этого шлюз и нужен), команды должны выполняться строго по одной, а
ошибка сервера - дойти до той мухи, чья команда упала, не роняя остальных.

Часть 2 - настоящая гонка: demos/race.py запускает три моста отдельными
процессами (allay, chicken, bat), шлюз говорит с заглушкой из
smoke_bridge в режиме нескольких мух - у каждой своё тело и свои счета.
Проверяется проводка, а не поведение:
  - все три мухи прошли весь путь и вышли без ошибок;
  - каждая видела ВСЕХ других мух гонки: мосты идут с --debug, и в логе
    видно, кого муха видит на каждом опросе (строка состояния показывает
    только ближайшего, по ней всех не проверить);
  - каждая щупала рельеф от СВОЕГО тела и двигала только своё тело;
  - счета у мух свои (#probe в flybrain_a, flybrain_b, flybrain_c);
  - общий опрос работает: мухи берут игрока, существ, стрелы и запах из
    снимков шлюза, а своих запросов существ почти нет. Что отбор из снимка
    совпадает с собственным запросом мухи, проверяет tools/check_shared.py.

Сервер не нужен. Займёт пару минут: три мозга калибруются параллельно.
Запуск:  python tools/smoke_race.py
"""

import math
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mcworld.gateway import KEY_ENV, GatewayConnection, GatewayServer  # noqa: E402

FLIES = (("a", "allay", "0,70,0,0"), ("b", "chicken", "4,70,0,0"),
         ("c", "bat", "8,70,0,0"))
KINDS = {"allay": "Allay", "chicken": "Chicken", "bat": "Bat"}


class EchoServer:
    """Отвечает текстом команды; считает, сколько команд шло одновременно."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.most = 0

    def command(self, text):
        with self.lock:
            self.active += 1
            self.most = max(self.most, self.active)
        time.sleep(0.0003)
        with self.lock:
            self.active -= 1
        if text == "boom":
            raise ConnectionError("сервер упал")
        return f"echo:{text}"


def part_gateway():
    key = secrets.token_hex(8)
    os.environ[KEY_ENV] = key
    server = EchoServer()
    gate = GatewayServer(server, key)
    wrong, done = [], []

    def client(i):
        conn = GatewayConnection(gate.address, f"t{i}").connect()
        for j in range(300):
            text = f"{i}:{j}"
            answer = conn.command(text)
            if answer != f"echo:{text}":
                wrong.append((text, answer))
        try:
            conn.command("boom")
            wrong.append(("boom", "ошибка не дошла"))
        except RuntimeError:
            pass
        if conn.command(f"{i}:after") != f"echo:{i}:after":
            wrong.append((f"{i}:after", "после ошибки связь не работает"))
        conn.disconnect()
        done.append(i)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(4)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    spent = time.perf_counter() - started
    per_fly, total, _ = gate.snapshot()
    gate.close()
    print(f"Шлюз: 4 клиента по 300 команд за {spent:.1f} с, всего прошло {total}, "
          f"чужих или неверных ответов {len(wrong)}, одновременно на сервере "
          f"не больше {server.most}, ошибок сервера учтено {len(gate.errors)}")
    errors = []
    if len(done) != 4:
        errors.append("не все клиенты закончили")
    if wrong:
        errors.append(f"неверные ответы: {wrong[:3]}")
    if server.most != 1:
        errors.append("команды шли на сервер одновременно")
    if len(gate.errors) != 4:
        errors.append("ошибка сервера не дошла до каждого клиента")
    return errors


def tp_latency(priority, polls=3, command_s=0.003):
    """Сколько ждёт телепорт, пока polls клиентов без перерыва шлют опросы."""
    key = secrets.token_hex(8)
    os.environ[KEY_ENV] = key

    class Slow:
        def command(self, text):
            time.sleep(command_s)
            return text

    gate = GatewayServer(Slow(), key, priority=priority)
    stop = threading.Event()

    def poller(i):
        conn = GatewayConnection(gate.address, f"p{i}").connect()
        while not stop.is_set():
            conn.command(f"execute at @e[tag=x] run data get entity @s Pos {i}")
        conn.disconnect()

    threads = [threading.Thread(target=poller, args=(i,)) for i in range(polls)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    mover = GatewayConnection(gate.address, "tp").connect()
    waits = []
    for _ in range(120):
        started = time.perf_counter()
        mover.command("tp @e[tag=x] 0 70 0 0 0")
        waits.append(time.perf_counter() - started)
        time.sleep(0.02)
    stop.set()
    for t in threads:
        t.join(5)
    mover.disconnect()
    gate.close()
    waits.sort()
    return waits[len(waits) // 2], waits[int(0.9 * (len(waits) - 1))], waits[-1]


def part_priority():
    """Телепорт вне очереди ждёт не больше одной чужой команды.

    Судим по медиане и 90-му процентилю: редкие выбросы под 50 мс бывают в
    обоих режимах - здесь клиенты и шлюз в одном процессе делят GIL, а в
    гонке мухи - отдельные процессы.
    """
    command_s = 0.003
    fair = tp_latency(False, command_s=command_s)
    fast = tp_latency(True, command_s=command_s)
    ms = lambda v: f"{v * 1000:.1f}"
    print(f"Телепорт при трёх непрерывных опросах, команда {command_s * 1000:.0f} мс: "
          f"в общей очереди медиана {ms(fair[0])} мс, 90% {ms(fair[1])}; "
          f"вне очереди медиана {ms(fast[0])} мс, 90% {ms(fast[1])} "
          f"(максимумы {ms(fair[2])} и {ms(fast[2])})")
    errors = []
    # Вне очереди: дождаться команды, которая уже выполняется, плюс своя.
    if fast[1] > 2 * command_s + 0.003:
        errors.append("телепорт вне очереди ждёт дольше двух команд")
    if fast[0] >= fair[0]:
        errors.append("вне очереди телепорт ждёт не меньше, чем в общей")
    return errors


def part_race():
    import smoke_bridge as stub
    stub.install_stub()
    stub.WORLD.update(multi=True, calm=True)
    import demos.race as race
    sys.argv = ["race.py"] + [a for name, body, at in FLIES
                              for a in ("--fly", f"{name}:{body}:{at}")] + [
                                  "--seconds", "20", "--", "--debug"]
    race.main()

    errors = []
    print()
    for name, body, _ in FLIES:
        log = (ROOT / "output" / f"race_{name}.log").read_text(encoding="utf-8",
                                                              errors="replace")
        statuses = [p for line in log.split("\n") for p in line.split("\r")
                    if race.STATUS_RE.match(p)]
        sights = [line for line in log.split("\n") if "] видит: " in line]
        others = {KINDS[b] for n, b, _ in FLIES if n != name}
        def sees(kind):
            return any(re.search(rf"\b{kind}(#\d+)?@", s) for s in sights)
        seen = {kind for kind in others if sees(kind)}
        if sees(KINDS[body]):
            errors.append(f"{name}: видит саму себя")
        own = [t for t in stub.SENT if f"tag=flybrain_active_{name}," in t]
        probes = [t for t in own if t.startswith("execute at") and " positioned " in t]
        moves = [t for t in stub.SENT if t.startswith(f"tp @e[tag=flybrain_active_{name},")]
        scores = [k for k in stub.WORLD["scores"] if k.endswith(f" flybrain_{name}")]
        finished = "За сессию" in log
        taken = re.search(r"за сессию ОБЩИЙ ОПРОС: снимков (\d+).*?существа (\d+)", log)
        taken, asked = (int(taken.group(1)), int(taken.group(2))) if taken else (0, 0)
        print(f"муха {name} ({body}): строк состояния {len(statuses)}, видела "
              f"{', '.join(sorted(seen)) or 'никого'}, щупов рельефа {len(probes)}, "
              f"телепортов {len(moves)}, своих счетов {len(scores)}, "
              f"снимков {taken}, существ спрашивала сама {asked}, "
              + ("закончила" if finished else "НЕ закончила"))
        if not taken:
            errors.append(f"{name}: не взяла ни одного общего снимка")
        elif asked > taken / 5:
            errors.append(f"{name}: существ спрашивала сама {asked} раз "
                          f"при {taken} снимках - общий опрос почти не работает")
        if "Traceback" in log:
            errors.append(f"{name}: в логе ошибка Python")
        if not finished:
            errors.append(f"{name}: не дошла до конца сессии")
        if seen != others:
            errors.append(f"{name}: не видела {', '.join(sorted(others - seen))}")
        if not probes or not moves:
            errors.append(f"{name}: не щупала рельеф или не двигалась")
        if not scores:
            errors.append(f"{name}: нет своего счёта")

    checked = {name: sum(1 for t in stub.SENT
                         if f"store result score #probe flybrain_{name} if block" in t)
               for name, _, _ in FLIES}
    print("проверка проходимых блоков, команд: "
          + ", ".join(f"{n} {c}" for n, c in checked.items()))
    if sum(1 for c in checked.values() if c) != 1:
        errors.append("проверку проходимых блоков делала не ровно одна муха")
    snapshots = [t for t in stub.SENT
                 if t.startswith("execute positioned") and " as @e[type=!" in t]
    print(f"общих запросов существ: {len(snapshots)}")
    if not snapshots:
        errors.append("шлюз не снимал общий опрос")
    stray = [t for t in stub.SENT if t.startswith("tp @e[tag=flybrain_active,")]
    if stray:
        errors.append(f"телепорт без имени мухи: {stray[0][:80]}")
    return errors


def part_blind():
    """Гонка с --no-see-flies и --own-food: мухи не видят друг друга, а зомби
    видят; у каждой свой вид еды."""
    import smoke_bridge as stub
    import demos.race as race
    stub.SENT.clear()
    stub.WORLD.update(bodies={}, types={}, scores={}, current="",
                      mobs=[("Zombie", (4.0, 70.0, 8.0))])
    sys.argv = ["race.py"] + [a for name, body, at in FLIES
                              for a in ("--fly", f"{name}:{body}:{at}")] + [
                                  "--seconds", "12", "--no-see-flies", "--own-food",
                                  "--", "--debug"]
    race.main()
    errors, foods = [], []
    print()
    final = stub.WORLD["bodies"]
    print("разлёт к концу, блоков: " + ", ".join(
        f"{a}-{b} {math.dist(final[a], final[b]):.0f}"
        for a, b in (("a", "b"), ("b", "c"), ("a", "c")) if a in final and b in final))
    for name, body, _ in FLIES:
        log = (ROOT / "output" / f"race_{name}.log").read_text(encoding="utf-8",
                                                              errors="replace")
        sights = [line for line in log.split("\n") if "] видит: " in line]
        flies = {KINDS[b] for _n, b, _ in FLIES}
        saw = {k for k in flies | {"Zombie"}
               if any(re.search(rf"\b{k}(#\d+)?@", s) for s in sights)}
        print(f"муха {name} без соседей: опросов {len(sights)}, видела "
              f"{', '.join(sorted(saw)) or 'никого'}")
        if "Других мух не видит" not in log:
            errors.append(f"{name}: ключ --no-see-flies не дошёл до моста")
        food = re.search(r"Ест только: ([^(]+) \(--food-only\)", log)
        foods.append(food.group(1).strip() if food else None)
        if saw & flies:
            errors.append(f"{name}: без соседей видела {', '.join(sorted(saw & flies))}")
        if "Zombie" not in saw:
            errors.append(f"{name}: без соседей не видела зомби")
        if "Traceback" in log:
            errors.append(f"{name}: в логе ошибка Python")
    print("своя еда: " + ", ".join(f"{n} - {f}" for (n, _, _), f in zip(FLIES, foods)))
    if None in foods or len(set(foods)) != len(foods):
        errors.append("--own-food: не у каждой мухи свой вид еды")
    shared = [t for t in stub.SENT if "type=!minecraft:item" in t and "as @e[" in t]
    if not shared or any("tag=!flybrain," not in t and "tag=!flybrain_" in t
                         for t in shared):
        errors.append("запросы существ в режиме без соседей не исключают всех мух")
    return errors


def main():
    errors = part_gateway()
    errors += part_priority()
    errors += part_race()
    errors += part_blind()
    if errors:
        sys.exit("ОШИБКА: " + "; ".join(errors))
    print("\nгонка проходит целиком")


if __name__ == "__main__":
    main()
