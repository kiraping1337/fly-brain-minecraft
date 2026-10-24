"""Общий опрос гонки даёт мухе то же, что её собственные запросы.

На подставном сервере из smoke_bridge (режим нескольких мух: отбор
селектора по расстоянию, пределу и тегу - как у сервера) раскладывается
случайный мир: три мухи, мобы, летящие стрелы, куски еды всех видов. Для
каждой мухи считается дважды:

  - сама: MinecraftWorld.scan() и probe_odor() своими запросами, как одна муха;
  - из общего снимка: шлюзовой SharedSight снимает мир один раз на всех,
    муха берёт из снимка свой отбор (там, где снимок её накрывает).

Сервер не нужен. Запуск:
    python tools/check_shared.py
"""

import copy
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import smoke_bridge as stub  # noqa: E402

stub.install_stub()
PRISTINE = copy.deepcopy(stub.WORLD)

from flygame.constants import FOOD_ITEMS  # noqa: E402
from mcworld import ident, shared  # noqa: E402
from mcworld.world import MinecraftWorld  # noqa: E402

FLIES = (("a", "minecraft:allay"), ("b", "minecraft:chicken"), ("c", "minecraft:bat"))
MOB_NAMES = ("Zombie", "Skeleton", "Cow", "Pig", "Spider")
TRIALS = 300


class SnapshotLink:
    """Подставное соединение мухи со шлюзом: отдаёт один готовый снимок."""

    def __init__(self, packaged):
        self.packaged = packaged

    def shared(self, last_seq, timeout):
        return ("ok", copy.deepcopy(self.packaged))

    def disconnect(self):
        pass


def spot(rng, center, half, y_low=69.0, y_high=80.0):
    return (round(center[0] + rng.uniform(-half, half), 3),
            round(rng.uniform(y_low, y_high), 3),
            round(center[2] + rng.uniform(-half, half), 3))


def lay_out(rng, spread=None):
    """Случайный мир в заглушке. Возвращает места мух (до сотых, как body_at).

    spread - насколько мухи разлетелись; по умолчанию случайно: тесно (одна
    кучка), средне и широко (каждая сама по себе).
    """
    stub.WORLD.clear()
    stub.WORLD.update(copy.deepcopy(PRISTINE))
    stub.WORLD.update(multi=True, calm=True)
    base = (rng.uniform(-200, 200), 0.0, rng.uniform(-200, 200))
    spread = spread or rng.choice((2.0, 5.0, 8.0, 20.0, 40.0))
    places = {}
    for name, kind in FLIES:
        place = (round(base[0] + rng.uniform(-spread, spread), 2),
                 round(rng.uniform(71.0, 78.0), 2),
                 round(base[2] + rng.uniform(-spread, spread), 2))
        places[name] = place
        stub.WORLD["bodies"][name] = list(place)
        stub.WORLD["types"][name] = kind
    stub.WORLD["mobs"] = [(rng.choice(MOB_NAMES), spot(rng, base, 40.0))
                          for _ in range(rng.choice((0, 3, 10, 30, 60)))]
    stub.WORLD["arrows"] = [("Arrow", spot(rng, base, 30.0))
                            for _ in range(rng.choice((0, 2, 6, 15)))]
    stub.WORLD["items"] = [{"id": item_id, "count": 1, "tags": [],
                            "pos": spot(rng, base, 45.0, 70.0, 70.0)}
                           for item_id in FOOD_ITEMS
                           for _ in range(rng.choice((0, 4, 30)))]
    return places


def tp_text(name, place):
    x, y, z = place
    return f"tp @e[tag=flybrain_active_{name},limit=1] {x:.3f} {y:.3f} {z:.3f} 0.0 0"


def make_sight(places, see_flies, jump=None):
    """Шлюзовой общий опрос на заглушке: каждая команда проходит через observe.

    jump=(имя, место) - эта муха перелетает туда прямо перед первым запросом
    существ: разброс стаи посчитан по старому месту.
    """
    conn = stub.FakeConnection()
    sight = None
    moved = []

    def command(text):
        if jump and not moved and "type=!minecraft:item" in text:
            moved.append(True)
            name, place = jump
            jumped = tp_text(name, place)
            sight.observe(jumped, conn.command(jumped))
        answer = conn.command(text)
        sight.observe(text, answer)
        return answer

    sight = shared.SharedSight(command, see_flies=see_flies)
    sight._check_multi_type()
    for name, place in places.items():
        text = tp_text(name, place)
        sight.observe(text, conn.command(text))
    return sight


def fly_world(name, multi_type_ok, place, link, see_flies):
    ident.apply_name(name)
    world = MinecraftWorld()
    world.multi_type_ok = multi_type_ok
    world.body_at = place
    world._shared_link = link
    world.see_flies = see_flies
    return world


def compare(name, place, packaged, multi_type_ok, see_flies, fly_mode=None):
    """Сама и из снимка. Возвращает (расхождение или None, счётчики, что видела)."""
    fly_mode = see_flies if fly_mode is None else fly_mode
    own = fly_world(name, multi_type_ok, place, False, fly_mode)
    alone = own.scan(), own.probe_odor()
    via = fly_world(name, multi_type_ok, place, SnapshotLink(packaged), fly_mode)
    together = via.scan(), via.probe_odor()
    seen = {s.name.split("#")[0] for s in alone[0]}
    if alone[0] != together[0]:
        return (f"существа: сама {[(s.name, s.position) for s in alone[0]]}, "
                f"из снимка {[(s.name, s.position) for s in together[0]]}"), \
            via.shared_stats, seen
    if alone[1] != together[1]:
        return (f"запах: сама {len(alone[1])}, из снимка {len(together[1])}",
                via.shared_stats, seen)
    return None, via.shared_stats, seen


def main():
    rng = random.Random(7)
    totals = {}
    kinds = {"single": 0, "cluster": 0}
    split = 0                  # кучка упёрлась в предел и разбилась на свои запросы
    fly_kinds = {k.split(":")[1].capitalize() for _n, k in FLIES}
    seen_flies = {True: 0, False: 0}
    checked = 0
    errors = []
    for trial in range(TRIALS):
        see = trial % 2 == 0
        places = lay_out(rng)
        jump = None
        if trial % 5 == 4:

            name = "c"
            x, y, z = places[name]
            jump = (name, (round(x + 30.0, 2), y, z))
        sight = make_sight(places, see, jump)
        mismatch = trial % 7 == 6
        before = {}
        for round_no in (1, 2):
            sight._take()
            if jump:
                places[jump[0]] = jump[1]
            parts = sight.snapshot["creatures"]
            for name in parts:
                kinds[parts[name]["kind"]] += 1
                if (round_no == 2 and before.get(name) == "cluster-crowded"
                        and parts[name]["kind"] == "single"):
                    split += 1
                if parts[name]["kind"] == "cluster" and not parts[name]["full"]:
                    before[name] = "cluster-crowded"
            packaged = sight.package(sight.snapshot)
            for name, _kind in FLIES:
                multi = (not sight.multi_type_ok) if mismatch else sight.multi_type_ok
                problem, stats, seen = compare(name, places[name], packaged, multi, see)
                checked += 1
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0) + value
                if problem:
                    errors.append(f"опыт {trial}.{round_no}, муха {name}, "
                                  f"{'видят' if see else 'не видят'}: {problem}")
                if seen & fly_kinds:
                    seen_flies[see] += 1

    for see in (True, False):
        lay_out(random.Random(5), spread=2.0)
        places = {"a": (100.0, 72.0, 100.0), "b": (109.0, 72.0, 100.0),
                  "c": (118.0, 72.0, 100.0)}
        for name, place in places.items():
            stub.WORLD["bodies"][name] = list(place)
        crowd = random.Random(9)
        stub.WORLD["mobs"] = [("Cow", (round(109.0 + crowd.uniform(-1, 1), 3), 72.0,
                                       round(100.0 + crowd.uniform(-1, 1), 3)))
                              for _ in range(40)]
        stub.WORLD["mobs"] += [("Zombie", (97.0, 72.0, 100.0 + k)) for k in range(4)]
        sight = make_sight(places, see)
        for round_no in (1, 2):
            sight._take()
            kind = sight.snapshot["creatures"]["a"]["kind"]
            if round_no == 1 and kind != "cluster":
                errors.append("людная середина: мухи не в одной кучке")
            if round_no == 2 and kind != "single":
                errors.append("людная середина: кучка не разбилась на свои запросы")
            problem, _stats, seen = compare("a", places["a"],
                                            sight.package(sight.snapshot),
                                            sight.multi_type_ok, see)
            if problem:
                errors.append(f"людная середина, снимок {round_no}: {problem}")
            if "Zombie" not in seen:
                errors.append("людная середина: крайняя муха не видит своих зомби")

    places = lay_out(random.Random(3), spread=2.0)
    sight = make_sight(places, True)
    sight._take()
    problem, stats, _ = compare("a", places["a"], sight.package(sight.snapshot),
                                sight.multi_type_ok, True, fly_mode=False)
    if problem:
        errors.append(f"режимы разошлись: {problem}")
    if not stats["creatures"]:
        errors.append("режимы разошлись, а муха взяла существ из снимка")

    places = lay_out(random.Random(1), spread=2.0)
    stub.WORLD["mobs"] = []
    del stub.WORLD["bodies"]["b"]
    sight = make_sight(places, True)
    sight._take()
    snap = sight.package(sight.snapshot)
    if snap["creatures"]["b"]["kind"] != "cluster":
        errors.append("проверка гибели: мухи не в одной кучке")
    dead = shared.view_for(snap, "b", sight.multi_type_ok)
    alive = shared.view_for(snap, "a", sight.multi_type_ok)
    if dead is not None:
        errors.append("погибшая муха взяла снимок вместо своего запроса")
    if alive is None or alive["creatures"] is None:
        errors.append("живая муха не взяла существ из снимка")
    elif any(name == "Chicken" for name, _ in alive["creatures"]):
        errors.append("живая муха видит погибшую (Chicken)")

    used = totals.get("taken", 0)
    print(f"опытов {TRIALS} по два снимка, сравнений {checked}: снимков взято {used}; "
          f"сама спрашивала - всё {totals.get('whole', 0)}, "
          f"существа {totals.get('creatures', 0)}, стрелы {totals.get('projectiles', 0)}, "
          f"запах {totals.get('odor', 0)} из {totals.get('odor_calls', 0)}")
    print(f"существа в снимках: своим запросом {kinds['single']}, кучкой "
          f"{kinds['cluster']}; людная кучка разбилась на свои запросы {split} раз; "
          f"видела других мух: в режиме с соседями {seen_flies[True]}, "
          f"без соседей {seen_flies[False]}")
    if seen_flies[False]:
        errors.append("в режиме без соседей муха видела другую муху")
    if not seen_flies[True]:
        errors.append("в обычном режиме муха ни разу не видела других мух")
    if not kinds["single"] or not kinds["cluster"] or not split:
        errors.append("не проверены все виды запроса существ")
    for key in ("creatures", "projectiles", "odor"):
        if not totals.get(key):
            errors.append(f"ни разу не спрашивала сама: {key}")
        if totals.get(key, 0) >= used:
            errors.append(f"ни разу не взяла из снимка: {key}")
    if errors:
        for line in errors[:10]:
            print("ОШИБКА:", line)
        sys.exit(f"расхождений: {len(errors)}")
    print("общий опрос совпадает с запросами мухи")


if __name__ == "__main__":
    main()
