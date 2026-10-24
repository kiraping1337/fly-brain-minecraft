"""Сверка главного цикла моста: старое дерево проекта против нынешнего.

Мост целиком (main) гоняется на подставном сервере из tools/smoke_bridge.py
по набору сценариев: мобы рядом, еда, вода, стена, террасы, удары с гибелью,
голодание, самопроверка, другие тела, опыты brain_lab. Сравнивается всё, что
мост отправил серверу, весь его вывод и запись мозга. Так проверяется
переделка самого цикла: поведение не должно измениться ни на одну команду.

Чтобы прогон повторялся до команды:
  - часы заморожены и идут только по sleep главного цикла;
  - фоновый опрос мира идёт в ногу с циклом: раз в POLL_INTERVAL_S по этим
    часам один круг опроса, пока цикл ждёт;
  - зёрна шума и numpy закреплены, удары по мухе - по расписанию часов.
Настоящее время и настоящий порядок потоков тут не проверяются - это
делают смоуки.

Каждое дерево гоняется в своём процессе. Калибровка мозга одинаковая для
одинаковых ключей, поэтому в процессе она делается один раз на набор ключей
и копируется в каждый сценарий (её вывод повторяется как есть).

Запуск: скопируйте ДО правки папки flybrain, flygame, mcworld, demos и tools
целиком, например в output/before/, а после правки:
    python tools/compare_loop.py output/before
    python tools/compare_loop.py output/before --only food,damage
    python tools/compare_loop.py --twice      # нынешнее дерево против себя же:
                                              # проверка, что сверка повторяема
"""
import argparse
import atexit
import contextlib
import copy
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = {
    "mobs": ({}, ["--record", "{tmp}/mobs.npz"], None, 25),
    "food": ({"food": True, "melon": True, "calm": True,
              "side": "minecraft:pumpkin"}, ["--hunger", "0.8"], None, 40),
    "flood": ({"food": True, "melon": True, "calm": True, "flood": True,
               "side": "minecraft:pumpkin"}, ["--hunger", "0.8"], None, 40),
    "water": ({"water": True, "calm": True}, [], None, 30),
    "wall": ({"wall": True, "player": (2.0, 2.0)}, ["--leash", "0"], None, 30),
    "steps": ({"relief": "terraces", "choke": True, "calm": True},
              ["--leash", "0", "--hunger", "1.0", "--starve", "0"], None, 45),
    "damage": ({"damage": True, "beat": [(3.0, 1.0), (9.0, 4.0), (15.0, 1.0),
                                         (22.0, 100.0)]},
               ["--respawn"], None, 30),
    "starve": ({"calm": True}, ["--hunger", "0.99", "--hunger-minutes", "0.3",
                                "--hunger-label", "--hunger-file", "{tmp}/h.txt"],
               None, 25),
    "chicken": ({}, ["--body", "chicken", "--name", "kur", "--ignore", "player"],
                None, 20),
    "bat": ({"calm": True}, ["--body", "bat", "--no-act"], None, 10),
    "simulate": ({}, ["--simulate"], None, 15),
    "pesticide": ({}, ["--record", "{tmp}/pest.npz"],
                  ["--drug", "pesticide", "--after", "3", "--ramp", "0.1",
                   "--paralysis", "4", "--dose-file", "{tmp}/dose.txt"], 25),
    "alcohol": ({"calm": True}, ["--hunger", "0.8"],
                ["--drug", "alcohol", "--after", "2", "--ramp", "0.1",
                 "--dose-file", ""], 25),
    "synesthesia": ({"food": True, "melon": True, "calm": True,
                     "side": "minecraft:pumpkin"},
                    ["--hunger", "0.8", "--record", "{tmp}/syn.npz"],
                    ["--synesthesia", "--after", "5", "--dose-file", ""], 30),
    "syn_input": ({"food": True, "melon": True, "calm": True},
                  ["--hunger", "0.8"],
                  ["--synesthesia", "--via", "input", "--after", "3",
                   "--dose-file", ""], 20),
}
SEED = "42"
# События в выводе: по ним видно, какие ветки цикла сценарии прошли.
EVENTS = ("ВЗЛЁТ", "ПОСАДКА", "УКУС", "ЕСТ:", "ГОЛОДАЕТ", "УДАР", "оглушена",
          "ПОГИБЛА", "ЗАСТРЯЛА", "НА ВОДЕ", "НАД ВОДОЙ", "ПАРАЛИЧ", "ПЬЯНАЯ",
          "СИНЕСТЕЗИЯ", "Запись активности")


#прогон одного дерева (в своём процессе)

CLOCK = [1000.0]


def install_clock(stub, poller_cls):
    """Замороженные часы и опрос в ногу с циклом."""
    real_run = poller_cls.run
    real_latest = poller_cls.latest
    loop = {"start": None}

    def fake_sleep(seconds):
        here = threading.current_thread()
        if here is threading.main_thread():
            CLOCK[0] += max(0.0, float(seconds))
            beat(stub, loop)
        elif sys._getframe(1).f_code is real_run.__code__:
            here._done.release()
            here._go.acquire()

    def run(self):
        self._go.acquire()
        real_run(self)

    def latest(self):
        if self._due is None:
            self._due = CLOCK[0]
        while self._running and CLOCK[0] >= self._due - 1e-9:
            self._go.release()
            self._done.acquire()
            self._due += self.interval
        return real_latest(self)

    real_init = poller_cls.__init__

    def init(self, *a, **kw):
        real_init(self, *a, **kw)
        self._go, self._done = threading.Semaphore(0), threading.Semaphore(0)
        self._due = None

    poller_cls.__init__ = init
    poller_cls.run = run
    poller_cls.latest = latest
    time.perf_counter = lambda: CLOCK[0]
    time.time = lambda: CLOCK[0]
    time.sleep = fake_sleep
    return loop


def beat(stub, loop):
    """Удары по расписанию: секунды от начала цикла (как beat_the_fly)."""
    world = stub.WORLD
    plan = world.get("beat")
    if not plan or not world["watching"]:
        return
    if loop["start"] is None:
        loop["start"] = CLOCK[0]
    while plan and CLOCK[0] - loop["start"] >= plan[0][0]:
        _, amount = plan.pop(0)
        if not world["alive"]:
            continue
        world["health"] = max(0.0, world["health"] - amount)
        world["hurt_until"] = CLOCK[0] + 0.5
        world["hits"].append(amount)
        if world["health"] <= 0:
            world["alive"] = False
            world["deaths"] += 1


def caching(bridge):
    """Калибровка один раз на набор ключей; её вывод повторяется как был."""
    original = bridge.calibrate
    cache = {}

    def calibrate(args):
        key = json.dumps({k: str(v) for k, v in sorted(vars(args).items())
                          if k not in ("seconds", "record", "spawn", "simulate",
                                       "name", "body", "hunger_file", "seed",
                                       "no_act", "ignore", "respawn", "leash",
                                       "hunger_label")}
                         | {"record": bool(args.record)})
        if key not in cache:
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                cal = original(args)
            cache[key] = (copy.deepcopy(cal), text.getvalue())
        cal, text = cache[key]
        print(text, end="")
        return copy.deepcopy(cal)
    bridge.calibrate = calibrate


def digest(path):
    import numpy as np
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".npz":
        data = np.load(path)
        return {k: hashlib.sha256(np.ascontiguousarray(data[k]).tobytes()).hexdigest()[:16]
                for k in sorted(data.files)}
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def dump(root, out, only):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tools"))
    sys.path.insert(0, str(root / "demos"))
    import numpy as np
    import smoke_bridge as stub
    stub.install_stub()
    pristine = copy.deepcopy(stub.WORLD)
    import mcworld.poller
    loop = install_clock(stub, mcworld.poller.WorldPoller)
    import demos.minecraft_bridge as bridge
    import demos.brain_lab as lab
    caching(bridge)
    base = dict(vars(bridge))
    tmp = Path(tempfile.mkdtemp(prefix="compare_loop_"))
    result = {}
    for name, (setup, argv, lab_argv, seconds) in SCENARIOS.items():
        if only and name not in only:
            continue
        stub.WORLD.clear()
        stub.WORLD.update(copy.deepcopy(pristine))
        setup = dict(setup)
        if setup.pop("wall", False):
            setup["blocks"] = {(x, y, z): "minecraft:stone"
                               for x in range(-5, 6) for z in range(-5, 6)
                               if max(abs(x), abs(z)) == 5 for y in range(69, 80)}
        if setup.get("relief") == "terraces":
            setup["relief"] = stub.terraces
            setup["choke_log"] = []
        setup["beat"] = list(setup.get("beat", []))
        stub.WORLD.update(setup)
        stub.SENT.clear()
        CLOCK[0] = 1000.0
        loop["start"] = None
        np.random.seed(0)
        real_rng = np.random.default_rng
        np.random.default_rng = (lambda seed=None, _r=real_rng:
                                 _r(12345 if seed is None else seed))
        argv = [a.replace("{tmp}", str(tmp)) for a in argv]
        bridge_argv = (["--spawn", "--seconds", str(seconds), "--seed", SEED,
                        "--hunger-file", ""] + argv)
        text = io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(text):
                if lab_argv is None:
                    sys.argv = ["minecraft_bridge.py"] + bridge_argv
                    bridge.main()
                else:
                    own = lab.parse_lab_args(
                        [a.replace("{tmp}", str(tmp)) for a in lab_argv])
                    lab.play(own, bridge_argv)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            np.random.default_rng = real_rng
            for key in list(vars(bridge)):
                if key not in base:
                    delattr(bridge, key)
            for key, value in base.items():
                setattr(bridge, key, value)
        files = {}
        for arg in argv:
            if str(tmp) in arg:
                files[Path(arg).name] = digest(arg)
        world = stub.WORLD
        result[name] = {
            "out": text.getvalue().replace(str(tmp), "{tmp}"),
            "sent": [c.replace(str(tmp), "{tmp}") for c in stub.SENT], "error": error,
            "files": files,
            "world": {k: repr(world.get(k)) for k in
                      ("body", "health", "bites", "eaten", "eaten_ids", "deaths",
                       "hits", "flip", "flips", "alive", "choked", "items")},
        }
        print(f"  {name}: команд {len(stub.SENT)}, строк вывода "
              f"{text.getvalue().count(chr(10))}" + (f", ОШИБКА {error}" if error else ""),
              file=sys.stderr, flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        atexit._run_exitfuncs()
    for name, (setup, argv, lab_argv, seconds) in SCENARIOS.items():
        if name in result:
            for arg in argv:
                if "{tmp}" in arg and arg.endswith(".npz"):
                    sidecar = Path(arg.replace("{tmp}", str(tmp))).with_suffix(".dose.npz")
                    if sidecar.exists():
                        result[name]["files"][sidecar.name] = digest(sidecar)
    shutil.rmtree(tmp, ignore_errors=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


#сравнение
def run_tree(root, only):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "loop.json"
        print(f"{root}:", flush=True)
        done = subprocess.run([sys.executable, __file__, "--dump", str(root), str(out),
                               ",".join(only)], encoding="utf-8")
        if done.returncode != 0 or not out.exists():
            sys.exit(f"прогон на {root} упал")
        return json.loads(out.read_text(encoding="utf-8"))


def around(x, y, width=100):
    """Кусок строк вокруг первого отличия, а не их начало."""
    at = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    start = max(0, at - width // 2)
    return x[start:start + width], y[start:start + width]


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (i,) + around(x, y)
    if len(a) != len(b):
        i = min(len(a), len(b))
        return (i,) + around(a[i] if i < len(a) else "<конец>",
                             b[i] if i < len(b) else "<конец>")
    return None


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--dump":
        only = [s for s in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if s]
        dump(Path(sys.argv[2]).resolve(), sys.argv[3], only)
        return
    p = argparse.ArgumentParser(description="сверка главного цикла моста")
    p.add_argument("old", nargs="?", help="копия дерева проекта ДО правки")
    p.add_argument("--only", default="", help="сценарии через запятую: "
                   + ",".join(SCENARIOS))
    p.add_argument("--twice", action="store_true",
                   help="нынешнее дерево против себя: сверка повторяема?")
    args = p.parse_args()
    only = [s for s in args.only.split(",") if s]
    unknown = [s for s in only if s not in SCENARIOS]
    if unknown:
        sys.exit(f"нет сценариев: {', '.join(unknown)}")
    if args.twice:
        old_root = ROOT
    elif args.old:
        old_root = Path(args.old).resolve()
        if not (old_root / "demos" / "minecraft_bridge.py").exists():
            sys.exit(f"в {old_root} нет demos/minecraft_bridge.py - нужна копия "
                     f"ДЕРЕВА проекта (flybrain, flygame, mcworld, demos, tools)")
    else:
        sys.exit(__doc__)
    old, new = run_tree(old_root, only), run_tree(ROOT, only)
    bad = 0
    for name in old:
        a, b = old[name], new.get(name)
        if b is None:
            continue
        problems = []
        if a["error"] != b["error"]:
            problems.append(f"падение: было {a['error']}, стало {b['error']}")
        diff = first_difference(a["sent"], b["sent"])
        if diff:
            i, x, y = diff
            problems.append(f"команда {i}: было {x!r}\n{' ' * 14}стало {y!r}")
        diff = first_difference(a["out"].splitlines(), b["out"].splitlines())
        if diff:
            i, x, y = diff
            problems.append(f"строка вывода {i}: было {x!r}\n{' ' * 14}стало {y!r}")
        if a["files"] != b["files"]:
            problems.append(f"файлы: было {a['files']}, стало {b['files']}")
        if a["world"] != b["world"]:
            keys = [k for k in a["world"] if a["world"][k] != b["world"].get(k)]
            problems.append(f"мир в конце различается: {', '.join(keys)}")
        lines = a["out"].count("\n")
        if problems:
            bad += 1
            print(f"{name:12s}: РАСХОДИТСЯ")
            for line in problems:
                print(f"    {line}")
        else:
            print(f"{name:12s}: совпадает - команд {len(a['sent'])}, строк вывода "
                  f"{lines}" + (f", файлов {len(a['files'])}" if a["files"] else "")
                  + (f" (оба упали: {a['error']})" if a["error"] else ""))
    seen = {e: sum(old[n]["out"].count(e) for n in old) for e in EVENTS}
    print("события во всех сценариях: " + ", ".join(f"{e.rstrip(':')} {c}"
                                                    for e, c in seen.items()))
    missed = [e.rstrip(":") for e, c in seen.items() if not c]
    if missed and not only:
        print(f"!! ни разу не случилось: {', '.join(missed)} - эти ветки не сверены")
    print("РАСХОЖДЕНИЙ НЕТ" if not bad else f"расходятся: {bad}")
    sys.exit(bool(bad))


if __name__ == "__main__":
    main()
