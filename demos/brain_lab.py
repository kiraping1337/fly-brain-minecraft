"""Опыты над мозгом мухи. Отдельно от моста: одиночная муха не меняется.

Файл запускает обычный мост (demos/minecraft_bridge.py), но подменяет у
него сборку мозга: мозг собирается и калибруется как всегда, и
только потом над ним делается операция.

Опыты

Отравление (--drug pesticide). Фосфорорганические пестициды (и боевые
отравляющие вещества того же класса) не дают ферменту убрать
ацетилхолин из синапса - каждый выброс бьёт сильнее и дольше.

Опьянение (--drug alcohol). Этанол усиливает тормозной ГАМК-рецептор и
ослабляет возбуждающую передачу. Здесь ГАМК-синапсы усиливаются, а
ацетилхолиновые слабеют

Синестезия (--synesthesia")

  brain (по умолчанию) - правка САМОГО мозга. Нейроны обонятельной доли,
  которые на чистом воздухе молчат, а на запах фрукта включаются (их
  находит замер при старте - в нашей схеме это 4 местных нейрона доли),
  получают копию выходных связей детекторов надвигания (LPLC2, LC4,
  LPLC1) своей стороны. Новые синапсы в матрице связей, как пестицид - но
  не сила старых, а новые. Запах теперь сам, внутри мозга, доходит до
  пути паники. Направления у запаха почти нет (антенны различают стороны
  на четверть), поэтому и паника без направления: на чистом воздухе муха
  как обычно, на запах издалека - беспокойно ходит, вблизи еды - взлетает
  и не может на неё сесть.

  input - переподключение на ВХОДЕ, мозг не тронут: запах ещё и подаётся
  на сами детекторы надвигания, с той стороны, откуда пахнет. Детекторы в
  нашей схеме - первый слой, выше них ничего нет, так что это как
  срастить обонятельный нерв со зрительным. Убегает чётко прочь от еды.
  --no-smell - нос молчит, остаётся только "вид" еды.


Запуск
В игре - всё после -- уходит мосту как есть:
    python demos/brain_lab.py --drug pesticide -- --spawn --record output/pest.npz
    python demos/brain_lab.py --drug alcohol --dose 0.6 --ramp 2 -- --spawn
    python demos/brain_lab.py --synesthesia -- --spawn --record output/syn.npz

Рендер. Лучше всего видно в режиме --vs-sober: картинка в обычных цветах,
но нейрон, который работает заметно сильнее, чем у этой же мухи в трезвом
начале записи, становится оранжевым, а заметно слабее - фиолетовым. Доза и "парализована"/"на спине" подписываются в
кадре сами, из ЗАПИСЬ.dose.npz:
    python tools/render_brain.py output/pest.npz --vs-sober
    python tools/render_brain.py output/syn.npz --vs-sober --show-smell
Обычный рендер тоже можно, но с одной яркостью для всех записей и своей
шкалой для промежуточных (иначе их перемен не видно):
    python tools/render_brain.py output/pest.npz --max-hz 36 --inner-max-hz 8
"""

import argparse
import atexit
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos"))

# Действие при полной дозе
DRUGS = {
    "pesticide": {"what": "отравление пестицидом", "effect": {"cholinergic": 1.0},
                  "paralysis_s": 20.0, "stagger": 0.0},
    "alcohol": {"what": "опьянение", "effect": {"gaba": 4.0, "cholinergic": -0.3},
                "paralysis_s": -1.0, "stagger": 1.0},
}

#действие на тело
STAGGER_YAW = 150.0       # градусов/с - разброс скорости поворота при полной дозе
STAGGER_YAW_TAU = 0.4     # с - как долго держится один "занос"
STAGGER_BOB = 1.2         # блоков/с - разброс вертикальной скорости в полёте
STAGGER_BOB_TAU = 0.6
TUMBLE_FROM = 0.5         # с какой дозы начинает заваливаться
TUMBLE_EVERY_S = 12.0     # раз в столько секунд при полной дозе (в среднем)
TUMBLE_S = (1.5, 3.0)     # сколько лежит
REFLIP_S = 3.0            # переворот подтверждать: удар мог его снять
# Рендер
INNER_MAX_HZ = 8.0
# Синестезия
SYN_LOOM_HZ = 50.0
GRAFT_K = 12.0
GRAFT_AIR_HZ = 3.0
GRAFT_FOOD_HZ = 10.0
GRAFT_LOOM_TYPES = ("LPLC2", "LC4", "LPLC1")
SYN_FROM = 0.25

SYN_EYE_EDGE = 150.0
SYN_NOTE_S = 3.0          # печатать "шарахнулась" не чаще
SYN_NOTE_HZ = 20.0        # с какой силы печатать
REBUILD_STEP = 0.02       # пересобирать синапсы, когда доза ушла на столько
REPORT_STEP = 0.10        # печатать дозу каждые 10%


def transmitter_masks(circuit, circuit_path):
    """Маски нейронов по медиатору, в нумерации схемы."""
    raw = np.load(circuit_path, allow_pickle=True)
    nt = raw["nt"].astype(str)
    forced = raw["sign_forced"].astype(bool) if "sign_forced" in raw.files \
        else np.zeros(nt.size, bool)
    sensory = np.zeros(nt.size, bool)
    sensory[circuit.input_idx] = True
    excit = circuit.sign > 0
    return {
        "cholinergic": excit & ((nt == "ACH") | (forced & sensory)),
        "gaba": (circuit.sign < 0) & (nt == "GABA"),
        "glutamate": (circuit.sign < 0) & (nt == "GLUT"),
    }


class Dose:
    """Сила синапсов мозга как функция дозы. Доза 0 - мозг как был, бит в бит."""

    def __init__(self, brain, masks, effect):
        self.brain = brain
        circuit = brain.circuit
        self.sober_in, self.sober_net = brain._base_in, brain._base_net
        slope = np.zeros(circuit.n_neurons, dtype=np.float32)
        touched = np.zeros(circuit.n_neurons, bool)
        for group, k in effect.items():
            slope[masks[group]] += k
            touched |= masks[group]
        self.slope_in = slope[circuit.input_idx]
        self.slope_net = slope[circuit.net_idx]
        self.level = 0.0
        self.neurons = int(touched.sum())

    def set(self, level):
        level = float(max(0.0, level))
        brain = self.brain
        if level == 0.0:
            brain._base_in, brain._base_net = self.sober_in, self.sober_net
        else:
            scale_in = sparse.diags(np.maximum(0.0, 1.0 + level * self.slope_in)
                                    .astype(np.float32))
            scale_net = sparse.diags(np.maximum(0.0, 1.0 + level * self.slope_net)
                                     .astype(np.float32))
            brain._base_in = (self.sober_in @ scale_in).tocsr().astype(np.float32)
            brain._base_net = (self.sober_net @ scale_net).tocsr().astype(np.float32)
        brain._apply_gain()
        self.level = level


class Schedule:
    """Доза во времени: трезвая after секунд, потом за ramp секунд до top."""

    def __init__(self, top, after_s, ramp_s):
        self.top, self.after_s, self.ramp_s = float(top), float(after_s), float(ramp_s)

    def at(self, t):
        if t <= self.after_s:
            return 0.0
        if self.ramp_s <= 0:
            return self.top
        return self.top * min(1.0, (t - self.after_s) / self.ramp_s)


class Intake:
    """Подменяет brain.advance: перед каждым тактом доза по расписанию.

    Пишет в консоль каждые 10%, в файл для OBS - текущую дозу, а к записи
    мозга (--record моста) - файл ЗАПИСЬ.dose.npz со временем и дозой.
    """

    def __init__(self, dose, schedule, what, dose_file=None, tape=None,
                 word="доза", label="Доза"):
        self.dose, self.schedule, self.what = dose, schedule, what
        self.word, self.label = word, label
        self.dose_file = Path(dose_file) if dose_file else None
        self.tape = Path(tape) if tape else None
        self.started = None
        self.reported = -1
        self.times, self.levels, self.codes = [], [], []
        self.body_code = 0          # 0 - как обычно, 1 - паралич, 2 - на спине
        self.marked = []            # нейроны, которые подсветить на рендере
        brain = dose.brain
        self._advance = brain.advance
        brain.advance = self.advance
        if self.tape:
            atexit.register(self.save)
        self._show(0.0)

    def advance(self, rates, duration_ms):
        now = time.perf_counter()
        if self.started is None:
            self.started = now
        t = now - self.started
        level = self.schedule.at(t)
        if (abs(level - self.dose.level) >= REBUILD_STEP
                or (level != self.dose.level and level in (0.0, self.schedule.top))):
            self.dose.set(level)
            self._show(level)
        if self.tape:
            self.times.append(t)
            self.levels.append(self.dose.level)
            self.codes.append(self.body_code)
        return self._advance(rates, duration_ms)

    def _show(self, level):
        step = int(round(level / REPORT_STEP))
        if step != self.reported:
            self.reported = step
            if level > 0:
                print(f"\n  {self.what.upper()}: {self.word} {level:.0%}")
        if self.dose_file:
            try:
                self.dose_file.parent.mkdir(parents=True, exist_ok=True)
                self.dose_file.write_text(f"{self.label}: {level:.0%}", encoding="utf-8")
            except OSError:
                pass

    def save(self):
        if not self.times:
            return
        path = self.tape.with_suffix(".dose.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=np.asarray(self.times, np.float32),
                            dose=np.asarray(self.levels, np.float32),
                            body=np.asarray(self.codes, np.int8),
                            label=np.asarray(self.label.lower()),
                            marked=np.asarray(self.marked, np.int32))
        print(f"Доза по времени: {path}")


#аргументы

def parse_lab_args(argv):
    p = argparse.ArgumentParser(description="Опыты над мозгом мухи",
                                epilog="всё после -- уходит мосту")
    p.add_argument("--drug", choices=sorted(DRUGS), default=None)
    p.add_argument("--synesthesia", action="store_true",
                   help="запах еды ещё и на детекторы надвигания (см. в начале файла)")
    p.add_argument("--via", choices=("brain", "input"), default="brain",
                   help="синестезия: новые связи в мозге (brain) или на входе (input)")
    p.add_argument("--graft-k", type=float, default=None,
                   help="синестезия в мозге: сколько детекторов надвигания 'весит' "
                        "один вживлённый нейрон (по умолчанию GRAFT_K)")
    p.add_argument("--loom-hz", type=float, default=SYN_LOOM_HZ,
                   help="синестезия: сила на детекторах надвигания у самой еды, Гц")
    p.add_argument("--loom-from", type=float, default=SYN_FROM,
                   help="синестезия: с какого отклика на запах (0..1) начинается")
    p.add_argument("--no-smell", action="store_true",
                   help="синестезия: нос молчит, запах идёт только в зрение")
    p.add_argument("--dose", type=float, default=1.0,
                   help="итоговая доза, 1.0 - полная (см. DRUGS)")
    p.add_argument("--after", type=float, default=30.0,
                   help="сколько секунд муха трезвая")
    p.add_argument("--ramp", type=float, default=None,
                   help="за сколько минут доза дорастает до --dose (0 - сразу; "
                        "по умолчанию у препаратов 3, у синестезии 0)")
    p.add_argument("--effect", default=None, metavar="ГРУППА=K,...",
                   help="своё действие вместо DRUGS, например "
                        "gaba=4,cholinergic=-0.3 (группы: cholinergic, gaba, glutamate)")
    p.add_argument("--dose-file", default=str(ROOT / "output" / "dose.txt"),
                   help="файл для OBS с текущей дозой ('' - не писать)")
    p.add_argument("--paralysis", type=float, default=None, metavar="СЕК",
                   help="паралич через столько секунд полной дозы (-1 - без "
                        "паралича; по умолчанию у пестицида 20, у алкоголя нет)")
    p.add_argument("--stagger", type=float, default=None,
                   help="сила шатания, 0 - без него (по умолчанию у алкоголя 1)")
    p.add_argument("--offline", action="store_true",
                   help="без игры: замер мозга при разных дозах")
    p.add_argument("--seconds", type=float, default=20.0,
                   help="--offline: секунд тишины на каждую дозу")
    lab = p.parse_args(argv)
    if bool(lab.drug) == bool(lab.synesthesia):
        p.error("нужен ровно один опыт: --drug ... или --synesthesia")
    if lab.ramp is None:
        lab.ramp = 0.0 if lab.synesthesia else 3.0
    return lab


def effect_of(lab):
    """Действие препарата: из DRUGS или из --effect."""
    if not lab.effect:
        return dict(DRUGS[lab.drug]["effect"])
    effect = {}
    for part in lab.effect.split(","):
        group, k = part.split("=")
        effect[group.strip()] = float(k)
    return effect


def describe(effect):
    return ", ".join(f"{g} x{1 + k:g}" for g, k in effect.items())


def split_argv(argv):
    if "--" in argv:
        at = argv.index("--")
        return argv[:at], argv[at + 1:]
    return argv, []


def bridge_arg(argv, name):
    """Значение ключа моста (последнее, как у argparse), или None."""
    found = [i for i, a in enumerate(argv[:-1]) if a == name]
    return argv[found[-1] + 1] if found else None


#без игры

def measure(cal, seconds, noise_hz, threat):
    """Мозг в тишине (и под угрозой спереди) - что увидел бы мост.

    Возбуждение считается так же, как в мосте и калибровке: быстро
    растёт, медленно спадает. Доли - сколько времени муха была бы выше
    порога посадки (ходила бы) и выше порога взлёта (взлетела бы).
    """
    from flygame.constants import CONTROL_HZ, TAU_AROUSAL_FALL, TAU_AROUSAL_RISE
    brain, decoder, circuit = cal.brain, cal.decoder, cal.circuit
    dt = 1.0 / CONTROL_HZ
    silence = np.zeros(circuit.input_idx.size)
    saved = brain.spontaneous_hz
    brain.spontaneous_hz = noise_hz
    brain.reset()
    arousal, rows = 0.0, []
    escapes, was_escape = 0, False
    warmup = int(1.0 * CONTROL_HZ)
    whole = []
    for step in range(int(seconds * CONTROL_HZ)):
        brain.advance(silence, duration_ms=dt * 1000.0)
        command = decoder.decode(brain.output_rates_hz)
        tau = TAU_AROUSAL_RISE if command.thrust > arousal else TAU_AROUSAL_FALL
        arousal += (command.thrust - arousal) * min(1.0, dt / tau)
        escape = brain.escape_triggered() is not None
        if step >= warmup:
            escapes += escape and not was_escape
            rows.append((arousal, command.climb, command.lateral, command.thrust))
            whole.append(brain.all_rates_hz)
        was_escape = escape
    rows = np.asarray(rows)
    whole = np.asarray(whole)
    result = {
        "brain_hz": float(whole.mean()),
        "lively_p99": float(np.percentile(whole[whole > 0], 99)) if (whole > 0).any() else 0.0,
        "arousal": float(rows[:, 0].mean()),
        "walks": float((rows[:, 0] >= cal.perch_thrust).mean()),
        "flies": float(((rows[:, 0] > cal.takeoff_thrust)
                        | (rows[:, 1] > cal.args.takeoff_climb)).mean()),
        "escapes_min": escapes / max(seconds - 1.0, 1e-6) * 60.0,
        "wobble": float(rows[:, 2].std()),
    }
    if threat:
        stim = cal.visual_field.visibility(0.0) * 30.0
        brain.reset()
        first = None
        for step in range(int(2.0 * CONTROL_HZ)):
            rates = stim if step >= CONTROL_HZ else silence
            brain.advance(rates, duration_ms=dt * 1000.0)
            if step >= CONTROL_HZ and first is None and brain.escape_triggered():
                first = (step - CONTROL_HZ + 1) * dt
        result["escape_s"] = first
    brain.spontaneous_hz = saved
    brain.reset()
    return result


def offline(lab, bridge_argv):
    import demos.minecraft_bridge as bridge
    sys.argv = ["minecraft_bridge.py", "--record", "offline"] + bridge_argv
    args = bridge.parse_args()
    bridge.apply_body(args.body)
    cal = bridge.calibrate(args)
    cal.args = args
    spec = DRUGS[lab.drug]
    effect = effect_of(lab)
    dose = Dose(cal.brain, transmitter_masks(cal.circuit, args.circuit), effect)
    print(f"\n{spec['what']}: нейронов {dose.neurons}, при полной дозе {describe(effect)}")
    noises = (("сытая", cal.base_noise), ("голодная", cal.base_noise + cal.hungry_extra))
    print(f"пороги: ходит выше {cal.perch_thrust:.2f}, взлетает выше "
          f"{cal.takeoff_thrust:.2f} или при подъёме {args.takeoff_climb:.2f}")
    head = (f"{'доза':>5} {'муха':>9} {'мозг Гц':>8} {'p99 Гц':>7} {'возб':>5} "
            f"{'ходит':>6} {'летит':>6} {'побегов/мин':>11} {'шатание':>7} {'побег через':>11}")
    print(head)
    ceiling = None
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        dose.set(level * lab.dose)
        for name, noise in noises:
            r = measure(cal, lab.seconds, noise, threat=(name == "сытая"))
            if level == 0.0 and name == "сытая":
                ceiling = r["lively_p99"]
            late = r.get("escape_s")
            late = "-" if "escape_s" not in r else ("нет" if late is None else f"{late:.2f} с")
            print(f"{level * lab.dose:>5.2f} {name:>9} {r['brain_hz']:>8.2f} "
                  f"{r['lively_p99']:>7.1f} {r['arousal']:>5.2f} {r['walks']:>6.0%} "
                  f"{r['flies']:>6.0%} {r['escapes_min']:>11.1f} {r['wobble']:>7.3f} {late:>11}",
                  flush=True)
    dose.set(0.0)
    if ceiling:
        print(f"\nРендер с одинаковой яркостью для трезвой и отравленной: "
              f"--max-hz {ceiling:.0f} --inner-max-hz {INNER_MAX_HZ:.0f}")


#в игре

class BodyEffects:
    """Паралич и шатание: подмена steer моста (см. блок констант выше)."""

    def __init__(self, bridge, intake, paralysis_s, stagger):
        from flygame.constants import STUN_FALL_MAX, STUN_FALL_NEAR, STUN_GRAVITY
        self.bridge, self.intake = bridge, intake
        self.gravity, self.fall_max, self.fall_near = STUN_GRAVITY, STUN_FALL_MAX, STUN_FALL_NEAR
        schedule = intake.schedule
        self.paralysis_at = (schedule.after_s + schedule.ramp_s + paralysis_s
                             if paralysis_s is not None and paralysis_s >= 0 else None)
        self.stagger = float(stagger or 0.0)
        self.rng = np.random.default_rng()
        self.world = None
        self.flipped = False
        self.flipped_at = 0.0
        self.paralyzed = False
        self.tumble_until = 0.0
        self.fall = 0.0
        self.checked = {}
        self.yaw_rate = 0.0
        self.bob = 0.0
        self._steer = bridge.steer
        bridge.steer = self.steer

    # переворот через тот же мир, что у моста
    def flip(self, on, now):
        if self.world is None or (on == self.flipped and not on):
            return
        self.bridge.safe_flip(self.world, on)
        self.flipped, self.flipped_at = on, now

    def unflip_before_close(self):
        if self.flipped and self.world is not None:
            self.bridge.safe_flip(self.world, False)
            self.flipped = False

    def lie(self, body, dt, terrain):
        """Лежит: не управляет телом и падает, как оглушённая."""
        body.vx = body.vz = body.vy = 0.0
        body.cruise = 0.0
        body.pending_steps = 0
        if terrain is None:
            return
        if terrain.stuck:
            body.y += 0.5
            self.fall = 0.0
        elif terrain.clear_below:
            if terrain.above_ground:
                self.fall = min(self.fall + self.gravity * dt, self.fall_max)
            else:
                self.fall = min(max(self.fall, 1.0), self.fall_near)
            drop = self.fall * dt
            if terrain.at is not None:
                depth = 2.5 if terrain.above_ground else 1.0
                known = math.floor(terrain.at[1] - depth)
                if drop > body.y - known:
                    drop = max(0.0, body.y - known)
                    self.fall = min(self.fall, self.fall_near)   # ждём снимок
            body.y -= drop
        else:
            self.fall = 0.0
            floor = math.floor(body.y)
            if body.y - floor > 0.02 and self.room_below(body, floor + 0.01):
                body.y = max(floor + 0.01, body.y - self.fall_near * dt)

    def room_below(self, body, target_y):
        """Свободно ли квадрату глаз, если опустить тело до target_y."""
        from mcworld import shape
        ey = target_y + shape.EYE_HEIGHT
        half = shape.EYE_BOX_HALF
        cells = sorted({(math.floor(body.x + dx), math.floor(ey), math.floor(body.z + dz))
                        for dx in (-half, half) for dz in (-half, half)})
        key = tuple(cells)
        if key not in self.checked:
            if self.world is None or not hasattr(self.world, "command"):
                return False
            chain = "".join(f"positioned {x + 0.5:.1f} {y + 0.5:.1f} {z + 0.5:.1f} "
                            f"if block ~ ~ ~ minecraft:air " for x, y, z in cells)
            try:
                answer = self.world.command(f"execute {chain}") or ""
            except Exception:
                answer = ""
            self.checked[key] = "passed" in answer.lower()
        return self.checked[key]

    def steer(self, body, motor, dt, home, args, terrain=None, state="flying",
              hold=False):
        now = time.perf_counter()
        started = self.intake.started
        t = 0.0 if started is None else now - started
        level = self.intake.dose.level

        self.intake.body_code = (1 if self.paralyzed else
                                 2 if now < self.tumble_until else 0)
        if self.paralysis_at is not None and t >= self.paralysis_at:
            self.intake.body_code = 1
            if not self.paralyzed:
                self.paralyzed = True
                print("\n  ПАРАЛИЧ: тело больше не слушается, мозг работает дальше")
            if not self.flipped or now - self.flipped_at > REFLIP_S:
                self.flip(True, now)
            self.lie(body, dt, terrain)
            return

        if self.stagger > 0 and level > 0:
            k = self.stagger * level
            self.yaw_rate += (-self.yaw_rate / STAGGER_YAW_TAU * dt
                              + STAGGER_YAW * k * math.sqrt(2.0 * dt / STAGGER_YAW_TAU)
                              * self.rng.standard_normal())
            self.bob += (-self.bob / STAGGER_BOB_TAU * dt
                         + STAGGER_BOB * k * math.sqrt(2.0 * dt / STAGGER_BOB_TAU)
                         * self.rng.standard_normal())
            if now < self.tumble_until:
                self.lie(body, dt, terrain)
                return
            if self.flipped:
                self.flip(False, now)
            if (state in ("perched", "walking") and level >= TUMBLE_FROM
                    and self.rng.random() < dt * self.stagger * (level - 0.4)
                    / (0.6 * TUMBLE_EVERY_S)):
                self.tumble_until = now + self.rng.uniform(*TUMBLE_S)
                self.intake.body_code = 2
                self.flip(True, now)
                print("\n  ПЬЯНАЯ: завалилась на спину")
                self.lie(body, dt, terrain)
                return
            if state in ("walking", "flying", "landing"):
                body.yaw = (body.yaw + self.yaw_rate * dt) % 360.0

        self._steer(body, motor, dt, home, args, terrain, state, hold=hold)

        if (self.stagger > 0 and level > 0 and state == "flying" and terrain is not None
                and (terrain.above_ground or self.bob > 0)):
            body.y += self.bob * dt


class Graft:
    """Синестезия в мозге: новые синапсы от нейронов доли к цели детекторов."""

    def __init__(self, cal, k):
        from scipy import sparse as sp
        brain, c, odor = cal.brain, cal.circuit, cal.odor
        self.brain = brain
        self.sources = self.pick_sources(cal)
        types_in, side_in = c.cell_type[c.input_idx], c.side[c.input_idx]
        mean_out = {}
        for side in ("left", "right"):
            loom = np.flatnonzero(np.isin(types_in, GRAFT_LOOM_TYPES) & (side_in == side))
            mean_out[side] = np.asarray(brain._base_in[:, loom].mean(axis=1)).ravel()
        rows, cols, vals = [], [], []
        where = {int(n): i for i, n in enumerate(c.net_idx)}
        for n in self.sources:
            column = mean_out.get(str(c.side[n]))
            if column is None:
                continue
            hit = np.flatnonzero(column)
            rows += hit.tolist()
            cols += [where[int(n)]] * hit.size
            vals += (column[hit] * k).tolist()
        self.extra = sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                                   shape=brain._base_net.shape)
        self.sober_net = brain._base_net
        self.level = 0.0
        self.synapses = int(self.extra.nnz)

    @staticmethod
    def pick_sources(cal):
        """Нейроны доли: молчат на чистом воздухе, работают в 2 блоках от еды."""
        from flygame.constants import ODOR_STRENGTH
        brain, odor = cal.brain, cal.odor
        found = []
        rates = {}
        for name, concentration in (("air", 0.0), ("food", ODOR_STRENGTH / 4.0)):
            brain.reset()
            acc = []
            for step in range(60):
                brain.advance(odor.project({"fruit": concentration}), 50.0)
                if step >= 20:
                    acc.append(brain.watch_rates_hz.copy())
            rates[name] = np.mean(acc, axis=0)
        brain.reset()
        for i, n in enumerate(odor.level_idx):
            if (rates["air"][i] < GRAFT_AIR_HZ and rates["food"][i] > GRAFT_FOOD_HZ
                    and cal.circuit.sign[n] >= 0):
                found.append(int(n))
        return found

    def set(self, level):
        brain = self.brain
        if level <= 0.0:
            brain._base_net = self.sober_net
        else:
            brain._base_net = (self.sober_net + self.extra * np.float32(level)).tocsr()
        brain._apply_gain()
        self.level = float(level)


class Synesthesia:
    """Запах еды -> ещё и детекторы надвигания, с той стороны, откуда пахнет."""

    def __init__(self, bridge, loom_hz, start, no_smell):
        self.loom_hz, self.start, self.no_smell = float(loom_hz), float(start), no_smell
        self.level = 0.0
        self.brain = None
        self.body, self.sources = None, []
        self.noted = 0.0
        self.peak = 0.0
        syn = self
        original = bridge.odor_concentration

        def odor_concentration(body, sources):
            syn.body, syn.sources = body, sources
            return original(body, sources)
        bridge.odor_concentration = odor_concentration

        class SynesthesiaMind(bridge.Mind):
            def smell(self, rates, concentration):
                nose = self.odor.project({"fruit": concentration})
                if syn.level <= 0.0:
                    return rates + nose
                if not syn.no_smell:
                    rates = rates + nose
                drive = syn.drive(concentration, self.odor) * syn.level
                if drive > 0.0 and syn.body is not None and syn.sources:
                    seen = self.encoder.field.visibility(syn.azimuth())
                    rates = np.maximum(rates, drive * seen)
                    syn.note(drive)
                return rates
        bridge.Mind = SynesthesiaMind

    def set(self, level):
        self.level = float(level)

    def drive(self, concentration, odor):
        """Гц на детекторах: отклик рецепторов на запах, выше порога start."""
        if concentration <= 0.0:
            return 0.0
        top = math.log1p(odor.reference / odor.half_scale)
        response = min(1.0, math.log1p(concentration / odor.half_scale) / top)
        if response <= self.start:
            return 0.0
        return self.loom_hz * (response - self.start) / (1.0 - self.start)

    def azimuth(self):
        """Откуда пахнет: вклады кусков с весом как у запаха (1/d^2)."""
        body = self.body
        vx = vz = 0.0
        for x, _y, z in self.sources:
            dx, dz = x - body.x, z - body.z
            d = max(math.hypot(dx, dz), 0.7)
            vx += dx / d / d ** 2
            vz += dz / d / d ** 2
        if math.hypot(vx, vz) < 1e-6:
            return 0.0                  # еда прямо под ногами
        az = body.azimuth_to((body.x + vx, body.y, body.z + vz))
        return max(-SYN_EYE_EDGE, min(SYN_EYE_EDGE, az))

    def note(self, drive):
        now = time.perf_counter()
        self.peak = max(self.peak, drive)
        if drive >= SYN_NOTE_HZ and now - self.noted >= SYN_NOTE_S:
            self.noted = now
            print(f"\n  СИНЕСТЕЗИЯ: еда \"надвигается\" - {drive:.0f} Гц на детекторах")


def play_graft(lab, bridge_argv):
    import demos.minecraft_bridge as bridge
    original = bridge.calibrate
    k = GRAFT_K if lab.graft_k is None else lab.graft_k

    def calibrate_then_graft(args):
        cal = original(args)
        graft = Graft(cal, k)
        names = ", ".join(f"{cal.circuit.cell_type[n]}/{cal.circuit.side[n]}"
                          for n in graft.sources)
        print(f"ОПЫТ: синестезия в мозге - {len(graft.sources)} нейронов доли ({names}) "
              f"получают выходы детекторов надвигания, x{k:g}: {graft.synapses} новых "
              f"связей; обычная муха {lab.after:.0f} с")
        schedule = Schedule(1.0, lab.after, lab.ramp * 60.0)
        intake = Intake(graft, schedule, "синестезия", dose_file=lab.dose_file or None,
                        tape=getattr(args, "record", None), word="сила", label="Синестезия")
        intake.marked = list(graft.sources)     # рендер подсветит их
        return cal

    bridge.calibrate = calibrate_then_graft
    sys.argv = ["minecraft_bridge.py"] + bridge_argv
    bridge.main()


def play_synesthesia(lab, bridge_argv):
    if lab.via == "brain":
        play_graft(lab, bridge_argv)
        return
    import demos.minecraft_bridge as bridge
    syn = Synesthesia(bridge, lab.loom_hz, lab.loom_from, lab.no_smell)
    original = bridge.calibrate

    def calibrate_then_rewire(args):
        cal = original(args)
        syn.brain = cal.brain
        schedule = Schedule(1.0, lab.after, lab.ramp * 60.0)
        print(f"ОПЫТ: синестезия - запах еды ещё и на детекторы надвигания, у самой "
              f"еды до {lab.loom_hz:g} Гц" + ("; нос выключен" if lab.no_smell else "")
              + f"; обычная муха {lab.after:.0f} с")
        Intake(syn, schedule, "синестезия", dose_file=lab.dose_file or None,
               tape=getattr(args, "record", None), word="сила", label="Синестезия")
        return cal

    bridge.calibrate = calibrate_then_rewire
    sys.argv = ["minecraft_bridge.py"] + bridge_argv
    try:
        bridge.main()
    finally:
        print(f"Синестезия: сильнее всего еда \"надвигалась\" на {syn.peak:.0f} Гц")


def play(lab, bridge_argv):
    if lab.synesthesia:
        play_synesthesia(lab, bridge_argv)
        return
    import demos.minecraft_bridge as bridge
    spec = DRUGS[lab.drug]
    effect = effect_of(lab)
    original = bridge.calibrate
    paralysis_s = spec["paralysis_s"] if lab.paralysis is None else lab.paralysis
    stagger = spec["stagger"] if lab.stagger is None else lab.stagger
    holder = {}

    def capturing(cls):
        class Captured(cls):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                if "world" not in holder:
                    holder["world"] = self
                    real_close = self.close

                    def close():
                        effects = holder.get("effects")
                        if effects is not None:
                            effects.unflip_before_close()
                        return real_close()
                    self.close = close
                    if holder.get("effects") is not None:
                        holder["effects"].world = self
        Captured.__name__ = cls.__name__
        return Captured

    bridge.MinecraftWorld = capturing(bridge.MinecraftWorld)
    bridge.FakeWorld = capturing(bridge.FakeWorld)

    def calibrate_then_dose(args):
        cal = original(args)
        dose = Dose(cal.brain, transmitter_masks(cal.circuit, args.circuit), effect)
        schedule = Schedule(lab.dose, lab.after, lab.ramp * 60.0)
        print(f"ОПЫТ: {spec['what']}, нейронов {dose.neurons}, при полной дозе "
              f"{describe(effect)}; трезвая {lab.after:.0f} с, "
              f"доза {lab.dose:.0%} за {lab.ramp:g} мин")
        intake = Intake(dose, schedule, spec["what"], dose_file=lab.dose_file or None,
                        tape=getattr(args, "record", None))
        effects = BodyEffects(bridge, intake, paralysis_s, stagger)
        effects.world = holder.get("world")
        holder["effects"] = effects
        if effects.paralysis_at is not None:
            print(f"ОПЫТ: паралич через {paralysis_s:g} с полной дозы "
                  f"(на {effects.paralysis_at:.0f}-й секунде опыта)")
        if effects.stagger > 0:
            print(f"ОПЫТ: шатание x{effects.stagger:g}, с дозы {TUMBLE_FROM:.0%} "
                  f"заваливается на спину")
        return cal

    bridge.calibrate = calibrate_then_dose
    sys.argv = ["minecraft_bridge.py"] + bridge_argv
    bridge.main()


def main():
    own, bridge_argv = split_argv(sys.argv[1:])
    lab = parse_lab_args(own)
    if lab.offline and lab.synesthesia:
        sys.exit("--offline - для препаратов; силу синестезии подбирали замером "
                 "(см. SYN_LOOM_HZ)")
    if lab.offline:
        offline(lab, bridge_argv)
    else:
        play(lab, bridge_argv)


if __name__ == "__main__":
    main()
