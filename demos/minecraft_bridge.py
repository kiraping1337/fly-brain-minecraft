"""Мост между Minecraft и схемой убегания.

Запуск:
    python demos/minecraft_bridge.py --spawn      # создать муху и начать
    python demos/minecraft_bridge.py              # муха уже есть в мире
    python demos/minecraft_bridge.py --simulate   # без сервера, самопроверка

Ctrl+C для выхода.

"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain import Threat  # noqa: E402
from flygame.airflow import air_movers  # noqa: E402
from flygame.calibration import calibrate  # noqa: E402
from flygame.constants import (  # noqa: E402
    BODY_MAX_HEALTH, BOLDNESS, CLIMB_TAKEOFF, CONTROL_HZ, EAT_HOLD_S,
    FLIGHT_HUNGER, FLOOR_BELOW_HOME, FOOD_ITEMS, FOOD_NAMES, HUNGER_CURVE,
    HUNGER_MINUTES, HUNGER_STARVE_AT, HUNGER_STARVE_DAMAGE,
    HUNGER_STARVE_EVERY_S, HUNGRY, K_AIRFLOW, K_SATURATION, KNOCKBACK_MAX,
    LAND_ON_ODOR, LEASH, MAX_SPEED, MEAL_SEARCH_S, ODOR_DIVE_RETREAT,
    ODOR_EVERY, ODOR_SLOW_S, ODOR_SMOOTH_S, PERCH_AFTER_S, PERCH_FACTOR,
    RECHECK_MIN_S, SPONTANEOUS_HZ, STARVE_REFERENCE_HEALTH, STEP_FREQ_MAX,
    STEP_LENGTH, STUN_FALL_MAX, STUN_FALL_NEAR, STUN_GRAVITY, STUN_MAX_S,
    STUN_MIN_S, STUN_THRESHOLD, TAKEOFF_FACTOR, TAU_AROUSAL_FALL,
    TAU_AROUSAL_RISE, TAU_TRAVEL, TERRAIN_RECHECK_DEG, VIGILANCE,
    WATER_NOTE_S, WOUND_FADE_S, WOUND_NOISE_SHARE, WOUND_PER_HIT)
from flygame.geometry import wrap_deg  # noqa: E402
from flygame.memory import remember_meal  # noqa: E402
from flygame.mind import Mind  # noqa: E402
from flygame.motion import push_body, steer  # noqa: E402
from flygame.odor import (  # noqa: E402
    bilateral_odor, odor_concentration, touching_food)
from mcworld import gateway, ident, shape  # noqa: E402
from mcworld.constants import (  # noqa: E402
    BODY_TAG, PASSABLE_BLOCKS, PROJECTILE_TYPES, RCON_HOST, RCON_PORT)
from mcworld.display import (  # noqa: E402
    HUNGER_FORMAT, STATS_EVERY_S, board_hide, board_set, board_show,
    label_create, label_follow, label_remove, label_update, report_commands,
    report_loop, report_shared, safe_flip, show_status, write_hunger_file)
from mcworld.fake import FakeWorld  # noqa: E402
from mcworld.ident import apply_name  # noqa: E402
from mcworld.link import QUIET_RULES  # noqa: E402
from mcworld.poller import WorldPoller  # noqa: E402
from mcworld.shape import BODIES, apply_body  # noqa: E402
from mcworld.world import MinecraftWorld  # noqa: E402


#главный цикл

# Сколько насыщает свой кусок в гонке (--food-only): как ломтик арбуза.
RACE_FOOD_WORTH = "minecraft:melon_slice"


def parse_args():
    parser = argparse.ArgumentParser(description="Мост Minecraft <-> схема убегания")
    parser.add_argument("--spawn", action="store_true",
                        help="создать муху перед игроком и начать")
    parser.add_argument("--simulate", action="store_true",
                        help="без сервера: проверка геометрии и цикла")
    parser.add_argument("--no-act", action="store_true",
                        help="не двигать тело, только показывать")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="остановиться через N секунд (0 = до Ctrl+C)")
    parser.add_argument("--k-sat", type=float, default=K_SATURATION,
                        help="чувствительность: меньше - пугливее")
    parser.add_argument("--spontaneous", default=SPONTANEOUS_HZ,
                        help='фоновый шум нейронов: число в Гц или "auto" '
                             '(0 = муха замирает полностью)')
    parser.add_argument("--k-wind", type=float, default=K_AIRFLOW,
                        help="чувствительность к потоку воздуха: меньше - острее")
    parser.add_argument("--ignore", choices=("none", "mobs", "player", "all"),
                        default="none",
                        help="ОТЛАДКА: кого муха не видит и не чувствует. "
                             "Нужно, чтобы проверять поиск еды отдельно от "
                             "побега. В логах чаще сгоняет игрок, а не мобы, "
                             "поэтому есть и player, и all")
    parser.add_argument("--keep-feedback", action="store_true",
                        help="не выключать вывод команд в чат и лог")
    parser.add_argument("--perch-factor", type=float, default=PERCH_FACTOR,
                        help="во сколько раз выше фона держится порог посадки")
    parser.add_argument("--takeoff-factor", type=float, default=TAKEOFF_FACTOR,
                        help="во сколько раз выше фона нужно, чтобы взлететь")
    parser.add_argument("--record", default=None, metavar="ФАЙЛ",
                        help="писать активность всех нейронов для визуализации")
    parser.add_argument("--turn-smoothing", type=float, default=TAU_TRAVEL,
                        help="сглаживание поворотов, с: больше - плавнее")
    parser.add_argument("--step-length", type=float, default=STEP_LENGTH,
                        help="длина одного шага в блоках")
    parser.add_argument("--step-rate", type=float, default=STEP_FREQ_MAX,
                        help="шагов в секунду на полном ходу")
    parser.add_argument("--takeoff-climb", type=float, default=CLIMB_TAKEOFF,
                        help="подъём выше этого - взлетает, ниже - уходит пешком")
    parser.add_argument("--idle-hz", type=float, default=60.0,
                        help="какую активность выхода держать в покое, Гц")
    parser.add_argument("--speed", type=float, default=MAX_SPEED,
                        help="скорость при полной тяге, блоков/с")
    parser.add_argument("--leash", type=float, default=LEASH,
                        help="радиус привязи, блоков (0 = отпустить совсем)")
    parser.add_argument("--floor", type=float, default=FLOOR_BELOW_HOME,
                        help="на сколько блоков ниже места старта может "
                             "опуститься полёт; для холмистого острова поднять")
    parser.add_argument("--hunger", type=float, default=0.0,
                        help="голод на старте: 0 сытая .. 1 голодная")
    parser.add_argument("--hunger-minutes", type=float, default=HUNGER_MINUTES,
                        help="за сколько минут сидя муха проголодается "
                             "(в полёте вдвое быстрее; 0 = голод не растёт)")
    parser.add_argument("--body", choices=sorted(BODIES), default="bee",
                        help="тело мухи в игре: bee - пчела, chicken - курица, "
                             "chick - цыплёнок, allay - тихоня, bat - летучая мышь")
    parser.add_argument("--hunger-file", default=None,
                        help="куда писать голод для OBS; пустая строка - не писать. "
                             "По умолчанию output/hunger.txt, у мухи с --name - "
                             "output/hunger_ИМЯ.txt")
    parser.add_argument("--name", default="",
                        help="имя мухи (a-z, 0-9, _): свои теги тела, счета, строка "
                             "табло, надпись и файл голода. Нужно, когда мух "
                             "несколько; без имени - одна муха, как раньше")
    parser.add_argument("--seed", type=int, default=None,
                        help="зерно шума нейронов; по умолчанию случайное, "
                             "печатается при старте, чтобы прогон можно было повторить")
    parser.add_argument("--gateway", default=None, metavar="ХОСТ:ПОРТ",
                        help="говорить с сервером через шлюз гонки, а не напрямую; "
                             "ставит demos/race.py, руками не нужно")
    parser.add_argument("--spawn-at", type=float, nargs=4, default=None,
                        metavar=("X", "Y", "Z", "YAW"),
                        help="где создавать тело (с --spawn и при возрождении); "
                             "YAW - поворот как в игре по F3: 0 юг, 90 запад, "
                             "180 север, -90 восток. По умолчанию - перед игроком")
    parser.add_argument("--hunger-format", default=HUNGER_FORMAT,
                        help="что писать в файл; {percent} - голод в процентах. "
                             "Например \"{percent}\" - только число")
    parser.add_argument("--no-hunger-board", action="store_true",
                        help="не показывать таблицу с голодом сбоку экрана")
    parser.add_argument("--hunger-label", action="store_true",
                        help="надпись с голодом над самой мухой, в мире: её не "
                             "прячет F1, но она попадает в кадр")
    parser.add_argument("--starve", type=float, default=HUNGER_STARVE_DAMAGE,
                        help=f"урон от голода раз в "
                             f"{HUNGER_STARVE_EVERY_S:.0f} с при полном "
                             f"голоде; 0 - выключить")
    parser.add_argument("--hunger-curve", type=float, default=HUNGER_CURVE,
                        help="показатель кривой голода: 1.0 - линейно, "
                             "0.5 - поиск начинается заметно раньше")
    parser.add_argument("--boldness", type=float, default=BOLDNESS,
                        help="насколько голод ослабляет вход от угрозы, 0..1")
    parser.add_argument("--hungry-noise", default="auto",
                        help='прибавка шума у голодной мухи, Гц, или "auto"')
    parser.add_argument("--invulnerable", action="store_true",
                        help="сделать тело неуязвимым: урона не будет совсем")
    parser.add_argument("--wound-fade", type=float, default=WOUND_FADE_S,
                        help="за сколько секунд забывается рана (0 = сразу)")
    parser.add_argument("--vigilance", type=float, default=VIGILANCE,
                        help="насколько рана обостряет чувство угрозы")
    parser.add_argument("--respawn", action="store_true",
                        help="после гибели создавать тело заново")
    parser.add_argument("--no-meal-search", action="store_true",
                        help="без локального поиска после укуса (25 с медленного "
                             "кружения у съеденного куска). Для гоночной дорожки: "
                             "замер field_scenarios trail - до финиша на треть "
                             "быстрее, с дорожки не сходит. Память о местах "
                             "остаётся: без неё муха теряет дорожку")
    parser.add_argument("--food-only", default=None, metavar="ПРЕДМЕТ",
                        help="ГОНКА: чуять и есть только этот вид кусков "
                             "(minecraft:apple и т.п.), блоки еды не трогать; "
                             "сытость от куска - как от ломтика арбуза. Ставит "
                             "race.py --own-food, одной мухе не нужен")
    parser.add_argument("--no-see-flies", action="store_true",
                        help="не видеть других мух гонки (ни глазами, ни потоком "
                             "воздуха); мобы и игрок остаются")
    parser.add_argument("--debug", action="store_true",
                        help="печатать всё, что муха видит на каждом опросе")
    parser.add_argument("--circuit", default=str(ROOT / "data" / "circuit.npz"))
    return parser.parse_args()


def main():
    Session(parse_args()).run()


# Код состояния в записи мозга (--record): его читает tools/render_brain.py.
STATE_CODE = {"flying": 0, "landing": 1, "perched": 2, "walking": 3,
              "eating": 4, "stunned": 5}


class Session:
    """Одна муха в мире: подготовка, главный цикл и итоги."""

    def __init__(self, args):
        self.args = args

    def run(self):
        if not self.setup():
            return
        try:
            while self.tick():
                pass
        except KeyboardInterrupt:
            print("\n\nостановлено")
        finally:
            self.finish()

    #подготовка

    def setup(self):
        """Всё до цикла. False - тела в мире нет, работать не с чем."""
        self.apply_identity()
        self.build_mind()
        self.connect()
        if not self.find_body():
            return False
        self.check_server()
        self.start_loop()
        return True

    def apply_identity(self):
        args = self.args
        apply_body(args.body)
        try:
            apply_name(args.name)
        except ValueError as exc:
            sys.exit(f"--name: {exc}")
        if args.hunger_file is None:
            args.hunger_file = ident.HUNGER_FILE
        if ident.NAME:
            print(f"Муха {ident.NAME}: тело {shape.BODY_ENTITY}, тег {ident.OWN_TAG}, "
                  f"счёт {ident.PROBE_OBJECTIVE}")

    def build_mind(self):
        args = self.args
        cal = calibrate(args)
        self.circuit, self.encoder, self.wind, self.odor = (
            cal.circuit, cal.encoder, cal.wind, cal.odor)
        self.brain, self.decoder, self.odor_rest = cal.brain, cal.decoder, cal.odor_rest
        self.perch_thrust, self.takeoff_thrust = cal.perch_thrust, cal.takeoff_thrust
        self.base_noise, self.hungry_extra = cal.base_noise, cal.hungry_extra
        self.perch_hungry = cal.perch_hungry
        self.mind = Mind(self.brain, self.decoder, self.encoder, self.wind, self.odor)
        seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
        self.brain.rng = np.random.default_rng(seed)
        print(f"Зерно шума: {seed} (повторить: --seed {seed})")

    def connect(self):
        args = self.args
        if args.simulate:
            self.world = FakeWorld()
            print("\nРежим самопроверки: сервер не нужен.\n")
        else:
            if args.gateway:
                host, _, port = args.gateway.rpartition(":")
                gateway.GATEWAY = (host, int(port))
                print(f"\nПодключаюсь к шлюзу гонки {args.gateway} ...")
            else:
                print(f"\nПодключаюсь к {RCON_HOST}:{RCON_PORT} ...")
            world = self.world = MinecraftWorld()
            sender = self.sender = MinecraftWorld()
            world.spawn_at = sender.spawn_at = args.spawn_at
            world.see_flies = not args.no_see_flies
            if args.no_meal_search:
                print("Без локального поиска после укуса (--no-meal-search)")
            if args.food_only:
                if args.food_only not in FOOD_ITEMS:
                    sys.exit(f"--food-only: нет такой еды {args.food_only}; "
                             f"есть: {', '.join(FOOD_ITEMS)}")
                world.food_items = (args.food_only,)
                world.food_only = True
                print(f"Ест только: {FOOD_NAMES.get(args.food_only, args.food_only)} "
                      f"(--food-only), сытость как от ломтика арбуза")
            if args.no_see_flies:
                print("Других мух не видит (--no-see-flies)")
            if not args.keep_feedback:
                muted = world.quiet()
                if len(muted) == len(QUIET_RULES):
                    print(f"Вывод команд в чат и лог выключен: {', '.join(muted)} "
                          f"(--keep-feedback вернёт)")
                else:
                    print(f"!! Сервер принял не все правила, которые глушат вывод команд "
                          f"(принял: {', '.join(muted) or 'ни одного'}) - команды мухи "
                          f"будут сыпаться в чат и лог сервера и нагружать его")
            if args.spawn:
                world.spawn_body(invulnerable=args.invulnerable)
            print("Связь есть. Ctrl+C для выхода.\n")
            if not args.no_hunger_board:
                board_show(world)

        if not args.simulate:
            count = self.world.command(f"execute if entity @e[tag={BODY_TAG}]") or ""
            if "count" in count.lower():
                print(f"Тел с тегом {BODY_TAG} в мире: {count.strip()}")

        if args.ignore != "none":
            who = {"mobs": "мобов", "player": "игрока", "all": "всех"}[args.ignore]
            print(f"!! ОТЛАДКА: муха не видит и не чувствует {who}. "
                  f"Это НЕ рабочий режим, для съёмки выключить.")

    def find_body(self):
        args = self.args
        body = self.body = self.world.locate_body()
        if body is None:
            print(f"Муха в мире не найдена. Создайте её командой в чате:\n"
                  f"  /summon {shape.BODY_ENTITY} ~ ~1 ~3 "
                  f'{{NoAI:1b,NoGravity:1b,Tags:[{ident.SPAWN_TAGS}]}}\n'
                  f"или запустите с --spawn")
            return False
        body.odor_rest = self.odor_rest
        body.visual_mask = self.circuit.visual_mask
        if args.simulate:
            self.sender = self.world
        self.home = body.position
        body.hunger = float(np.clip(args.hunger, 0.0, 1.0))
        return True

    def check_server(self):
        """Здоровье тела и что сервер умеет: переворот, блоки, снаряды."""
        args, world, body = self.args, self.world, self.body
        self.full_health = BODY_MAX_HEALTH
        first = world.health()
        if first is not None:
            self.full_health = max(first, 1e-3)
            body.health = 1.0
        can_flip = (not args.invulnerable) and world.pick_flip_name()
        print("Оглушённая переворачивается вверх ногами"
              if can_flip else
              "Переворот вверх ногами недоступен: сервер не принял CustomName")
        unknown = world.check_passable()
        if unknown is None:
            print("Проходимые блоки: проверка не сработала, список оставлен целиком")
        elif unknown:
            print(f"Проходимые блоки: сервер не знает {len(unknown)} из "
                  f"{len(PASSABLE_BLOCKS)} - {', '.join(unknown)}")
        else:
            print(f"Проходимые блоки: сервер знает все {len(PASSABLE_BLOCKS)}")
        print("Снаряды: " + (f"{len(PROJECTILE_TYPES)} типа одним запросом"
                             if world.check_multi_type() else
                             "сервер не принял перечисление типов в селекторе, "
                             "отдельным запросом идёт только стрела"))
        print(f"Тело: здоровье {self.full_health:g}"
              + (", неуязвимо (--invulnerable)" if args.invulnerable else
                 f", рана забывается за {args.wound_fade:.0f} с, "
                 f"чувство угрозы до x{1 + args.vigilance:.1f}"))

    def start_loop(self):
        """Опрос мира в фоне и переменные цикла."""
        args, body = self.args, self.body
        self.tape_times, self.tape_rates, self.tape_state, self.tape_hunger = [], [], [], []

        poller = self.poller = WorldPoller(self.world, debug=args.debug)
        poller.watch_health = not args.invulnerable
        poller.full_health = self.full_health
        poller.start()

        self.control_dt = 1.0 / CONTROL_HZ
        self.last_index = 0
        self.last_poll_time = time.perf_counter()
        self.last_tick = time.perf_counter()
        started = self.started = time.perf_counter()
        self.peak_thrust = 0.0
        self.state = "flying"
        self.last_place = None
        self.calm_since = started
        self.landings = 0
        self.eating_until = 0.0
        self.stunned_until = 0.0
        self.fall_speed = 0.0
        self.flipped = False
        self.holding = False
        self.odor_sources = []
        self.mob_last = {}        # прошлое мировое положение каждого объекта
        self.poll_count = 0
        self.checked_yaw = body.travel_yaw
        self.last_starve = started
        self.last_recheck = started
        self.water_noted = -1e9
        self.water_refused = -1e9  # когда последний раз отказалась садиться на воду
        self.board_shown = None
        self.file_shown = None
        self.hunger_path = None
        if args.hunger_file:
            self.hunger_path = Path(args.hunger_file).resolve()
            self.hunger_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Голод пишется в файл: {self.hunger_path}")
            print("  в OBS: источник 'Текст (GDI+)' -> 'Читать из файла' -> этот файл")
        self.label_on = args.hunger_label and not args.simulate
        self.label_made = False
        self.stats_at = (started, 0, 0.0, {}, 0)
        self.loop = dict(ticks=0, worst_ms=0.0, brain_s=0.0, move_s=0.0, moves=0,
                         move_max=0.0)
        self.starve_ok = True
        self.shared_at = {}       # счётчики общего опроса на прошлой сводке

    #такт

    def tick(self):
        """Один такт управления. False - пора выходить."""
        args = self.args
        now = time.perf_counter()
        self.poll_count += 1
        self.report(now)
        if args.seconds and now - self.started > args.seconds:
            return False
        self.aim_probes(now)
        scan = self.poller.latest()
        threats = self.see(scan, now)
        self.starve(now)
        self.digest(now)
        if not self.survive(now):
            return False
        self.take_hits(scan, threats, now)
        motor, rates, wind_rates = self.think(threats, now)
        self.keep_safe(scan, now)
        self.decide(motor, scan, now)
        shown = self.act(motor, scan, now)
        self.move_body()
        self.show(threats, rates, motor, wind_rates, shown, now)
        self.record(shown, now)
        time.sleep(max(0.0, self.control_dt - (time.perf_counter() - now)))
        return True

    def report(self, now):
        """Раз в STATS_EVERY_S: команды на сервер, такт, общий опрос."""
        if now - self.stats_at[0] >= STATS_EVERY_S:
            span = now - self.stats_at[0]
            line, self.stats_at = report_commands(self.stats_at, now, self.poller.polls)
            print(f"\n  {line}")
            print(f"  {report_loop(self.loop, span, CONTROL_HZ)}")
            shared_line = report_shared(getattr(self.world, "shared_stats", {}),
                                        self.shared_at)
            if shared_line:
                print(f"  {shared_line}")

    def aim_probes(self, now):
        """Куда щупать рельеф и когда перемерить его вне очереди."""
        body, poller, state = self.body, self.poller, self.state
        self.world.set_lookahead(body.travel_yaw, body.speed,
                                 walking=state in ("walking", "perched", "eating"))

        moving = state in ("flying", "landing", "walking")
        if (moving and abs(wrap_deg(body.travel_yaw - self.checked_yaw)) > TERRAIN_RECHECK_DEG
                and now - self.last_recheck >= RECHECK_MIN_S):
            self.checked_yaw = body.travel_yaw
            self.last_recheck = now
            poller.recheck_terrain = True
        if body.want_recheck and now - self.last_recheck >= RECHECK_MIN_S:
            body.want_recheck = False
            self.last_recheck = now
            poller.recheck_terrain = True

    def see(self, scan, now):
        """Снимок мира -> угрозы относительно тела; свежий снимок - в энкодеры."""
        args, body = self.args, self.body
        visible = [
            s for s in scan.sightings
            if not (args.ignore == "all"
                    or (args.ignore == "player" and s.is_player)
                    or (args.ignore == "mobs" and not s.is_player))
        ]
        threats = [
            Threat(azimuth_deg=body.azimuth_to(s.position),
                   distance=body.distance_to(s.position),
                   size=s.size, key=s.name)
            for s in visible
        ]

        if scan.index != self.last_index:
            poll_dt_ms = (now - self.last_poll_time) * 1000.0
            self.encoder.update(threats, dt_ms=poll_dt_ms)
            self.wind.update(air_movers(visible, threats, body, self.mob_last,
                                        poll_dt_ms / 1000.0),
                             dt_ms=poll_dt_ms)
            self.last_poll_time = now
            self.last_index = scan.index
        return threats

    def starve(self, now):
        """Голодание: урон, пока голод упёрся в потолок."""
        args, body = self.args, self.body

        if (args.starve > 0 and body.hunger >= HUNGER_STARVE_AT
                and self.starve_ok and now - self.last_starve >= HUNGER_STARVE_EVERY_S):
            self.last_starve = now

            bite = args.starve * self.full_health / STARVE_REFERENCE_HEALTH
            self.poller.expect_damage(bite)
            if self.world.starve(bite):
                body.starved += bite
                print(f"\n  ГОЛОДАЕТ: снято {bite:.1f} здоровья "
                      f"из {self.full_health:g}, всего {body.starved:.1f}")
            else:
                self.starve_ok = False
                print("\n  Голодание недоступно: сервер не принял "
                      "команду damage")

    def digest(self, now):
        """Голод растёт со временем; съеденное его снимает."""
        args, body = self.args, self.body
        if args.hunger_minutes > 0:
            pace = FLIGHT_HUNGER if self.state in ("flying", "landing") else 1.0
            body.hunger = min(1.0, body.hunger + pace * self.control_dt
                              / (args.hunger_minutes * 60.0))
        for _, (kind, what, note) in self.poller.meals():
            self.eating_until = now + EAT_HOLD_S
            if kind == "block":

                print(f"\n  УКУС: {FOOD_NAMES.get(what, what)} {note}, "
                      f"голод {body.hunger:.2f}")
            else:
                before = body.hunger

                worth = (FOOD_ITEMS[RACE_FOOD_WORTH] if args.food_only
                         else FOOD_ITEMS.get(what, 0.1))
                body.hunger = max(0.0, body.hunger - worth)
                body.meals += 1

                body.meal_search = 0.0 if args.no_meal_search else MEAL_SEARCH_S

                remember_meal(body)
                print(f"\n  ЕСТ: {FOOD_NAMES.get(what, what)}, "
                      f"голод {before:.2f} -> {body.hunger:.2f}")

    def survive(self, now):
        """Гибель и возрождение. False - муха погибла и возрождаться не будет."""
        args, body, poller = self.args, self.body, self.poller

        if poller.died():

            print(f"\n  ПОГИБЛА: перед этим здоровье было "
                  f"{body.health:.0%}, ударов {body.hits}, "
                  f"прожила {now - self.started:.0f} с")
            if not args.respawn:
                return False
            fresh = self.world.spawn_body(invulnerable=args.invulnerable)
            if fresh is None:
                return False
            body.x, body.y, body.z = fresh.position
            body.health, body.wounded = 1.0, 0.0

            body.starved = 0.0
            body.step_terrain = None
            body.safe_pos = None
            body.places = []
            body.search_time = body.search_hold = 0.0
            self.flipped = False
            poller.revive()
            self.state, self.calm_since = "flying", now
        return True

    def take_hits(self, scan, threats, now):
        """Здоровье с сервера, удары: толчок, оглушение, рана."""
        args, body, poller = self.args, self.body, self.poller
        full_health = self.full_health
        known = poller.health_now()
        if known is not None:
            body.health = known / full_health
        for amount, left in poller.hits():
            share = amount / full_health
            body.health = left / full_health
            body.hits += 1
            body.wounded = min(1.0, body.wounded + share * WOUND_PER_HIT)

            push_body(body, threats, KNOCKBACK_MAX * min(share, 1.0))

            if scan.terrain.stuck:
                self.stunned_until = 0.0
                print(f"\n  УДАР {amount:.1f} в блоке: не оглушает, "
                      f"выбираюсь; здоровье {body.health:.0%}")
            elif share >= STUN_THRESHOLD:
                over = (share - STUN_THRESHOLD) / (1.0 - STUN_THRESHOLD)
                span = STUN_MIN_S + (STUN_MAX_S - STUN_MIN_S) * min(over, 1.0)
                self.stunned_until = max(self.stunned_until, now + span)
                print(f"\n  УДАР {amount:.1f}: оглушена на {span:.1f} с, "
                      f"здоровье {body.health:.0%}, рана {body.wounded:.2f}")
            else:
                print(f"\n  УДАР {amount:.1f}: здоровье {body.health:.0%}, "
                      f"рана {body.wounded:.2f}")
        if args.wound_fade > 0:
            body.wounded *= math.exp(-self.control_dt / args.wound_fade)
        else:
            body.wounded = 0.0

    def think(self, threats, now):
        """Чувства -> мозг -> команда. Возвращает команду и входы для вывода."""
        args, body, poller, mind = self.args, self.body, self.poller, self.mind

        poller.feeding = (not args.no_act and self.state in ("perched", "walking")
                          and body.hunger >= HUNGRY)

        urge = body.hunger ** args.hunger_curve
        self.brain.spontaneous_hz = (self.base_noise + self.hungry_extra * urge
                                     + self.hungry_extra * WOUND_NOISE_SHARE
                                     * body.wounded)
        boldness = ((1.0 - args.boldness * urge)
                    * (1.0 + args.vigilance * body.wounded))

        rates, wind_rates = mind.sense(threats, boldness)

        if self.odor.active:
            if self.poll_count % ODOR_EVERY == 0:
                self.odor_sources = self.world.probe_odor()
            concentration = odor_concentration(body, self.odor_sources)
            rates = mind.smell(rates, concentration)
            body.odor_side = bilateral_odor(body, self.odor_sources)
            body.touching = touching_food(body, self.odor_sources)
            if body.touching and body.hunger >= HUNGRY:
                poller.feed_now = True

        loop = self.loop
        elapsed_ms = (now - self.last_tick) * 1000.0
        self.last_tick = now
        loop["ticks"] += 1
        loop["worst_ms"] = max(loop["worst_ms"], elapsed_ms)
        thought = time.perf_counter()
        motor = mind.step(rates, min(elapsed_ms, 250.0))
        loop["brain_s"] += time.perf_counter() - thought

        if self.odor.active:
            level = mind.odor_level()
            if body.odor_level is None:
                body.odor_level = level
            if body.odor_slow == 0.0:
                body.odor_slow = level
            body.odor_level += (level - body.odor_level) * (
                1.0 - math.exp(-self.control_dt / ODOR_SMOOTH_S))
            body.odor_slow += (level - body.odor_slow) * (
                1.0 - math.exp(-self.control_dt / ODOR_SLOW_S))
        mind.adapt(motor, self.control_dt)
        self.peak_thrust = max(self.peak_thrust, motor.thrust)
        return motor, rates, wind_rates

    def keep_safe(self, scan, now):
        """Последнее безопасное место, возврат из блока, взлёт с воды."""
        body = self.body

        terrain_now = scan.terrain
        if terrain_now.at is not None and terrain_now is not body.safe_from:
            body.safe_from = terrain_now
            if not terrain_now.stuck:
                body.safe_pos = terrain_now.at
        if terrain_now.stuck and body.rescued_by is not terrain_now:
            body.rescued_by = terrain_now
            if body.safe_pos is not None:
                body.x, body.y, body.z = body.safe_pos
                body.vx = body.vz = body.vy = 0.0

                body.cruise = 0.0
                body.yaw = body.travel_yaw = (body.travel_yaw + 180.0) % 360.0
                print("\n  ЗАСТРЯЛА в блоке, возвращаю на прошлое безопасное место "
                      "и разворачиваю")
                self.poller.recheck_terrain = True

                self.state, self.calm_since = "flying", now
            else:
                print("\n  ЗАСТРЯЛА в блоке, поднимаюсь")
                self.state, self.calm_since = "flying", now

        if (terrain_now.wet_below and terrain_now is not body.lifted_by
                and self.state in ("perched", "walking", "eating")):
            body.lifted_by = terrain_now
            self.state, self.calm_since = "flying", now
            if now - self.water_noted >= WATER_NOTE_S:
                self.water_noted = now
                print("\n  НА ВОДЕ: взлетает")

    def decide(self, motor, scan, now):
        """Возбуждение и переходы: полёт, посадка, сидит, идёт."""
        args, body = self.args, self.body

        tau = (TAU_AROUSAL_RISE if motor.thrust > body.arousal
               else TAU_AROUSAL_FALL)
        body.arousal += (motor.thrust - body.arousal) * min(
            1.0, self.control_dt / tau)

        airborne = (motor.climb > args.takeoff_climb
                    or body.arousal > self.takeoff_thrust)

        state = self.state
        if state == "flying":
            urge_now = body.hunger ** args.hunger_curve
            land_at = self.perch_thrust + (self.perch_hungry - self.perch_thrust) * urge_now

            over_food = (body.hunger >= HUNGRY
                         and motor.retreat <= ODOR_DIVE_RETREAT
                         and body.odor_level is not None
                         and body.odor_level > body.odor_rest * LAND_ON_ODOR
                         and now - self.water_refused > PERCH_AFTER_S)
            if over_food:
                self.state = "landing"
            elif body.arousal >= land_at:
                self.calm_since = now
            elif now - self.calm_since > PERCH_AFTER_S:
                self.state = "landing"
        elif airborne:
            if state in ("perched", "walking"):
                print(f"\n  ВЗЛЁТ: возбуждение {body.arousal:.2f}, "
                      f"подъём {motor.climb:.2f}")
            self.state, self.calm_since = "flying", now
        elif state in ("perched", "walking"):

            if body.arousal >= self.perch_thrust:
                self.calm_since = now
                if state == "perched":
                    self.state = "walking"
            elif now - self.calm_since > PERCH_AFTER_S:
                self.state = "perched"
        elif state == "landing" and not scan.terrain.clear_below:
            if scan.terrain.wet_below:

                self.state, self.calm_since = "flying", now
                self.water_refused = now
                if now - self.water_noted >= WATER_NOTE_S:
                    self.water_noted = now
                    print("\n  НАД ВОДОЙ: не садится, летит дальше")
            else:
                self.state = "perched"
                self.landings += 1
                print(f"\n  ПОСАДКА #{self.landings}: возбуждение {body.arousal:.2f}")

    def act(self, motor, scan, now):
        """Тело: оглушённое падает, иначе - по команде мозга. Что показать."""
        args, body = self.args, self.body
        stunned = now < self.stunned_until
        if stunned:
            body.vx = body.vz = 0.0
            body.pending_steps = 0
            if scan.terrain.stuck:
                body.y += 0.5
                self.fall_speed = 0.0
            elif scan.terrain.above_ground:
                self.fall_speed = min(self.fall_speed + STUN_GRAVITY * self.control_dt,
                                      STUN_FALL_MAX)
                body.y -= self.fall_speed * self.control_dt
            elif scan.terrain.clear_below:
                self.fall_speed = min(self.fall_speed, STUN_FALL_NEAR)
                body.y -= self.fall_speed * self.control_dt
            else:
                self.fall_speed = 0.0
            if self.state in ("perched", "walking"):
                self.calm_since = now
            self.poller.watch_ground = True
            if not self.flipped:
                safe_flip(self.world, True)
                self.flipped = True
        else:
            self.fall_speed = 0.0
            self.poller.watch_ground = False
            if self.flipped:
                safe_flip(self.world, False)
                self.flipped = False
            self.holding = ((now < self.eating_until or
                             (body.touching and body.hunger >= HUNGRY))
                            and self.state in ("perched", "walking"))
            steer(body, motor, self.control_dt, self.home, args, scan.terrain,
                  self.state, hold=self.holding)
        return ("stunned" if stunned else
                "eating" if self.holding and not stunned else self.state)

    def move_body(self):
        """Телепорт тела, если оно сдвинулось, и надпись над ним."""
        if self.args.no_act:
            return
        body, sender, loop = self.body, self.sender, self.loop
        place = (round(body.x, 2), round(body.y, 2), round(body.z, 2),
                 round(body.yaw, 1))
        if place != self.last_place:
            moving = time.perf_counter()
            sender.move(body)
            waited = time.perf_counter() - moving
            loop["move_s"] += waited
            loop["moves"] += 1
            loop["move_max"] = max(loop["move_max"], waited)
            self.last_place = place

            self.world.body_at = place[:3]
            if self.label_on:
                if self.label_made:
                    label_follow(sender, place[:3])
                else:
                    label_create(sender, place[:3], body.hunger)
                    self.label_made = True

    def show(self, threats, rates, motor, wind_rates, shown, now):
        """Голод на табло, над мухой и в файле; строка состояния."""
        args, body = self.args, self.body
        percent = int(round(max(0.0, min(1.0, body.hunger)) * 100))
        if percent != self.board_shown:
            self.board_shown = percent
            if not args.no_hunger_board:
                board_set(self.sender, body.hunger)
            if self.label_made:
                label_update(self.sender, body.hunger)
        if self.hunger_path is not None and percent != self.file_shown:
            if write_hunger_file(self.hunger_path,
                                 args.hunger_format.format(percent=percent)):
                self.file_shown = percent

        show_status(now - self.started, body, threats, rates, motor,
                    wind_rates, shown, body.hunger, body.wounded,
                    args.ignore)

    def record(self, shown, now):
        """Кадр записи мозга (--record)."""
        if self.args.record:
            self.tape_times.append(now - self.started)
            self.tape_rates.append(self.brain.all_rates_hz.astype(np.float16))
            self.tape_state.append(STATE_CODE[shown])
            self.tape_hunger.append(self.body.hunger)

    #итоги

    def finish(self):
        """После цикла: вернуть тело, убрать надписи, сохранить запись, итоги."""
        args, body, world, poller = self.args, self.body, self.world, self.poller
        poller.stop()
        if self.flipped:
            safe_flip(world, False)
        try:
            if not args.no_hunger_board:
                board_hide(world)
            if self.label_made:
                label_remove(world)
        except Exception:
            pass
        world.close()
        if not args.simulate:
            self.sender.close()
        if args.record and self.tape_times:
            self.save_tape()
        line, _ = report_commands((self.started, 0, 0.0, {}, 0),
                                  time.perf_counter(), poller.polls)
        print(f"\nЗа сессию: {line}")
        shared_line = report_shared(getattr(world, "shared_stats", {}), {})
        if shared_line:
            print(f"  за сессию {shared_line}")
        print(f"Максимальная тяга за сессию: {self.peak_thrust:.2f}, "
              f"посадок: {self.landings}, рывков: {body.saccades}, "
              f"шагов: {body.steps}, съедено: {body.meals}, "
              f"голод в конце: {body.hunger:.2f}, "
              f"ударов получено: {body.hits}, "
              f"здоровье в конце: {body.health:.0%}"
              + (f", испорченных чтений здоровья: {poller.bad_readings}"
                 if poller.bad_readings else ""))
        for value, answer in poller.bad_values:
            print(f"Вместо здоровья пришло {value}: {answer!r}")
        if getattr(world, "mismatched", 0):
            print(f"Ответов не на свою команду: {world.mismatched} "
                  f"(отброшены)")
            for answer in world.bad_answers:
                print(f"  {answer!r}")

    def save_tape(self):
        path = Path(self.args.record)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times=np.asarray(self.tape_times, dtype=np.float32),
            rates=np.asarray(self.tape_rates, dtype=np.float16),
            state=np.asarray(self.tape_state, dtype=np.int8),
            hunger=np.asarray(self.tape_hunger, dtype=np.float16),
        )
        print(f"Запись активности: {path} "
              f"({len(self.tape_times)} кадров, {path.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
