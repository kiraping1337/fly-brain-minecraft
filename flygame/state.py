"""Состояние мухи и снимки мира: тело, кормные места, рельеф."""

import math
from dataclasses import dataclass, field

from .geometry import bearing_deg, wrap_deg


#состояние тела

@dataclass
class Body:
    """Кинематическое состояние мухи. Хозяин состояния - мост, не сервер."""
    x: float = 0.0
    y: float = 70.0
    z: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0        # текущая скорость, блоков/с
    vy: float = 0.0
    vz: float = 0.0
    arousal: float = 0.0   # сглаженная активность нисходящих нейронов
    landed: bool = False
    odor_level: float | None = None   # быстрый след уровня запаха, Гц на нейрон
    odor_slow: float = 0.0            # медленный след, к нему нос привыкает
    odor_rest: float = 0.0            # уровень доли на чистом воздухе
    cast_hold: float = 0.0            # с: после рывка поиска сначала пройти
    meal_search: float = 0.0          # с: локальный поиск после еды
    starved: float = 0.0              # сколько здоровья снял голод
    step_terrain: object = None       # снимок, по которому уже шагнули вверх
    safe_pos: tuple | None = None     # где снимок последний раз видел глаза в воздухе
    safe_from: object = None          # по какому снимку это место записано
    rescued_by: object = None         # по какому снимку уже спасали
    lifted_by: object = None          # по какому снимку уже взлетала с воды
    want_recheck: bool = False        # попросить внеочередной опрос рельефа
    places: list = field(default_factory=list)  # кормные места, см. Place
    place_idle: float = 0.0           # с у ближайшего места без еды
    search_time: float = 0.0          # с поиска без запаха и без цели
    search_hold: float = 0.0          # с до конца прямой пробежки поиска
    odor_side: float = 0.0            # разница запаха у антенн, + слева
    touching: bool = False            # ноги на еде
    visual_mask: object = None        # какие входные нейроны зрительные
    turn_debt: float = 0.0  # накопленное рассогласование курса, градусы
    saccade: float = 0.0    # сколько градусов осталось довернуть в рывке
    saccades: int = 0       # сколько рывков сделано за сессию
    step_phase: float = 0.0  # фаза шагательного цикла, 0..1
    pending_steps: int = 0   # сколько шагов осталось при устройстве на месте
    cruise: float = 0.0      # величина скорости полёта, блоков/с
    travel_yaw: float = 0.0  # сглаженное направление движения, градусы
    steps: int = 0           # сколько шагов сделано за сессию
    hunger: float = 0.0      # 0 сытая .. 1 голодная; НАШЕ число, не мозг
    meals: int = 0           # сколько кусков съедено за сессию
    health: float = 1.0      # доля здоровья по данным сервера
    wounded: float = 0.0     # 0 цела .. 1 только что крепко получила
    hits: int = 0            # сколько раз ударили за сессию

    @property
    def speed(self):
        return math.hypot(self.vx, self.vz)

    @property
    def position(self):
        return (self.x, self.y, self.z)

    def azimuth_to(self, target):
        """Куда объект относительно взгляда: 0 спереди, -90 слева, +90 справа."""
        return wrap_deg(bearing_deg(target[0] - self.x, target[2] - self.z) - self.yaw)

    def distance_to(self, target):
        return max(math.dist(self.position, target), 0.4)


@dataclass
class Sighting:
    """Существо в мировых координатах. Азимут считается позже, от тела."""
    name: str
    position: tuple
    size: float
    is_player: bool = False   # чтобы --ignore мог отличить игрока от мобов


@dataclass
class Place:
    """Кормное место в памяти: вектор от мухи до него и его история."""
    dx: float = 0.0
    dz: float = 0.0
    age: float = 0.0                  # с с тех пор, как она тут ела
    empty_since: float | None = None  # с с тех пор, как нашла тут пусто
    meals: int = 1


@dataclass
class Terrain:
    """Что вокруг тела. Всё True - открытое пространство, лететь можно куда угодно."""
    stuck: bool = False         # тело замуровано в блоках, надо вверх
    free_ahead: bool = True     # по курсу пусто
    above_ground: bool = True   # под телом есть запас высоты
    clear_below: bool = True    # прямо под телом пусто (для посадки)
    step_up: bool = False       # впереди ступенька в блок, и на неё можно встать
    wet_ahead: bool = False     # впереди под ногами вода или лава
    probe_yaw: float | None = None  # по какому курсу снят снимок
    edge_left: bool = False     # впереди левым боком заденет блок
    edge_right: bool = False    # и правым
    wet_below: bool = False     # прямо под брюхом вода или лава
    at: tuple | None = None     # откуда снят снимок: там, где не застряла, - безопасно
    # Пешком: на сколько блоков от at по курсу снимка можно уйти, не задев
    # ничего квадратом глаз, пока не придёт свежий. None - без предела.
    walk_room: float | None = None


@dataclass
class Scan:
    index: int = 0
    sightings: list = field(default_factory=list)
    terrain: Terrain = field(default_factory=Terrain)
