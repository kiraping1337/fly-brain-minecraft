"""Константы сервера: теги, селекторы, счета, биты опросов, блоки."""

#соединение

RCON_HOST = "127.0.0.1"
RCON_PORT = 25575
RCON_PASSWORD = "flyproject"

#темпы

POLL_INTERVAL_S = 0.10    # опрос сервера
RCON_PAUSE_S = 0.0

TERRAIN_EVERY = 3         # проверка блоков раз в N опросов
TERRAIN_LAG_S = 0.20

STILL_TERRAIN_S = 1.0
STILL_SPEED = 0.05
STILL_MOVE = 0.25
STILL_TURN_DEG = 15.0
# сопоставление существ между опросами
TRACK_GATE = 2.0
TRACK_GATE_FAST = 10.0
INVISIBILITY_EVERY = 10

#тело

BODY_TAG = "flybrain"      # общий тег всех мух
PROBE_PLAYER = "#probe"

BERRY_BUSH = "minecraft:sweet_berry_bush"
BERRY_YIELD = {3: (2, 3), 2: (1, 2)}

#восприятие

MAX_DISTANCE = 24         # блоков; дальше не смотрим
NEARBY_LIMIT = 4          # сколько существ брать за опрос, от ближайших

PROJECTILE_TYPES = ("minecraft:arrow", "minecraft:spectral_arrow",
                    "minecraft:trident")
PROJECTILE_LIMIT = 3
PROJECTILE_DISTANCE = 20  # блоков; дальше стрелу всё равно не почувствовать
PLAYER_SIZE = 1.0

# Что живым существом не считается и муху не пугает.
IGNORED_TYPES = [
    "minecraft:item", "minecraft:experience_orb",
    "minecraft:item_frame", "minecraft:glow_item_frame", "minecraft:painting",
    "minecraft:armor_stand", "minecraft:marker", "minecraft:interaction",
    "minecraft:text_display", "minecraft:item_display", "minecraft:block_display",
    "minecraft:area_effect_cloud", "minecraft:falling_block", "minecraft:tnt",
    "minecraft:eye_of_ender", "minecraft:firework_rocket",
    "minecraft:leash_knot",
    "minecraft:player",
    "minecraft:cod", "minecraft:salmon", "minecraft:tropical_fish",
    "minecraft:pufferfish", "minecraft:squid", "minecraft:glow_squid",
    "minecraft:tadpole",
]

#голод
FOOD_BLOCKS = ("minecraft:melon", "minecraft:pumpkin")
FOOD_DROPS = {"minecraft:melon": "minecraft:melon_slice",
              "minecraft:pumpkin": "minecraft:pumpkin"}
FOOD_SPOTS = ((0, -1, 0, "под ногами"), (1, 0, 0, "рядом"), (-1, 0, 0, "рядом"),
              (0, 0, 1, "рядом"), (0, 0, -1, "рядом"))
REACH = 1.0               # блоков: дальше этого муха предмет ногами не чувствует
CRUMB_TAG = "flybrain_crumb"   # метка раскиданных нами ломтиков
CRUMB_SPEED = 0.07        # блоков/тик: разлёт ломтиков, чтобы легли рядом
# Блоки, сквозь которые муха проходит
PASSABLE_BLOCKS = (
    "#minecraft:pressure_plates",   # нажимные плиты всех видов
    "minecraft:tripwire",           # нитка растяжки
    "minecraft:tripwire_hook",
    "minecraft:short_grass", "minecraft:tall_grass",
    "minecraft:fern", "minecraft:large_fern", "minecraft:dead_bush",
    "minecraft:snow",
    "minecraft:sweet_berry_bush",
    "minecraft:cave_vines", "minecraft:cave_vines_plant",
    "#minecraft:flowers", "#minecraft:small_flowers",
    "minecraft:sunflower", "minecraft:lilac", "minecraft:rose_bush",
    "minecraft:peony", "minecraft:pitcher_plant",
    "minecraft:leaf_litter", "minecraft:pink_petals", "minecraft:wildflowers",
    "minecraft:bush", "minecraft:firefly_bush",
    "minecraft:short_dry_grass", "minecraft:tall_dry_grass",
    "minecraft:vine", "minecraft:glow_lichen",
    "#minecraft:wool_carpets", "minecraft:moss_carpet", "minecraft:pale_moss_carpet",
    "#minecraft:buttons",
    "minecraft:lever",
    "#minecraft:rails",
    "#minecraft:signs",
    "minecraft:torch",
    "minecraft:redstone_wire",
)

# Ступени досмотра по частоте
_TIER_1 = ("minecraft:short_grass", "minecraft:tall_grass", "#minecraft:flowers",
           "#minecraft:small_flowers", "minecraft:fern", "minecraft:large_fern")
_TIER_2 = ("minecraft:sweet_berry_bush", "minecraft:leaf_litter",
           "minecraft:pink_petals", "minecraft:wildflowers", "minecraft:bush",
           "minecraft:firefly_bush", "minecraft:short_dry_grass",
           "minecraft:tall_dry_grass", "minecraft:dead_bush", "minecraft:snow",
           "minecraft:vine", "minecraft:glow_lichen", "minecraft:cave_vines",
           "minecraft:cave_vines_plant", "minecraft:sunflower", "minecraft:lilac",
           "minecraft:rose_bush", "minecraft:peony", "minecraft:pitcher_plant")

SOLID_BLOCKS = (
    "#minecraft:dirt", "minecraft:grass_block", "#minecraft:sand",
    "minecraft:sandstone", "minecraft:gravel", "minecraft:clay",
    "#minecraft:base_stone_overworld", "#minecraft:logs", "#minecraft:leaves",
)

PASSABLE_TIERS = (
    ("minecraft:short_grass",),
    tuple(e for e in _TIER_1 if e in PASSABLE_BLOCKS and e != "minecraft:short_grass"),
    tuple(e for e in _TIER_2 if e in PASSABLE_BLOCKS),
    tuple(e for e in PASSABLE_BLOCKS if e not in _TIER_1 and e not in _TIER_2),
)

FOOD_HOLDER = "#food"     # счёт для проверок еды
PASS_HOLDER = "#pass"
SIGHT_HOLDER = "#sight"
# Досмотр проходимости
PASS_CACHE_S = 30.0
PASS_CACHE_MAX = 4000
SIGHT_EVERY = 10          # проверять раз в N опросов (раз в секунду)
SIGHT_MAX = 8             # не больше стольких за раз - бит на каждого
SIGHT_LIFT = 0.8          # блоков: смотреть поверх травы и цветов

AHEAD_BITS = (4, 32, 64, 128)
STEP_BIT = 256
WALK_BIT = 512            # на своём уровне впереди пусто

EYE_BITS = (1, 16384, 32768, 65536)
WATER_BIT = 1024          # впереди под ногами вода
LAVA_BIT = 2048           # впереди под ногами лава

UNDER_WATER_BIT = 131072
UNDER_LAVA_BIT = 262144
# Края тела впереди
WALK_L_BIT = 4096         # левый край впереди пусто
WALK_R_BIT = 8192         # правый край впереди пусто

#обоняние

ODOR_RANGE = 32.0         # блоков: дальше источник не ищем
# Сколько ближайших источников каждого вида брать.
ODOR_LIMIT = 12

#голодание
HEALTH_HOLDER = "#hp"

#урон
FLIP_NAMES = ("Dinnerbone", "Grumm")
