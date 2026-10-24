"""
Проверка связи с сервером Minecraft по RCON.

Ничего не симулирует. Просто подключается, задаёт серверу несколько вопросов
и печатает ответы как есть - чтобы увидеть, в каком формате ваша версия
отдаёт координаты.

Перед запуском:
  1. в server.properties выставлены enable-rcon=true, rcon.password, rcon.port
  2. сервер запущен
  3. вы зашли в игру и стоите в мире

Запуск:  python demos/rcon_probe.py
"""

from mcrcon import MCRcon

HOST = "127.0.0.1"
PORT = 25575
PASSWORD = "flyproject"   # то же, что в server.properties

# Мобы, которых будем искать. Спавните любого рядом с собой командой в игровом чате, например:  /summon minecraft:spider ~3 ~ ~
MOBS = ["minecraft:spider", "minecraft:zombie", "minecraft:creeper"]

QUERIES = [
    ("кто онлайн", "list"),
    ("позиция игрока", "data get entity @p Pos"),
    ("поворот игрока", "data get entity @p Rotation"),
    ("здоровье игрока", "data get entity @p Health"),
]

for mob in MOBS:
    QUERIES.append((
        f"ближайший {mob.split(':')[-1]}",
        f"execute at @p run data get entity "
        f"@e[type={mob},limit=1,sort=nearest] Pos",
    ))

print(f"Подключаюсь к {HOST}:{PORT} ...\n")

try:
    with MCRcon(HOST, PASSWORD, port=PORT) as mc:
        for label, cmd in QUERIES:
            try:
                answer = mc.command(cmd)
            except Exception as e:
                answer = f"<ошибка: {e}>"
            print(f"--- {label}")
            print(f"    команда: {cmd}")
            print(f"    ответ:   {answer!r}\n")

except ConnectionRefusedError:
    print("Сервер не отвечает. Проверьте:")
    print("  - сервер запущен?")
    print("  - enable-rcon=true в server.properties?")
    print("  - сервер перезапущен ПОСЛЕ правки server.properties?")
except Exception as e:
    print(f"Не подключился: {type(e).__name__}: {e}")
    print("Если это про авторизацию - не совпадает rcon.password.")
