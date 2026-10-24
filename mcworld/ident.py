"""Чья это муха: личные теги тела, счета, табло, надпись и файл голода.

Без имени - одна муха, как до гонки, и все значения ровно прежние. С именем
(--name) у каждой мухи свои: она убивает, захватывает и опрашивает только
своё тело, пишет только в свои счета, и несколько мух в одном мире друг
другу не мешают. Общим остаётся тег BODY_TAG - по нему видно, что это муха."""

import re

from .constants import BODY_TAG

NAME = ""
# Личный тег тела
OWN_TAG = BODY_TAG
# Теги, которые вешаются на тело при появлении (содержимое Tags:[...]).
SPAWN_TAGS = f'"{BODY_TAG}"'
# Личный тег, который мост вешает на ОДНО выбранное тело
ACTIVE_TAG = "flybrain_active"
BODY_SELECTOR = f"@e[tag={ACTIVE_TAG},limit=1]"

# Счёт, через который читаются проверки блоков
PROBE_OBJECTIVE = "flybrain"

# Табло голода сбоку экрана общее на всех мух
BOARD_TITLE = "Муха"
BOARD_HOLDER = "hunger"
BOARD_LINE = "Голод, %"

# Надпись с голодом над мухой.
LABEL_TAG = "flyhungerlabel"

# Голод в текстовый файл для OBS.
HUNGER_FILE = "output/hunger.txt"

NAME_RE = re.compile(r"[a-z0-9_]{1,12}")


def apply_name(name):
    """Дать мухе имя: свои теги, счета, строка табло, надпись и файл."""
    global NAME, OWN_TAG, SPAWN_TAGS, ACTIVE_TAG, BODY_SELECTOR
    global PROBE_OBJECTIVE, BOARD_TITLE, BOARD_HOLDER, BOARD_LINE
    global LABEL_TAG, HUNGER_FILE
    if not name:
        return
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"имя мухи {name!r}: только a-z, 0-9 и _, до 12 знаков")
    NAME = name
    OWN_TAG = f"{BODY_TAG}_{name}"
    SPAWN_TAGS = f'"{BODY_TAG}","{OWN_TAG}"'
    ACTIVE_TAG = f"flybrain_active_{name}"
    BODY_SELECTOR = f"@e[tag={ACTIVE_TAG},limit=1]"
    PROBE_OBJECTIVE = f"flybrain_{name}"
    BOARD_TITLE = "Мухи"
    BOARD_HOLDER = name
    BOARD_LINE = f"{name}: голод, %"
    LABEL_TAG = f"flyhungerlabel_{name}"
    HUNGER_FILE = f"output/hunger_{name}.txt"