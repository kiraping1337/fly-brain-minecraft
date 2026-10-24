"""Разбор текстовых ответов сервера"""

import re

from .constants import DEFAULT_SIZE, ENTITY_SIZES


#разбор ответов сервера

NAME_RE = re.compile(r"^(.*?) has the following entity data:", re.S)
DATA_RE = re.compile(r"entity data:\s*(.+)$", re.S)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_numbers(response):
    if not response:
        return None
    match = DATA_RE.search(response)
    if not match:
        return None
    values = [float(x) for x in NUMBER_RE.findall(match.group(1))]
    return values or None


SIGHTING_RE = re.compile(
    r"(.*?) has the following entity data:\s*\[([^\]]*)\]", re.S)


def parse_sightings(response):
    """Все пары (имя, координаты) из ответа."""
    found = []
    for name, numbers in SIGHTING_RE.findall(response or ""):
        values = [float(x) for x in NUMBER_RE.findall(numbers)]
        if len(values) < 3:
            continue
        clean = name.rsplit("]", 1)[-1].strip() or "существо"
        found.append((clean, tuple(values[:3])))
    return found


def parse_name(response):
    """Имя существа стоит в начале ответа: 'Spider has the following...'."""
    if not response:
        return None
    match = NAME_RE.search(response)
    return match.group(1).strip() if match else None


def size_of(name):
    return ENTITY_SIZES.get((name or "").lower(), DEFAULT_SIZE)
