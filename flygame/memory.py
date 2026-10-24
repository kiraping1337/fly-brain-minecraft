"""Память о кормных местах (путевая интеграция) и поиск вслепую."""

import math

import numpy as np

from .constants import (
    BILATERAL_MIN, HUNGRY, ODOR_GRADIENT_DEAD, PI_DRIFT_DEG_PER_BLOCK,
    PI_GIVE_UP_RADIUS, PI_GIVE_UP_S, PI_PULL_HUNGER, PLACES_MAX,
    PLACE_FORGET_S, PLACE_MERGE_R, PLACE_REGROW_S)
from .geometry import bearing_deg, wrap_deg
from .state import Place
from .odor import odor_informative


def update_path_integral(body, moved_x, moved_z, dt):
    """Вычесть пройденное из вектора до КАЖДОГО запомненного места"""
    distance = math.hypot(moved_x, moved_z)
    for place in body.places:
        place.dx -= moved_x
        place.dz -= moved_z
        if distance > 0.0:
            wobble = math.radians(np.random.normal(
                0.0, PI_DRIFT_DEG_PER_BLOCK * math.sqrt(distance)))
            cos_w, sin_w = math.cos(wobble), math.sin(wobble)
            place.dx, place.dz = (place.dx * cos_w - place.dz * sin_w,
                                  place.dx * sin_w + place.dz * cos_w)


def remember_meal(body):
    """Съела - запомнить место или обновить известное рядом."""
    body.place_idle = 0.0
    body.search_time = 0.0
    for place in body.places:
        if math.hypot(place.dx, place.dz) <= PLACE_MERGE_R:
            place.dx = place.dz = 0.0
            place.age = 0.0
            place.empty_since = None
            place.meals += 1
            return
    body.places.append(Place())
    if len(body.places) > PLACES_MAX:
        worst = max(body.places, key=lambda p: (p.empty_since is not None, p.age))
        body.places.remove(worst)


def target_place(body):
    """К какому месту идти: ближайшее из годных. None - идти некуда."""
    good = [p for p in body.places
            if p.empty_since is None or p.empty_since >= PLACE_REGROW_S]
    if not good:
        return None
    return min(good, key=lambda p: math.hypot(p.dx, p.dz))


def tend_places(body, dt):
    """Время для памяти: состарить места, забыть давние, сдаться у пустого."""
    for place in body.places:
        place.age += dt
        if place.empty_since is not None:
            place.empty_since += dt
    body.places = [p for p in body.places if p.age < PLACE_FORGET_S]
    place = target_place(body)
    if (place is not None and body.hunger >= HUNGRY and body.meal_search <= 0.0
            and math.hypot(place.dx, place.dz) <= PI_GIVE_UP_RADIUS):
        body.place_idle += dt
        if body.place_idle >= PI_GIVE_UP_S:
            place.empty_since = 0.0
            body.place_idle = 0.0
    else:
        body.place_idle = 0.0


def path_integral_pull(body):
    """Куда тянет память о еде, в градусах относительно курса движения.

    Ноль - не тянет никуда. Тянет только голодную и только когда запах не
    даёт направления: есть градиент - идти по нему точнее любой памяти.
    """
    place = target_place(body)
    if place is None or body.hunger < PI_PULL_HUNGER:
        return 0.0
    if abs(body.odor_side) > BILATERAL_MIN:
        return 0.0
    if body.odor_level is not None:
        if body.odor_level > body.odor_rest * 2.0:
            return 0.0
        if abs(body.odor_level - body.odor_slow) > ODOR_GRADIENT_DEAD:
            return 0.0
    far = math.hypot(place.dx, place.dz)
    if far < 1.0:
        return 0.0
    want = bearing_deg(place.dx, place.dz)
    return wrap_deg(want - body.travel_yaw)


def searching(body):
    """Ищет вслепую: голодна, запах молчит, идти по памяти некуда."""
    return (body.hunger >= HUNGRY and body.meal_search <= 0.0
            and not odor_informative(body)
            and (target_place(body) is None or body.hunger < PI_PULL_HUNGER))
