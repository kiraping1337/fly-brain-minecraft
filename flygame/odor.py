"""Запах: концентрация, две антенны, касание, правило поворота."""

import math

from .constants import (
    ANTENNA_SPAN, BILATERAL_MIN, CONTACT_REACH, MEAL_SEARCH_TURN, ODOR_CAST,
    ODOR_DIVE_CLIMB, ODOR_DIVE_RETREAT, ODOR_GRADIENT_DEAD, ODOR_STRAIGHTEN,
    ODOR_STRENGTH, SEARCH_STRAIGHTEN)
from .geometry import heading_vector


def odor_concentration(body, sources):
    """Концентрация запаха в точке, где сейчас муха"""
    if not sources:
        return 0.0
    total = 0.0
    for x, y, z in sources:
        distance = math.dist((body.x, body.y, body.z), (x, y, z))
        total += ODOR_STRENGTH / max(distance, 0.7) ** 2
    return total


def odor_turn_bias(body):
    """Во сколько раз труднее свернуть при нынешнем градиенте запаха."""
    if body.meal_search > 0.0:
        return MEAL_SEARCH_TURN
    if body.search_hold > 0.0:
        return SEARCH_STRAIGHTEN
    if body.odor_level is None:
        return 1.0
    if abs(body.odor_side) > BILATERAL_MIN:
        return 1.0
    if body.cast_hold > 0.0:
        return 1.0
    change = body.odor_level - body.odor_slow
    if change > ODOR_GRADIENT_DEAD:
        return ODOR_STRAIGHTEN
    if change < -ODOR_GRADIENT_DEAD:
        return ODOR_CAST
    return 1.0

def bilateral_odor(body, sources):
    """Разница запаха у левой и правой антенны, в логарифме. + слева сильнее."""
    if not sources:
        return 0.0
    forward_x, forward_z = heading_vector(body.yaw)
    left_x, left_z = forward_z, -forward_x
    sides = []
    for sign in (1.0, -1.0):
        at = (body.x + sign * left_x * ANTENNA_SPAN, body.y,
              body.z + sign * left_z * ANTENNA_SPAN)
        total = sum(ODOR_STRENGTH / max(math.dist(at, p), 0.3) ** 2 for p in sources)
        sides.append(total)
    if min(sides) <= 0.0:
        return 0.0
    return math.log(sides[0]) - math.log(sides[1])


def touching_food(body, sources):
    """Стоит ли муха на куске еды - то, что почувствовали бы ноги."""
    here = (body.x, body.y, body.z)
    return any(math.dist(here, p) <= CONTACT_REACH for p in sources)


def odor_informative(body):
    """Говорит ли запах хоть что-то: сильный или меняется."""
    if body.odor_level is None:
        return False
    return (body.odor_level > body.odor_rest * 2.0
            or abs(body.odor_level - body.odor_slow) > ODOR_GRADIENT_DEAD
            or abs(body.odor_side) > BILATERAL_MIN)


def odor_dive_allowed(body, motor):
    """Можно ли сейчас проседать на запах.

    Три условия, и любое снимает снижение МГНОВЕННО, без сглаживания:
      - запах действительно есть (выше уровня чистого воздуха);
      - мозг не уходит от угрозы (retreat почти ноль);
      - мозг не просит вверх (climb почти ноль).
    """
    if body.odor_level is None or body.odor_level <= body.odor_rest:
        return False
    if motor.retreat > ODOR_DIVE_RETREAT:
        return False
    return motor.climb <= ODOR_DIVE_CLIMB
