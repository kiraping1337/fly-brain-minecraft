"""Поток воздуха от движущихся объектов - вход джонстонова органа."""

import math

from flybrain import Threat

from .constants import MOVER_MAX_SPEED


def air_movers(visible, threats, body, last, dt_s):
    """Те же угрозы, но с собственной скоростью каждого объекта к мухе."""
    out, seen = [], set()
    for s, t in zip(visible, threats):
        seen.add(s.name)
        before = last.get(s.name)
        last[s.name] = s.position
        approach = 0.0
        if before is not None and dt_s > 0:
            moved = [s.position[i] - before[i] for i in range(3)]
            if math.hypot(*moved) / dt_s <= MOVER_MAX_SPEED:
                to_fly = (body.x - s.position[0], body.y - s.position[1],
                          body.z - s.position[2])
                reach = math.hypot(*to_fly)
                if reach > 1e-6:
                    approach = sum(m * f for m, f in zip(moved, to_fly)) / reach / dt_s
        out.append(Threat(azimuth_deg=t.azimuth_deg, distance=t.distance,
                          size=t.size, key=t.key, approach=approach))
    for name in set(last) - seen:
        last.pop(name, None)
    return out
