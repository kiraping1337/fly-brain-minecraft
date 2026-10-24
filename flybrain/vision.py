"""Зрение: превращает объекты в мире в частоты спайков LC-нейронов. Это слой между игрой и мозгом."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import Circuit

# Дуга обзора каждого глаза в градусах
DEFAULT_EYES = {"left": (-150.0, 15.0), "right": (-15.0, 150.0)}

# Зрительные типы
DEFAULT_VISUAL_TYPES = ("LC4", "LPLC1", "LPLC2")


@dataclass
class Threat:
    """Объект, который муха может увидеть."""
    azimuth_deg: float   # 0 спереди, -90 слева, +90 справа
    distance: float      # в тех же единицах, что и size
    size: float
    key: str = "default"  # чтобы отслеживать один и тот же объект между кадрами

    approach: float | None = None

    @property
    def angular_size(self) -> float:
        """Угловой размер в радианах."""
        return 2.0 * np.arctan(self.size / (2.0 * max(self.distance, 1e-6)))


def angular_difference(a, b):
    """Разница углов в градусах с учётом того, что 350 и 10 - это рядом."""
    return (a - b + 180.0) % 360.0 - 180.0


class VisualField:
    """Карта направлений взгляда входных нейронов."""

    def __init__(self, circuit: Circuit, eyes=None, tuning_width_deg=35.0,
                 visual_types=None):
        self.circuit = circuit
        self.eyes = dict(eyes or DEFAULT_EYES)
        self.tuning_width = float(tuning_width_deg)
        self.visual_types = tuple(
            t for t in (visual_types or DEFAULT_VISUAL_TYPES)
            if t in set(circuit.input_types))

        n = circuit.n_neurons
        self.preferred_azimuth = np.zeros(n)
        self._eye_lo = np.full(n, np.nan)
        self._eye_hi = np.full(n, np.nan)

        # только зрительные нейроны среди входных
        is_input = np.isin(self.circuit.cell_type, self.visual_types)
        is_input[~np.isin(np.arange(n), circuit.input_idx)] = False
        self.member = is_input[circuit.input_idx]

        for eye_side, (lo, hi) in self.eyes.items():
            on_this_side = is_input & (circuit.side == eye_side)
            for cell_type in self.visual_types:
                m = on_this_side & (circuit.cell_type == cell_type)
                if m.any():
                    self.preferred_azimuth[m] = np.linspace(lo, hi, int(m.sum()))
            self._eye_lo[on_this_side] = lo
            self._eye_hi[on_this_side] = hi

    def visibility(self, azimuth_deg: float) -> np.ndarray:
        """Насколько каждый входной нейрон видит объект: от 0 до 1."""
        idx = self.circuit.input_idx
        offset = angular_difference(self.preferred_azimuth[idx], azimuth_deg)
        seen = np.exp(-0.5 * (offset / self.tuning_width) ** 2)
        # объект вне дуги обзора глаза этот глаз просто не видит
        in_arc = (azimuth_deg >= self._eye_lo[idx]) & (azimuth_deg <= self._eye_hi[idx])
        return seen * in_arc * self.circuit.visual_mask * self.member

    def n_seeing(self, azimuth_deg: float, cutoff=0.05) -> int:
        return int((self.visibility(azimuth_deg) > cutoff).sum())


class LoomingEncoder:
    """Скорость расширения объекта -> частоты спайков LC-нейронов.

    Нейроны реагируют не на дистанцию, а на то, как быстро объект растёт
    в поле зрения. Энкодер помнит предыдущий угловой размер каждого объекта,
    поэтому работает и на дискретных замерах из игры, а не только на
    аналитической траектории.
    """

    def __init__(self, visual_field: VisualField, f_max_hz=150.0, k_saturation=1.0,
                 max_growth_ratio=2.0):
        self.field = visual_field
        self.f_max = float(f_max_hz)
        self.k_sat = float(k_saturation)
        self.max_growth_ratio = float(max_growth_ratio)
        self._previous: dict[str, float] = {}
        self._levels: dict[str, float] = {}

    def reset(self) -> None:
        self._previous.clear()
        self._levels.clear()

    def update(self, threats, dt_ms: float) -> dict:
        """Пересчитать «уровень тревоги» по каждому объекту. Вызывается, когда пришли свежие координаты."""
        if dt_ms <= 0:
            return self._levels

        dt_s = dt_ms / 1000.0
        seen_keys = set()

        for threat in threats:
            theta = threat.angular_size
            seen_keys.add(threat.key)
            previous = self._previous.get(threat.key)
            self._previous[threat.key] = theta

            if previous is None:
                self._levels[threat.key] = 0.0
                continue  # первый кадр: скорость расширения ещё неизвестна

            ratio = theta / previous if previous > 0 else float("inf")
            if ratio > self.max_growth_ratio or ratio < 1.0 / self.max_growth_ratio:
                self._levels[threat.key] = 0.0
                continue  # подмена объекта, скорость считать не по чему

            expansion = (theta - previous) / dt_s          # рад/с
            self._levels[threat.key] = self.f_max * float(
                np.clip(expansion / self.k_sat, 0.0, 1.0))

        # объекты, пропавшие из виду, забываем
        for key in set(self._previous) - seen_keys:
            del self._previous[key]
            self._levels.pop(key, None)

        return self._levels

    def project(self, threats) -> np.ndarray:
        """Вектор частот (Гц) по числу входных нейронов.

        Берёт уровни, посчитанные в update(), и раскладывает их по нейронам
        согласно текущему направлению на объект. Вызывается чаще, чем update:
        когда муха поворачивается, координаты объекта те же, а вот кто из
        нейронов его видит - уже другие.
        """
        rates = np.zeros(self.field.circuit.input_idx.size)
        for threat in threats:
            level = self._levels.get(threat.key, 0.0)
            if level > 0:
                # несколько объектов не суммируем, берём самый пугающий
                rates = np.maximum(
                    rates, level * self.field.visibility(threat.azimuth_deg))
        return rates

    def encode(self, threats, dt_ms: float) -> np.ndarray:
        """update + project за один вызов. Для оффлайн-прогонов."""
        self.update(threats, dt_ms)
        return self.project(threats)