"""Механосенсорика: поток воздуха от приближающегося объекта"""

from __future__ import annotations

import math

import numpy as np

from .circuit import Circuit


class WindEncoder:
    """Скорость сближения -> частоты спайков джонстонова органа."""

    def __init__(self, circuit: Circuit, f_max_hz=120.0, k_saturation=0.35,
                 lateral_bias=0.5, transient_tau_ms=180.0, front_bias=0.4):
        self.circuit = circuit
        self.f_max = float(f_max_hz)
        self.k_sat = float(k_saturation)
        self.lateral_bias = float(lateral_bias)
        self.transient_tau = float(transient_tau_ms)
        self.front_bias = float(front_bias)

        types_of_input = circuit.cell_type[circuit.input_idx]
        sides_of_input = circuit.side[circuit.input_idx]
        self.mask_fast = types_of_input == "JO-A"    # быстрые колебания
        self.mask_slow = types_of_input == "JO-B"    # медленные смещения
        self.is_left = sides_of_input == "left"

        self.active = bool(self.mask_fast.any() or self.mask_slow.any())
        self._previous: dict[str, float] = {}
        self._sustained: dict[str, float] = {}
        self._slow: dict[str, float] = {}

    def reset(self) -> None:
        self._previous.clear()
        self._sustained.clear()
        self._slow.clear()

    #обновление по свежим координатам

    def update(self, threats, dt_ms: float) -> dict:
        """Пересчитать уровень потока по каждому объекту."""
        if not self.active or dt_ms <= 0:
            return self._sustained

        dt_s = dt_ms / 1000.0
        decay = math.exp(-dt_ms / self.transient_tau)
        seen = set()

        for threat in threats:
            seen.add(threat.key)
            previous = self._previous.get(threat.key)
            self._previous[threat.key] = threat.distance
            if previous is None:
                self._sustained[threat.key] = 0.0
                self._slow[threat.key] = 0.0
                continue

            if threat.approach is not None:
                closing = threat.approach                      # блоков/с
            else:
                closing = (previous - threat.distance) / dt_s  # блоков/с

            level = self.f_max * float(np.clip(
                closing / (self.k_sat * threat.distance ** 2), 0.0, 1.0))

            self._sustained[threat.key] = level
            self._slow[threat.key] = (self._slow.get(threat.key, 0.0) * decay
                                      + level * (1.0 - decay))

        for key in set(self._previous) - seen:
            self._previous.pop(key, None)
            self._sustained.pop(key, None)
            self._slow.pop(key, None)

        return self._sustained

    # раскладка по нейронам

    def project(self, threats) -> np.ndarray:
        """Вектор частот по числу входных нейронов; ненулевой только у JO."""
        rates = np.zeros(self.circuit.input_idx.size)
        if not self.active:
            return rates

        for threat in threats:
            sustained = self._sustained.get(threat.key, 0.0)
            if sustained <= 0:
                continue

            transient = max(0.0, sustained - self._slow.get(threat.key, 0.0))

            front = 1.0 - self.front_bias * (
                1.0 - math.cos(math.radians(threat.azimuth_deg))) / 2.0
            align = -math.sin(math.radians(threat.azimuth_deg))
            gain_left = max(0.0, 1.0 + self.lateral_bias * align)
            gain_right = max(0.0, 1.0 - self.lateral_bias * align)
            gain = np.where(self.is_left, gain_left, gain_right) * front

            contribution = np.zeros_like(rates)
            contribution[self.mask_slow] = sustained
            contribution[self.mask_fast] = sustained * 0.4 + transient * 1.6
            rates = np.maximum(rates, contribution * gain)

        return rates

    def encode(self, threats, dt_ms: float) -> np.ndarray:
        self.update(threats, dt_ms)
        return self.project(threats)