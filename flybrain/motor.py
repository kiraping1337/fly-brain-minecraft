"""Декодер моторной команды: активность нисходящих нейронов -> движение тела"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .circuit import Circuit


@dataclass
class MotorCommand:
    escape_deg: float   # куда уходить относительно взгляда: 0 вперёд, 180 назад
    thrust: float       # 0 .. 1, насколько быстро
    climb: float        # 0 .. 1, рывок ВВЕРХ по гигантскому волокну
    retreat: float      # составляющая "назад", для индикации
    lateral: float      # составляющая "вбок", для индикации
    left_hz: float
    right_hz: float
    steer: float = 0.0       # 0 .. 1, сила рулевого сигнала (DNb05)
    steer_side: float = 0.0  # -1 .. 1, куда рулить: + вправо, - влево


class MotorDecoder:
    def __init__(self, circuit: Circuit, retreat_scale_hz=60.0,
                 lateral_scale_hz=100.0, speed_scale_hz=25.0, cruise=0.30,
                 climb_scale_hz=45.0, steer_scale_hz=60.0):

        self.giant_fiber = np.zeros(circuit.output_idx.size, dtype=bool)
        self.giant_fiber[circuit.command_idx] = True
        self.others = ~self.giant_fiber

        escape = circuit.escape_mask
        sides = circuit.side[circuit.output_idx]
        self.left_mask = escape & (sides == "left")
        self.right_mask = escape & (sides == "right")
        self.steer_left = circuit.other_mask & (sides == "left")
        self.steer_right = circuit.other_mask & (sides == "right")
        self.has_steer = bool(circuit.other_mask.any())

        per_side = max(int(self.left_mask.sum()), 1)
        self.retreat_scale = float(retreat_scale_hz) * per_side
        self.lateral_scale = float(lateral_scale_hz) * per_side
        self.speed_scale = float(speed_scale_hz) * per_side * 2
        self.climb_scale = float(climb_scale_hz)

        self.cruise = float(cruise)
        self.baseline = 0.0
        self.lateral_baseline = 0.0
        self.climb_baseline = 0.0

        per_steer = max(int(self.steer_left.sum()), 1)
        self.steer_scale = float(steer_scale_hz) * per_steer
        self.steer_baseline = 0.0

    def calibrate_baseline(self, idle_rates_hz, margin=1.3) -> float:
        """Запомнить, сколько общей активности даёт один фоновый шум"""
        rates = np.asarray(idle_rates_hz, dtype=np.float64)
        left = float(rates[self.left_mask].sum())
        right = float(rates[self.right_mask].sum())
        self.baseline = min(left, right) * float(margin)
        self.lateral_baseline = (left - right) / self.lateral_scale
        if self.has_steer:
            self.steer_baseline = float(rates[self.steer_left].sum()
                                        + rates[self.steer_right].sum())
        return self.baseline

    def adapt_lateral(self, lateral, dt, tau_s=15.0):
        """Медленно доснимать остаточный перекос половин"""
        self.lateral_baseline += lateral * min(1.0, dt / tau_s)

    def _giant_fiber_rate(self, rates_hz):
        """Средняя частота гигантского волокна."""
        if not self.giant_fiber.any():
            return 0.0
        return float(np.mean(rates_hz[self.giant_fiber]))

    def adapt_climb(self, rates_hz, dt, tau_s=0.9):
        """Подтягивать опорный уровень волокна к текущему."""
        current = self._giant_fiber_rate(rates_hz)
        self.climb_baseline += (current - self.climb_baseline) * min(1.0, dt / tau_s)

    def decode(self, rates_hz) -> MotorCommand:
        rates_hz = np.asarray(rates_hz, dtype=np.float64)
        left = float(rates_hz[self.left_mask].sum())
        right = float(rates_hz[self.right_mask].sum())

        common = max(0.0, min(left, right) - self.baseline)
        retreat = float(np.clip(common / self.retreat_scale, 0.0, 1.0))
        lateral = float(np.clip(
            (left - right) / self.lateral_scale - self.lateral_baseline,
            -1.0, 1.0))

        escape = math.degrees(math.atan2(lateral, self.cruise - retreat))
        thrust = float(np.clip((left + right) / self.speed_scale, 0.0, 1.0))

        lead = self._giant_fiber_rate(rates_hz) - self.climb_baseline
        climb = float(np.clip(lead / self.climb_scale, 0.0, 1.0))

        steer = steer_side = 0.0
        if self.has_steer:
            s_left = float(rates_hz[self.steer_left].sum())
            s_right = float(rates_hz[self.steer_right].sum())
            steer = float(np.clip((s_left + s_right - self.steer_baseline)
                                  / self.steer_scale, 0.0, 1.0))
            steer_side = float(np.clip((s_right - s_left) / self.steer_scale,
                                       -1.0, 1.0))

        return MotorCommand(escape_deg=escape, thrust=thrust, climb=climb,
                            retreat=retreat, lateral=lateral,
                            left_hz=left, right_hz=right,
                            steer=steer, steer_side=steer_side)