"""Обоняние: концентрация пахучего вещества -> частоты рецепторов антенны."""

from __future__ import annotations

import numpy as np

from .circuit import Circuit

# Какой тип рецептора на что отзывается. Ключи - обозначения запахов, значения - типы клеток из коннектома.
ODOR_TYPES = {
    "fruit": ("ORN_DM1", "ORN_DM2", "ORN_VA2", "ORN_DM4"),
    "co2": ("ORN_V",),
    "geosmin": ("ORN_DA2",),
    "foul": ("ORN_DM5",),
}


class OdorEncoder:
    """Концентрации запахов -> частоты обонятельных рецепторов.
    Концентрация приходит в условных единицах 0..1: 1 - рецепторы насыщены,
    0 - чистый воздух."""

    def __init__(self, circuit: Circuit, f_max_hz=90.0, reference=4.0,
                 half_scale=0.05, lateral_bias=0.25):
        self.circuit = circuit
        self.f_max = float(f_max_hz)

        self.reference = float(reference)
        self.half_scale = float(half_scale)
        self.lateral_bias = float(lateral_bias)

        types_of_input = circuit.cell_type[circuit.input_idx]
        sides_of_input = circuit.side[circuit.input_idx]
        self.is_left = sides_of_input == "left"
        self.masks = {}
        for odor, cell_types in ODOR_TYPES.items():
            mask = np.isin(types_of_input, cell_types)
            if mask.any():
                self.masks[odor] = mask

        self.level_idx = self._downstream_of(ODOR_TYPES["fruit"])

        self.active = bool(self.masks)

    def _downstream_of(self, cell_types, min_synapses=10.0):
        source = np.isin(self.circuit.cell_type, cell_types)
        if not source.any():
            return np.zeros(0, dtype=int)
        incoming = self.circuit.weights[source].sum(axis=0)
        return np.flatnonzero((incoming >= min_synapses) & (~source))

    def project(self, concentrations, heading_offsets=None) -> np.ndarray:
        """Частоты входных нейронов по концентрациям запахов."""
        rates = np.zeros(self.circuit.input_idx.size)
        if not self.active:
            return rates
        heading_offsets = heading_offsets or {}
        for odor, mask in self.masks.items():
            level = float(concentrations.get(odor, 0.0))
            if level <= 0.0:
                continue
            top = np.log1p(self.reference / self.half_scale)
            base = self.f_max * min(1.0, np.log1p(level / self.half_scale) / top)
            side = float(np.clip(heading_offsets.get(odor, 0.0), -1.0, 1.0))
            gain_left = 1.0 + self.lateral_bias * side
            gain_right = 1.0 - self.lateral_bias * side
            rates[mask & self.is_left] += base * gain_left
            rates[mask & (~self.is_left)] += base * gain_right
        return rates

    def odor_level(self, rates_hz) -> float:
        """Уровень запаха по активности антеннальной доли, Гц на нейрон."""
        rates = np.asarray(rates_hz, dtype=np.float64)
        if self.level_idx.size == 0 or rates.size == 0:
            return 0.0
        if rates.size == self.level_idx.size:
            return float(rates.mean())
        return float(rates[self.level_idx].mean())