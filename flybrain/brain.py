"""Ядро симуляции: спайковые нейроны на матрице коннектома"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy import sparse

from .circuit import Circuit


@dataclass
class NeuronParams:
    """Ориентиры из статьи Shiu et al. 2024."""
    v_rest: float = -52.0
    v_threshold: float = -45.0
    v_reset: float = -52.0
    tau_mem_ms: float = 20.0
    refractory_ms: float = 2.2
    synapse_mv: float = 0.275


@dataclass
class StepResult:
    time_ms: float
    input_spikes: int
    output_spikes: np.ndarray
    voltages: np.ndarray
    escape: str | None = None


@dataclass
class History:
    """Запись для графиков"""
    time_ms: list = dc_field(default_factory=list)
    voltages: list = dc_field(default_factory=list)
    input_rate_hz: list = dc_field(default_factory=list)
    input_spike_t: list = dc_field(default_factory=list)
    input_spike_i: list = dc_field(default_factory=list)
    output_spike_t: list = dc_field(default_factory=list)
    output_spike_i: list = dc_field(default_factory=list)

    def as_arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


#шаг мозга
PUSH_COLUMNS_MAX = 24
RNG_CHUNK = 128


class FlyBrain:
    def __init__(self, circuit: Circuit, calibration_visibility=None,
                 trigger_rate_hz=30.0, dt_ms=0.1, params: NeuronParams | None = None,
                 seed=42, record=False, spontaneous_hz=0.0, tau_rate_ms=80.0,
                 track_all=False,
                 watch_idx=None,
                 gain=None):
        self.circuit = circuit
        self.params = params or NeuronParams()
        self.dt = float(dt_ms)
        self.record = bool(record)
        self.rng = np.random.default_rng(seed)
        self.spontaneous_hz = float(spontaneous_hz)
        self.tau_rate_ms = float(tau_rate_ms)
        self.track_all = bool(track_all)
        self._watch_idx = None if watch_idx is None else np.asarray(watch_idx, int)

        self.n_in = circuit.input_idx.size
        self.n_net = circuit.net_idx.size
        self.n_out = circuit.output_idx.size
        self.output_labels = circuit.labels(circuit.output_idx)
        self._out_rows = circuit.output_in_net
        self._command_rows = set(int(self._out_rows[k])
                                 for k in circuit.command_idx)
        if self._watch_idx is None:
            self._watch_rows = None
        else:
            where = {int(j): i for i, j in enumerate(circuit.net_idx)}
            rows = [where[int(j)] for j in self._watch_idx if int(j) in where]
            self._watch_rows = np.asarray(rows, dtype=int) if rows else None
        self._base_in = self._matrix(circuit, circuit.input_idx, circuit.net_idx)
        self._base_net = self._matrix(circuit, circuit.net_idx, circuit.net_idx)

        self.gain = float(gain) if gain is not None else \
            self._autocalibrate(calibration_visibility, trigger_rate_hz)
        self._apply_gain()
        self.reset()

    #сборка матриц
    @staticmethod
    def _matrix(circuit, pre_idx, post_idx):
        """Разреженная матрица (post x pre) в милливольтах на спайк"""
        dense = circuit.signed_weights(pre_idx, post_idx)
        return sparse.csr_matrix(dense.T.astype(np.float32))

    def _apply_gain(self):
        scale = np.float32(self.params.synapse_mv * self.gain)
        self._w_in = self._base_in * scale
        self._w_net = self._base_net * scale
        self._cols_in = self._columns(self._w_in)
        self._cols_net = self._columns(self._w_net)
        self._push_buffer = np.zeros(self.n_net, dtype=np.float32)

    @staticmethod
    def _columns(matrix):
        csc = matrix.tocsc()
        return csc.indptr, csc.indices, csc.data

    def _push(self, matrix, columns, spikes, fired):
        """Вклад сработавших нейронов в напряжение"""
        if fired.size > PUSH_COLUMNS_MAX:
            return matrix @ spikes
        indptr, rows, values = columns
        out = self._push_buffer
        out.fill(0.0)
        for j in fired:
            start, stop = indptr[j], indptr[j + 1]
            out[rows[start:stop]] += values[start:stop]
        return out

    #калибровка

    def _autocalibrate(self, visibility, trigger_rate_hz,
                       target_spikes=4, test_ms=300.0):
        """Подобрать общий множитель силы синапсов"""
        if visibility is None:
            visibility = np.ones(self.n_in)
        rates = np.asarray(visibility, dtype=np.float64) * trigger_rate_hz
        if rates.max() <= 0:
            return 0.0

        low, high = -4.0, 1.0     # десятичный логарифм множителя
        for _ in range(16):
            middle = 0.5 * (low + high)
            self.gain = 10.0 ** middle
            self._apply_gain()
            self.reset()
            spikes = self.advance(rates, duration_ms=test_ms).output_spikes
            fired = sum(int(spikes[k]) for k in self.circuit.command_idx)
            if fired >= target_spikes:
                high = middle
            else:
                low = middle
        return 10.0 ** high

    def autotune_spontaneous(self, target_idle_hz=25.0, test_ms=500.0,
                             ceiling_hz=80.0) -> float:
        """Подобрать фоновый шум так, чтобы в покое была слабая активность"""
        silence = np.zeros(self.n_in)
        low, high = 0.0, float(ceiling_hz)
        for _ in range(12):
            middle = 0.5 * (low + high)
            self.spontaneous_hz = middle
            self.reset()
            self.advance(silence, duration_ms=test_ms)
            if float(self.output_rates_hz.sum()) > target_idle_hz:
                high = middle
            else:
                low = middle
        self.spontaneous_hz = 0.5 * (low + high)
        self.reset()
        return self.spontaneous_hz

    #состояние

    def reset(self) -> None:
        p = self.params
        self.time_ms = 0.0
        self._v = np.full(self.n_net, p.v_rest, dtype=np.float32)
        self._refractory_until = np.full(self.n_net, -1.0, dtype=np.float32)
        self._spikes_net = np.zeros(self.n_net, dtype=np.float32)
        self._spikes_in = np.zeros(self.n_in, dtype=np.float32)
        self._command_spikes: deque = deque()
        self._rate_trace = np.zeros(self.n_out)
        self._watch_trace = (np.zeros(self._watch_rows.size)
                             if self._watch_rows is not None else None)
        self._trace_net = np.zeros(self.n_net) if self.track_all else None
        self._trace_in = np.zeros(self.n_in) if self.track_all else None
        self.history = History()

    @property
    def voltages(self) -> np.ndarray:
        return self._v[self._out_rows].astype(np.float64)

    @property
    def output_rates_hz(self) -> np.ndarray:
        """Сглаженная частота каждого выходного нейрона"""
        return self._rate_trace * (1000.0 / self.tau_rate_ms)

    @property
    def watch_rates_hz(self) -> np.ndarray:
        """Сглаженные частоты наблюдаемых нейронов, в порядке watch_idx. Дешёвая замена all_rates_hz, когда нужен не весь мозг, а один слой."""
        if self._watch_rows is None:
            return np.zeros(0)
        return self._watch_trace * (1000.0 / self.tau_rate_ms)

    @property
    def all_rates_hz(self) -> np.ndarray:
        """Сглаженная частота каждого нейрона схемы, в порядке circuit. Доступно только при track_all=True."""
        if not self.track_all:
            raise RuntimeError("нужен FlyBrain(track_all=True)")
        rates = np.zeros(self.circuit.n_neurons)
        scale = 1000.0 / self.tau_rate_ms
        rates[self.circuit.net_idx] = self._trace_net * scale
        rates[self.circuit.input_idx] = self._trace_in * scale
        return rates

    #основной цикл
    def advance(self, input_rates_hz, duration_ms: float) -> StepResult:
        p = self.params
        rates = np.asarray(input_rates_hz, dtype=np.float64)
        if rates.shape != (self.n_in,):
            raise ValueError(f"Ожидался вектор длины {self.n_in}, пришёл {rates.shape}")

        n_steps = max(1, int(round(duration_ms / self.dt)))
        probability = (rates + self.spontaneous_hz) * self.dt / 1000.0
        leak = np.float32(self.dt / p.tau_mem_ms)
        rate_decay = math.exp(-self.dt / self.tau_rate_ms)

        output_counts = np.zeros(self.n_out, dtype=int)
        input_total = 0
        mean_rate = float(rates.mean())

        draws = None
        for step in range(n_steps):
            self.time_ms += self.dt
            t = self.time_ms
            row = step % RNG_CHUNK
            if row == 0:
                draws = self.rng.random((min(RNG_CHUNK, n_steps - step), self.n_in))
            self._spikes_in[:] = draws[row] < probability
            fired_in = np.flatnonzero(self._spikes_in)
            n_fired_in = fired_in.size
            input_total += n_fired_in

            self._v += leak * (p.v_rest - self._v)
            if n_fired_in:
                self._v += self._push(self._w_in, self._cols_in, self._spikes_in, fired_in)
            fired_net = np.flatnonzero(self._spikes_net)
            if fired_net.size:
                self._v += self._push(self._w_net, self._cols_net, self._spikes_net,
                                      fired_net)

            in_refractory = t < self._refractory_until
            self._v[in_refractory] = p.v_reset

            fired = (self._v >= p.v_threshold) & ~in_refractory
            self._spikes_net[:] = fired
            self._rate_trace *= rate_decay
            if self._watch_rows is not None:
                self._watch_trace *= rate_decay
                self._watch_trace += fired[self._watch_rows]
            if self.track_all:
                self._trace_net *= rate_decay
                self._trace_net += fired
                self._trace_in *= rate_decay
                self._trace_in += self._spikes_in

            if fired.any():
                self._v[fired] = p.v_reset
                self._refractory_until[fired] = t + p.refractory_ms
                out_fired = np.flatnonzero(fired[self._out_rows])
                if out_fired.size:
                    output_counts[out_fired] += 1
                    self._rate_trace[out_fired] += 1.0
                    for k in out_fired:
                        if int(self._out_rows[k]) in self._command_rows:
                            self._command_spikes.append((t, int(k)))
                    if self.record:
                        self.history.output_spike_t.extend([t] * out_fired.size)
                        self.history.output_spike_i.extend(out_fired.tolist())

            if self.record:
                self.history.time_ms.append(t)
                self.history.voltages.append(self.voltages)
                self.history.input_rate_hz.append(mean_rate)
                if n_fired_in:
                    k = np.flatnonzero(self._spikes_in)
                    self.history.input_spike_t.extend([t] * k.size)
                    self.history.input_spike_i.extend(k.tolist())

        self._forget_old_command_spikes()
        return StepResult(
            time_ms=self.time_ms,
            input_spikes=input_total,
            output_spikes=output_counts,
            voltages=self.voltages,
            escape=self.escape_triggered(),
        )

    #чтение команды
    def _forget_old_command_spikes(self, keep_ms=200.0) -> None:
        cutoff = self.time_ms - keep_ms
        while self._command_spikes and self._command_spikes[0][0] < cutoff:
            self._command_spikes.popleft()

    def escape_triggered(self, window_ms=50.0, min_spikes=3) -> str | None:
        """Был ли за последние window_ms залп гигантского волокна."""
        if not self._command_spikes:
            return None
        cutoff = self.time_ms - window_ms
        counts: dict[int, int] = {}
        for t, k in self._command_spikes:
            if t >= cutoff:
                counts[k] = counts.get(k, 0) + 1
        if not counts:
            return None
        best = max(counts, key=counts.get)
        if counts[best] < min_spikes:
            return None
        return str(self.circuit.side[self.circuit.output_idx[best]])