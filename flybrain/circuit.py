"""Загрузка схемы нейронов из circuit.npz."""

from __future__ import annotations

import numpy as np

# Типы клеток, которые считаются входом схемы (детекторы приближения)
DEFAULT_INPUT_TYPES = ("LC4", "LPLC1", "LPLC2")

# Зрительные типы
DEFAULT_VISUAL_TYPES = ("LC4", "LPLC1", "LPLC2")

# Хемо- и гигро-рецепторы антенны: обоняние (ORN), влажность (HRN), температура (TRN)
DEFAULT_CHEMO_PREFIXES = ("ORN_", "HRN_", "TRN_")

# Выходы побега
DEFAULT_ESCAPE_TYPES = ("DNp01", "DNp02", "DNp11")

# Всё, что начинается с этого префикса, считается выходом (нисходящие нейроны)
DEFAULT_OUTPUT_PREFIX = "DNp"

# Нейрон, спайк которого означает команду "прыгай". Гигантское волокно.
COMMAND_TYPE = "DNp01"

# Механосенсорные типы: джонстонов орган, он же слух и датчик потока воздуха
DEFAULT_MECHANO_TYPES = ("JO-A", "JO-B")


class Circuit:
    """Коннектом: кто с кем связан, сколько синапсов, какой знак.
    Знак берётся из предсказанного нейромедиатора: ацетилхолин возбуждает
    (+1), ГАМК и глутамат тормозят (-1), модуляторы игнорируются (0).
    """

    def __init__(self, root_ids, cell_type, side, sign, weights,
                 input_types=DEFAULT_INPUT_TYPES,
                 output_prefix=DEFAULT_OUTPUT_PREFIX,
                 output_types=None,
                 escape_types=None,
                 visual_types=None,
                 mechano_types=DEFAULT_MECHANO_TYPES):
        self.root_ids = np.asarray(root_ids, dtype=np.int64)
        self.cell_type = np.array([str(x) for x in cell_type])
        self.side = np.array([str(x) for x in side])
        self.sign = np.asarray(sign, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)

        self.input_types = tuple(input_types)
        self.mechano_types = tuple(t for t in (mechano_types or ())
                                   if t in self.input_types)
        self.visual_types = tuple(t for t in (visual_types or DEFAULT_VISUAL_TYPES)
                                  if t in self.input_types)
        self.chemo_types = tuple(t for t in self.input_types
                                 if t.startswith(DEFAULT_CHEMO_PREFIXES))
        self.output_prefix = output_prefix

        self.input_idx = np.flatnonzero(np.isin(self.cell_type, self.input_types))

        self.output_types = tuple(output_types) if output_types else None
        if self.output_types:
            self.output_idx = np.flatnonzero(
                np.isin(self.cell_type, self.output_types))
        else:
            self.output_idx = np.flatnonzero(
                np.char.startswith(self.cell_type, output_prefix))

        types_of_output = self.cell_type[self.output_idx]
        wanted = escape_types or self.output_types or DEFAULT_ESCAPE_TYPES
        self.escape_types = tuple(t for t in wanted if (types_of_output == t).any())
        self.escape_mask = np.isin(types_of_output, self.escape_types)
        self.other_mask = ~self.escape_mask

        types_of_input = self.cell_type[self.input_idx]
        self.visual_mask = np.isin(types_of_input, self.visual_types)
        self.mechano_mask = np.isin(types_of_input, self.mechano_types)
        self.chemo_mask = np.isin(types_of_input, self.chemo_types)

        is_input = np.zeros(self.n_neurons, dtype=bool)
        is_input[self.input_idx] = True
        self.net_idx = np.flatnonzero(~is_input)

        position = {j: i for i, j in enumerate(self.net_idx)}
        self.output_in_net = np.array([position[j] for j in self.output_idx])

        if self.input_idx.size == 0:
            raise ValueError(f"В схеме нет входных нейронов типов {self.input_types}")
        if self.output_idx.size == 0:
            raise ValueError(f"В схеме нет нейронов с префиксом {output_prefix!r}")

    #загрузка
    @classmethod
    def load(cls, path, **kwargs) -> "Circuit":
        data = np.load(path, allow_pickle=True)
        if "anchor_types" in data.files and "output_types" not in kwargs:
            kwargs["output_types"] = [str(x) for x in data["anchor_types"]]
        if "escape_types" in data.files and "escape_types" not in kwargs:
            kwargs["escape_types"] = [str(x) for x in data["escape_types"]]
        if "sensory_types" in data.files and "input_types" not in kwargs:
            kwargs["input_types"] = [str(x) for x in data["sensory_types"]]
        return cls(
            root_ids=data["root_ids"],
            cell_type=data["cell_type"],
            side=data["side"],
            sign=data["sign"],
            weights=data["W"],
            **kwargs,
        )

    #удобные срезы

    @property
    def n_neurons(self) -> int:
        return len(self.cell_type)

    @property
    def command_idx(self) -> np.ndarray:
        """Позиции гигантского волокна внутри output_idx."""
        return np.flatnonzero(self.cell_type[self.output_idx] == COMMAND_TYPE)

    @property
    def hidden_count(self) -> int:
        """Сколько нейронов между входом и выходом."""
        return int(self.net_idx.size - self.output_idx.size)

    def polarity_counts(self):
        """Сколько возбуждающих, тормозных и неопределённых."""
        return (int((self.sign > 0).sum()), int((self.sign < 0).sum()),
                int((self.sign == 0).sum()))

    def labels(self, idx) -> np.ndarray:
        """Читаемые имена вида 'DNp01/left'."""
        idx = np.asarray(idx)
        return np.array([f"{self.cell_type[i]}/{self.side[i]}" for i in idx])

    def signed_weights(self, pre_idx, post_idx) -> np.ndarray:
        """Матрица синапсов pre -> post, умноженная на знак пресинапса."""
        return self.weights[np.ix_(pre_idx, post_idx)] * self.sign[pre_idx, None]

    def describe(self, max_types=8) -> str:
        excitatory, inhibitory, unknown = self.polarity_counts()
        senses = (int(self.visual_mask.sum()) + int(self.mechano_mask.sum())
                  + int(self.chemo_mask.sum()))
        silent = int(self.input_idx.size) - senses
        lines = [
            f"Схема: {self.n_neurons} нейронов "
            f"(зрение {int(self.visual_mask.sum())}, "
            f"механосенсорика {int(self.mechano_mask.sum())}, "
            f"хеморецепция {int(self.chemo_mask.sum())}, "
            + (f"вход без энкодера {silent}, " if silent else "")
            + f"промежуточных {self.hidden_count}, "
            + (f"выход {int(self.escape_mask.sum())} побег + "
               f"{int(self.other_mask.sum())} прочих)"
               if self.other_mask.any() else f"выход {self.output_idx.size})"),
            f"  возбуждающих {excitatory}, тормозных {inhibitory}, "
            f"неизвестных {unknown}",
        ]
        names, counts = np.unique(self.cell_type, return_counts=True)
        order = np.argsort(-counts)
        shown = [f"{names[i]} {counts[i]}" for i in order[:max_types]]
        extra = len(names) - len(shown)
        lines.append("  типы: " + ", ".join(shown)
                     + (f" и ещё {extra}" if extra > 0 else ""))
        return "\n".join(lines)