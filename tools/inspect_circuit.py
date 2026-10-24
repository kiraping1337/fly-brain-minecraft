"""Показывает, что лежит внутри circuit.npz.

Запуск:  python tools/inspect_circuit.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain import Circuit  # noqa: E402

PATH = ROOT / "data" / "circuit.npz"


def main():
    if not PATH.exists():
        sys.exit(f"Нет файла {PATH}. Сначала: python tools/extract_circuit.py")

    raw = np.load(PATH, allow_pickle=True)
    print(f"{PATH}\n")
    print(f"{'массив':12s} {'форма':>12s}  тип")
    for key in raw.files:
        print(f"{key:12s} {str(raw[key].shape):>12s}  {raw[key].dtype}")

    circuit = Circuit.load(PATH)
    print()
    print(circuit.describe())

    print("\nНейромедиаторы:")
    forced = raw["sign_forced"] if "sign_forced" in raw.files else None
    for nt_name in dict.fromkeys(raw["nt"]):
        same = raw["nt"] == nt_name
        note = ""
        if forced is not None and int((same & forced).sum()):
            note = f"  (знак поставлен снаружи у {int((same & forced).sum())})"
        print(f"  {str(nt_name):6s} {int(same.sum()):4d}{note}")
    if forced is not None and forced.any():
        print(f"\nЗнак проставлен снаружи у {int(forced.sum())} первичных "
              f"рецепторов: предсказание медиатора у них не сошлось, а "
              f"холинергичность\n  сенсорных нейронов известна независимо от "
              f"коннектома. Это НАШЕ, не из данных.")

    print("\nВход схемы:", ", ".join(circuit.input_types))
    print("Выход схемы:", ", ".join(circuit.labels(circuit.output_idx)))


if __name__ == "__main__":
    main()