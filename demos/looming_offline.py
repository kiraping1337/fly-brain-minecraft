"""Оффлайн-прогон: объект летит на муху, смотрим на реакцию схемы.

Игра не нужна, траектория задана формулой. Это эталонный тест: если после
подключения к Minecraft что-то сломается, запустите этот скрипт и станет
понятно, мозг сломался или мост.

Запуск:  python demos/looming_offline.py
         python demos/looming_offline.py --azimuth 90
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain import Circuit, FlyBrain, LoomingEncoder, Threat, VisualField  # noqa: E402

TYPE_COLORS = {"LC4": "#2ca02c", "LPLC1": "#9467bd", "LPLC2": "#17becf"}


def parse_args():
    p = argparse.ArgumentParser(description="Оффлайн-прогон схемы убегания")
    p.add_argument("--azimuth", type=float, default=-40.0,
                   help="откуда летит объект: 0 спереди, -90 слева, 180 сзади")
    p.add_argument("--size", type=float, default=30.0, help="размер объекта, мм")
    p.add_argument("--distance", type=float, default=300.0, help="старт, мм")
    p.add_argument("--speed", type=float, default=200.0, help="сближение, мм/с")
    p.add_argument("--onset", type=float, default=200.0, help="старт движения, мс")
    p.add_argument("--duration", type=float, default=1800.0, help="длительность, мс")
    p.add_argument("--sample", type=float, default=1.0,
                   help="шаг пересчёта стимула, мс (эмулирует опрос игры)")
    p.add_argument("--circuit", default=str(ROOT / "data" / "circuit.npz"))
    p.add_argument("--out", default=str(ROOT / "output" / "looming_offline.png"))
    return p.parse_args()


def main():
    args = parse_args()

    circuit = Circuit.load(args.circuit)
    print(circuit.describe())

    field = VisualField(circuit)
    encoder = LoomingEncoder(field)
    brain = FlyBrain(circuit,
                     calibration_visibility=field.visibility(0.0),
                     record=True)

    seeing = field.n_seeing(args.azimuth)
    print(f"\nОбъект на азимуте {args.azimuth:+.0f}°: "
          f"его заметно видят {seeing} из {int(circuit.visual_mask.sum())} "
          f"зрительных нейронов")
    if seeing == 0:
        print("  -> объект в мёртвой зоне, схема его не увидит")
    print(f"Калибровка: GAIN = {brain.gain:.4f}")

    collision_ms = args.onset + (args.distance - args.size) / args.speed * 1000.0
    n_samples = int(args.duration / args.sample)

    for k in range(n_samples):
        t_ms = k * args.sample
        moved_s = max(0.0, (t_ms - args.onset) / 1000.0)
        distance = max(args.distance - args.speed * moved_s, 5.0)
        threat = Threat(azimuth_deg=args.azimuth, distance=distance, size=args.size)
        rates = encoder.encode([threat], dt_ms=args.sample)
        brain.advance(rates, duration_ms=args.sample)

    report(circuit, brain, collision_ms)
    plot(circuit, field, brain, args, collision_ms)


def report(circuit, brain, collision_ms):
    h = brain.history.as_arrays()
    out_t, out_i = h["output_spike_t"], h["output_spike_i"]

    print(f"\nСпайков LC: {len(h['input_spike_t']):,}\n")
    print(f"{'нейрон':14s} {'спайков':>8s} {'первый':>10s}")
    for k, label in enumerate(brain.output_labels):
        m = out_i == k
        first = f"{out_t[m].min():.0f} мс" if m.any() else "-"
        print(f"{label:14s} {int(m.sum()):8d} {first:>10s}")

    fired = [k for k in circuit.command_idx if (out_i == k).any()]
    if fired:
        winner = min(fired, key=lambda k: out_t[out_i == k].min())
        t_first = out_t[out_i == winner].min()
        print(f"\nПервым сработало гигантское волокно: "
              f"{circuit.side[circuit.output_idx[winner]]}")
        print(f"Запас до столкновения: {collision_ms - t_first:.0f} мс")
    else:
        print("\nГигантское волокно молчит.")


def plot(circuit, field, brain, args, collision_ms):
    h = brain.history.as_arrays()
    time = h["time_ms"]
    voltages = np.array(brain.history.voltages)

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    moved = np.clip((time - args.onset) / 1000.0, 0, None)
    distance = np.maximum(args.distance - args.speed * moved, 5.0)
    theta = np.degrees(2 * np.arctan(args.size / (2 * distance)))

    ax[0].plot(time, theta, color="#1f77b4")
    ax[0].set_ylabel("размер объекта, °")
    twin = ax[0].twinx()
    twin.plot(time, h["input_rate_hz"], color="#ff7f0e", ls="--")
    twin.set_ylabel("средняя частота LC, Гц", color="#ff7f0e")
    ax[0].set_title(f"Объект приближается с азимута {args.azimuth:+.0f}°")

    idx = circuit.input_idx
    order = np.argsort(field.preferred_azimuth[idx])
    rank = np.empty(idx.size, dtype=int)
    rank[order] = np.arange(idx.size)

    in_t, in_i = h["input_spike_t"], h["input_spike_i"]
    if in_t.size:
        for cell_type, color in TYPE_COLORS.items():
            m = circuit.cell_type[idx][in_i] == cell_type
            if m.any():
                ax[1].scatter(in_t[m], rank[in_i[m]], s=1.2, color=color, label=cell_type)
        ax[1].legend(loc="upper left", markerscale=6, fontsize=8)
    ax[1].set_ylabel("LC-нейроны\n(по направлению взгляда)")
    ax[1].set_title("Спайки детекторов приближения")

    out_t, out_i = h["output_spike_t"], h["output_spike_i"]
    for k in circuit.command_idx:
        side = circuit.side[circuit.output_idx[k]]
        color = "#d62728" if side == "left" else "#8c564b"
        ax[2].plot(time, voltages[:, k], color=color, lw=0.8, label=f"DNp01 {side}")
        for t_spike in out_t[out_i == k]:
            ax[2].axvline(t_spike, color=color, alpha=0.2, lw=1)
    ax[2].axhline(brain.params.v_threshold, color="k", ls=":", lw=1)
    ax[2].set_ylabel("потенциал, мВ")
    ax[2].set_xlabel("время, мс")
    ax[2].set_title("Гигантское волокно DNp01: вертикальные линии = команда ПРЫГАЙ")
    ax[2].legend(loc="upper left", fontsize=8)

    for a in ax:
        a.axvline(args.onset, color="gray", ls="--", lw=1)
        a.axvline(collision_ms, color="black", lw=1.5, alpha=0.6)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"\nГрафик сохранён: {args.out}")
    plt.show()


if __name__ == "__main__":
    main()