"""Рисует активность нейронов в трёх измерениях по записи из моста.

Мост с флагом --record пишет частоту каждого нейрона двадцать раз в секунду.
Этот скрипт раскладывает нейроны по их настоящим координатам в мозге мухи
и подсвечивает те, что сейчас работают.

Запуск:
    python demos/minecraft_bridge.py --record output/tape.npz
    python tools/render_brain.py output/tape.npz

    python tools/render_brain.py output/tape.npz --zoom 2.0   # крупнее
    python tools/render_brain.py output/tape.npz --no-context # без силуэта

Кадры складываются в output/brain/ по одному файлу на кадр - дальше их
кладут дорожкой в монтаж. Отрисовка не в реальном времени: так качество
выше.
"""

import argparse
import glob
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain import Circuit  # noqa: E402

BACKGROUND = "#0a0d13"
SHELL_COLOR = "#5b6a86"    # остальной мозг
TEXT_COLOR = "#9aa6b8"

# Роли нейронов: цвет и подпись. Зрение холодное, механосенсорика зелёная, промежуточные тусклые, команда красная.
ROLES = {
    "vision": ((0.35, 0.72, 1.00), "зрение"),
    "mechano": ((0.35, 0.95, 0.55), "слух и поток воздуха"),
    "hidden": ((0.62, 0.64, 0.74), "промежуточные"),
    "output": ((1.00, 0.32, 0.30), "команда телу"),
}
STATE_NAMES = {0: "в полёте", 1: "снижается", 2: "сидит", 3: "идёт", 4: "ест"}

# Режим --vs-sober
HOTTER = (1.00, 0.52, 0.12)   # сильнее, чем трезвая
COLDER = (0.72, 0.40, 1.00)   # слабее, чем трезвая
VS_FULL = 1.0                 # log2 перемены, при которой цвет полный (в 2 раза)
VS_FLOOR_HZ = 0.5             # прибавка к частоте, чтобы молчащие не мигали
VS_MARK = math.log2(1.5)      # "заметно изменился" для полосок - в 1.5 раза


def parse_args():
    p = argparse.ArgumentParser(description="Трёхмерная картинка активности")
    p.add_argument("tape", help="запись из моста (--record)")
    p.add_argument("--circuit", default=str(ROOT / "data" / "circuit.npz"))
    p.add_argument("--out", default=str(ROOT / "output" / "brain"))
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--zoom", default="auto",
                   help="приближение: число или auto (подобрать по кадру)")
    p.add_argument("--spin", type=float, default=30.0,
                   help="на сколько градусов повернуть вид за всю запись")
    p.add_argument("--elev", type=float, default=16.0)
    p.add_argument("--azim", type=float, default=-72.0)
    p.add_argument("--size", type=float, default=17.0, help="размер точки")
    p.add_argument("--inner-max-hz", type=float, default=0.0,
                   help="своя полная яркость для промежуточных и команды (0 - общая "
                        "с --max-hz). Сенсорные нейроны шумят на ~10 Гц и задают общую "
                        "яркость, а промежуточные в покое еле тлеют - на общей шкале их "
                        "изменения не видно (опыты demos/brain_lab.py)")
    p.add_argument("--vs-sober", action="store_true",
                   help="обычные цвета, но нейрон, который работает заметно сильнее, "
                        "чем у трезвой мухи, - оранжевый, слабее - фиолетовый. Трезвая - начало этой же записи, "
                        "пока доза 0 (нужен ЗАПИСЬ.dose.npz), или --sober")
    p.add_argument("--show-smell", action="store_true",
                   help="обонятельные рецепторы своим цветом (обычно они серые, как "
                        "промежуточные) - для синестезии demos/brain_lab.py")
    p.add_argument("--sober", default=None,
                   help="--vs-sober: отдельная запись трезвой мухи")
    p.add_argument("--smooth", type=float, default=1.5,
                   help="--vs-sober: сглаживание, секунд")
    p.add_argument("--dose", default=None,
                   help="доза по времени из demos/brain_lab.py (по умолчанию "
                        "ЗАПИСЬ.dose.npz рядом с записью, если есть)")
    p.add_argument("--max-hz", type=float, default=0.0,
                   help="частота полной яркости (0 = взять из записи)")
    p.add_argument("--no-context", action="store_true",
                   help="не рисовать серый силуэт остального мозга")
    p.add_argument("--context-points", type=int, default=60000,
                   help="сколько нейронов брать на силуэт мозга")
    p.add_argument("--shell-alpha", type=float, default=0.42,
                   help="насколько плотный силуэт")
    p.add_argument("--shell-size", type=float, default=1.7,
                   help="размер точки силуэта")
    p.add_argument("--width", type=int, default=1600, help="ширина кадра")
    p.add_argument("--height", type=int, default=900, help="высота кадра")
    p.add_argument("--limit", type=int, default=0, help="сколько кадров рисовать")
    return p.parse_args()


def roles_of(circuit, smell=False):
    roles = np.full(circuit.n_neurons, "hidden", dtype=object)
    roles[circuit.input_idx[circuit.visual_mask]] = "vision"
    roles[circuit.input_idx[circuit.mechano_mask]] = "mechano"
    if smell:
        roles[circuit.input_idx[circuit.chemo_mask]] = "smell"
    roles[circuit.output_idx] = "output"
    return roles


SMELL = ((1.00, 0.84, 0.30), "обоняние")   # --show-smell.
MARKED = ((1.00, 0.30, 0.85), "вживлённые связи")


def load_context(points):
    """Серое облако остального мозга - чтобы был виден силуэт."""
    matches = []
    for pattern in ("coordinates*.csv", "*coordinates*.csv"):
        for suffix in ("", ".gz"):
            matches += glob.glob(str(ROOT / "data" / "raw" / (pattern + suffix)))
    if not matches:
        print("  силуэт пропущен: файла координат в data/raw/ нет")
        return None

    import pandas as pd
    table = pd.read_csv(sorted(matches)[0])
    column = next((c for c in table.columns
                   if c.lower() in ("position", "pos", "coordinates")), None)
    if column is None:
        return None
    sample = table[column].dropna()
    if len(sample) > points:
        sample = sample.sample(points, random_state=0)
    found = sample.astype(str).str.findall(r"-?\d+(?:\.\d+)?")
    return np.array([[float(v[0]), float(v[1]), float(v[2])]
                     for v in found if len(v) >= 3], dtype=np.float32)


def content_extent(figure):
    """Границы нарисованного в пикселях: где вообще есть что-то ярче фона."""
    figure.canvas.draw()
    frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].astype(np.float32)
    lit = frame.mean(axis=2) > 24.0
    rows = np.flatnonzero(lit.any(axis=1))
    cols = np.flatnonzero(lit.any(axis=0))
    if not rows.size or not cols.size:
        return None
    return cols[0], cols[-1], rows[0], rows[-1]


def axes_box(scale, width, height):
    """Прямоугольник области рисования в долях кадра, увеличенный в scale раз."""
    tall = scale
    wide = scale * height / width
    return [0.5 - wide / 2.0, 0.5 - tall / 2.0, wide, tall]


def fit_zoom(figure, ax, args, fill=0.96):
    """Подобрать увеличение по пробным кадрам."""
    height, width = figure.canvas.get_width_height()[::-1]
    center_x, center_y = width / 2.0, height / 2.0
    worst = None
    for share in (0.0, 0.5, 1.0):
        ax.view_init(elev=args.elev, azim=args.azim + args.spin * share)
        box = content_extent(figure)
        if box is None:
            continue
        left, right, top, bottom = box
        reach_x = max(abs(left - center_x), abs(right - center_x))
        reach_y = max(abs(top - center_y), abs(bottom - center_y))
        scale = min(center_x * fill / max(reach_x, 1.0),
                    center_y * fill / max(reach_y, 1.0))
        worst = scale if worst is None else min(worst, scale)
    ax.view_init(elev=args.elev, azim=args.azim)
    return float(np.clip(worst if worst else 1.0, 0.4, 12.0))


def main():
    args = parse_args()

    circuit = Circuit.load(args.circuit)
    raw = np.load(args.circuit, allow_pickle=True)
    if "positions" not in raw.files:
        sys.exit("В схеме нет координат. Скачайте Marked Neuron Coordinates "
                 "в data/raw/ и пересоберите: python tools/extract_circuit.py")

    positions = np.asarray(raw["positions"], dtype=np.float64)
    tape = np.load(args.tape)
    rates = np.asarray(tape["rates"], dtype=np.float32)
    times = np.asarray(tape["times"], dtype=np.float32)
    states = tape["state"] if "state" in tape.files else None
    hunger = tape["hunger"] if "hunger" in tape.files else None

    if rates.shape[1] != circuit.n_neurons:
        sys.exit(f"Запись на {rates.shape[1]} нейронов, а схема на "
                 f"{circuit.n_neurons}. Запись сделана другой схемой.")

    known = ~np.isnan(positions).any(axis=1)
    print(f"Нейронов с координатами: {known.sum()} из {circuit.n_neurons}")
    print(f"Кадров в записи: {len(times)} ({times[-1]:.1f} с)")

    context = None if args.no_context else load_context(args.context_points)
    reference = positions[known] if context is None else np.vstack(
        [positions[known], context])
    center = np.median(reference, axis=0)
    span = float(np.percentile(np.abs(reference - center), 97))

    points = (positions[known] - center) / span
    if args.show_smell:
        ROLES["smell"] = SMELL
    roles = roles_of(circuit, smell=args.show_smell)[known]
    base = np.array([ROLES[r][0] for r in roles])
    activity = rates[:, known]

    lively = activity[activity > 0]
    ceiling = args.max_hz or (float(np.percentile(lively, 99)) if lively.size else 1.0)
    print(f"Полная яркость при {ceiling:.0f} Гц")
    if args.inner_max_hz > 0:
        inner = np.zeros(circuit.n_neurons, bool)
        inner[circuit.net_idx] = True
        ceiling = np.where(inner[known], args.inner_max_hz, ceiling)
        print(f"Промежуточные и команда: полная яркость при {args.inner_max_hz:.0f} Гц")

    dose_path = Path(args.dose) if args.dose else Path(args.tape).with_suffix(".dose.npz")
    dose_at = body_at = None
    dose_word = "доза"
    if dose_path.exists():
        dose_tape = np.load(dose_path)
        dose_at = np.interp(times, dose_tape["times"], dose_tape["dose"])
        dose_word = str(dose_tape["label"]) if "label" in dose_tape.files else "доза"
        print(f"Доза из {dose_path}: до {float(dose_at.max()):.0%}")
        if "body" in dose_tape.files and len(dose_tape["times"]):
            where = np.clip(np.searchsorted(dose_tape["times"], times),
                            0, len(dose_tape["times"]) - 1)
            body_at = np.asarray(dose_tape["body"])[where]

    marked = np.zeros(int(known.sum()), bool)
    if dose_at is not None and "marked" in dose_tape.files and dose_tape["marked"].size:
        ids = np.zeros(circuit.n_neurons, bool)
        ids[np.asarray(dose_tape["marked"], int)] = True
        marked = ids[known]
        base[marked] = MARKED[0]
        ROLES["marked"] = MARKED
        print(f"Подсвечено нейронов: {int(marked.sum())} ({MARKED[1]})")

    sober_mean = None
    if args.vs_sober:
        if args.sober:
            sober_rates = np.asarray(np.load(args.sober)["rates"], dtype=np.float32)[:, known]
        elif dose_at is not None and (dose_at == 0).sum() >= 40:
            sober_rates = activity[dose_at == 0]
        else:
            sys.exit("--vs-sober: не с чем сравнивать - нет трезвого начала "
                     "(ЗАПИСЬ.dose.npz) и не задан --sober")
        sober_mean = sober_rates.mean(axis=0)
        inner_vs = np.zeros(circuit.n_neurons, bool)
        inner_vs[circuit.net_idx] = True
        inner_vs = inner_vs[known]
        print(f"Сравнение с трезвой: {len(sober_rates)} кадров, сглаживание {args.smooth:g} с")

    dpi = 100
    figure = plt.figure(figsize=(args.width / dpi, args.height / dpi),
                        dpi=dpi, facecolor=BACKGROUND)

    ax = figure.add_axes(axes_box(1.0, args.width, args.height),
                         projection="3d", facecolor=BACKGROUND)
    ax.set_axis_off()
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0, 0, 0, 0))

    if context is not None:
        shell = (context - center) / span

        ax.scatter(shell[:, 0], shell[:, 1], shell[:, 2],
                   s=args.shell_size, c=SHELL_COLOR, alpha=args.shell_alpha,
                   linewidths=0, depthshade=False)

    cloud = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       s=args.size, c=base, linewidths=0, depthshade=False)

    everything = np.vstack([points] + ([shell] if context is not None else []))

    low, high = everything.min(axis=0), everything.max(axis=0)
    middle = (low + high) / 2.0
    extent = np.maximum(high - low, 1e-6)
    margin = extent * 0.02
    ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
    ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
    ax.set_zlim(low[2] - margin[2], high[2] + margin[2])
    ax.set_box_aspect(tuple(extent / extent.max()))


    ax.view_init(elev=args.elev, azim=args.azim)
    if str(args.zoom).lower() == "auto":
        zoom = fit_zoom(figure, ax, args)
        print(f"Приближение подобрано: {zoom:.2f}")
    else:
        zoom = float(args.zoom)
    ax.set_position(axes_box(zoom, args.width, args.height))

    legend = list(ROLES.items())
    if sober_mean is not None:
        legend += [("hot", (HOTTER, "сильнее, чем трезвая")),
                   ("cold", (COLDER, "слабее, чем трезвая"))]
    for row, (role, (color, title)) in enumerate(
            r for r in legend if r[0] != "hidden"):
        line = 0.945 - row * 0.052
        figure.text(0.028, line, "\u25cf", color=color, fontsize=13,
                    va="center")
        figure.text(0.045, line, title, color=TEXT_COLOR, fontsize=11,
                    va="center")

    panel = figure.add_axes([0.135, 0.018, 0.30, 0.055 if args.vs_sober is False else 0.085],
                        facecolor=BACKGROUND)
    panel.set_axis_off()
    groups = [r for r in ("vision", "mechano", "smell", "hidden", "output")
              if (roles == r).any()]
    bar_colors = [ROLES[r][0] for r in groups]
    bar_titles = [ROLES[r][1] for r in groups]
    if sober_mean is not None:
        groups = ["cold", "hot"] + groups
        bar_colors = [COLDER, HOTTER] + bar_colors
        bar_titles = ["слабее трезвой", "сильнее трезвой"] + bar_titles
    bars = panel.barh(range(len(groups)), [0] * len(groups),
                      color=bar_colors, height=0.62)
    panel.set_xlim(0, 1)
    panel.set_ylim(-0.7, len(groups) - 0.3)
    for row, title in enumerate(bar_titles):
        panel.text(-0.03, row, title, color=TEXT_COLOR, fontsize=9,
                   ha="right", va="center")

    caption = figure.text(0.975, 0.042, "", color=TEXT_COLOR, fontsize=12,
                          ha="right")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):
        old.unlink()

    masks = {role: roles == role for role in groups}
    total = len(times) if not args.limit else min(args.limit, len(times))
    smooth = None
    blend = 1.0 - math.exp(-1.0 / max(args.smooth * args.fps, 1e-6))
    for index in range(total):
        strength = np.clip(activity[index] / ceiling, 0.0, 1.0)
        colors = np.concatenate(
            [base, (0.20 + 0.80 * strength)[:, None]], axis=1)
        sizes = args.size * (0.28 + 1.8 * strength)
        for bar, role in zip(bars, groups):
            if role in masks:
                bar.set_width(float(np.clip(strength[masks[role]].mean() * 3.0, 0, 1)))
        if sober_mean is not None:
            smooth = activity[index].copy() if smooth is None else (
                smooth + (activity[index] - smooth) * blend)
            change = np.log2((smooth + VS_FLOOR_HZ) / (sober_mean + VS_FLOOR_HZ))

            power = np.where(inner_vs, np.clip(np.abs(change) / VS_FULL, 0.0, 1.0), 0.0)
            tint = np.where((change > 0)[:, None], np.array(HOTTER), np.array(COLDER))
            mix = power[:, None]
            colors[:, :3] = np.clip(base * (1.0 - mix) + tint * mix, 0.0, 1.0)
            colors[:, 3] = np.maximum(colors[:, 3], 0.20 + 0.80 * power)
            sizes = np.maximum(sizes, args.size * (0.28 + 2.2 * power))
            share = inner_vs.sum()
            for bar, role in zip(bars, groups):
                if role in ("cold", "hot"):
                    moved = (change < -VS_MARK) if role == "cold" else (change > VS_MARK)
                    bar.set_width(float(np.clip((moved & inner_vs).sum() / share * 4.0, 0, 1)))
        if marked.any():
            colors[marked, :3] = MARKED[0]
            colors[marked, 3] = np.maximum(colors[marked, 3], 0.55)
            sizes = np.array(sizes, dtype=float)
            sizes[marked] = args.size * (1.6 + 4.4 * strength[marked])
        cloud.set_facecolors(colors)
        cloud.set_sizes(sizes)

        ax.view_init(elev=args.elev,
                     azim=args.azim + args.spin * index / max(total - 1, 1))
        state = (f"    {STATE_NAMES.get(int(states[index]), '')}"
                 if states is not None else "")
        if body_at is not None and body_at[index]:
            state = "    парализована" if body_at[index] == 1 else "    на спине"
        fed = (f"    голод {float(hunger[index]):.2f}"
               if hunger is not None else "")
        dosed = (f"    {dose_word} {float(dose_at[index]):.0%}"
                 if dose_at is not None else "")
        caption.set_text(f"{times[index]:6.2f} с{state}{fed}{dosed}")
        figure.savefig(out_dir / f"frame_{index:05d}.png", dpi=dpi,
                       facecolor=BACKGROUND)
        if index % 20 == 0:
            print(f"\r  кадр {index + 1}/{total}", end="", flush=True)

    print(f"\rГотово: {total} кадров в {out_dir}")
    print(f"Собрать ролик: ffmpeg -framerate {args.fps:.0f} "
          f"-i {out_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p brain.mp4")


if __name__ == "__main__":
    main()