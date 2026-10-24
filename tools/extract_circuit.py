"""Собирает схему из коннектома FlyWire (FAFB v783) обходом графа.

Положите в data/raw/ файлы из Codex -> Info -> Download Data:
    consolidated_cell_types.csv[.gz]
    classification.csv[.gz]
    connections_princeton.csv[.gz]
    coordinates.csv[.gz]          (необязательно, нужен для 3D-картинки)

Запуск:  python tools/extract_circuit.py
         python tools/extract_circuit.py --hops 2 --min-synapses 20

"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "circuit.npz"

# Якоря
ESCAPE_TYPES = ["DNp01", "DNp02", "DNp11"]
ODOR_TYPES = ["DNb05"]
ANCHOR_TYPES = ESCAPE_TYPES + ODOR_TYPES

# Сенсорные типы
SENSORY_TYPES = [
    "LC4", "LPLC1", "LPLC2", "JO-A", "JO-B",
    "ORN_V", "ORN_DA2", "ORN_DM1", "ORN_DM2", "ORN_VA2", "ORN_DM4", "ORN_DM5",
    "HRN_VP1d", "HRN_VP1l", "HRN_VP4", "HRN_VP5",
    "TRN_VP1m", "TRN_VP2", "TRN_VP3a", "TRN_VP3b",
]

CHUNK = 2_000_000
SYN_THRESHOLD = 5   # минимум синапсов на пару нейронов в итоговой матрице

# У мух ацетилхолин возбуждает, ГАМК и глутамат (через GluCl) тормозят.
NT_SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0,
           "DA": 0.0, "SER": 0.0, "OCT": 0.0, "UNK": 0.0}

READ_KWARGS = dict(
    usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"],
    dtype={"pre_root_id": np.int64, "post_root_id": np.int64,
           "syn_count": np.int32},
    chunksize=CHUNK,
)


def find_file(*patterns):
    for pattern in patterns:
        for suffix in ("", ".gz"):
            hits = sorted(glob.glob(str(RAW_DIR / (pattern + suffix))))
            if hits:
                return hits[0]
    return None


def load_catalogue(path_types, path_class):
    """Полный список нейронов: root_id -> тип, сторона."""
    types = pd.read_csv(path_types)
    column = next((c for c in ("primary_type", "cell_type", "type")
                   if c in types.columns), None)
    if column is None:
        sys.exit(f"Нет колонки с типом клетки. Есть: {list(types.columns)}")

    catalogue = types[["root_id", column]].dropna()
    catalogue.columns = ["root_id", "cell_type"]
    catalogue["root_id"] = catalogue["root_id"].astype(np.int64)
    catalogue["cell_type"] = catalogue["cell_type"].astype(str).str.strip()

    if path_class:
        cls = pd.read_csv(path_class,
                          usecols=lambda c: c in ("root_id", "side"))
        cls["root_id"] = cls["root_id"].astype(np.int64)
        catalogue = catalogue.merge(cls, on="root_id", how="left")
    if "side" not in catalogue.columns:
        catalogue["side"] = "unknown"
    catalogue["side"] = catalogue["side"].fillna("unknown")
    return catalogue


def scan_upstream(path_conn, targets, label):
    """Кто и сколько синапсов даёт на заданное множество нейронов."""
    print(f"  {label}: ищу входы для {len(targets):,} нейронов", end="", flush=True)
    found, total = [], 0
    for chunk in pd.read_csv(path_conn, **READ_KWARGS):
        total += len(chunk)
        hit = chunk[chunk.post_root_id.isin(targets)]
        if len(hit):
            found.append(hit.groupby("pre_root_id", as_index=False).syn_count.sum())
        print(f"\r  {label}: ищу входы для {len(targets):,} нейронов "
              f"({total:,} строк)", end="", flush=True)
    print()
    if not found:
        return pd.DataFrame(columns=["pre_root_id", "syn_count"])
    return pd.concat(found).groupby("pre_root_id", as_index=False).syn_count.sum()


def scan_sensory_outputs(path_conn, sensory):
    """Все исходящие связи сенсорных нейронов, парами (кто -> кому)."""
    print(f"  проход вперёд: исходящие у {len(sensory):,} сенсорных",
          end="", flush=True)
    found, total = [], 0
    for chunk in pd.read_csv(path_conn, **READ_KWARGS):
        total += len(chunk)
        hit = chunk[chunk.pre_root_id.isin(sensory)]
        if len(hit):
            found.append(hit.groupby(["pre_root_id", "post_root_id"],
                                     as_index=False).syn_count.sum())
        print(f"\r  проход вперёд: исходящие у {len(sensory):,} сенсорных "
              f"({total:,} строк)", end="", flush=True)
    print()
    if not found:
        return pd.DataFrame(columns=["pre_root_id", "post_root_id", "syn_count"])
    return (pd.concat(found)
            .groupby(["pre_root_id", "post_root_id"], as_index=False)
            .syn_count.sum())


def scan_downstream(path_conn, sources, label):
    """Кому и сколько синапсов отдаёт заданное множество нейронов."""
    print(f"  {label}: ищу выходы у {len(sources):,} нейронов", end="", flush=True)
    found, total = [], 0
    for chunk in pd.read_csv(path_conn, **READ_KWARGS):
        total += len(chunk)
        hit = chunk[chunk.pre_root_id.isin(sources)]
        if len(hit):
            found.append(hit.groupby("post_root_id", as_index=False).syn_count.sum())
        print(f"\r  {label}: ищу выходы у {len(sources):,} нейронов "
              f"({total:,} строк)", end="", flush=True)
    print()
    if not found:
        return pd.DataFrame(columns=["post_root_id", "syn_count"])
    return pd.concat(found).groupby("post_root_id", as_index=False).syn_count.sum()


def scan_internal(path_conn, ids):
    """Связи внутри схемы плюс голоса за нейромедиатор по всем выходам."""
    print("  финальный проход: связи внутри схемы", end="", flush=True)
    edges, votes, total = [], [], 0
    for chunk in pd.read_csv(path_conn, **READ_KWARGS):
        total += len(chunk)
        pre_ours = chunk.pre_root_id.isin(ids)
        if pre_ours.any():
            votes.append(chunk.loc[pre_ours].groupby(
                ["pre_root_id", "nt_type"], as_index=False).syn_count.sum())
            inner = pre_ours & chunk.post_root_id.isin(ids)
            if inner.any():
                edges.append(chunk.loc[inner, ["pre_root_id", "post_root_id",
                                               "syn_count"]])
        print(f"\r  финальный проход: связи внутри схемы ({total:,} строк)",
              end="", flush=True)
    print()
    if not edges:
        sys.exit("Ни одной связи внутри схемы. Проверьте файл.")
    edges = (pd.concat(edges, ignore_index=True)
             .groupby(["pre_root_id", "post_root_id"], as_index=False).syn_count.sum())
    votes = (pd.concat(votes)
             .groupby(["pre_root_id", "nt_type"], as_index=False).syn_count.sum())
    return edges, votes


def neurotransmitters(votes, root_ids):
    """Медиатор по большинству исходящих синапсов нейрона + доля победителя."""
    best = votes.loc[votes.groupby("pre_root_id").syn_count.idxmax()] \
                .set_index("pre_root_id")
    totals = votes.groupby("pre_root_id").syn_count.sum()

    nt, conf = [], []
    for root_id in root_ids:
        if root_id in best.index:
            nt.append(str(best.at[root_id, "nt_type"]).upper())
            conf.append(float(best.at[root_id, "syn_count"]) / float(totals.at[root_id]))
        else:
            nt.append("UNK")
            conf.append(0.0)
    nt = np.array(nt)
    return nt, np.array(conf, dtype=np.float32), \
        np.array([NT_SIGN.get(x, 0.0) for x in nt], dtype=np.float32)


def load_positions(root_ids):
    """Координаты нейронов в мозге, если файл с ними лежит в data/raw/."""
    path = find_file("coordinates*.csv", "*coordinates*.csv", "positions*.csv")
    if not path:
        print(f"\nКоординат нет: файла coordinates*.csv в {RAW_DIR} не нашлось.")
        print("  Скачайте его из Codex (Info -> Download Data) и запустите снова,")
        print("  если нужна трёхмерная визуализация.")
        return None

    table = pd.read_csv(path)
    print(f"\nКоординаты: {Path(path).name}, колонки {list(table.columns)}")
    if "root_id" not in table.columns:
        print("  Нет колонки root_id, пропускаю.")
        return None
    table["root_id"] = table["root_id"].astype(np.int64)

    trio = [c for c in table.columns if c.lower() in ("x", "y", "z")]
    if len(trio) == 3:
        order = sorted(trio, key=lambda c: "xyz".index(c.lower()))
        coords = table[["root_id"] + order].copy()
        coords.columns = ["root_id", "x", "y", "z"]
    else:
        column = next((c for c in table.columns
                       if c.lower() in ("position", "pos", "coordinates", "xyz")), None)
        if column is None:
            print(f"  Не понял, где координаты. Есть: {list(table.columns)}")
            return None
        numbers = table[column].astype(str).str.findall(r"-?\d+(?:\.\d+)?")
        keep = numbers.str.len() >= 3
        coords = pd.DataFrame({
            "root_id": table.root_id[keep].values,
            "x": [float(v[0]) for v in numbers[keep]],
            "y": [float(v[1]) for v in numbers[keep]],
            "z": [float(v[2]) for v in numbers[keep]],
        })

    coords = coords.groupby("root_id", as_index=False)[["x", "y", "z"]].mean()
    lookup = coords.set_index("root_id")

    found = sum(1 for r in root_ids if r in lookup.index)
    print(f"  координаты нашлись у {found} из {len(root_ids)} нейронов схемы")
    if not found:
        return None

    positions = np.full((len(root_ids), 3), np.nan, dtype=np.float32)
    for i, root_id in enumerate(root_ids):
        if root_id in lookup.index:
            positions[i] = lookup.loc[root_id, ["x", "y", "z"]].to_numpy()
    return positions


def parse_args():
    p = argparse.ArgumentParser(description="Собрать схему обходом коннектома")
    p.add_argument("--hops", type=int, default=1,
                   help="на сколько синаптических шагов вверх от якорей")
    p.add_argument("--forward-hops", type=int, default=1,
                   help="шагов ВНИЗ от сенсорных, которые не дотягиваются до "
                        "схемы сами (обоняние). Второй шаг затаскивает "
                        "грибовидное тело, см. комментарий в коде")
    p.add_argument("--min-synapses", type=int, default=10,
                   help="минимум синапсов на текущий фронт, чтобы попасть в схему")
    p.add_argument("--max-neurons", type=int, default=5000,
                   help="предохранитель: больше этого не берём")
    return p.parse_args()


def main():
    args = parse_args()

    path_types = find_file("consolidated_cell_types*.csv")
    path_class = find_file("classification*.csv")
    path_conn = find_file("connections_princeton*.csv", "connections*.csv")
    if not path_types or not path_conn:
        sys.exit(f"Не хватает файлов. Положите их в {RAW_DIR}")

    catalogue = load_catalogue(path_types, path_class)
    print(f"В каталоге: {len(catalogue):,} нейронов")

    anchors = set(catalogue.loc[catalogue.cell_type.isin(ANCHOR_TYPES), "root_id"])
    sensory = set(catalogue.loc[catalogue.cell_type.isin(SENSORY_TYPES), "root_id"])
    if not anchors:
        sys.exit(f"Якорные типы не найдены: {ANCHOR_TYPES}")
    print(f"Якоря ({', '.join(ANCHOR_TYPES)}): {len(anchors)} нейронов")

    #обход графа вверх от якорей
    selected = set(anchors)
    frontier = set(anchors)
    for hop in range(1, args.hops + 1):
        upstream = scan_upstream(path_conn, frontier, f"шаг {hop}")
        strong = upstream[upstream.syn_count >= args.min_synapses]
        strong = strong.sort_values("syn_count", ascending=False)

        room = args.max_neurons - len(selected)
        if len(strong) > room:
            print(f"  предохранитель: из {len(strong):,} кандидатов беру "
                  f"{room:,} самых сильных")
            strong = strong.head(max(room, 0))

        frontier = set(strong.pre_root_id) - selected
        selected |= frontier
        print(f"  шаг {hop}: добавлено {len(frontier):,}, всего {len(selected):,}")
        if not frontier:
            break

    selected |= sensory

    outgoing = scan_sensory_outputs(path_conn, sensory)
    inside = selected - sensory
    reach = outgoing[outgoing.post_root_id.isin(inside)]
    reach = reach.groupby("pre_root_id").syn_count.sum()
    connected = set(reach[reach >= args.min_synapses].index)
    orphans = sensory - connected
    print(f"Сенсорных: {len(sensory)}, дотягиваются до схемы сами "
          f"{len(connected)}, идут вперёд {len(orphans)}")

    frontier = orphans
    for hop in range(1, args.forward_hops + 1):
        if not frontier:
            break
        if hop == 1:
            step = outgoing[outgoing.pre_root_id.isin(frontier)]
            step = step.groupby("post_root_id", as_index=False).syn_count.sum()
        else:
            step = scan_downstream(path_conn, frontier, f"вперёд, шаг {hop}")
        strong = step[step.syn_count >= args.min_synapses]
        strong = strong.sort_values("syn_count", ascending=False)

        room = args.max_neurons - len(selected)
        if len(strong) > room:
            print(f"  предохранитель: из {len(strong):,} кандидатов беру "
                  f"{room:,} самых сильных")
            strong = strong.head(max(room, 0))

        frontier = set(strong.post_root_id) - selected
        selected |= frontier
        print(f"  вперёд, шаг {hop}: добавлено {len(frontier):,}, "
              f"всего {len(selected):,}")

    neurons = catalogue[catalogue.root_id.isin(selected)].copy()
    neurons = neurons.sort_values(["cell_type", "side", "root_id"]).reset_index(drop=True)
    n = len(neurons)
    print(f"\nВ схеме: {n:,} нейронов")

    #связи и медиаторы
    id_to_index = {rid: i for i, rid in enumerate(neurons.root_id.values)}
    edges, votes = scan_internal(path_conn, set(id_to_index))
    nt, conf, sign = neurotransmitters(votes, neurons.root_id.values)

    cell_type_arr = neurons.cell_type.values
    receptors = np.isin(cell_type_arr, SENSORY_TYPES)
    sign_forced = receptors & (nt != "ACH")
    if sign_forced.any():
        sign = sign.copy()
        sign[sign_forced] = 1.0
        by_type = pd.Series(cell_type_arr[sign_forced]).value_counts()
        print(f"\nЗнак проставлен снаружи: {int(sign_forced.sum())} первичных "
              f"рецепторов из {int(receptors.sum())}")
        print("  предсказание дало не ацетилхолин у: "
              + ", ".join(f"{k} {v}" for k, v in by_type.head(8).items()))
        print("  первичные сенсорные нейроны насекомых холинергические - "
              "это НАШЕ знание, не из коннектома")

    strong_edges = edges[edges.syn_count >= SYN_THRESHOLD]
    weights = np.zeros((n, n), dtype=np.float32)
    np.add.at(weights,
              (strong_edges.pre_root_id.map(id_to_index).values,
               strong_edges.post_root_id.map(id_to_index).values),
              strong_edges.syn_count.values)

    #отчёт
    cell_type = neurons.cell_type.values
    excitatory = int((sign > 0).sum())
    inhibitory = int((sign < 0).sum())
    unknown = int((sign == 0).sum())

    print(f"\nМедиаторы: возбуждающих {excitatory:,}, тормозных {inhibitory:,}, "
          f"неизвестных {unknown:,}")
    if inhibitory:
        print(f"  торможение составляет {inhibitory / n:.0%} схемы - "
              f"именно его не хватало в ручном списке")

    counts = pd.Series(cell_type).value_counts()
    print(f"\nТипов клеток: {len(counts)}. Самые многочисленные:")
    for name, count in counts.head(15).items():
        mark = ""
        if name in ANCHOR_TYPES:
            mark = "  <- якорь"
        elif name in SENSORY_TYPES:
            mark = "  <- вход"
        print(f"  {name:22s} {count:5d}{mark}")

    print(f"\nСвязей: {len(edges):,} пар, {len(strong_edges):,} "
          f"после порога >= {SYN_THRESHOLD}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    positions = load_positions(neurons.root_id.values)

    extra = {} if positions is None else {"positions": positions}
    np.savez_compressed(
        OUT_PATH,
        root_ids=neurons.root_id.values,
        cell_type=cell_type.astype(str),
        side=neurons.side.values.astype(str),
        nt=nt, sign=sign, conf=conf, W=weights,
        sign_forced=sign_forced,
        threshold=np.array(SYN_THRESHOLD),
        sensory_types=np.array(SENSORY_TYPES),
        anchor_types=np.array(ANCHOR_TYPES),
        escape_types=np.array(ESCAPE_TYPES),
        **extra,
    )
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nСохранено: {OUT_PATH} ({size_mb:.1f} МБ)")
    print("Проверить: python tools/inspect_circuit.py")


if __name__ == "__main__":
    main()