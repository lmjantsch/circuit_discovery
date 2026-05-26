"""
Compute RelP-style PCC between exact component AP and local attribution scores.

The AP CSV is produced by run_component_activation_patching.py. Attribution
scores are loaded from an MIB/EAP importances.json and aggregated from edges to
component outputs by default:

  attn_out(layer) = sum over outgoing scores from all a{layer}.h* nodes
  mlp_out(layer)  = sum over outgoing scores from m{layer}

Residual-stream AP rows are kept in the AP CSV but skipped here unless a score
source provides matching residual component scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ComponentKey = Tuple[str, int]


ATTN_NODE_RE = re.compile(r"^a(?P<layer>\d+)\.h(?P<head>\d+)$")
MLP_NODE_RE = re.compile(r"^m(?P<layer>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ap-csv", required=True)
    parser.add_argument("--importances-json", required=True)
    parser.add_argument(
        "--edge-direction",
        choices=["outgoing", "incoming"],
        default="outgoing",
        help="Aggregate edge scores around the component as a source or destination.",
    )
    parser.add_argument(
        "--score-agg",
        choices=["signed_sum", "abs_sum"],
        default="signed_sum",
    )
    parser.add_argument(
        "--position-agg",
        choices=["final", "mean", "sum"],
        default="mean",
        help="How to reduce AP rows over token positions for each layer/component.",
    )
    parser.add_argument(
        "--ap-value",
        choices=["delta_metric_patch", "patched_metric"],
        default="delta_metric_patch",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["attn_out", "mlp_out"],
        help="Component types to include in PCC.",
    )
    parser.add_argument("--joined-out", default=None)
    args = parser.parse_args()

    ap_scores = load_ap_component_scores(
        Path(args.ap_csv),
        ap_value=args.ap_value,
        position_agg=args.position_agg,
        components=set(args.components),
    )
    attribution_scores = load_component_scores_from_importances(
        Path(args.importances_json),
        edge_direction=args.edge_direction,
        score_agg=args.score_agg,
    )
    rows = join_scores(ap_scores, attribution_scores)
    if args.joined_out:
        write_rows(Path(args.joined_out), rows)

    report(rows)


def load_ap_component_scores(
    path: Path,
    ap_value: str,
    position_agg: str,
    components: set[str],
) -> Dict[ComponentKey, float]:
    by_batch_component: Dict[Tuple[str, int, int], List[Tuple[int, float]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            component_type = row["component_type"]
            if component_type not in components:
                continue
            layer = int(row["layer"])
            batch_idx = int(row["batch_idx"])
            pos = int(row["pos"])
            value = float(row[ap_value])
            by_batch_component[(component_type, layer, batch_idx)].append((pos, value))

    reduced_by_component: Dict[ComponentKey, List[float]] = defaultdict(list)
    for (component_type, layer, _batch_idx), pos_values in by_batch_component.items():
        values = [value for _pos, value in pos_values]
        if position_agg == "mean":
            reduced = sum(values) / len(values)
        elif position_agg == "sum":
            reduced = sum(values)
        elif position_agg == "final":
            final_pos = max(pos for pos, _value in pos_values)
            final_values = [value for pos, value in pos_values if pos == final_pos]
            reduced = sum(final_values) / len(final_values)
        else:
            raise ValueError(f"Unknown position aggregation: {position_agg}")
        reduced_by_component[(component_type, layer)].append(reduced)

    return {
        key: sum(values) / len(values)
        for key, values in reduced_by_component.items()
    }


def load_component_scores_from_importances(
    path: Path,
    edge_direction: str,
    score_agg: str,
) -> Dict[ComponentKey, float]:
    with path.open() as f:
        data = json.load(f)

    scores: Dict[ComponentKey, float] = defaultdict(float)
    for edge_name, info in data["edges"].items():
        src, dst = edge_name.split("->", 1)
        node = src if edge_direction == "outgoing" else strip_qkv_suffix(dst)
        key = node_to_component_key(node)
        if key is None:
            continue
        score = float(info["score"])
        if score_agg == "abs_sum":
            score = abs(score)
        elif score_agg != "signed_sum":
            raise ValueError(f"Unknown score aggregation: {score_agg}")
        scores[key] += score
    return dict(scores)


def strip_qkv_suffix(node: str) -> str:
    return node.split("<", 1)[0]


def node_to_component_key(node: str) -> ComponentKey | None:
    attn_match = ATTN_NODE_RE.match(node)
    if attn_match:
        return ("attn_out", int(attn_match.group("layer")))
    mlp_match = MLP_NODE_RE.match(node)
    if mlp_match:
        return ("mlp_out", int(mlp_match.group("layer")))
    return None


def join_scores(
    ap_scores: Dict[ComponentKey, float],
    attribution_scores: Dict[ComponentKey, float],
) -> List[dict]:
    rows = []
    for key in sorted(ap_scores):
        if key not in attribution_scores:
            continue
        component_type, layer = key
        rows.append(
            {
                "component_type": component_type,
                "layer": layer,
                "ap_score": ap_scores[key],
                "attribution_score": attribution_scores[key],
            }
        )
    return rows


def report(rows: List[dict]) -> None:
    if not rows:
        raise ValueError("No overlapping component scores found.")

    all_ap = [float(row["ap_score"]) for row in rows]
    all_attr = [float(row["attribution_score"]) for row in rows]
    print(f"n = {len(rows)}")
    print(f"PCC(all) = {pearson(all_ap, all_attr):.6g}")
    print(f"PCC(abs all) = {pearson([abs(x) for x in all_ap], [abs(y) for y in all_attr]):.6g}")

    by_component: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_component[row["component_type"]].append(row)
    for component_type in sorted(by_component):
        component_rows = by_component[component_type]
        xs = [float(row["ap_score"]) for row in component_rows]
        ys = [float(row["attribution_score"]) for row in component_rows]
        print(f"{component_type}: n = {len(component_rows)}")
        print(f"{component_type}: PCC = {pearson(xs, ys):.6g}")
        print(f"{component_type}: PCC(abs) = {pearson([abs(x) for x in xs], [abs(y) for y in ys]):.6g}")


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Pearson inputs must be non-empty and equal length.")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
