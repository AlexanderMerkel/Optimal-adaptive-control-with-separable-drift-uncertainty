from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from common import DEPTH, WIDTH, load_rows, mse, normal_pdf, relu_gaussian_covariance


def gaussian_states(weights: np.ndarray, order: int) -> dict:
    nodes, qw = np.polynomial.legendre.leggauss(order)
    mean = np.zeros(WIDTH, dtype=np.float64)
    cov = np.eye(WIDTH, dtype=np.float64)
    state = {k: [] for k in (
        "pre_mean", "pre_cov", "post_mean", "gain1", "gain2", "phi"
    )}
    t0 = time.perf_counter()
    for w32 in weights:
        w = np.asarray(w32, dtype=np.float64)
        pre_mean = w.T @ mean
        pre_cov = w.T @ cov @ w
        pre_cov = 0.5 * (pre_cov + pre_cov.T)
        var = np.maximum(np.diag(pre_cov), 1e-18)
        sd = np.sqrt(var)
        alpha = pre_mean / sd
        phi = normal_pdf(alpha)
        gain1 = ndtr(alpha)
        gain2 = phi / sd
        mean, cov, *_ = relu_gaussian_covariance(
            pre_mean, pre_cov, nodes=nodes, weights=qw
        )
        state["pre_mean"].append(pre_mean)
        state["pre_cov"].append(pre_cov)
        state["post_mean"].append(mean.copy())
        state["gain1"].append(gain1)
        state["gain2"].append(gain2)
        state["phi"].append(phi)
    state["post_mean"] = np.stack(state["post_mean"])
    state["elapsed_s"] = time.perf_counter() - t0
    return state


def k3_diagonal_at(weights: np.ndarray, state: dict, target: int) -> tuple[np.ndarray, float]:
    if target == 0:
        return np.zeros(WIDTH), 0.0
    directions = np.asarray(weights[target], dtype=np.float64).copy()
    result = np.zeros(WIDTH, dtype=np.float64)
    t0 = time.perf_counter()
    for source in range(target - 1, -1, -1):
        q = state["pre_cov"][source] @ (
            state["gain1"][source][:, None] * directions
        )
        result += 3.0 * np.sum(
            state["gain2"][source][:, None] * directions * q * q,
            axis=0,
        )
        if source > 0:
            directions = np.asarray(weights[source], dtype=np.float64) @ (
                state["gain1"][source][:, None] * directions
            )
    return result, time.perf_counter() - t0


def response_matrices(weights: np.ndarray, state: dict) -> list[np.ndarray]:
    response = [np.empty((0, 0)) for _ in range(DEPTH)]
    response[-1] = np.eye(WIDTH)
    for layer in range(DEPTH - 2, -1, -1):
        response[layer] = np.asarray(weights[layer + 1], dtype=np.float64) @ (
            state["gain1"][layer + 1][:, None] * response[layer + 1]
        )
    return response


def coefficient(state: dict, layer: int) -> np.ndarray:
    mean = state["pre_mean"][layer]
    var = np.maximum(np.diag(state["pre_cov"][layer]), 1e-18)
    alpha = mean / np.sqrt(var)
    return -alpha * state["phi"][layer] / (6.0 * var)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=16)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--order", type=int, default=12)
    p.add_argument("--depths", default="1,2,4,8,16")
    p.add_argument("--scales", default="0.5,0.75,1,1.25,1.5,2")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    scales = [float(x) for x in a.scales.split(",")]
    records = []
    wall = time.perf_counter()
    for local, row in enumerate(load_rows(a.start, a.rows), 1):
        weights = np.ascontiguousarray(row["weights"], dtype=np.float32)
        truth = np.asarray(row["all_layer_means"], dtype=np.float64)
        state = gaussian_states(weights, a.order)
        response = response_matrices(weights, state)
        contributions = []
        timings = []
        for layer in range(DEPTH):
            k3, elapsed = k3_diagonal_at(weights, state, layer)
            local_mean = coefficient(state, layer) * k3
            final = local_mean @ response[layer]
            contributions.append(final)
            timings.append(elapsed)
            print(layer + 1, elapsed, np.sqrt(np.mean(final * final)), flush=True)
        contributions = np.stack(contributions)
        base = state["post_mean"][-1]
        metrics = {"k2": mse(base, truth[-1])}
        predictions = {"k2": base.astype(np.float32)}
        for depth in depths:
            correction = np.sum(contributions[-depth:], axis=0)
            for scale in scales:
                pred = base + scale * correction
                name = f"last{depth}_scale{scale:g}"
                metrics[name] = mse(pred, truth[-1])
                predictions[name] = pred.astype(np.float32)
        row_index = a.start + local - 1
        np.savez_compressed(
            a.output.with_name(f"{a.output.stem}_{row_index}.npz"),
            truth=truth[-1].astype(np.float32),
            contributions=contributions.astype(np.float32),
            **predictions,
        )
        record = {
            "row_index": row_index,
            "mlp_id": int(row["mlp_id"]),
            "mlp_name": str(row["mlp_name"]),
            "gaussian_elapsed_s": state["elapsed_s"],
            "directional_elapsed_s": timings,
            "metrics": metrics,
            "top": sorted(metrics.items(), key=lambda kv: kv[1]),
        }
        records.append(record)
        print(json.dumps(record["top"][:20], indent=2), flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({"complete": False, "records": records}, indent=2))
    common = sorted(set.intersection(*(set(r["metrics"]) for r in records)))
    aggregate = {
        name: {
            "mean_mse": float(np.mean([r["metrics"][name] for r in records])),
            "rows": [r["metrics"][name] for r in records],
        }
        for name in common
    }
    result = {
        "complete": True,
        "config": vars(a) | {"output": str(a.output)},
        "records": records,
        "aggregate": aggregate,
        "top": sorted(aggregate.items(), key=lambda kv: kv[1]["mean_mse"]),
        "elapsed_s": time.perf_counter() - wall,
    }
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["top"][:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
