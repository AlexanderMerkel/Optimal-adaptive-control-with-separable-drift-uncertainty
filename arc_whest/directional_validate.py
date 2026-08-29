from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from mlp_kprop.factor_k3 import FactoredTensor
from mlp_kprop.kprop_harmonic import Kind, coerce_input, linear_kprop, nonlin_kprop
from mlp_kprop.mlp import MLP
from mlp_kprop.wick import relu_wick_coef

from common import DEPTH, WIDTH, load_rows, mse
from directional_multilayer import gaussian_states, k3_diagonal_at


def make_mlp(weights: np.ndarray) -> MLP:
    mlp = MLP(
        input_dim=WIDTH, hidden_dim=WIDTH, output_dim=WIDTH,
        num_layers=DEPTH + 1, nonlin="relu", init_kind="manual",
        w_var=[2.0] * DEPTH + [1.0], b_var=0.0, b_mean=0.0,
    ).to(device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for layer in range(DEPTH):
            mlp.Ws[layer].weight.copy_(
                torch.from_numpy(np.ascontiguousarray(weights[layer].T))
            )
        mlp.Ws[DEPTH].weight.copy_(torch.eye(WIDTH))
    return mlp


def exact_pre(weights: np.ndarray):
    mlp = make_mlp(weights)
    K = coerce_input(
        {1: torch.zeros(WIDTH), 2: torch.eye(WIDTH)},
        k_max=3, kind=Kind.SIMPLE,
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        for layer in range(DEPTH - 1):
            K = linear_kprop(K, mlp.Ws[layer].weight, k_max=3, set_metric=None, bias=None)
            K = nonlin_kprop(
                K, nonlin_wick_coef=relu_wick_coef, k_max=3,
                kind=Kind.SIMPLE, use_pK=True, factor=True,
            )
        K = linear_kprop(K, mlp.Ws[-2].weight, k_max=3, set_metric=None, bias=None)
    ft = K[3]
    if not isinstance(ft, FactoredTensor):
        raise TypeError(type(ft))
    f = ft._factors
    diagonal = torch.sum(f[0] * f[1] * f[2], dim=1)
    return (
        K[1].core.detach().double().numpy(),
        K[2].to_tensor().detach().double().numpy(),
        diagonal.detach().double().numpy(),
        time.perf_counter() - t0,
    )


def compare(a: np.ndarray, e: np.ndarray) -> dict[str, float]:
    scale = float(np.sum(a * e) / max(np.sum(a * a), 1e-300))
    return {
        "relative_l2": float(np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-300)),
        "correlation": float(np.corrcoef(a, e)[0, 1]),
        "optimal_scale": scale,
        "scaled_relative_l2": float(np.linalg.norm(scale * a - e) / max(np.linalg.norm(e), 1e-300)),
        "approx_rms": float(np.sqrt(np.mean(a * a))),
        "exact_rms": float(np.sqrt(np.mean(e * e))),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=16)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--order", type=int, default=12)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    records = []
    for local, row in enumerate(load_rows(a.start, a.rows), 1):
        weights = np.ascontiguousarray(row["weights"], dtype=np.float32)
        state = gaussian_states(weights, a.order)
        approximate, elapsed = k3_diagonal_at(weights, state, DEPTH - 1)
        exact_mean, exact_cov, exact, exact_elapsed = exact_pre(weights)
        record = {
            "row_index": a.start + local - 1,
            "mlp_id": int(row["mlp_id"]),
            "mlp_name": str(row["mlp_name"]),
            "comparison": compare(approximate, exact),
            "elapsed_s": {"directional": elapsed, "full": exact_elapsed},
        }
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)
    result = {"complete": True, "records": records}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
