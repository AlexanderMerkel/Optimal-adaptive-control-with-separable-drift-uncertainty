import argparse
from pathlib import Path

import numpy as np
import torch as th
from scipy.integrate import odeint

from common import (
    ProblemConfig,
    ensure_outputs_dir,
    load_yaml_config,
    log_run_metadata,
    resolve_device,
    setup_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Riccati table for benchmark controls")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--dlambda", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefer-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_yaml_config(pre_args.config)
    if cfg:
        parser.set_defaults(**cfg)

    return parser.parse_args()


def riccati(y, _t, a, c):
    return -a * y**2 + c


def solve_riccati_odeint(a, c, y0, T, dt):
    num_steps = int(T / dt)
    t = np.linspace(0.0, T, num_steps + 1)
    y = odeint(riccati, y0, t, args=(a, c))
    return t, y


def main() -> None:
    args = parse_args()
    device = resolve_device(prefer_cuda=args.prefer_cuda)
    setup_seed(args.seed, deterministic=False)
    output_dir = ensure_outputs_dir(args.output_dir)

    T = ProblemConfig.T
    yT = ProblemConfig.C
    rho = ProblemConfig.rho
    c = ProblemConfig.c

    lam_range = np.arange(0.0, 1.0, args.dlambda)
    solution = th.zeros((len(lam_range), int(T / args.dt) + 1), dtype=th.float32, device=device)

    for idx, lam in enumerate(lam_range):
        a = lam**2 / (rho**2)
        _, y = solve_riccati_odeint(a, c, yT, T, args.dt)
        # Reverse to index by forward time t while preserving terminal-value condition.
        y_tensor = th.from_numpy(y).float().squeeze().flip(0).to(device)
        solution[idx, :] = y_tensor

    out_path = output_dir / "fd_riccati.pt"
    th.save(solution, out_path)
    log_run_metadata(output_dir, "fd_riccati_solver", vars(args), device)
    print(f"Saved Riccati solution: {out_path} with shape={tuple(solution.shape)}")


if __name__ == "__main__":
    main()
