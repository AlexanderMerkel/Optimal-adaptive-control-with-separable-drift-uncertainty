import argparse
import csv
from pathlib import Path

import torch as th

from common import (
    CustomNeuralNetwork,
    ProblemConfig,
    G,
    ensure_outputs_dir,
    eta,
    log_run_metadata,
    load_yaml_config,
    problem_bounds,
    resolve_device,
    safe_load_tensor,
    setup_seed,
    terminal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train benchmark cost PDE (naive or CE control)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--mode", choices=["naive", "ce"], default=None)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-epochs", type=int, default=30000)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-v", type=int, default=512)
    parser.add_argument("--layers-v", type=int, default=2)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--dlambda", type=float, default=1.0e-4)
    parser.add_argument("--dl", type=float, default=1.0e-3)
    parser.add_argument("--naive-lambda", type=float, default=0.5)
    parser.add_argument("--include-lambda-endpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--prefer-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--riccati-file", type=str, default="outputs/fd_riccati.pt")

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_yaml_config(pre_args.config)
    if cfg:
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    if args.mode is None:
        parser.error("`--mode` is required (set via CLI or YAML config).")
    return args


def riccati_lookup(riccati_solution: th.Tensor, t: th.Tensor, lam: th.Tensor, dt: float, dlambda: float):
    n_lam = riccati_solution.shape[0]
    n_t = riccati_solution.shape[1]

    lam_idx = th.floor_divide(lam, dlambda).long().clamp_(0, n_lam - 1)
    t_idx = th.floor_divide(t, dt).long().clamp_(0, n_t - 1)
    return riccati_solution[lam_idx, t_idx]


def u_lqr(x: th.Tensor, lam: th.Tensor, riccati_solution: th.Tensor, dt: float, dlambda: float) -> th.Tensor:
    return -(lam / ProblemConfig.rho) * riccati_lookup(riccati_solution, x[:, 0], lam, dt, dlambda) * x[:, 1]


def hjb_operator(u, x, mode: str, riccati_solution: th.Tensor, args: argparse.Namespace):
    Du = th.autograd.grad(u(x).sum(dim=0), x, create_graph=True)[0]
    Dtu, Dyu, Dmu, Dgu = Du.unbind(1)
    Dyyu, Dymu = th.autograd.grad(Dyu.sum(dim=0), x, create_graph=True)[0][:, 1:3].unbind(1)
    Dmmu = th.autograd.grad(Dmu.sum(dim=0), x, create_graph=True)[0][:, 2]

    G_out = G(x[:, 2:4], args.dl, include_endpoint=args.include_lambda_endpoint)
    if mode == "ce":
        lam = G_out
    else:
        lam = th.full_like(G_out, args.naive_lambda)

    ctrl_eval = u_lqr(x, lam, riccati_solution, args.dt, args.dlambda)
    sig = ProblemConfig.sig

    term1 = ctrl_eval * (G_out * Dyu + Dymu) + ctrl_eval**2 * (
        ((1.0 / sig**2) * G_out * Dmu)
        + ((1.0 / sig**2) * Dgu)
        + ((1.0 / sig**2) * Dmmu / 2.0)
        + ProblemConfig.rho
    )
    term3 = Dtu + sig**2 * Dyyu / 2.0 + ProblemConfig.c * x[:, 1] ** 2
    return term1 + term3


def loss_embed(value_net, sample, mode: str, riccati_solution: th.Tensor, args: argparse.Namespace):
    def u_composed(x):
        return value_net(x) * eta(x).unsqueeze(1) + terminal(x).unsqueeze(1)

    loss_int = hjb_operator(u_composed, sample, mode, riccati_solution, args)
    return th.mean(loss_int**2)


def save_loss_history(path: Path, rows: list[tuple[int, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_value"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(prefer_cuda=args.prefer_cuda)
    setup_seed(args.seed, deterministic=args.deterministic)
    output_dir = ensure_outputs_dir(args.output_dir)

    riccati_path = Path(args.riccati_file)
    if not riccati_path.exists():
        raise FileNotFoundError(
            f"Missing Riccati file: {riccati_path}. Run fd_riccati_solver.py first."
        )
    riccati_solution = safe_load_tensor(riccati_path, device=device)

    neural_v = CustomNeuralNetwork(ProblemConfig.d, args.hidden_v, 1, args.layers_v).to(device)
    opt_v = th.optim.Adam(neural_v.parameters(), lr=args.lr)
    scheduler_v = th.optim.lr_scheduler.MultiStepLR(opt_v, milestones=[8000, 13000], gamma=0.1)

    min_point, max_point = problem_bounds(device)
    dist = th.distributions.Uniform(min_point, max_point)

    script_name = f"cost_{args.mode}_control"
    log_run_metadata(output_dir, script_name, vars(args), device)

    value_loss = th.tensor(th.inf, device=device)
    history: list[tuple[int, float]] = []

    for epoch in range(1, args.max_epochs + 1):
        opt_v.zero_grad()
        sample = dist.sample((args.batch_size,)).requires_grad_(True)
        value_loss = loss_embed(neural_v, sample, args.mode, riccati_solution, args)

        if not th.isfinite(value_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {value_loss.item()}")

        value_loss.backward()
        opt_v.step()
        scheduler_v.step()

        if epoch % args.print_interval == 0:
            print(f"epoch={epoch} value_loss={value_loss.item():.3e}")

        history.append((epoch, float(value_loss.item())))

        if value_loss.item() <= args.tol:
            print(f"Early stop at epoch {epoch} (value loss <= {args.tol})")
            break

    th.save(neural_v.state_dict(), output_dir / f"neural_v_{args.mode}.pt")
    save_loss_history(output_dir / f"loss_{args.mode}.csv", history)
    print(f"Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
