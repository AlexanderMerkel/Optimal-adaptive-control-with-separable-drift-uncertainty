import argparse
import csv
from pathlib import Path

import torch as th

from common import (
    CustomNeuralNetwork,
    ProblemConfig,
    build_control_network,
    ensure_outputs_dir,
    eta,
    G,
    log_run_metadata,
    load_yaml_config,
    problem_bounds,
    setup_seed,
    terminal,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive DGM + control network")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--batch-size", type=int, default=7500)
    parser.add_argument("--max-epochs", type=int, default=30000)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-v", type=int, default=512)
    parser.add_argument("--hidden-ctrl", type=int, default=512)
    parser.add_argument("--layers-v", type=int, default=2)
    parser.add_argument("--layers-ctrl", type=int, default=2)
    parser.add_argument("--dl", type=float, default=1.0e-3)
    parser.add_argument("--include-lambda-endpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--prefer-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-prefix", type=str, default="adaptive")

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_yaml_config(pre_args.config)
    if cfg:
        parser.set_defaults(**cfg)

    return parser.parse_args()


def hjb_operator(u, x, ctrl, dl, include_lambda_endpoint):
    Du = th.autograd.grad(u(x).sum(dim=0), x, create_graph=True)[0]
    Dtu, Dyu, Dmu, Dgu = Du.unbind(1)
    Dyyu, Dymu = th.autograd.grad(Dyu.sum(dim=0), x, create_graph=True)[0][:, 1:3].unbind(1)
    Dmmu = th.autograd.grad(Dmu.sum(dim=0), x, create_graph=True)[0][:, 2]

    G_out = G(x[:, 2:4], dl, include_endpoint=include_lambda_endpoint)
    ctrl_eval = ctrl(x).squeeze(-1)

    sig = ProblemConfig.sig
    term1 = ctrl_eval * (G_out * Dyu + Dymu) + ctrl_eval**2 * (
        ((1.0 / sig**2) * G_out * Dmu)
        + ((1.0 / sig**2) * Dgu)
        + ((1.0 / sig**2) * Dmmu / 2.0)
        + ProblemConfig.rho
    )
    term3 = Dtu + sig**2 * Dyyu / 2.0 + ProblemConfig.c * x[:, 1] ** 2
    return term1, term1 + term3


def loss_embed(value_net, ctrl_net, sample, dl, include_lambda_endpoint):
    def u_composed(x):
        return value_net(x) * eta(x).unsqueeze(1) + terminal(x).unsqueeze(1)

    loss_ctrl, loss_int = hjb_operator(u_composed, sample, ctrl_net, dl, include_lambda_endpoint)
    return th.mean(loss_ctrl), th.mean(loss_int**2)


def save_loss_history(path: Path, rows: list[tuple[int, float, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_value", "loss_ctrl"])
        writer.writerows(rows)


def train(args: argparse.Namespace) -> None:
    device = resolve_device(prefer_cuda=args.prefer_cuda)
    setup_seed(args.seed, deterministic=args.deterministic)
    output_dir = ensure_outputs_dir(args.output_dir)

    neural_v = CustomNeuralNetwork(ProblemConfig.d, args.hidden_v, 1, args.layers_v).to(device)
    neural_ctrl = build_control_network(ProblemConfig.d, args.hidden_ctrl, args.layers_ctrl).to(device)

    opt_v = th.optim.Adam(neural_v.parameters(), lr=args.lr)
    opt_ctrl = th.optim.Adam(neural_ctrl.parameters(), lr=args.lr)
    scheduler_v = th.optim.lr_scheduler.MultiStepLR(opt_v, milestones=[8000, 13000], gamma=0.1)
    scheduler_ctrl = th.optim.lr_scheduler.MultiStepLR(opt_ctrl, milestones=[8000, 13000], gamma=0.1)

    min_point, max_point = problem_bounds(device)
    dist = th.distributions.Uniform(min_point, max_point)

    log_run_metadata(output_dir, "dgm_train", vars(args), device)

    value_loss = th.tensor(th.inf, device=device)
    history: list[tuple[int, float, float]] = []

    for epoch in range(1, args.max_epochs + 1):
        opt_v.zero_grad()
        opt_ctrl.zero_grad()

        sample = dist.sample((args.batch_size,)).requires_grad_(True)
        ctrl_loss, value_loss = loss_embed(
            neural_v,
            neural_ctrl,
            sample,
            args.dl,
            args.include_lambda_endpoint,
        )

        if not th.isfinite(value_loss) or not th.isfinite(ctrl_loss):
            raise RuntimeError(
                f"Non-finite loss encountered at epoch {epoch}: "
                f"value={value_loss.item()}, ctrl={ctrl_loss.item()}"
            )

        if epoch % 2 == 0:
            value_loss.backward()
            opt_v.step()
            scheduler_v.step()
        else:
            ctrl_loss.backward()
            opt_ctrl.step()
            scheduler_ctrl.step()

        if epoch % args.print_interval == 0:
            print(
                f"epoch={epoch} value_loss={value_loss.item():.3e} "
                f"ctrl_loss={ctrl_loss.item():.3e}"
            )

        history.append((epoch, float(value_loss.item()), float(ctrl_loss.item())))

        if value_loss.item() <= args.tol:
            print(f"Early stop at epoch {epoch} (value loss <= {args.tol})")
            break

    th.save(neural_v.state_dict(), output_dir / f"neural_v_{args.model_prefix}.pt")
    th.save(neural_ctrl.state_dict(), output_dir / f"neural_ctrl_{args.model_prefix}.pt")
    save_loss_history(output_dir / f"loss_{args.model_prefix}.csv", history)
    print(f"Saved artifacts to: {output_dir}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
