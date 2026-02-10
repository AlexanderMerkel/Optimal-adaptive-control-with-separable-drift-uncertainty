import json
import random
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import yaml


def ensure_outputs_dir(path: str = "outputs") -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def setup_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    if deterministic:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        th.use_deterministic_algorithms(True)


def resolve_device(prefer_cuda: bool = True) -> th.device:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cuda_available = th.cuda.is_available()
    if prefer_cuda and cuda_available:
        return th.device("cuda")
    return th.device("cpu")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_yaml_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping/dictionary: {path}")
    return payload


def log_run_metadata(output_dir: Path, script_name: str, cfg: dict, device: th.device) -> None:
    payload = {
        "script": script_name,
        "git_commit": get_git_commit(),
        "device": str(device),
        "torch_version": th.__version__,
        "torch_cuda": th.version.cuda,
        "config": cfg,
    }
    write_json(output_dir / f"{script_name}_run_config.json", payload)


class CustomNeuralNetwork(nn.Module):
    # DGM architecture for value function approximation.
    def __init__(self, input_size: int, hidden_size: int, output_size: int, depth: int):
        super().__init__()
        self.depth = depth
        self.sigmoid = nn.Sigmoid()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.Uz = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(depth)])
        self.Wz = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.Ug = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(depth)])
        self.Wg = nn.Linear(hidden_size, hidden_size)
        self.Ur = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(depth)])
        self.Wr = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.Uh = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(depth)])
        self.Wh = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: th.Tensor) -> th.Tensor:
        state = self.sigmoid(self.W1(x))
        for layer_idx in range(self.depth):
            z_gate = self.sigmoid(self.Uz[layer_idx](x) + self.Wz[layer_idx](state))
            g_gate = self.sigmoid(self.Ug[layer_idx](x) + self.Wg(state))
            r_gate = self.sigmoid(self.Ur[layer_idx](x) + self.Wr[layer_idx](state))
            h_tilde = self.sigmoid(self.Uh[layer_idx](x) + self.Wh[layer_idx](state * r_gate))
            state = (1 - g_gate) * h_tilde + z_gate * state
        return self.output(state)


def build_control_network(input_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    net = th.nn.Sequential()
    net.add_module("linear0", th.nn.Linear(input_dim, hidden_dim))
    net.add_module("sigmoid0", th.nn.Sigmoid())
    for i in range(1, depth):
        net.add_module(f"linear{i}", th.nn.Linear(hidden_dim, hidden_dim))
        net.add_module(f"sigmoid{i}", th.nn.Sigmoid())
    net.add_module(f"linear{depth}", th.nn.Linear(hidden_dim, 1))
    return net


class ProblemConfig:
    d = 4
    y_min, y_max = -3.0, 3.0
    m_min, m_max = -5.0, 5.0
    g_min, g_max = 0.0, 10.0
    T = 1.0
    sig = 1.0
    c = 2.0
    rho = 2.0
    C = 5.0


def problem_bounds(device: th.device) -> tuple[th.Tensor, th.Tensor]:
    min_point = th.tensor(
        [0.0, ProblemConfig.y_min, ProblemConfig.m_min, ProblemConfig.g_min],
        device=device,
    )
    max_point = th.tensor(
        [
            ProblemConfig.T,
            ProblemConfig.y_max,
            ProblemConfig.m_max,
            ProblemConfig.g_max,
        ],
        device=device,
    )
    return min_point, max_point


def density(lmbd: th.Tensor) -> th.Tensor:
    return th.ones_like(lmbd)


def F_m(x: th.Tensor, lmbd: th.Tensor) -> th.Tensor:
    return lmbd * th.exp(lmbd * x[:, 0] - 0.5 * lmbd**2 * x[:, 1]) * density(lmbd)


def F(x: th.Tensor, lmbd: th.Tensor) -> th.Tensor:
    return th.exp(lmbd * x[:, 0] - 0.5 * lmbd**2 * x[:, 1]) * density(lmbd)


def G(x: th.Tensor, dl: float, include_endpoint: bool = False) -> th.Tensor:
    stop = 1.0 + (dl if include_endpoint else 0.0)
    lmbd = th.arange(0.0, stop, dl, device=x.device, dtype=x.dtype)
    integral1 = th.trapezoid(F_m(x.unsqueeze(-1), lmbd), lmbd)
    integral2 = th.trapezoid(F(x.unsqueeze(-1), lmbd), lmbd)
    return integral1 / integral2


def terminal(x: th.Tensor) -> th.Tensor:
    return ProblemConfig.C * x[:, 1] ** 2


def eta(x: th.Tensor) -> th.Tensor:
    return ProblemConfig.T - x[:, 0]


def safe_load_tensor(path: Path, device: th.device) -> th.Tensor:
    return th.load(path, map_location=device)
