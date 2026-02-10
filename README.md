# Optimal adaptive control with separable drift uncertainty

Companion code for the paper: https://arxiv.org/abs/2309.07091

This repository implements the Section 6 numerical workflow:
- adaptive control HJB via DGM + policy iteration style alternating updates,
- naive and certainty-equivalent (CE) benchmark cost PDEs,
- Riccati table precomputation for the benchmark feedback maps.

## Environment

- Python: 3.13+
- PyTorch: CUDA-enabled wheel recommended for GPU runs
- Core packages: `torch`, `numpy`, `scipy`, `matplotlib`

Install:

```bash
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130
```

## Files

- `common.py`: shared constants, architectures, utilities
- `fd_riccati_solver.py`: precompute Riccati table (`outputs/fd_riccati.pt`)
- `dgm_train.py`: train adaptive value + control networks
- `benchmark_cost.py`: train benchmark cost PDE (`--mode naive|ce`)
- `cost_naive_control.py`: wrapper for `benchmark_cost.py --mode naive`
- `cost_ce_control.py`: wrapper for `benchmark_cost.py --mode ce`

## Local workflow

1. Precompute Riccati table:

```bash
python fd_riccati_solver.py --config configs/paper_fd_riccati.yaml
```

2. Train adaptive model:

```bash
python dgm_train.py --config configs/paper_adaptive.yaml
```

3. Train benchmark costs:

```bash
python benchmark_cost.py --config configs/paper_naive.yaml
python benchmark_cost.py --config configs/paper_ce.yaml
```

All scripts auto-create `outputs/` and write:
- model weights (`.pt`),
- loss history (`.csv`),
- run metadata (`*_run_config.json`) with git commit, device, and config.

YAML configs are optional; CLI flags override YAML values when both are provided.

## Reproducibility and runtime controls

Every script supports:
- `--seed` and `--deterministic`
- `--print-interval`
- `--batch-size`, `--max-epochs`, `--lr`

Suggested quick debug run (CPU/GPU):

```bash
python fd_riccati_solver.py --config configs/local_fd_riccati.yaml
python dgm_train.py --config configs/local_adaptive.yaml
python benchmark_cost.py --config configs/local_naive.yaml
python benchmark_cost.py --config configs/local_ce.yaml
```

## Notes on expected resources

Full paper-scale runs are compute intensive:
- adaptive training default batch size: 7500,
- benchmark training default batch size: 10000,
- epochs up to 30000.

A recent CUDA GPU is strongly recommended for paper-scale timings.
