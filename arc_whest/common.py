from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np
from scipy.special import ndtr

DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v2-phase2"
WIDTH = 1024
DEPTH = 16


def load_rows(start: int, rows: int) -> Iterable[dict[str, object]]:
    from datasets import load_dataset

    ds = load_dataset(
        DATASET, revision=REVISION, split="mini", streaming=True
    ).with_format("numpy")
    return itertools.islice(ds, start, start + rows)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d * d))


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def relu_gaussian_covariance(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    nodes: np.ndarray,
    weights: np.ndarray,
):
    var = np.maximum(np.diag(cov), 1e-18)
    sigma = np.sqrt(var)
    alpha = mu / sigma
    phi = normal_pdf(alpha)
    Phi = ndtr(alpha)
    mean = mu * Phi + sigma * phi
    second = (mu * mu + var) * Phi + mu * sigma * phi
    post_var = np.maximum(second - mean * mean, 0.0)
    sigma_outer = np.outer(sigma, sigma)
    rho = np.clip(cov / np.maximum(sigma_outer, 1e-30), -1 + 1e-10, 1 - 1e-10)
    a = alpha[:, None]
    b = alpha[None, :]
    integral = np.zeros_like(cov)
    for node, weight in zip(nodes, weights):
        s = 0.5 * rho * (node + 1.0)
        one_minus = np.maximum(1.0 - s * s, 1e-18)
        density = np.exp(
            -(a * a - 2.0 * s * a * b + b * b) / (2.0 * one_minus)
        ) / (2.0 * math.pi * np.sqrt(one_minus))
        integral += weight * (rho - s) * density
    integral *= 0.5 * rho
    out = sigma_outer * (rho * np.outer(Phi, Phi) + integral)
    np.fill_diagonal(out, post_var)
    return mean, 0.5 * (out + out.T), second, phi, Phi
