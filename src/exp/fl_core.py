"""Model, data partitioning, attacks, plaintext robust aggregators, sketches and DP accounting."""

from __future__ import annotations

import json
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as _norm


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_processed(name: str, data_dir: str | Path, allow_synthetic: bool = False) -> dict:
    p = Path(data_dir).expanduser() / f"{name}_processed.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return {
            "X_train": z["X_train"].astype(np.float32), "y_train": z["y_train"].astype(np.int64),
            "X_test": z["X_test"].astype(np.float32), "y_test": z["y_test"].astype(np.int64),
            "class_names": [str(c) for c in z["class_names"]],
            "source": f"real:{p.name}", "source_sha256": meta.get("source_sha256"), "meta": meta,
        }
    if not allow_synthetic:
        raise FileNotFoundError(
            f"{p} not found. Run `python -m src.data.prepare_datasets` on the real CSVs first. "
            "(Synthetic data is only allowed with --allow-synthetic for smoke tests.)")
    rng = np.random.default_rng(1234)
    n_feat, n_cls = {"edge_iiotset": (48, 15), "ton_iot": (40, 10), "unsw_nb15": (56, 10)}[name]
    cen = rng.normal(0, 1.5, (n_cls, n_feat))
    y = rng.choice(n_cls, 12000, p=rng.dirichlet(np.ones(n_cls) * 1.2))
    X = (cen[y] + rng.normal(0, 0.65, (len(y), n_feat))).astype(np.float32)
    return {"X_train": X[:10000], "y_train": y[:10000], "X_test": X[10000:], "y_test": y[10000:],
            "class_names": [f"c{i}" for i in range(n_cls)], "source": "SYNTHETIC", "source_sha256": None,
            "meta": {}}


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int,
                        min_size: int = 20, max_tries: int = 100) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(max_tries):
        parts: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for cid, chunk in enumerate(np.split(idx, cuts)):
                parts[cid].extend(chunk.tolist())
        if min(len(pp) for pp in parts) >= min_size:
            return [rng.permutation(np.array(pp, dtype=np.int64)) for pp in parts]
    raise RuntimeError("Dirichlet partition: could not satisfy min_size; raise alpha or lower min_size")


# ----------------------------------------------------------------------------
# Model and local training
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in: int, n_cls: int, h1: int = 128, h2: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, n_cls))

    def forward(self, x):
        return self.net(x)


def get_flat(model: nn.Module) -> np.ndarray:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy().astype(np.float32).copy()


def set_flat(model: nn.Module, flat: np.ndarray) -> None:
    torch.nn.utils.vector_to_parameters(torch.from_numpy(flat.astype(np.float32)), model.parameters())


def local_update(model: nn.Module, theta: np.ndarray, X: np.ndarray, y: np.ndarray, *, lr: float,
                 batch: int, epochs: int, max_steps: int, gen: torch.Generator) -> np.ndarray:
    """Return delta = theta_local - theta after local SGD."""
    set_flat(model, theta)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=gen)
        for s in range(0, len(yt), batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            lossf(model(Xt[b]), yt[b]).backward()
            opt.step()
            steps += 1
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
    return get_flat(model) - theta


@torch.no_grad()
def evaluate(model: nn.Module, theta: np.ndarray, X: np.ndarray, y: np.ndarray, n_cls: int) -> dict:
    set_flat(model, theta)
    model.eval()
    pred = model(torch.from_numpy(X)).argmax(1).numpy()
    acc = float(np.mean(pred == y))
    f1s = []
    for c in range(n_cls):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue  # class absent from test set and never predicted
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"acc": acc, "macro_f1": float(np.mean(f1s)) if f1s else 0.0}


# ----------------------------------------------------------------------------
# Attacks (Byzantine clients know all benign updates: omniscient, standard for ALIE/IPM)
# ----------------------------------------------------------------------------
def alie_z(n: int, f: int) -> float:
    """Baruch et al. (NeurIPS 2019): s = floor(n/2 + 1) - f, z = Phi^{-1}((n - s) / n)."""
    s = int(np.floor(n / 2 + 1)) - f
    return float(_norm.ppf((n - s) / n))


def apply_attack(attack: str, benign: dict[int, np.ndarray], poisoned: dict[int, np.ndarray],
                 byz: list[int], n: int, cfg: dict) -> dict[int, np.ndarray]:
    """benign: every client's honest update. poisoned: label-flipped updates of Byzantine clients
    (only needed for 'scaling'/'label_flip'). Returns the submitted updates."""
    out = dict(benign)
    if attack == "none" or not byz:
        return out
    mat = np.stack([benign[i] for i in sorted(benign)], 0)
    mu, sd = mat.mean(0), mat.std(0)
    for b in byz:
        if attack == "sign_flip":
            out[b] = (-cfg["sign_flip_scale"] * benign[b]).astype(np.float32)
        elif attack == "scaling":
            out[b] = (cfg["scaling_factor"] * poisoned[b]).astype(np.float32)
        elif attack == "label_flip":
            out[b] = poisoned[b]
        elif attack == "alie":
            z = cfg.get("alie_z") or alie_z(n, len(byz))
            out[b] = (mu - z * sd).astype(np.float32)
        elif attack == "minmax":
            out[b] = minmax_update(mat, cfg.get("minmax_dir", "unit_vec"))
        elif attack == "ipm":
            out[b] = (-cfg["ipm_eps"] * mu).astype(np.float32)
        else:
            raise ValueError(attack)
    return out


def minmax_update(benign: np.ndarray, direction: str = "unit_vec") -> np.ndarray:
    """Min-Max attack (Shejwalkar & Houmansadr, NDSS 2021): m = mu + gamma * p with the largest
    gamma such that max_i ||m - g_i|| <= max_{i,j} ||g_i - g_j|| (stays inside the benign spread)."""
    B = benign.astype(np.float64)
    mu = B.mean(0)
    if direction == "std":
        p = -B.std(0)
    elif direction == "sign":
        p = -np.sign(mu)
    else:
        p = -mu / (np.linalg.norm(mu) + 1e-12)
    sq = np.sum(B ** 2, 1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * B @ B.T, 0))
    bound = D.max()
    ok = lambda g: np.max(np.linalg.norm(B - (mu + g * p), axis=1)) <= bound
    lo, hi = 0.0, 1.0
    while ok(hi) and hi < 1e6:
        lo, hi = hi, hi * 2
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return (mu + lo * p).astype(np.float32)


# ----------------------------------------------------------------------------
# Plaintext robust aggregators (baselines; they need to see every update in the clear)
# ----------------------------------------------------------------------------
def agg_fedavg(U: np.ndarray, **_) -> np.ndarray:
    return U.mean(0)


def agg_median(U: np.ndarray, **_) -> np.ndarray:
    return np.median(U, 0)


def agg_trimmed_mean(U: np.ndarray, f: int, **_) -> np.ndarray:
    n = U.shape[0]
    k = min(f, (n - 1) // 2)
    S = np.sort(U, 0)
    return S[k:n - k].mean(0)


def _krum_scores(U: np.ndarray, f: int) -> np.ndarray:
    n = U.shape[0]
    sq = np.sum(U.astype(np.float64) ** 2, 1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (U.astype(np.float64) @ U.T.astype(np.float64)), 0.0)
    m = max(1, n - f - 2)
    return np.array([np.sort(np.delete(D[i], i))[:m].sum() for i in range(n)])


def agg_krum(U: np.ndarray, f: int, **_) -> np.ndarray:
    return U[int(np.argmin(_krum_scores(U, f)))].copy()


def agg_multi_krum(U: np.ndarray, f: int, **_) -> np.ndarray:
    sel = np.argsort(_krum_scores(U, f))[: max(1, U.shape[0] - f)]
    return U[sel].mean(0)


def agg_fltrust(U: np.ndarray, server_update: np.ndarray, **_) -> np.ndarray:
    """Cao et al., NDSS 2021: ReLU(cos) trust scores, magnitudes normalised to the server update."""
    g0 = server_update.astype(np.float64)
    n0 = np.linalg.norm(g0) + 1e-12
    norms = np.linalg.norm(U, axis=1) + 1e-12
    ts = np.maximum((U @ g0) / (norms * n0), 0.0)
    if ts.sum() <= 0:
        return np.zeros_like(server_update)
    Un = U * (n0 / norms)[:, None]
    return (ts[:, None] * Un).sum(0) / ts.sum()


PLAINTEXT_AGGS = {
    "fedavg": agg_fedavg, "median": agg_median, "trimmed_mean": agg_trimmed_mean,
    "krum": agg_krum, "multi_krum": agg_multi_krum, "fltrust": agg_fltrust,
}


# ----------------------------------------------------------------------------
# Sketch + DP accounting
# ----------------------------------------------------------------------------
def projection_matrix(m: int, d: int, seed: int, kind: str = "rademacher") -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "rademacher":
        return (rng.integers(0, 2, size=(m, d), dtype=np.int8) * 2 - 1).astype(np.float32) / np.float32(np.sqrt(m))
    return rng.normal(0, 1 / np.sqrt(m), size=(m, d)).astype(np.float32)


def clip_norm(g: np.ndarray, c: float) -> np.ndarray:
    n = float(np.linalg.norm(g))
    return (g * (c / n)).astype(np.float32) if n > c else g.astype(np.float32)


def make_sketch(gbar: np.ndarray, P: np.ndarray, clip_s: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """s = clip(P g_bar, C_s) + N(0, sigma^2 I)  (post-projection clip: L2 sensitivity 2 C_s for any P)."""
    v = clip_norm(P @ gbar.astype(np.float32), clip_s)
    return (v + rng.normal(0, sigma, P.shape[0]).astype(np.float32)).astype(np.float32)


def zcdp_eps(clip_s: float, sigma: float, rounds: int, delta: float) -> dict:
    """Client-level replace-one: Delta_2 = 2 C_s, rho_1 = Delta^2/(2 sigma^2) = 2 C_s^2/sigma^2;
    T-fold composition rho = T rho_1; eps = rho + 2 sqrt(rho ln(1/delta)) (Bun & Steinke 2016)."""
    rho1 = 2 * clip_s ** 2 / sigma ** 2
    rho = rounds * rho1
    return {"rho_1": rho1, "rho_T": rho, "eps_T": rho + 2 * sqrt(rho * log(1 / delta)),
            "eps_1_analytic": analytic_gaussian_eps(2 * clip_s, sigma, delta), "delta": delta, "rounds": rounds}


def sigma_for_eps(clip_s: float, eps: float, rounds: int, delta: float) -> float:
    """Inverse of zcdp_eps: rho = (sqrt(ln(1/delta) + eps) - sqrt(ln(1/delta)))^2 (conversion
    eps = rho + 2 sqrt(rho ln(1/delta))), rho = T * 2 C_s^2 / sigma^2  ->  sigma = C_s sqrt(2 T / rho)."""
    L = log(1.0 / delta)
    rho = (sqrt(L + eps) - sqrt(L)) ** 2
    return float(clip_s * sqrt(2.0 * rounds / rho))


def classic_gaussian_eps(sens: float, sigma: float, delta: float) -> float:
    """eps from sigma = sens*sqrt(2 ln(1.25/delta))/eps (Dwork-Roth Thm A.1; only valid for eps < 1)."""
    return sens * sqrt(2 * log(1.25 / delta)) / sigma


def analytic_gaussian_eps(sens: float, sigma: float, delta: float) -> float:
    """Smallest eps such that N(0, sigma^2) with L2 sensitivity `sens` is (eps, delta)-DP
    (Balle & Wang, ICML 2018, Thm 8). Bisection on the exact condition."""
    Phi = lambda t: 0.5 * (1 + erf(t / sqrt(2)))

    def delta_of(eps: float) -> float:
        a = sens / (2 * sigma)
        b = eps * sigma / sens
        try:
            return Phi(a - b) - exp(eps) * Phi(-a - b)
        except OverflowError:
            return 0.0

    lo, hi = 0.0, 1.0
    while delta_of(hi) > delta:
        hi *= 2
        if hi > 1e7:
            return float("inf")
    for _ in range(200):
        mid = (lo + hi) / 2
        if delta_of(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi
