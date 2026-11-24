import numpy as np


def create_dp_sketch(
    gradient: np.ndarray,
    sketch_dim: int = 128,
    clip_norm: float = 5.0,
    dp_sigma: float = 0.05,
    seed: int = 2026,
) -> np.ndarray:
    """
    Create a privacy-preserving low-dimensional sketch via L2 clipping,
    scaled random projection, and Gaussian DP noise without exposing raw gradients.
    """
    g = gradient.astype(np.float32).ravel()
    norm = float(np.linalg.norm(g))
    if norm > clip_norm and norm > 0.0:
        g = g * (clip_norm / norm)

    rng = np.random.default_rng(seed)
    proj = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(sketch_dim),
        size=(sketch_dim, g.size),
    ).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=dp_sigma, size=(sketch_dim,)).astype(np.float32)
    return (proj @ g + noise).astype(np.float32)


def sketch_scores(
    sketches: dict[int, np.ndarray],
    directional_weight: float = 0.5,
    reference: np.ndarray | None = None,
    eps: float = 1e-8,
) -> tuple[list[int], np.ndarray]:
    """
    Per-client outlier scores (larger = more suspicious): a hybrid of the normalized L2 distance to the
    coordinate-wise median sketch and the normalized cosine distance to a reference direction.
    reference=None uses the median sketch itself (original PQ-Full). A public, noise-free reference
    (e.g. the sketch P*Delta_{t-1} of the previous, already released global update, or the sketch of a
    server root-set update) removes the noise of the median from the directional term. Using it is
    post-processing of the DP sketches, so it costs no privacy.
    """
    client_ids = sorted(sketches.keys())
    n = len(client_ids)
    if n == 0:
        return [], np.zeros(0)

    mat = np.stack([sketches[cid].astype(np.float64).ravel() for cid in client_ids], axis=0)
    median_vec = np.median(mat, axis=0)

    # 1. Normalized L2 deviation from median sketch
    l2_dists = np.linalg.norm(mat - median_vec, axis=1)
    med_l2 = float(np.median(l2_dists))
    norm_l2 = l2_dists / (med_l2 + eps)

    # 2. Directional / cosine distance from the reference (default: median sketch): 1 - cos_sim
    ref = median_vec if reference is None else np.asarray(reference, np.float64).ravel()
    mat_norms = np.linalg.norm(mat, axis=1)
    ref_norm = float(np.linalg.norm(ref))
    cos_sim = (mat @ ref) / (mat_norms * ref_norm + eps)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    cos_dists = 1.0 - cos_sim
    med_cos = float(np.median(cos_dists))
    norm_cos = cos_dists / (med_cos + eps)

    w_dir = float(np.clip(directional_weight, 0.0, 1.0))
    scores = (1.0 - w_dir) * norm_l2 + w_dir * norm_cos
    return client_ids, scores


def reject_top(client_ids: list[int], scores: np.ndarray, num_rejected: int) -> tuple[list[int], list[int]]:
    n = len(client_ids)
    num_rejected = min(int(num_rejected), max(0, n - 1))
    order = np.argsort(scores)
    rejected_idx = set(order[-num_rejected:].tolist()) if num_rejected > 0 else set()
    accepted = [client_ids[i] for i in range(n) if i not in rejected_idx]
    rejected = [client_ids[i] for i in range(n) if i in rejected_idx]
    return accepted, rejected


def filter_byzantine_sketches(
    sketches: dict[int, np.ndarray],
    max_byzantine_fraction: float = 0.25,
    directional_weight: float = 0.5,
    eps: float = 1e-8,
    reference: np.ndarray | None = None,
) -> tuple[list[int], list[int]]:
    """
    Filter Byzantine outliers using a hybrid of normalized L2 sketch distance
    and directional (cosine) consistency relative to the coordinate-wise median sketch
    (or to a public reference direction, see sketch_scores).
    """
    client_ids, scores = sketch_scores(sketches, directional_weight, reference, eps)
    if not client_ids:
        return [], []
    n = len(client_ids)
    return reject_top(client_ids, scores, int(np.floor(n * max_byzantine_fraction)))


class ReputationFilter:
    """Multi-round reputation: R_i <- lam * R_i + (1 - lam) * score_i (first observation: R_i = score_i);
    each round the floor(beta*N) clients with the largest R_i among the current senders are rejected.
    Pure post-processing of the released sketches (no extra privacy cost). Per-round noise in the
    scores averages out over an effective window of about (1 + lam) / (1 - lam) rounds, while a
    consistently malicious client keeps a high reputation score. lam = 0 gives the single-round filter."""

    def __init__(self, lam: float):
        self.lam = float(lam)
        self.R: dict[int, float] = {}

    def update(self, client_ids: list[int], scores: np.ndarray) -> np.ndarray:
        out = np.empty(len(client_ids))
        for k, (cid, sc) in enumerate(zip(client_ids, scores)):
            prev = self.R.get(cid)
            self.R[cid] = float(sc) if prev is None else self.lam * prev + (1.0 - self.lam) * float(sc)
            out[k] = self.R[cid]
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    dim = 4096
    honest_base = rng.normal(loc=0.0, scale=0.2, size=dim).astype(np.float32)

    grads: dict[int, np.ndarray] = {}
    for cid in range(15):
        grads[cid] = honest_base + rng.normal(0.0, 0.02, size=dim).astype(np.float32)
    # Stealthy ALIE-style shifted clients (15..19)
    for cid in range(15, 20):
        grads[cid] = (-1.5 * honest_base - 0.65 * np.std(honest_base)).astype(np.float32)

    sketches = {cid: create_dp_sketch(grads[cid], sketch_dim=128, dp_sigma=0.05, seed=99) for cid in grads}
    accepted, rejected = filter_byzantine_sketches(sketches, max_byzantine_fraction=0.25, directional_weight=0.5)
    print(f"Accepted ({len(accepted)}): {accepted}")
    print(f"Rejected ({len(rejected)}): {rejected}")
