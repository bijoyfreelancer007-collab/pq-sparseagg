"""Unit tests for the 2026-11-23 filter additions (run: python tests/test_improve.py)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.exp import fl_core as F  # noqa: E402
from src.fl.byzantine_filter import (ReputationFilter, filter_byzantine_sketches, reject_top,  # noqa: E402
                                     sketch_scores)


def test_sigma_for_eps_inverts_zcdp():
    for C in (0.5, 1.0, 5.0):
        for eps in (1.0, 4.0, 8.0, 32.0):
            s = F.sigma_for_eps(math.sqrt(2) * C, eps, 30, 1e-5)
            assert abs(F.zcdp_eps(math.sqrt(2) * C, s, 30, 1e-5)["eps_T"] - eps) < 1e-9
    assert abs(F.sigma_for_eps(math.sqrt(2) * 5, 8.0, 30, 1e-5) - 53.5) < 0.05  # Remark 1 value


def test_default_reference_equals_original_filter():
    rng = np.random.default_rng(0)
    for _ in range(50):
        sk = {i: rng.normal(size=32) for i in range(20)}
        ids, sc = sketch_scores(sk, 0.5)
        assert reject_top(ids, sc, 5) == filter_byzantine_sketches(sk, 0.25 + 1e-12, 0.5)


def test_public_reference_beats_noisy_median():
    rng = np.random.default_rng(1)
    m, sigma, hits_med, hits_ref = 128, 3.0, 0, 0
    for _ in range(200):
        u = rng.normal(size=m); u /= np.linalg.norm(u)
        sk = {i: (1.0 if i >= 5 else -1.0) * u + rng.normal(0, sigma / math.sqrt(m) * 4, m) for i in range(20)}
        a, r = reject_top(*sketch_scores(sk, 1.0), 5)
        hits_med += len(set(r) & set(range(5)))
        a, r = reject_top(*sketch_scores(sk, 1.0, reference=u), 5)
        hits_ref += len(set(r) & set(range(5)))
    assert hits_ref > hits_med, (hits_ref, hits_med)


def test_reputation_ema():
    rf = ReputationFilter(0.8)
    assert np.allclose(rf.update([0, 1], np.array([1.0, 3.0])), [1.0, 3.0])
    assert np.allclose(rf.update([0, 1], np.array([2.0, 0.0])), [1.2, 2.4])
    assert np.allclose(rf.update([1], np.array([1.0])), [2.12])
    assert np.allclose(ReputationFilter(0.0).update([0], np.array([5.0])), [5.0])


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
