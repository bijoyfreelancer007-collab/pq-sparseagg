"""Correctness tests for the PQ secure-aggregation simulation.  Run:  python -m pytest -q tests  (or python tests/test_secagg.py)"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp import fl_core as F  # noqa: E402
from src.secagg.protocol import FixedPoint, PQSecureAggregator, shamir_combine, shamir_split  # noqa: E402

N, D, M, SIG = 20, 2000, 64, 0.05


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    g = {i: F.clip_norm(rng.normal(0, 0.05, D).astype(np.float32), 5.0) for i in range(N)}
    agg = PQSecureAggregator(N, None, graph_seed=seed, fp=FixedPoint(16, 5.0))
    return rng, g, agg


def _round(agg, g, rng, select, sketch_src=None, dropped=None, tamper=None):
    proj = lambda s: F.projection_matrix(M, D, s)
    src = sketch_src or g
    sk = lambda i, P: F.make_sketch(src[i], P, np.sqrt(2) * 5.0, SIG, rng)
    return agg.run_round(1, g, proj, sk, select, rng, sigma=SIG, dropped=dropped, tamper=tamper)


def test_shamir_roundtrip():
    rng = np.random.default_rng(1)
    sec = os.urandom(32)
    sh = shamir_split(sec, 5, 3, rng)
    assert shamir_combine(sh[:3]) == sec
    assert shamir_combine([sh[0], sh[2], sh[4]]) == sec
    try:
        assert shamir_combine(sh[:2]) != sec  # below threshold: wrong value
    except OverflowError:
        pass  # below threshold: value outside the secret range


def test_graph_is_regular_and_log_degree():
    _, _, agg = _setup()
    assert agg.degree == 5 and all(len(v) == 5 for v in agg.graph.values())
    agg.close()


def test_exact_unmask_with_rejections():
    rng, g, agg = _setup()
    rejected = {1, 4, 7, 12, 19}
    sel = lambda sk: ([i for i in sk if i not in rejected], sorted(rejected))
    mean, A, R, st = _round(agg, g, rng, sel)
    assert st.unmask_exact and st.check_passed and not st.aborted
    assert set(R) == rejected
    ref = np.mean([g[i] for i in A], 0)
    assert np.max(np.abs(mean - ref)) < 2 ** -16
    agg.close()


def test_dropout_handled():
    rng, g, agg = _setup(2)
    mean, A, R, st = _round(agg, g, rng, lambda sk: (sorted(sk), []), dropped={3})
    assert 3 in R and st.unmask_exact and not st.aborted
    agg.close()


def test_tampered_upload_rejected():
    rng, g, agg = _setup(3)
    mean, A, R, st = _round(agg, g, rng, lambda sk: (sorted(sk), []), tamper={5})
    assert st.bad_signature == [5] and 5 in R and st.unmask_exact
    agg.close()


def test_masked_vectors_look_uniform():
    rng, g, agg = _setup(4)
    from src.secagg.protocol import prg_u32
    q = agg.fp.encode(g[0])
    y = q + prg_u32(os.urandom(32), b"SELF", D)
    frac = y.astype(np.float64) / 2 ** 32
    assert abs(frac.mean() - 0.5) < 0.03 and q.std() != y.std()
    agg.close()


def test_inconsistent_sketch_aborts_round():
    rng, g, agg = _setup(5)
    byz = {0, 1, 2, 3, 4}
    bad = {i: (F.clip_norm(-50 * g[i], 5.0) if i in byz else g[i]) for i in range(N)}
    # Byzantine clients upload `bad` but sketch their benign update -> check must fail
    mean, A, R, st = _round(agg, bad, rng, lambda sk: (sorted(sk), []), sketch_src=g)
    assert mean is None and not st.check_passed and st.aborted == "consistency check failed"
    agg.close()


def test_alie_z():
    assert abs(F.alie_z(20, 5) - 0.5244) < 1e-3


def test_zcdp_matches_theory_table():
    # theory thread: sigma/C_s = 27.7 -> eps ~ 1 over T=8, delta=1e-5
    C_s = np.sqrt(2) * 5
    e = F.zcdp_eps(C_s, 27.7 * C_s, 8, 1e-5)["eps_T"]
    assert 0.95 < e < 1.05, e


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
