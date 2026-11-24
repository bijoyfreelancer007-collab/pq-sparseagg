"""
Print compact summaries of the 2026-11-23 improvement grids (copy-paste friendly) and write a CSV.

  python -m src.exp.summarize_improve [--in experiments/improve]

TPR = rejected Byzantine / (f * rounds); FPR = rejected honest / ((N - f) * rounds).
bound = mean over rounds t of Phi(Phi^{-1}(FPR) + mu_t), mu_t = sqrt(2 t rho_1): the best TPR any
sketch-based filter can reach at the same FPR (Proposition "Detection Limit of DP Sketches").
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

ND = NormalDist()


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:
            print(f"  ! error row in {p.name}: {r['dataset']} {r['attack']} {r['variant']} s{r['seed']}: {r['error']}")
            continue
        out.append(r)
    return out


def rates(r: dict) -> tuple[float | None, float | None]:
    h = r["history"]
    R = len(h["byz_rejected"])
    f = r["byzantine_count"]
    N = r["cfg"]["clients"]
    if R == 0 or not r["variant"].startswith("pq") or r["variant"] == "pq_no_filter":
        return None, None
    tpr = sum(h["byz_rejected"]) / (f * R) if f else None
    fpr = sum(h["honest_rejected"]) / ((N - f) * R)
    return tpr, fpr


def bound(r: dict, fpr: float) -> float | None:
    dp = r.get("dp") or {}
    rho1 = dp.get("rho_1")
    if rho1 is None or fpr is None or fpr <= 0 or fpr >= 1:
        return None
    R = len(r["history"]["byz_rejected"])
    z = ND.inv_cdf(fpr)
    return float(np.mean([ND.cdf(z + math.sqrt(2 * t * rho1)) for t in range(1, R + 1)]))


def ms(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "   -       "
    return f"{np.mean(xs):.3f}±{np.std(xs):.3f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return f"{np.mean(xs):.2f}" if xs else "  - "


def summarize_grid(name: str, recs: list[dict], keys: list[str], writer) -> None:
    if not recs:
        return
    print(f"\n=== {name} ({len(recs)} runs) ===")
    g = defaultdict(list)
    for r in recs:
        c = r["cfg"]
        k = (r["dataset"], r["attack"], r["variant"]) + tuple(c.get(x) for x in keys)
        g[k].append(r)
    hdr = f"{'dataset':13s} {'attack':9s} {'variant':12s} " + " ".join(f"{x:>8s}" for x in keys) + \
          f" {'eps_T':>8s} {'n':>2s} {'macroF1':>12s} {'TPR':>5s} {'FPR':>5s} {'bound':>5s} {'div':>3s}"
    print(hdr)
    for k in sorted(g, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        rs = g[k]
        tf = [rates(r) for r in rs]
        tprs, fprs = [t for t, _ in tf], [f for _, f in tf]
        bnds = [bound(r, f) for r, f in zip(rs, fprs)]
        eps = (rs[0].get("dp") or {}).get("eps_T")
        eps_s = f"{eps:8.3g}" if eps is not None else "       -"
        ndiv = sum(bool(r["diverged"]) for r in rs)
        vals = " ".join(f"{'-' if v is None else v!s:>8s}" for v in k[3:])
        print(f"{k[0]:13s} {k[1]:9s} {k[2]:12s} {vals} {eps_s} {len(rs):2d} {ms([r['final_macro_f1'] for r in rs]):>12s} "
              f"{mean(tprs):>5s} {mean(fprs):>5s} {mean(bnds):>5s} {ndiv:3d}")
        writer.writerow([name, *k[:3], json.dumps(dict(zip(keys, k[3:]))), eps, len(rs),
                         np.mean([r["final_macro_f1"] for r in rs]), np.std([r["final_macro_f1"] for r in rs]),
                         mean(tprs), mean(fprs), mean(bnds), ndiv])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ind", default="experiments/improve")
    a = ap.parse_args(argv)
    ind = Path(a.ind)
    with open(ind / "summary_improve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grid", "dataset", "attack", "variant", "params", "eps_T", "n", "f1_mean", "f1_std",
                    "tpr", "fpr", "tpr_bound", "n_diverged"])
        for name, keys in [("privfront", ["clip_C", "dp_eps_target", "rep_lambda"]), ("clipbase", ["clip_C"]), ("replam", ["clip_C", "dp_eps_target", "rep_lambda"]), ("main2", ["clip_C", "dp_eps_target"]),
                           ("alpha2", ["alpha", "clip_C", "dp_eps_target"]), ("timing2", []), ("ton_main", []), ("ton_sigma", ["dp_sigma"]),
                           ("ton_alpha", ["alpha"]), ("ton_timing", [])]:
            recs = load(ind / f"{name}.jsonl")
            summarize_grid(name, recs, keys, w)
            if name in ("timing2", "ton_timing") and recs:
                g = defaultdict(list)
                for r in recs:
                    g[(r["dataset"], r["variant"])].append(r["avg_round_ms"])
                for k, v in sorted(g.items()):
                    print(f"  round time {k[0]:13s} {k[1]:12s} {np.mean(v):8.1f} ± {np.std(v):6.1f} ms")
    # honest update norms (to pick C)
    pf = load(ind / "privfront.jsonl")
    if pf:
        print("\n=== honest pre-clip update norms (median over runs of per-round q10/q50/q90) ===")
        g = defaultdict(list)
        for r in pf:
            g[r["dataset"]].append(np.array(r["history"].get("honest_norm_q", [])))
        for ds, arrs in g.items():
            arrs = [x for x in arrs if x.size]
            if not arrs:
                continue
            Rm = min(len(x) for x in arrs)
            med = np.median(np.stack([x[:Rm] for x in arrs]), 0)
            for t in (0, Rm // 2, Rm - 1):
                print(f"  {ds:13s} round {t + 1:2d}: q10={med[t, 0]:.3f} q50={med[t, 1]:.3f} q90={med[t, 2]:.3f}")
    lp = ind / "leakage.jsonl"
    if lp.exists():
        rows = [json.loads(x) for x in lp.read_text().splitlines() if x.strip()]
        print(f"\n=== label-inference leakage ({len(rows)} rows) ===")
        g = defaultdict(list)
        for r in rows:
            g[(r["dataset"], r.get("clip_C"), r["observation"])].append(r)
        print(f"{'dataset':13s} {'C':>4s} {'observation':18s} {'single':>7s} {'accum':>7s} {'prior':>7s} {'uniform':>7s} n")
        for k, rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2])):
            print(f"{k[0]:13s} {k[1]!s:>4s} {k[2]:18s} {np.mean([r['single_round_acc'] for r in rs]):7.3f} "
                  f"{np.mean([r['accum_acc'] for r in rs]):7.3f} {np.mean([r['prior_acc'] for r in rs]):7.3f} "
                  f"{np.mean([1 / r['n_proto_classes'] for r in rs]):7.3f} {len(rs)}")
    print(f"\nCSV: {ind / 'summary_improve.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
