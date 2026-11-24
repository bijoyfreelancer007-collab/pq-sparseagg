"""
Run the PQ-SparseAgg experiments on the preprocessed real datasets.

Each run (dataset, attack, variant, seed) executes in its own fresh process, so
the reported peak RSS belongs to that run only. Results are appended to a JSONL
file, one line per run; finished runs are skipped on restart (resume).

Examples
  python -m src.exp.run_experiments --grid main   --workers 4
  python -m src.exp.run_experiments --grid timing --workers 1
  python -m src.exp.run_experiments --grid main --datasets unsw_nb15 --seeds 1 --rounds 3   # quick check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

DATASETS = ["edge_iiotset", "ton_iot", "unsw_nb15"]
ATTACKS = ["none", "sign_flip", "scaling", "alie", "minmax"]
PLAIN = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum", "fltrust"]
# PQ variants: (directional weight, directional reference, multi-round reputation)
#   reference "median": coordinate-wise median of the noisy sketches (original PQ-Full)
#   reference "global": P * Delta_{t-1}, the sketch of the previous (already public) global update
#   reference "root":   P * g_0, the sketch of a server root-set update (same trust assumption as FLTrust)
PQ = {"pq_no_filter": (None, "median", False), "pq_l2": (0.0, "median", False),
      "pq_cos": (1.0, "median", False), "pq_full": (0.5, "median", False),
      "pq_rep": (0.5, "median", True), "pq_gref": (1.0, "global", False),
      "pq_grep": (1.0, "global", True), "pq_root": (1.0, "root", True)}
NEW_PQ = ["pq_rep", "pq_gref", "pq_grep", "pq_root"]
VARIANTS = PLAIN + list(PQ)
SEEDS = [1, 2, 3, 4, 5]
CODE_VERSION = "2026-11-23.rep-v1"  # + reputation / public-reference filter variants (fltrust-v2 unchanged)

DEFAULT_CFG = {
    "clients": 20,
    "byzantine_fraction": 0.25,
    "rounds": 30,                # convergence check 2026-11-22: UNSW plateaus ~20, Edge still rising at 30
    "alpha": 0.5,
    "local_lr": 0.05,
    "server_lr": 1.0,
    "batch": 64,
    "local_epochs": 1,
    "max_local_steps": 50,
    "hidden": [128, 64],
    "degree": None,              # None -> max(2, ceil(log2 N))
    "sketch_dim": 128,
    "clip_C": 5.0,               # update clip before masking (g_bar = clip(g, C))
    "clip_s": None,              # post-projection sketch clip; None -> sqrt(2) * C
    "dp_sigma": 0.05,            # 0.05 = NON-private (see Remark 1); at T=30 eps 8/4/2/1 needs ~53.5/100.4/193.6/379.6
    "dp_delta": 1e-5,
    "projection": "gaussian",    # i.i.d. N(0, 1/m), as in Sec. III (Prop. 3 tail bounds)
    "filter_beta": 0.25,         # n_reject = floor(beta * N) (fixed upper bound on f, not of |V|)
    "fp_frac_bits": 16,          # phi
    "check_gamma": 0.01,
    "dropout_rate": 0.0,         # fraction of clients offline after the sketch phase
    "byz_sketch": "consistent",  # "benign": Byzantine clients sketch their benign update (inconsistent pair)
    "sign_flip_scale": 5.0,
    "scaling_factor": 12.0,
    "alie_z": None,              # None -> Baruch et al. formula
    "ipm_eps": 0.5,
    "minmax_dir": "unit_vec",
    "fltrust_root_size": 100,
    "rep_lambda": 0.8,
    "dp_eps_target": None,       # if set: dp_sigma is derived from (clip_s, rounds, dp_delta) to reach this eps_T           # reputation EMA factor for pq_rep / pq_grep / pq_root
    "torch_threads": 1,
}

GRIDS = {
    "main": dict(datasets=DATASETS, attacks=ATTACKS, variants=VARIANTS, seeds=SEEDS),
    "timing": dict(datasets=DATASETS, attacks=["none"], variants=["fedavg", "trimmed_mean", "krum", "multi_krum",
                                                                   "fltrust", "pq_no_filter", "pq_full"], seeds=[1, 2, 3]),
    "sigma": dict(datasets=DATASETS, attacks=["sign_flip", "minmax"], variants=["pq_full"], seeds=SEEDS,
                  sweep=("dp_sigma", [0.05, 5.0, 53.5, 100.4, 193.6, 379.6])),
    "alpha": dict(datasets=DATASETS, attacks=["minmax"], variants=["fedavg", "multi_krum", "fltrust", "pq_full"],
                  seeds=SEEDS, sweep=("alpha", [0.1, 0.3, 1.0])),
    # ---- 2026-11-23 improvements: privacy-robustness frontier and new filter variants ----
    # privacy-robustness frontier: tight update clip C and sigma derived from a target eps_T
    "privfront": dict(datasets=DATASETS, attacks=["sign_flip", "minmax"], variants=["pq_full", "pq_rep", "pq_grep", "pq_root"],
                      seeds=[1, 2, 3], sweep=("multi", [{"clip_C": c, "dp_eps_target": e}
                                                        for c in (1.0, 0.5) for e in (None, 32.0, 16.0, 8.0, 4.0)])),
    # reputation-window ablation (lambda = 0.95 vs the default 0.8 used in privfront)
    "replam": dict(datasets=DATASETS, attacks=["sign_flip", "minmax"], variants=["pq_grep", "pq_root"], seeds=[1, 2, 3],
                   sweep=("multi", [{"clip_C": 0.5, "dp_eps_target": e, "rep_lambda": 0.95} for e in (16.0, 8.0)])),
    # SecAgg + tight clipping only (no sketches, so no eps): does the filter add anything beyond clipping?
    "clipbase": dict(datasets=DATASETS, attacks=ATTACKS, variants=["pq_no_filter"], seeds=SEEDS,
                     sweep=("clip_C", [1.0, 0.5])),
    # ---- stage B (after stage A): C = 1.0 (about the median honest update norm on both datasets) ----
    "main2": dict(datasets=DATASETS, attacks=ATTACKS, variants=["pq_full", "pq_grep", "pq_root"], seeds=SEEDS,
                  sweep=("clip_C", [1.0])),
    "alpha2": dict(datasets=DATASETS, attacks=["minmax"], variants=["pq_no_filter", "pq_full", "pq_grep", "pq_root"],
                   seeds=SEEDS, sweep=("multi", [{"clip_C": 1.0, "alpha": a} for a in (0.1, 0.3, 1.0)])),
    "timing2": dict(datasets=DATASETS, attacks=["none"], variants=["pq_full", "pq_grep", "pq_root"], seeds=[1, 2, 3],
                    sweep=("clip_C", [1.0])),
    # same configs as the C = 1.0 half of privfront (same cfg hash): run with --seeds 4 5 into privfront.jsonl
    "privfront_c1": dict(datasets=DATASETS, attacks=["sign_flip", "minmax"], variants=["pq_full", "pq_grep", "pq_root"],
                         seeds=[4, 5], sweep=("multi", [{"clip_C": 1.0, "dp_eps_target": e}
                                                        for e in (None, 32.0, 16.0, 8.0, 4.0)])),
}


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def env_info() -> dict:
    import oqs
    import torch
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_gib = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem_gib = round(int(line.split()[1]) / 1024 ** 2, 2)
    except OSError:
        pass
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "liboqs": oqs.oqs_version(), "liboqs_python": getattr(oqs, "oqs_python_version", lambda: "?")(),
            "os": platform.platform(), "cpu": cpu, "nproc": os.cpu_count(), "mem_gib": mem_gib}


# ----------------------------------------------------------------------------
def run_one(job: dict) -> dict:
    import torch
    from src.exp import fl_core as F
    from src.fl.byzantine_filter import ReputationFilter, reject_top, sketch_scores
    from src.secagg.protocol import FixedPoint, PQSecureAggregator

    cfg, ds, attack, variant, seed = job["cfg"], job["dataset"], job["attack"], job["variant"], job["seed"]
    torch.set_num_threads(int(cfg["torch_threads"]))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    data = F.load_processed(ds, job["data_dir"], job.get("allow_synthetic", False))
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    n_cls = len(data["class_names"])
    N = cfg["clients"]
    f = int(N * cfg["byzantine_fraction"])

    # server root set for FLTrust is removed from the client pool for every variant (same partitions)
    root_idx = rng.choice(len(ytr), size=cfg["fltrust_root_size"], replace=False)
    pool = np.setdiff1d(np.arange(len(ytr)), root_idx)
    parts = [pool[p] for p in F.dirichlet_partition(ytr[pool], N, cfg["alpha"], seed)]
    byz = sorted(rng.choice(N, size=f, replace=False).tolist()) if f > 0 else []
    if attack == "none":
        byz_active = []
    else:
        byz_active = byz

    model = F.MLP(Xtr.shape[1], n_cls, *cfg["hidden"])
    theta = F.get_flat(model)
    d = theta.size
    gen = torch.Generator().manual_seed(seed)

    secagg = None
    if variant in PQ:
        secagg = PQSecureAggregator(N, cfg["degree"], graph_seed=seed,
                                    fp=FixedPoint(cfg["fp_frac_bits"], cfg["clip_C"]))

    hist = {"acc": [], "macro_f1": [], "round_ms": [], "train_ms": [], "agg_ms": [], "filter_ms": [],
            "byz_rejected": [], "honest_rejected": [], "honest_norm_q": [], "unmask_exact": [], "check": [], "aborted": []}
    cost = {k: 0.0 for k in ["client_kem_ms", "client_sign_ms", "client_verify_ms", "client_mask_ms",
                             "client_share_ms", "server_verify_ms", "server_unmask_ms", "server_check_ms",
                             "uplink_bytes", "sig_bytes", "kem_bytes"]}
    eps_round = None
    diverged = False
    lr_kw = dict(lr=cfg["local_lr"], batch=cfg["batch"], epochs=cfg["local_epochs"], max_steps=cfg["max_local_steps"])
    root_kw = dict(lr_kw, epochs=int(np.ceil(cfg["max_local_steps"] * cfg["batch"] / len(root_idx))) + 1)
    w_dir, ref_kind, use_rep = PQ.get(variant, (None, "median", False))
    C_s_cfg = cfg["clip_s"] or float(np.sqrt(2.0) * cfg["clip_C"])
    sigma = (F.sigma_for_eps(C_s_cfg, cfg["dp_eps_target"], cfg["rounds"], cfg["dp_delta"])
             if cfg.get("dp_eps_target") else cfg["dp_sigma"])
    repf = ReputationFilter(cfg["rep_lambda"]) if use_rep else None
    prev_agg = None  # last released global update Delta_{t-1} (public)

    for r in range(1, cfg["rounds"] + 1):
        t_r = time.perf_counter()
        benign, poisoned = {}, {}
        for i in range(N):
            Xi, yi = Xtr[parts[i]], ytr[parts[i]]
            benign[i] = F.local_update(model, theta, Xi, yi, gen=gen, **lr_kw)
            if i in byz_active and attack in ("scaling", "label_flip"):
                poisoned[i] = F.local_update(model, theta, Xi, (n_cls - 1 - yi).astype(np.int64), gen=gen, **lr_kw)
        train_ms = (time.perf_counter() - t_r) * 1e3
        sent = F.apply_attack(attack, benign, poisoned, byz_active, N, cfg)
        _hn = [float(np.linalg.norm(benign[i])) for i in range(N) if i not in byz_active]
        hist["honest_norm_q"].append([float(np.quantile(_hn, q)) for q in (0.1, 0.5, 0.9)])  # pre-clip update norms

        t_a = time.perf_counter()
        filt_ms = 0.0
        rejected: list[int] = []
        if variant in PLAIN:
            U = np.stack([sent[i] for i in range(N)], 0)
            kw = {"f": f}
            if variant == "fltrust":
                # FLTrust: the server runs the SAME number of local SGD steps as a client on its root set
                # (several passes over the 100 root samples), so its update has a comparable norm.
                kw["server_update"] = F.local_update(model, theta, Xtr[root_idx], ytr[root_idx], gen=gen, **root_kw)
            agg = F.PLAINTEXT_AGGS[variant](U, **kw).astype(np.float32)
        else:
            C = cfg["clip_C"]
            C_s = cfg["clip_s"] or float(np.sqrt(2.0) * C)
            gbar = {i: F.clip_norm(sent[i], C) for i in range(N)}
            gbar_benign = {i: F.clip_norm(benign[i], C) for i in byz_active}
            n_rej = int(np.floor(cfg["filter_beta"] * N))
            g_root = (F.local_update(model, theta, Xtr[root_idx], ytr[root_idx], gen=gen, **root_kw)
                      if ref_kind == "root" and w_dir is not None else None)
            holder = {}

            def proj_fn(pseed, _r=r):
                if w_dir is None:
                    return None
                holder["P"] = F.projection_matrix(cfg["sketch_dim"], d, pseed, cfg["projection"])
                return holder["P"]

            def sketch_fn(i, P):
                if P is None:
                    return np.zeros(0, np.float32)
                src = gbar_benign[i] if (cfg["byz_sketch"] == "benign" and i in gbar_benign) else gbar[i]
                return F.make_sketch(src, P, C_s, sigma, rng)

            def select(sk):
                nonlocal filt_ms
                if w_dir is None:
                    return sorted(sk), []
                t = time.perf_counter()
                ref = None  # None -> median sketch
                if ref_kind == "global" and prev_agg is not None and np.linalg.norm(prev_agg) > 0:
                    ref = holder["P"] @ prev_agg
                elif ref_kind == "root" and g_root is not None:
                    ref = holder["P"] @ g_root
                ids, scores = sketch_scores(sk, directional_weight=w_dir, reference=ref)
                if repf is not None:
                    scores = repf.update(ids, scores)
                a, rj = reject_top(ids, scores, n_rej) if ids else ([], [])
                filt_ms += (time.perf_counter() - t) * 1e3
                return a, rj

            if eps_round is None and w_dir is not None:
                eps_round = F.zcdp_eps(C_s, sigma, cfg["rounds"], cfg["dp_delta"])
                eps_round["sigma"] = sigma
            n_drop = int(round(cfg["dropout_rate"] * N))
            dropped = set(rng.choice(N, size=n_drop, replace=False).tolist()) if n_drop else set()
            agg, accepted, rejected, st = secagg.run_round(
                r, gbar, proj_fn, sketch_fn, select, rng, sigma=sigma,
                check_gamma=cfg["check_gamma"], dropped=dropped)
            for k in cost:
                cost[k] += getattr(st, k)
            hist["unmask_exact"].append(st.unmask_exact)
            hist["check"].append([st.check_value, st.check_threshold, st.check_passed])
            hist["aborted"].append(st.aborted)
            if agg is None:
                agg = np.zeros(d, np.float32)  # aborted round: no model update
            elif np.all(np.isfinite(agg)):
                prev_agg = agg.astype(np.float32)
        agg_ms = (time.perf_counter() - t_a) * 1e3

        theta = (theta + cfg["server_lr"] * agg).astype(np.float32)
        if not np.all(np.isfinite(theta)):
            diverged = True
        hist["round_ms"].append((time.perf_counter() - t_r) * 1e3)
        hist["train_ms"].append(train_ms)
        hist["agg_ms"].append(agg_ms)
        hist["filter_ms"].append(filt_ms)
        hist["byz_rejected"].append(len(set(rejected) & set(byz_active)))
        hist["honest_rejected"].append(len(set(rejected) - set(byz_active)))
        ev = F.evaluate(model, np.nan_to_num(theta), Xte, yte, n_cls) if not diverged else {"acc": float("nan"), "macro_f1": float("nan")}
        hist["acc"].append(ev["acc"])
        hist["macro_f1"].append(ev["macro_f1"])
        if diverged:
            break

    max_byz_nbrs = None
    if secagg is not None:
        bset = set(byz_active)
        max_byz_nbrs = max(len(set(secagg.graph[i]) & bset) for i in range(N))
        k_deg = secagg.degree
        secagg.close()
    else:
        k_deg = 0
    R = len(hist["round_ms"])
    return {
        "dataset": ds, "attack": attack, "variant": variant, "seed": seed,
        "code_version": CODE_VERSION,
        "cfg_hash": cfg_hash(cfg), "cfg": cfg,
        "source": data["source"], "source_sha256": data["source_sha256"],
        "n_features": int(Xtr.shape[1]), "n_classes": n_cls, "n_params": int(d),
        "train_rows": int(len(ytr)), "test_rows": int(len(yte)),
        "byzantine_ids": byz_active, "byzantine_count": len(byz_active), "k_degree": k_deg,
        "max_byz_neighbors": max_byz_nbrs,
        "shamir_threshold": (k_deg // 2 + 1) if k_deg else None,
        "client_sizes": [int(len(p)) for p in parts],
        "final_acc": hist["acc"][-1], "final_macro_f1": hist["macro_f1"][-1], "diverged": diverged,
        "history": hist,
        "avg_round_ms": float(np.mean(hist["round_ms"])),
        "avg_train_ms": float(np.mean(hist["train_ms"])),
        "avg_agg_ms": float(np.mean(hist["agg_ms"])),
        "avg_filter_ms": float(np.mean(hist["filter_ms"])),
        "crypto_per_round": {k: v / R for k, v in cost.items()},
        "per_client_uplink_kib_per_round": (cost["uplink_bytes"] + cost["kem_bytes"]) / R / N / 1024,
        "rounds_aborted": sum(1 for a in hist["aborted"] if a),
        "dp": eps_round,
        "total_s": time.perf_counter() - t_start,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "workers": job["workers"],
        "env": env_info(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def _worker(job: dict) -> dict:
    try:
        return run_one(job)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                **{k: job[k] for k in ("dataset", "attack", "variant", "seed")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", choices=list(GRIDS), default="main")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--attacks", nargs="*")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON", help="override any config key, e.g. dp_sigma=1.0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", help="JSONL output (default experiments/real/<grid>.jsonl)")
    ap.add_argument("--allow-synthetic", action="store_true", help="smoke tests only; results are marked SYNTHETIC")
    a = ap.parse_args(argv)

    g = GRIDS[a.grid]
    base = dict(DEFAULT_CFG)
    if a.rounds:
        base["rounds"] = a.rounds
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in base:
            ap.error(f"unknown config key {k}")
        base[k] = json.loads(v)
    sweep_key, sweep_vals = g.get("sweep", (None, [None]))
    out = Path(a.out or f"experiments/real/{a.grid}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done.add((r["dataset"], r["attack"], r["variant"], r["seed"], r["cfg_hash"]))

    jobs = []
    for sv in sweep_vals:
        cfg = dict(base)
        if isinstance(sv, dict):
            cfg.update(sv)
        elif sweep_key:
            cfg[sweep_key] = sv
        h = cfg_hash(cfg)
        for ds in a.datasets or g["datasets"]:
            for atk in a.attacks or g["attacks"]:
                for var in a.variants or g["variants"]:
                    for s in a.seeds or g["seeds"]:
                        if (ds, atk, var, s, h) in done:
                            continue
                        jobs.append({"cfg": cfg, "dataset": ds, "attack": atk, "variant": var, "seed": s,
                                     "data_dir": a.data_dir, "allow_synthetic": a.allow_synthetic,
                                     "workers": a.workers})
    print(f"[{a.grid}] {len(jobs)} runs to do ({len(done)} already done) -> {out}", flush=True)
    if not jobs:
        return 0
    t0 = time.time()
    n_err = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=a.workers, maxtasksperchild=1) as pool, open(out, "a") as fh:
        for k, res in enumerate(pool.imap_unordered(_worker, jobs), 1):
            fh.write(json.dumps(res, default=_jsonable) + "\n")
            fh.flush()
            if "error" in res:
                n_err += 1
                print(f"ERROR {res['dataset']} {res['attack']} {res['variant']} s{res['seed']}: {res['error']}", flush=True)
                continue
            el = time.time() - t0
            print(f"[{k}/{len(jobs)} {el/60:.1f}m eta {el/k*(len(jobs)-k)/60:.1f}m] {res['dataset']:12s} "
                  f"{res['attack']:9s} {res['variant']:13s} s{res['seed']} acc={res['final_acc']:.4f} "
                  f"f1={res['final_macro_f1']:.4f} round={res['avg_round_ms']:.0f}ms src={res['source']}", flush=True)
    print(f"finished, {n_err} errors")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
