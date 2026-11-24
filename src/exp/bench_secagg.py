"""
Protocol-only scalability benchmark (no training): cost of one secure-aggregation
round for sparse degree k = max(2, ceil(log2 N)) vs dense k = N-1, at the model
size d of the experiments. Supports the complexity table with measurements.

  python -m src.exp.bench_secagg --clients 20 50 100 200 --dim 17167 --reps 3
Writes experiments/real/bench_secagg.jsonl and paper/generated/tab_scaling.tex
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.exp import fl_core as F
from src.secagg.protocol import FixedPoint, PQSecureAggregator


def one(n, k, d, m, rep):
    rng = np.random.default_rng(rep)
    g = {i: F.clip_norm(rng.normal(0, 0.01, d).astype(np.float32), 5.0) for i in range(n)}
    agg = PQSecureAggregator(n, k, graph_seed=rep, fp=FixedPoint(16, 5.0))
    f = n // 4
    rej = set(rng.choice(n, f, replace=False).tolist())
    t = time.perf_counter()
    mean, A, R, st = agg.run_round(1, g, lambda s: F.projection_matrix(m, d, s), 
                                   lambda i, P: F.make_sketch(g[i], P, np.sqrt(2) * 5, 0.05, rng),
                                   lambda sk: ([i for i in sk if i not in rej], sorted(rej)), rng, sigma=0.05)
    wall = (time.perf_counter() - t) * 1e3
    deg = agg.degree
    agg.close()
    client_ms = (st.client_kem_ms + st.client_sign_ms + st.client_verify_ms + st.client_mask_ms + st.client_share_ms) / n
    server_ms = st.server_verify_ms + st.server_unmask_ms + st.server_check_ms
    return {"clients": n, "degree": deg, "dim": d, "rep": rep, "wall_ms": wall, "client_ms_avg": client_ms,
            "server_ms": server_ms, "kem_encaps": n * deg // 2, "uplink_kib_per_client": (st.uplink_bytes + st.kem_bytes) / n / 1024,
            "unmask_exact": st.unmask_exact, "check_passed": st.check_passed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", nargs="*", type=int, default=[20, 50, 100, 200])
    ap.add_argument("--dim", type=int, default=17167)
    ap.add_argument("--sketch-dim", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="experiments/real/bench_secagg.jsonl")
    ap.add_argument("--tex", default="paper/generated/tab_scaling.tex")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(a.out, "w") as fh:
        for n in a.clients:
            for mode, k in [("sparse", None), ("dense", n - 1)]:
                for rep in range(a.reps):
                    r = one(n, k, a.dim, a.sketch_dim, rep)
                    r["mode"] = mode
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    print(json.dumps(r), flush=True)
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{Measured cost of one secure-aggregation round ($d={a.dim}$, 25\\% rejected; mean over {a.reps} runs)}}",
             r"\label{tab:scaling}", r"{\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{rlrrrr}", r"\toprule",
             r"$N$ & Graph & $k$ & Client (ms) & Server (ms) & Uplink (KiB/client) \\", r"\midrule"]
    for n in a.clients:
        for mode in ["sparse", "dense"]:
            rs = [r for r in rows if r["clients"] == n and r["mode"] == mode]
            lines.append(f"{n} & {mode} & {rs[0]['degree']} & {np.mean([r['client_ms_avg'] for r in rs]):.1f} & "
                         f"{np.mean([r['server_ms'] for r in rs]):.1f} & {np.mean([r['uplink_kib_per_client'] for r in rs]):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tex).write_text("\n".join(lines) + "\n")
    print("wrote", a.tex)


if __name__ == "__main__":
    main()
