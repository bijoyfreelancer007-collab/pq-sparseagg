"""
Empirical leakage of the per-client sketches: a dominant-label inference attack.

Threat: an honest-but-curious server wants to learn which traffic/attack class dominates a client's
local data (e.g. "site i is being hit by DDoS"). It holds a small auxiliary labelled set (same
assumption as FLTrust's root set) and, every round, computes one "class prototype" update g_c per class
on its auxiliary samples of class c from the current global model. It then scores each client by
cos(observation_i, prototype_c) and guesses argmax_c.

Observations compared (same clients, same rounds, same model trajectory):
  raw      : the clipped plaintext update g_i            (what a server sees WITHOUT secure aggregation)
  sketch   : clip(P g_i, C_s) + N(0, sigma^2 I), P fresh each round and known to the server; prototypes
             are sketched with the same P but without noise. sigma is derived from the target eps_T.
  secagg   : the server only sees the sum, so per-client inference cannot beat the prior; we report
             the prior (always guess the most common dominant class) as this reference.
Two attackers: "single" (one round) and "accum" (scores summed over all observed rounds; this is the
strongest attacker and is what eps_T bounds).

The FL trajectory is honest FedAvg over clipped updates (no Byzantine clients), so the attack sees
realistic updates. Output: one JSON line per (dataset, seed, observation), plus a printed summary.

  python -m src.exp.leakage_attack --datasets edge_iiotset unsw_nb15 --seeds 1 2 3 4 5 --rounds 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def run(ds: str, seed: int, a) -> list[dict]:
    import torch
    from src.exp import fl_core as F

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = F.load_processed(ds, a.data_dir, a.allow_synthetic)
    Xtr, ytr = data["X_train"], data["y_train"]
    n_cls = len(data["class_names"])
    N = a.clients
    aux_idx = rng.choice(len(ytr), size=a.aux_size, replace=False)
    pool = np.setdiff1d(np.arange(len(ytr)), aux_idx)
    parts = [pool[p] for p in F.dirichlet_partition(ytr[pool], N, a.alpha, seed)]
    dom = [int(np.bincount(ytr[p], minlength=n_cls).argmax()) for p in parts]
    prior_cls, prior_cnt = Counter(dom).most_common(1)[0]

    aux_by_c = {c: aux_idx[ytr[aux_idx] == c] for c in range(n_cls)}
    classes = [c for c in range(n_cls) if len(aux_by_c[c]) >= a.min_aux_per_class]

    model = F.MLP(Xtr.shape[1], n_cls, *a.hidden)
    theta = F.get_flat(model)
    d = theta.size
    gen = torch.Generator().manual_seed(seed)
    lr_kw = dict(lr=a.lr, batch=a.batch, epochs=1, max_steps=a.max_steps)
    C = a.clip_C
    C_s = math.sqrt(2.0) * C
    sigmas = {("eps", e): F.sigma_for_eps(C_s, e, a.account_rounds, a.delta) for e in a.eps}
    sigmas[("sigma", 0.05)] = 0.05  # the non-private main-grid setting

    obs_names = ["raw"] + [f"sketch_eps{k[1]:g}" if k[0] == "eps" else "sketch_sigma0.05" for k in sigmas]
    single_hits = defaultdict(int)
    single_tot = defaultdict(int)
    accum = {o: np.zeros((N, len(classes))) for o in obs_names}

    for r in range(1, a.rounds + 1):
        upd = [F.clip_norm(F.local_update(model, theta, Xtr[p], ytr[p], gen=gen, **lr_kw), C) for p in parts]
        proto = []
        for c in classes:
            ac = aux_by_c[c]
            proto.append(F.local_update(model, theta, Xtr[ac], ytr[ac], gen=gen,
                                        **dict(lr_kw, epochs=int(np.ceil(a.max_steps * a.batch / len(ac))) + 1)))
        proto = np.stack(proto, 0)
        P = F.projection_matrix(a.sketch_dim, d, int(rng.integers(2 ** 31)), "gaussian")
        Pproto = proto @ P.T  # (n_proto, m)

        def cos_rows(v, M):
            return (M @ v) / (np.linalg.norm(M, axis=1) * np.linalg.norm(v) + 1e-12)

        for i in range(N):
            ob = {"raw": cos_rows(upd[i], proto)}
            base = F.clip_norm(P @ upd[i], C_s)
            for k, sg in sigmas.items():
                name = f"sketch_eps{k[1]:g}" if k[0] == "eps" else "sketch_sigma0.05"
                s = base + rng.normal(0, sg, a.sketch_dim).astype(np.float32)
                ob[name] = cos_rows(s, Pproto)
            for o, sc in ob.items():
                accum[o][i] += sc
                single_tot[o] += 1
                single_hits[o] += int(classes[int(np.argmax(sc))] == dom[i])
        theta = (theta + np.mean(upd, 0)).astype(np.float32)

    out = []
    for o in obs_names:
        acc_hits = sum(int(classes[int(np.argmax(accum[o][i]))] == dom[i]) for i in range(N))
        sg = None
        for k, v in sigmas.items():
            if (k[0] == "eps" and o == f"sketch_eps{k[1]:g}") or (k[0] == "sigma" and o == "sketch_sigma0.05"):
                sg = v
        out.append({"dataset": ds, "seed": seed, "observation": o, "sigma": sg, "clip_C": C,
                    "eps_T": (float(o.split("eps")[1]) if "eps" in o else None),
                    "single_round_acc": single_hits[o] / single_tot[o], "accum_acc": acc_hits / N,
                    "prior_acc": prior_cnt / N, "n_clients": N, "rounds": a.rounds,
                    "account_rounds": a.account_rounds, "n_proto_classes": len(classes), "n_classes": n_cls,
                    "dominant_classes": dom, "source": data["source"], "d": int(d), "sketch_dim": a.sketch_dim})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["edge_iiotset", "unsw_nb15"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--rounds", type=int, default=10, help="rounds the attacker observes")
    ap.add_argument("--account-rounds", type=int, default=30, help="T used for the eps_T accounting")
    ap.add_argument("--eps", nargs="*", type=float, default=[32.0, 16.0, 8.0, 4.0, 1.0])
    ap.add_argument("--clip-C", type=float, default=5.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--aux-size", type=int, default=2000)
    ap.add_argument("--min-aux-per-class", type=int, default=8)
    ap.add_argument("--sketch-dim", type=int, default=128)
    ap.add_argument("--hidden", nargs=2, type=int, default=[128, 64])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="experiments/improve/leakage.jsonl")
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args(argv)
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    with open(outp, "a") as fh:
        for ds in a.datasets:
            for s in a.seeds:
                res = run(ds, s, a)
                for r in res:
                    r["clip_C"] = a.clip_C
                    fh.write(json.dumps(r) + "\n")
                rows += res
                print(f"{ds} seed {s} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"\nDominant-label inference accuracy (mean over seeds), C={a.clip_C}")
    print(f"{'dataset':14s} {'observation':18s} {'single':>7s} {'accum':>7s} {'prior':>7s} {'uniform':>7s}")
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["observation"])].append(r)
    for (ds, o), rs in g.items():
        print(f"{ds:14s} {o:18s} {np.mean([r['single_round_acc'] for r in rs]):7.3f} "
              f"{np.mean([r['accum_acc'] for r in rs]):7.3f} {np.mean([r['prior_acc'] for r in rs]):7.3f} "
              f"{np.mean([1.0 / r['n_proto_classes'] for r in rs]):7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
