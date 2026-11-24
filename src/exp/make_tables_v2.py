"""
Tables and figures for the revised paper (2026-11-23), generated from the JSONL files only.

Inputs:  experiments/real/{main,timing,alpha}.jsonl        (published grids, C = 5)
         experiments/improve/{main2,clipbase,alpha2,privfront,leakage,bench_*,timing2}.jsonl
Output:  paper/generated/

Variant keys used in the tables (relabelled by configuration, never by hand):
  pq_full  (C = 5, from main)        -> PQ-Med        median-reference hybrid filter (original PQ-Full)
  clip1    (pq_no_filter, C = 1)     -> Clip-only     secure aggregation + clipping, no filter
  pq_full1 (pq_full, C = 1)          -> PQ-Med$_1$
  pq_root1 (pq_root, C = 1)          -> PQ-Root
  pq_grep1 (pq_grep, C = 1)          -> PQ-GRep       global-update reference + reputation (proposed default)

  python -m src.exp.make_tables_v2 [--real experiments/real] [--improve experiments/improve] [--out paper/generated]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from src.exp import make_tables as M

ND = NormalDist()
fmt, cell_stats = M.fmt, M.cell_stats
NEW_LABEL = {"pq_full": "PQ-Med", "clip1": "Clip-only", "pq_full1": r"PQ-Med$_1$", "pq_root1": "PQ-Root",
             "pq_grep1": r"\textbf{PQ-GRep}", "pq_no_filter": "PQ-NoFilt"}
M.VAR_LABEL.update(NEW_LABEL)
MAIN_V2 = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum", "fltrust", "pq_full", "clip1", "pq_grep1"]
FIG_V2 = MAIN_V2


def relabel(recs, cond, new):
    out = []
    for r in recs:
        if cond(r):
            r = dict(r)
            r["variant"] = new
            out.append(r)
    return out


def c_is(x):
    return lambda r: abs(r["cfg"]["clip_C"] - x) < 1e-9


def rates(r):
    h = r["history"]
    R = len(h["byz_rejected"])
    f = r["byzantine_count"]
    N = r["cfg"]["clients"]
    tpr = sum(h["byz_rejected"]) / (f * R) if f and R else float("nan")
    fpr = sum(h["honest_rejected"]) / ((N - f) * R) if R else float("nan")
    return tpr, fpr


def bound(r, fpr):
    rho1 = (r.get("dp") or {}).get("rho_1")
    if rho1 is None or not (0 < fpr < 1):
        return float("nan")
    R = len(r["history"]["byz_rejected"])
    z = ND.inv_cdf(fpr)
    return float(np.mean([ND.cdf(z + math.sqrt(2 * t * rho1)) for t in range(1, R + 1)]))


def filter_table(recs, datasets) -> str:
    """C = 1 filter design: Clip-only, PQ-Med_1, PQ-Root, PQ-GRep; F1 and TPR/FPR."""
    g = M.group(recs, ("dataset", "attack", "variant"))
    atks = ["sign_flip", "scaling", "alie", "minmax"]
    vs = ["clip1", "pq_full1", "pq_root1", "pq_grep1"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Filter design at $C=1$: final macro-F1 (mean$\pm$std, 5 seeds) and Byzantine TPR/FPR of the filter}",
             r"\label{tab:filter}", r"{\scriptsize\setlength{\tabcolsep}{2pt}\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{llccccccc}", r"\toprule",
             r" & & Clip-only & \multicolumn{2}{c}{PQ-Med$_1$} & \multicolumn{2}{c}{PQ-Root} & \multicolumn{2}{c}{\textbf{PQ-GRep}} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Dataset & Attack & F1 & F1 & TPR/FPR & F1 & TPR/FPR & F1 & TPR/FPR \\", r"\midrule"]
    for di, ds in enumerate(datasets):
        for atk in atks:
            row = []
            for v in vs:
                rs = g.get((ds, atk, v), [])
                m, s, _ = M.cell_stats([r["final_macro_f1"] for r in rs])
                row.append(M.fmt(m, s))
                if v != "clip1":
                    tf = [rates(r) for r in rs]
                    row.append(f"{np.mean([t for t, _ in tf]):.2f}/{np.mean([f for _, f in tf]):.2f}" if rs else "--")
            lines.append(f"{M.DS_LABEL[ds] if atk == atks[0] else ''} & {M.ATK_LABEL[atk]} & " + " & ".join(row) + r" \\")
        if di < len(datasets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def privfront_table(pf, clipbase, datasets) -> str:
    g = defaultdict(list)
    for r in pf:
        if abs(r["cfg"]["clip_C"] - 1.0) > 1e-9:
            continue
        g[(r["dataset"], r["attack"], r["variant"], r["cfg"]["dp_eps_target"])].append(r)
    cb = M.group([r for r in clipbase if abs(r["cfg"]["clip_C"] - 1.0) < 1e-9], ("dataset", "attack"))
    eps_list = [None, 32.0, 16.0, 8.0, 4.0]
    atks = ["sign_flip", "minmax"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Privacy--robustness frontier at $C=1$ ($T=30$, $\delta=10^{-5}$, 5 seeds): final macro-F1, TPR of PQ-GRep, "
             r"and the largest TPR any sketch-based filter can reach at the same FPR (Proposition~\ref{prop:detect})}",
             r"\label{tab:privfront}", r"{\scriptsize\setlength{\tabcolsep}{1.8pt}\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{llcccccccc}", r"\toprule",
             r" & & \multicolumn{4}{c}{Sign-flip} & \multicolumn{4}{c}{Min-Max} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r"Dataset & $\varepsilon_T$ & PQ-Med$_1$ & \textbf{PQ-GRep} & TPR & bound & PQ-Med$_1$ & \textbf{PQ-GRep} & TPR & bound \\",
             r"\midrule"]
    for di, ds in enumerate(datasets):
        row = []
        for a in atks:
            m, s, _ = M.cell_stats([r["final_macro_f1"] for r in cb.get((ds, a), [])])
            row += [r"\multicolumn{2}{c}{" + M.fmt(m, s) + "}", "--", "--"]
        lines.append(f"{M.DS_LABEL[ds]} & no sketch (Clip-only) & " + " & ".join(row) + r" \\")
        for e in eps_list:
            row = []
            for a in atks:
                med = g.get((ds, a, "pq_full", e), [])
                gr = g.get((ds, a, "pq_grep", e), [])
                m1, s1, _ = M.cell_stats([r["final_macro_f1"] for r in med])
                m2, s2, _ = M.cell_stats([r["final_macro_f1"] for r in gr])
                tf = [rates(r) for r in gr]
                tpr = np.mean([t for t, _ in tf]) if gr else float("nan")
                bd = np.mean([bound(r, f) for r, (_, f) in zip(gr, tf)]) if gr else float("nan")
                row += [M.fmt(m1, s1), M.fmt(m2, s2), f"{tpr:.2f}", "--" if not np.isfinite(bd) else f"{bd:.2f}"]
            lab = r"$\infty^{\ast}$" if e is None else f"{e:g}"
            lines.append(f" & {lab} & " + " & ".join(row) + r" \\")
        if di < len(datasets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\multicolumn{10}{l}{\scriptsize $^{\ast}$\,$\sigma_{\mathrm{dp}}=0.05$ ($\varepsilon_T\approx 5\times10^{4}$), i.e.\ no meaningful privacy.}",
              r"\end{tabular}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def leakage_table(leak, datasets, clip=5.0) -> str:
    g = defaultdict(list)
    for r in leak:
        if abs((r.get("clip_C") or 0) - clip) < 1e-9:
            g[(r["dataset"], r["observation"])].append(r)
    cols = [("raw", "Raw update"), ("sketch_sigma0.05", r"Sketch, $\sigma_{\mathrm{dp}}{=}0.05$"),
            ("sketch_eps32", r"$\varepsilon_T{=}32$"), ("sketch_eps8", r"$\varepsilon_T{=}8$"), ("sketch_eps1", r"$\varepsilon_T{=}1$")]
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{Dominant-class inference by an honest-but-curious server from one client's messages (accuracy over all clients, "
             f"5 seeds, 10 observed rounds, $C={clip:g}$); prior = always guess the most common class, uniform = $1/\\#$classes}}",
             r"\label{tab:leakage}", r"{\scriptsize\setlength{\tabcolsep}{2pt}\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * (len(cols) + 2) + "}", r"\toprule",
             "Dataset & " + " & ".join(c for _, c in cols) + r" & Prior & Uniform \\", r"\midrule"]
    for ds in datasets:
        vals = []
        for k, _ in cols:
            rs = g.get((ds, k), [])
            m, s, _ = M.cell_stats([r["accum_acc"] for r in rs])
            vals.append(M.fmt(m, s, digits=2))
        rs = g.get((ds, "raw"), [])
        pri = np.mean([r["prior_acc"] for r in rs]) if rs else float("nan")
        uni = np.mean([1 / r["n_proto_classes"] for r in rs]) if rs else float("nan")
        lines.append(f"{M.DS_LABEL[ds]} & " + " & ".join(vals) + f" & {pri:.2f} & {uni:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def scaling_table(bench: dict) -> str:
    """bench: {d: rows}. Server and client ms, uplink, for sparse and dense graphs."""
    dims = sorted(bench)
    Ns = sorted({r["clients"] for rows in bench.values() for r in rows})
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Measured cost of one secure-aggregation round for the two model sizes (25\% rejected; mean over 3 runs)}",
             r"\label{tab:scaling}", r"{\scriptsize\setlength{\tabcolsep}{2pt}\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{rlr" + "rrr" * len(dims) + "}", r"\toprule",
             r" & & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$d={d:,}$}}".replace(",", "{,}") for d in dims) + r" \\",
             "".join(f"\\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(dims))),
             r"$N$ & Graph & $k$ & " + " & ".join(["Client & Server & Uplink"] * len(dims)) + r" \\",
             r" & & & " + " & ".join(["(ms) & (ms) & (KiB)"] * len(dims)) + r" \\", r"\midrule"]
    for n in Ns:
        for mode in ["sparse", "dense"]:
            cells = []
            k = None
            for d in dims:
                rs = [r for r in bench[d] if r["clients"] == n and r["mode"] == mode]
                k = rs[0]["degree"]
                cells += [f"{np.mean([r['client_ms_avg'] for r in rs]):.1f}", f"{np.mean([r['server_ms'] for r in rs]):.1f}",
                          f"{np.mean([r['uplink_kib_per_client'] for r in rs]):.1f}"]
            lines.append(f"{n} & {mode} & {k} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def frontier_figure(pf, out: Path, datasets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6, "pdf.fonttype": 42})
    eps_list = [32.0, 16.0, 8.0, 4.0]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.5, 1.55), sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        for atk, mk, col in [("sign_flip", "o", "#1b7837"), ("minmax", "s", "#d95f02")]:
            tp, bd = [], []
            for e in eps_list:
                rs = [r for r in pf if r["dataset"] == ds and r["attack"] == atk and r["variant"] == "pq_grep"
                      and abs(r["cfg"]["clip_C"] - 1) < 1e-9 and r["cfg"]["dp_eps_target"] == e]
                tf = [rates(r) for r in rs]
                tp.append(np.mean([t for t, _ in tf]))
                bd.append(np.mean([bound(r, f) for r, (_, f) in zip(rs, tf)]))
            ax.plot(eps_list, tp, marker=mk, color=col, lw=1, ms=3, label=f"PQ-GRep TPR ({M.ATK_LABEL[atk]})")
            ax.plot(eps_list, bd, ls="--", color=col, lw=0.8, label=f"bound ({M.ATK_LABEL[atk]})")
        ax.axhline(0.25, color="k", lw=0.5, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(eps_list)
        ax.set_xticklabels([f"{e:g}" for e in eps_list])
        ax.set_xlabel(r"$\varepsilon_T$")
        ax.set_title(M.DS_LABEL[ds])
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("TPR")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.30, wspace=0.08)
    fig.savefig(out / "fig_privfront.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_privfront.png", dpi=300, bbox_inches="tight")


def _minmax_by_seed(rows, ds, var, alpha, C=None):
    out = {}
    for r in rows:
        if (r["dataset"] == ds and r["attack"] == "minmax" and r["variant"] == var and abs(r["cfg"]["alpha"] - alpha) < 1e-9
                and (C is None or abs(r["cfg"]["clip_C"] - C) < 1e-9) and not r.get("diverged")):
            out[r["seed"]] = r["final_macro_f1"]
    return out


def seeds10_table(plain, pq, datasets) -> tuple[str, list[str]]:
    """Min-Max, seeds 1-10 (stage D adds seeds 6-10 with the same effective configuration): paired t-test of
    PQ-GRep against FLTrust and FedAvg, with Holm correction over all rows."""
    from scipy import stats
    rows, warn = [], []
    for ds in datasets:
        for a in (0.1, 0.3, 0.5, 1.0):
            fa = _minmax_by_seed(plain, ds, "fedavg", a)
            fl = _minmax_by_seed(plain, ds, "fltrust", a)
            cl = _minmax_by_seed(pq, ds, "pq_no_filter", a, 1.0)
            gr = _minmax_by_seed(pq, ds, "pq_grep", a, 1.0)
            s = sorted(set(fa) & set(fl) & set(cl) & set(gr))
            if len(s) < 10:
                warn.append(f"seeds10 {ds} alpha={a}: only {len(s)} paired seeds")
                if len(s) < 3:
                    continue
            X = {k: np.array([d[i] for i in s]) for k, d in (("fa", fa), ("cl", cl), ("fl", fl), ("gr", gr))}
            p_fl = float(stats.ttest_rel(X["gr"], X["fl"]).pvalue)
            p_fa = float(stats.ttest_rel(X["gr"], X["fa"]).pvalue)
            rows.append(dict(ds=ds, a=a, n=len(s), X=X, p_fl=p_fl, p_fa=p_fa,
                             wins=int((X["gr"] > X["fl"]).sum())))
    # Holm-Bonferroni over the PQ-GRep vs FLTrust tests
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_fl"])
    m, alive = len(rows), True
    for rank, i in enumerate(order):
        alive = alive and rows[i]["p_fl"] <= 0.05 / (m - rank)
        rows[i]["holm"] = alive
    n = rows[0]["n"] if rows else 0
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{Min-Max over {n} seeds (paired by seed; $C=1$ for Clip-only and PQ-GRep): final macro-F1, "
             r"difference to FLTrust, paired $t$-test $p$ against FLTrust ($^{\dagger}$: significant after Holm correction over all rows) "
             f"and seeds on which PQ-GRep beats FLTrust. Against FedAvg, PQ-GRep has $p < {_ceil_pow10(max(r['p_fa'] for r in rows))}$ in every row}}",
             r"\label{tab:seeds10}", r"{\scriptsize\setlength{\tabcolsep}{2pt}\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{llccccccc}", r"\toprule",
             r"Dataset & $\alpha$ & FedAvg & Clip-only & FLTrust & \textbf{PQ-GRep} & $\Delta_{\mathrm{FLTrust}}$ & $p$ & wins \\",
             r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["ds"] != prev:
            lines.append(r"\midrule")
        name = M.DS_LABEL[r["ds"]] if r["ds"] != prev else ""
        prev = r["ds"]
        X = r["X"]
        c = lambda v: fmt(float(v.mean()), float(v.std(ddof=1)))
        pstr = f"{r['p_fl']:.3f}" + (r"$^{\dagger}$" if r["holm"] else "")
        lines.append(f"{name} & {r['a']:g} & {c(X['fa'])} & {c(X['cl'])} & {c(X['fl'])} & {c(X['gr'])} & "
                     f"{float(X['gr'].mean() - X['fl'].mean()):+.3f} & {pstr} & {r['wins']}/{r['n']}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}}", r"\end{table}"]
    return "\n".join(lines) + "\n", warn


def _ceil_pow10(x: float) -> str:
    e = int(math.ceil(math.log10(x))) if x > 0 else -12
    return f"10^{{{e}}}"


def extra_attacks_table(rows_all, datasets) -> str:
    """Stage D: IPM, label-flip and ALIE with z = 1.5 (5 seeds)."""
    cols = [("fedavg", None, "FedAvg"), ("multi_krum", None, "M-Krum"), ("fltrust", None, "FLTrust"),
            ("pq_no_filter", 1.0, "Clip-only"), ("pq_root", 1.0, "PQ-Root"), ("pq_grep", 1.0, r"\textbf{PQ-GRep}")]
    atks = [("ipm", "IPM"), ("label_flip", "Label-flip"), ("alie", r"ALIE, $z{=}1.5$")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Additional attacks (5 seeds, $\alpha=0.5$): final macro-F1 and the Byzantine TPR/FPR of PQ-GRep. "
             r"IPM sends $-0.5$ times the benign mean; label-flip sends the update trained on flipped labels}",
             r"\label{tab:extraatk}", r"{\scriptsize\setlength{\tabcolsep}{2pt}\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{ll" + "c" * (len(cols) + 1) + "}", r"\toprule",
             "Dataset & Attack & " + " & ".join(c[2] for c in cols) + r" & TPR/FPR \\", r"\midrule"]
    for di, ds in enumerate(datasets):
        for ai, (atk, alab) in enumerate(atks):
            cells, means = [], []
            for var, C, _ in cols:
                rs = [r for r in rows_all if r["dataset"] == ds and r["attack"] == atk and r["variant"] == var
                      and (C is None or abs(r["cfg"]["clip_C"] - C) < 1e-9)
                      and ((r["cfg"].get("alie_z") == 1.5) if atk == "alie" else True)]
                m, sd, n = cell_stats([np.nan if r.get("diverged") else r["final_macro_f1"] for r in rs])
                cells.append((m, sd)); means.append(m)
            best = np.nanmax(means)
            txt = [fmt(m, sd, bold=abs(m - best) < 5e-4) for m, sd in cells]
            gr = [r for r in rows_all if r["dataset"] == ds and r["attack"] == atk and r["variant"] == "pq_grep"
                  and ((r["cfg"].get("alie_z") == 1.5) if atk == "alie" else True)]
            tf = [rates(r) for r in gr]
            tpr = f"{np.mean([t for t, _ in tf]):.2f}/{np.mean([f for _, f in tf]):.2f}"
            lines.append(f"{M.DS_LABEL[ds] if ai == 0 else ''} & {alab} & " + " & ".join(txt) + f" & {tpr}" + r" \\")
        if di < len(datasets) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def ablation_on(ds, fn):
    """The C = 5 score ablation (pq_l2 / pq_cos) exists only for the datasets of the published grid."""
    keep, M.DATASETS = M.DATASETS, ds
    try:
        return fn()
    finally:
        M.DATASETS = keep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="experiments/real")
    ap.add_argument("--improve", default="experiments/improve")
    ap.add_argument("--out", default="paper/generated")
    ap.add_argument("--extra", default="experiments/extraD", help="stage D results (seeds 6-10, extra attacks)")
    a = ap.parse_args(argv)
    real, imp, out = Path(a.real), Path(a.improve), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = lambda p: M.load(p, False)
    main_r, timing, alpha = L(real / "main.jsonl"), L(real / "timing.jsonl"), L(real / "alpha.jsonl")
    main2, clipbase, alpha2 = L(imp / "main2.jsonl"), L(imp / "clipbase.jsonl"), L(imp / "alpha2.jsonl")
    pf = L(imp / "privfront.jsonl")
    leak = [json.loads(x) for x in (imp / "leakage.jsonl").read_text().splitlines() if x.strip()]
    ton_main = L(imp / "ton_main.jsonl") if (imp / "ton_main.jsonl").exists() else []
    main_r = main_r + ton_main
    datasets = [d for d in ["edge_iiotset", "ton_iot", "unsw_nb15"] if any(r["dataset"] == d for r in main2)]
    M.DATASETS = datasets

    v2 = [r for r in main_r if r["variant"] in ("fedavg", "median", "trimmed_mean", "krum", "multi_krum", "fltrust",
                                                 "pq_no_filter", "pq_l2", "pq_cos", "pq_full")]
    v2 += relabel(clipbase, c_is(1.0), "clip1")
    v2 += relabel([r for r in main2 if r["variant"] == "pq_grep"], c_is(1.0), "pq_grep1")
    v2 += relabel([r for r in main2 if r["variant"] == "pq_root"], c_is(1.0), "pq_root1")
    v2 += relabel([r for r in main2 if r["variant"] == "pq_full"], c_is(1.0), "pq_full1")
    warn = M.check_cells(M.group(v2, ("dataset", "attack", "variant")), 5, "v2")

    attacks = [x for x in M.ATTACKS if any(r["attack"] == x for r in v2)]
    c0 = main_r[0]["cfg"]
    cap = lambda m: (f"Final {m} on the held-out test set, mean$\\pm$std over 5 seeds ($N={c0['clients']}$, $T={c0['rounds']}$, "
                     f"{int(c0['byzantine_fraction'] * 100)}\\% Byzantine, Dirichlet $\\alpha={c0['alpha']}$). PQ-Med uses $C=5$; "
                     f"Clip-only and PQ-GRep use $C=1$")
    files = {
        "tab_macro_f1.tex": M.metric_table(v2, "final_macro_f1", cap("macro-F1"), "tab:macrof1", MAIN_V2, attacks),
        "tab_accuracy.tex": M.metric_table(v2, "final_acc", cap("accuracy"), "tab:accuracy", MAIN_V2, attacks),
        "tab_ablation.tex": ablation_on([d for d in datasets if any(r["dataset"] == d and r["variant"] == "pq_l2" for r in v2)],
                                        lambda: M.ablation_table(v2, "final_macro_f1")).replace(r"\textbf{Hybrid}", r"Hybrid (PQ-Med)")
                             .replace("Filter ablation:", r"Score ablation at $C=5$ (median reference):"),
        "tab_filter.tex": filter_table(v2, datasets),
        "tab_privfront.tex": privfront_table(pf, clipbase, datasets),
        "tab_leakage.tex": leakage_table(leak, [d for d in datasets if any(x.get("dataset") == d for x in leak)]),
        "results_macros.tex": M.macros([r for r in main_r if r["dataset"] in datasets]),
    }
    # non-IID: published FedAvg / M-Krum / FLTrust / PQ-Med (C = 5) + Clip-only and PQ-GRep (C = 1)
    ton_alpha = L(imp / "ton_alpha.jsonl") if (imp / "ton_alpha.jsonl").exists() else []
    al = [r for r in alpha + ton_alpha if r["variant"] in ("fedavg", "multi_krum", "fltrust", "pq_full")]
    al += relabel([r for r in alpha2 if r["variant"] == "pq_no_filter"], c_is(1.0), "clip1")
    al += relabel([r for r in alpha2 if r["variant"] == "pq_grep"], c_is(1.0), "pq_grep1")
    M.MAIN_VARS = ["fedavg", "multi_krum", "fltrust", "pq_full", "clip1", "pq_grep1"]
    M.ABL_VARS = []
    files["tab_alpha.tex"] = M.sweep_table(al, "alpha", "tab:alpha",
                                           r"Non-IID strength (Dirichlet $\alpha$) under Min-Max (25\% Byzantine): final macro-F1, 5 seeds",
                                           "final_macro_f1")
    xd = Path(a.extra) if a.extra else None
    if xd and xd.exists():
        s10 = L(xd / "seeds10.jsonl") + L(xd / "seeds10_alpha.jsonl")
        plain = [r for r in main_r + alpha + s10 if r["variant"] in ("fedavg", "fltrust")]
        pqr = [r for r in main2 + alpha2 + clipbase + s10 if r["variant"] in ("pq_grep", "pq_no_filter")]
        d10 = [d for d in datasets if any(r["dataset"] == d for r in s10)]
        files["tab_seeds10.tex"], w10 = seeds10_table(plain, pqr, d10)
        warn += w10
        ex = L(xd / "attacks.jsonl") + L(xd / "alie15.jsonl")
        files["tab_extraatk.tex"] = extra_attacks_table(ex, [d for d in datasets if any(r["dataset"] == d for r in ex)])
    bench = {}
    for name in ["bench_edge.jsonl", "bench_unsw.jsonl"]:
        rows = [json.loads(x) for x in (imp / name).read_text().splitlines() if x.strip()]
        bench[rows[0]["dim"]] = rows
    files["tab_scaling.tex"] = scaling_table(bench)
    same = L(imp / "timing_same.jsonl") if (imp / "timing_same.jsonl").exists() else []
    if same:  # one VM session: FedAvg, FLTrust and PQ-Med measured back to back (Edge-IIoTset + UNSW-NB15)
        files["tab_overhead.tex"] = M.overhead_table(
            same, "same-session timing run, no attack, 3 seeds per dataset on Edge-IIoTset and UNSW-NB15, 1 worker, $C=5$")
    else:
        files["tab_overhead.tex"] = M.overhead_table(timing, "timing grid, 1 worker, $C=5$")

    for name, body in files.items():
        (out / name).write_text(body)
    M.MAIN_VARS = FIG_V2
    import inspect
    fsrc = inspect.getsource(M.figure).replace(
        '"pq_no_filter": "#9467bd", "pq_full": "#1b7837"}',
        '"pq_no_filter": "#9467bd", "pq_full": "#9467bd", "clip1": "#e7ba52", "pq_grep1": "#1b7837"}').replace(
        'ax.legend(ncol=4,', 'ax.legend(ncol=5,')
    ns = dict(M.__dict__)
    exec(fsrc, ns)
    ns["figure"](v2, same or timing, out, "final_macro_f1")
    frontier_figure(pf, out, datasets)
    prov = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(list(real.glob("*.jsonl")) + list(imp.glob("*.jsonl")))},
            "warnings": warn, "datasets": datasets,
            "note": "generated by src/exp/make_tables_v2.py; no number typed by hand"}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    for w in warn:
        print("WARNING", w, file=sys.stderr)
    print(f"wrote {len(files)} tables + 2 figures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
