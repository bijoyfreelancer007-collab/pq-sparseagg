"""
Generate every results table, the evaluation figure and a LaTeX macro file
directly from the experiment JSONL files. No number is typed by hand.

  python -m src.exp.make_tables                 # reads experiments/real/*.jsonl
  python -m src.exp.make_tables --metric acc    # accuracy instead of macro-F1 in the figure

Outputs (paper/generated/):
  tab_accuracy.tex  tab_macro_f1.tex  tab_ablation.tex  tab_overhead.tex
  tab_dp_sigma.tex (if sigma grid ran)  tab_alpha.tex (if alpha grid ran)
  results_macros.tex  fig_pq_sparseagg_eval.pdf/.png  summary.csv  provenance.json
In the paper:  \\input{generated/results_macros.tex}  and  \\input{generated/tab_accuracy.tex}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DS_LABEL = {"edge_iiotset": "Edge-IIoTset", "ton_iot": "ToN-IoT", "unsw_nb15": "UNSW-NB15"}
ATK_LABEL = {"none": "No attack", "sign_flip": "Sign-flip", "scaling": "Scaling", "alie": "ALIE",
             "ipm": "IPM", "minmax": "Min-Max", "label_flip": "Label-flip"}
VAR_LABEL = {"fedavg": "FedAvg", "median": "Median", "trimmed_mean": "TrMean", "krum": "Krum",
             "multi_krum": "M-Krum", "fltrust": "FLTrust", "pq_no_filter": "PQ-NoFilt",
             "pq_l2": r"PQ-$\ell_2$", "pq_cos": "PQ-Cos", "pq_full": r"\textbf{PQ-Full}"}
MAIN_VARS = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum", "fltrust", "pq_no_filter", "pq_full"]
ABL_VARS = ["pq_no_filter", "pq_l2", "pq_cos", "pq_full"]
DATASETS = ["edge_iiotset", "ton_iot", "unsw_nb15"]
ATTACKS = ["none", "sign_flip", "scaling", "alie", "minmax"]


def load(path: Path, allow_synth: bool) -> list[dict]:
    if not path.exists():
        return []
    out, errs = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            errs += 1
            continue
        if r["variant"] == "fltrust" and r.get("code_version", "") < "2026-11-23.fltrust-v2":
            sys.exit(f"{path}: contains FLTrust runs from before the 2026-11-23 fix (server update too short). "
                     "Remove them and rerun (see EXPERIMENTS_REAL_bn.md).")
        if r["source"] == "SYNTHETIC" and not allow_synth:
            sys.exit(f"{path}: contains SYNTHETIC runs. Refusing (use --allow-synthetic only for smoke tests).")
        out.append(r)
    if errs:
        print(f"warning: {path.name}: {errs} failed runs ignored", file=sys.stderr)
    return out


def cell_stats(vals: list[float]) -> tuple[float, float, int]:
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    n_nan = len(vals) - len(v)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v) + n_nan * 0


def fmt(m: float, s: float, bold: bool = False, digits: int = 3) -> str:
    if not np.isfinite(m):
        return "--"
    body = f"{m:.{digits}f}{{\\scriptsize$\\pm${s:.{digits}f}}}"
    return f"\\textbf{{{m:.{digits}f}}}{{\\scriptsize$\\pm${s:.{digits}f}}}" if bold else body


def group(recs: list[dict], keys: tuple[str, ...]) -> dict:
    g = defaultdict(list)
    for r in recs:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def check_cells(g: dict, expected_seeds: int, label: str) -> list[str]:
    msgs = []
    for k, rs in g.items():
        seeds = sorted({r["seed"] for r in rs})
        hashes = {r["cfg_hash"] for r in rs}
        if len(seeds) != expected_seeds:
            msgs.append(f"{label} {k}: {len(seeds)} seeds {seeds} (expected {expected_seeds})")
        if len(hashes) > 1:
            msgs.append(f"{label} {k}: mixed configs {hashes}")
        if len(rs) != len(seeds):
            msgs.append(f"{label} {k}: duplicate seeds")
    return msgs


def metric_table(recs, metric, caption, label, variants, attacks) -> str:
    g = group(recs, ("dataset", "attack", "variant"))
    cols = "ll" + "c" * len(variants)
    lines = [r"\begin{table*}[t]", r"\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"{\footnotesize\setlength{\tabcolsep}{2.6pt}", f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             "Dataset & Attack & " + " & ".join(VAR_LABEL[v] for v in variants) + r" \\", r"\midrule"]
    for di, ds in enumerate(DATASETS):
        for atk in attacks:
            stats = [cell_stats([r[metric] for r in g.get((ds, atk, v), [])]) for v in variants]
            ndiv = [sum(bool(r["diverged"]) for r in g.get((ds, atk, v), [])) for v in variants]
            means = [m for m, _, _ in stats]
            best = np.nanmax(means) if np.any(np.isfinite(means)) else np.nan
            cells = []
            for (m, s, n), nd in zip(stats, ndiv):
                if nd and not np.isfinite(m):
                    cells.append(r"div.")           # every seed diverged (non-finite weights)
                else:
                    c = fmt(m, s, bold=np.isfinite(m) and round(m, 3) == round(best, 3))
                    cells.append(c + (f"$^{{\\dagger{nd}}}$" if nd else ""))
            
            lines.append(f"{DS_LABEL[ds] if atk == attacks[0] else ''} & {ATK_LABEL[atk]} & " + " & ".join(cells) + r" \\")
        if di < len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def ablation_table(recs, metric) -> str:
    g = group(recs, ("dataset", "attack", "variant"))
    atks = [a for a in ATTACKS if a != "none"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Filter ablation: final macro-F1 (mean$\pm$std over seeds) and Byzantine rejection rate of the hybrid filter (25\% Byzantine, $N=20$)}"
             if metric == "final_macro_f1" else r"\caption{Filter ablation: final accuracy (mean$\pm$std over seeds)}",
             r"\label{tab:ablation}", r"{\scriptsize\setlength{\tabcolsep}{2pt}\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccccc}", r"\toprule",
             r"Dataset & Attack & NoFilter & $\ell_2$-only & Cos-only & \textbf{Hybrid} & TPR / FPR \\", r"\midrule"]
    for di, ds in enumerate(DATASETS):
        for atk in atks:
            row = []
            for v in ABL_VARS:
                m, s, _ = cell_stats([r[metric] for r in g.get((ds, atk, v), [])])
                row.append(fmt(m, s, digits=3))
            full = g.get((ds, atk, "pq_full"), [])
            if full:
                tp = np.mean([sum(r["history"]["byz_rejected"]) / max(1, r["byzantine_count"] * len(r["history"]["byz_rejected"])) for r in full])
                n_h = [r["cfg"]["clients"] - r["byzantine_count"] for r in full]
                fp = np.mean([sum(r["history"]["honest_rejected"]) / (nh * len(r["history"]["honest_rejected"])) for r, nh in zip(full, n_h)])
                row.append(f"{tp:.2f} / {fp:.2f}")
            else:
                row.append("--")
            lines.append(f"{DS_LABEL[ds] if atk == atks[0] else ''} & {ATK_LABEL[atk]} & " + " & ".join(row) + r" \\")
        if di < len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def overhead_table(recs, source_note) -> str:
    g = group(recs, ("variant",))
    order = [v for v in ["fedavg", "trimmed_mean", "krum", "multi_krum", "fltrust", "pq_no_filter", "pq_full"] if (v,) in g]
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{Per-round cost, mean$\\pm$std over datasets and seeds ({source_note})}}",
             r"\label{tab:overhead}", r"{\scriptsize\setlength{\tabcolsep}{2pt}\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Variant & Round (ms) & Crypto (ms) & Filter (ms) & Uplink (KiB/client) & Peak RSS (MiB) \\", r"\midrule"]
    for v in order:
        rs = g[(v,)]
        rnd = cell_stats([r["avg_round_ms"] for r in rs])
        if v.startswith("pq"):
            cr = cell_stats([sum(val for k, val in r["crypto_per_round"].items() if k.endswith("_ms")) for r in rs])
            up = cell_stats([r["per_client_uplink_kib_per_round"] for r in rs])
        else:
            cr = (0.0, 0.0, len(rs))
            up = cell_stats([r["n_params"] * 4 / 1024 for r in rs])
        fl = cell_stats([r["avg_filter_ms"] for r in rs])
        mem = cell_stats([r["peak_rss_mb"] for r in rs])
        name = VAR_LABEL[v].replace(r"\textbf{", "").replace("}", "")
        lines.append(f"{name} & {fmt(rnd[0], rnd[1], digits=1)} & {fmt(cr[0], cr[1], digits=1)} & "
                     f"{fmt(fl[0], fl[1], digits=1)} & {fmt(up[0], up[1], digits=1)} & {fmt(mem[0], mem[1], digits=0)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def sweep_table(recs, key, label, caption, metric) -> str:
    g = group(recs, ("dataset", "attack", "variant"))
    vals = sorted({r["cfg"][key] for r in recs})
    atks = sorted({r["attack"] for r in recs}, key=lambda a: ATTACKS.index(a) if a in ATTACKS else 9)
    variants = [v for v in MAIN_VARS + ABL_VARS if any(r["variant"] == v for r in recs)]
    variants = list(dict.fromkeys(variants))
    head = key.replace("dp_sigma", r"$\sigma$").replace("alpha", r"$\alpha$")
    extra = r" & $\varepsilon_T$" if key == "dp_sigma" else ""
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"{\scriptsize\setlength{\tabcolsep}{2pt}",
             f"\\begin{{tabular}}{{ll{'c' if extra else ''}{'c' * len(variants) * len(atks)}}}", r"\toprule",
             f"Dataset & {head}{extra} & " + " & ".join(f"{VAR_LABEL[v]} ({ATK_LABEL[a]})" for a in atks for v in variants) + r" \\",
             r"\midrule"]
    for ds in DATASETS:
        for x in vals:
            row = []
            eps = ""
            for a in atks:
                for v in variants:
                    rs = [r for r in g.get((ds, a, v), []) if r["cfg"][key] == x]
                    m, s, _ = cell_stats([r[metric] for r in rs])
                    row.append(fmt(m, s))
                    if key == "dp_sigma" and rs and rs[0].get("dp") and not eps:
                        eps = f" & {rs[0]['dp']['eps_T']:.3g}"
            lines.append(f"{DS_LABEL[ds] if x == vals[0] else ''} & {x:g}{eps} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def macros(main: list[dict]) -> str:
    r0 = main[0]
    c = r0["cfg"]
    by_ds = {r["dataset"]: r for r in main}
    seeds = sorted({r["seed"] for r in main})
    env = r0["env"]
    out = ["% generated by src/exp/make_tables.py -- do not edit by hand",
           f"\\newcommand{{\\expClients}}{{{c['clients']}}}",
           f"\\newcommand{{\\expRounds}}{{{c['rounds']}}}",
           f"\\newcommand{{\\expAlpha}}{{{c['alpha']}}}",
           f"\\newcommand{{\\expByzFrac}}{{{int(round(c['byzantine_fraction'] * 100))}\\%}}",
           f"\\newcommand{{\\expByzCount}}{{{int(c['clients'] * c['byzantine_fraction'])}}}",
           f"\\newcommand{{\\expSeeds}}{{{len(seeds)}}}",
           f"\\newcommand{{\\expLocalSteps}}{{{c['max_local_steps']}}}",
           f"\\newcommand{{\\expLocalLR}}{{{c['local_lr']}}}",
           f"\\newcommand{{\\expBatch}}{{{c['batch']}}}",
           f"\\newcommand{{\\expSketchDim}}{{{c['sketch_dim']}}}",
           f"\\newcommand{{\\expClipC}}{{{c['clip_C']}}}",
           f"\\newcommand{{\\expSigma}}{{{c['dp_sigma']}}}",
           f"\\newcommand{{\\expBeta}}{{{c['filter_beta']}}}",
           f"\\newcommand{{\\expPhi}}{{{c['fp_frac_bits']}}}",
           f"\\newcommand{{\\expSignFlip}}{{{c['sign_flip_scale']}}}",
           f"\\newcommand{{\\expScaling}}{{{c['scaling_factor']}}}",
           f"\\newcommand{{\\expDegree}}{{{max(r['k_degree'] for r in main)}}}",
           f"\\newcommand{{\\expCPU}}{{{env['cpu'].replace('(R)', '').replace('(TM)', '')}}}",
           f"\\newcommand{{\\expNproc}}{{{env['nproc']}}}",
           f"\\newcommand{{\\expMemGiB}}{{{env['mem_gib']}}}",
           f"\\newcommand{{\\expLiboqs}}{{{env['liboqs']}}}",
           f"\\newcommand{{\\expTorch}}{{{env['torch'].split('+')[0]}}}"]
    tag = {"edge_iiotset": "Edge", "ton_iot": "Ton", "unsw_nb15": "Unsw"}
    for ds, r in by_ds.items():
        t = tag[ds]
        out += [f"\\newcommand{{\\exp{t}Features}}{{{r['n_features']}}}",
                f"\\newcommand{{\\exp{t}Classes}}{{{r['n_classes']}}}",
                f"\\newcommand{{\\exp{t}Train}}{{{r['train_rows']:,}}}".replace(",", "{,}"),
                f"\\newcommand{{\\exp{t}Test}}{{{r['test_rows']:,}}}".replace(",", "{,}"),
                f"\\newcommand{{\\exp{t}Params}}{{{r['n_params']:,}}}".replace(",", "{,}")]
    mb = [r["max_byz_neighbors"] for r in main if r.get("max_byz_neighbors") is not None]
    if mb:
        out.append(f"\\newcommand{{\\expMaxByzNeighbors}}{{{max(mb)}}}")
        out.append(f"\\newcommand{{\\expRunsByzNeighborsAboveTwo}}{{{sum(1 for x in mb if x > 2)}/{len(mb)}}}")
    pq_runs = [r for r in main if r["variant"].startswith("pq")]
    if pq_runs:
        out.append(f"\\newcommand{{\\expAbortedRounds}}{{{sum(r.get('rounds_aborted', 0) for r in pq_runs)}}}")
        out.append(f"\\newcommand{{\\expPQRoundsTotal}}{{{sum(len(r['history']['round_ms']) for r in pq_runs)}}}")
    dp = next((r["dp"] for r in main if r.get("dp")), None)
    if dp:
        out.append(f"\\newcommand{{\\expEpsT}}{{{dp['eps_T']:.3g}}}")
    return "\n".join(out) + "\n"


def figure(main, timing, out_dir: Path, metric: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5, "pdf.fonttype": 42})
    g = group(main, ("dataset", "attack", "variant"))
    colors = {"fedavg": "#7f7f7f", "median": "#bcbd22", "trimmed_mean": "#1f77b4", "krum": "#8c564b",
              "multi_krum": "#ff7f0e", "fltrust": "#17becf", "pq_no_filter": "#9467bd", "pq_full": "#1b7837"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [2.3, 1.0]})
    scen = [(d, a) for d in DATASETS for a in ATTACKS]
    x = np.arange(len(scen))
    w = 0.84 / len(MAIN_VARS)
    for vi, v in enumerate(MAIN_VARS):
        ms, ss = [], []
        for d, a in scen:
            m, s, _ = cell_stats([r[metric] for r in g.get((d, a, v), [])])
            ms.append(m)
            ss.append(s)
        ax.bar(x - 0.42 + (vi + 0.5) * w, ms, w, yerr=ss, color=colors[v], label=VAR_LABEL[v].replace(r"\textbf{", "").replace("}", ""),
               error_kw={"elinewidth": 0.5, "capsize": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ATK_LABEL[a]}" for _, a in scen], rotation=45, ha="right", fontsize=6)
    for i, d in enumerate(DATASETS):
        ax.text(i * len(ATTACKS) + (len(ATTACKS) - 1) / 2, 1.04, DS_LABEL[d], ha="center", fontsize=7, transform=ax.get_xaxis_transform())
        if i:
            ax.axvline(i * len(ATTACKS) - 0.5, color="k", lw=0.4, ls=":")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Final macro-F1" if metric == "final_macro_f1" else "Final accuracy")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.38), frameon=False)
    ax.set_title("(a) Robustness under 25% Byzantine clients", pad=12)

    src = timing or main
    gt = group(src, ("variant",))
    vs = [v for v in ["fedavg", "trimmed_mean", "krum", "multi_krum", "fltrust", "pq_no_filter", "pq_full"] if (v,) in gt]
    train = [np.mean([r["avg_train_ms"] for r in gt[(v,)]]) for v in vs]
    crypto = [np.mean([sum(val for k, val in r["crypto_per_round"].items() if k.endswith("_ms")) for r in gt[(v,)]]) if v.startswith("pq") else 0 for v in vs]
    total = [np.mean([r["avg_round_ms"] for r in gt[(v,)]]) for v in vs]
    other = [max(0.0, t - a - b) for t, a, b in zip(total, train, crypto)]
    y = np.arange(len(vs))
    ax2.barh(y, train, color="#c7c7c7", label="local training")
    ax2.barh(y, crypto, left=train, color="#1b7837", label="PQ crypto + masking")
    ax2.barh(y, other, left=np.array(train) + np.array(crypto), color="#9ecae1", label="aggregation / filter")
    ax2.set_yticks(y)
    ax2.set_yticklabels([VAR_LABEL[v].replace(r"\textbf{", "").replace("}", "") for v in vs], fontsize=6.5)
    ax2.set_xlabel("Avg. round time (ms)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.4, -0.3), ncol=1, fontsize=6, frameon=False)
    ax2.set_title("(b) Per-round CPU time" + (" (timing grid)" if timing else ""))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pq_sparseagg_eval.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_pq_sparseagg_eval.png", dpi=300, bbox_inches="tight")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="experiments/real")
    ap.add_argument("--out-dir", default="paper/generated")
    ap.add_argument("--metric", choices=["macro_f1", "acc"], default="macro_f1")
    ap.add_argument("--seeds", type=int, default=5, help="expected seeds per cell (main grid)")
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args(argv)
    ind, out = Path(a.in_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main_r = load(ind / "main.jsonl", a.allow_synthetic)
    timing = load(ind / "timing.jsonl", a.allow_synthetic)
    sigma = load(ind / "sigma.jsonl", a.allow_synthetic)
    alpha = load(ind / "alpha.jsonl", a.allow_synthetic)
    if not main_r:
        sys.exit(f"no runs in {ind/'main.jsonl'}")
    warn = check_cells(group(main_r, ("dataset", "attack", "variant")), a.seeds, "main")
    synth = any(r["source"] == "SYNTHETIC" for r in main_r)
    banner = "% !!! SYNTHETIC SMOKE-TEST DATA -- NOT FOR THE PAPER !!!\n" if synth else ""

    attacks = [x for x in ATTACKS if any(r["attack"] == x for r in main_r)]
    cap = lambda m: (f"Final {m} on the held-out test set, mean$\\pm$std over {len({r['seed'] for r in main_r})} seeds "
                     f"($N={main_r[0]['cfg']['clients']}$, $T={main_r[0]['cfg']['rounds']}$, "
                     f"{int(main_r[0]['cfg']['byzantine_fraction']*100)}\\% Byzantine, Dirichlet $\\alpha={main_r[0]['cfg']['alpha']}$)")
    files = {
        "tab_accuracy.tex": metric_table(main_r, "final_acc", cap("accuracy"), "tab:accuracy", MAIN_VARS, attacks),
        "tab_macro_f1.tex": metric_table(main_r, "final_macro_f1", cap("macro-F1"), "tab:macrof1", MAIN_VARS, attacks),
        "tab_ablation.tex": ablation_table(main_r, "final_macro_f1"),
        "tab_overhead.tex": overhead_table(timing or main_r, "timing grid, 1 worker" if timing else "main grid, parallel workers"),
        "results_macros.tex": macros(main_r),
    }
    if sigma:
        files["tab_dp_sigma.tex"] = sweep_table(sigma, "dp_sigma", "tab:dpsigma",
                                                r"Sketch noise $\sigma$ vs.\ privacy $\varepsilon_T$ ($\delta=10^{-5}$, zCDP over $T$ rounds) and final macro-F1 of PQ-Full",
                                                "final_macro_f1")
    if alpha:
        files["tab_alpha.tex"] = sweep_table(alpha, "alpha", "tab:alpha",
                                             r"Non-IID strength (Dirichlet $\alpha$) under Min-Max (25\% Byzantine): final macro-F1", "final_macro_f1")
    for name, body in files.items():
        (out / name).write_text(banner + body)
    figure(main_r, timing, out, "final_macro_f1" if a.metric == "macro_f1" else "final_acc")

    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grid", "dataset", "attack", "variant", "param", "n_seeds", "acc_mean", "acc_std", "f1_mean", "f1_std",
                    "round_ms_mean", "rounds_aborted", "diverged_runs"])
        for gname, recs in [("main", main_r), ("timing", timing), ("sigma", sigma), ("alpha", alpha)]:
            key = {"sigma": "dp_sigma", "alpha": "alpha"}.get(gname)
            gg = defaultdict(list)
            for r in recs:
                gg[(r["dataset"], r["attack"], r["variant"], r["cfg"][key] if key else "")].append(r)
            for (ds, at, v, p), rs in sorted(gg.items(), key=lambda kv: str(kv[0])):
                am, asd, n = cell_stats([r["final_acc"] for r in rs])
                fm, fsd, _ = cell_stats([r["final_macro_f1"] for r in rs])
                w.writerow([gname, ds, at, v, p, n, f"{am:.4f}", f"{asd:.4f}", f"{fm:.4f}", f"{fsd:.4f}",
                            f"{np.mean([r['avg_round_ms'] for r in rs]):.1f}",
                            sum(r.get("rounds_aborted", 0) for r in rs), sum(bool(r["diverged"]) for r in rs)])

    prov = {"inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ind.glob("*.jsonl"))},
            "data_sources": sorted({(r["dataset"], r["source"], json.dumps(r["source_sha256"])) for r in main_r}),
            "runs": {"main": len(main_r), "timing": len(timing), "sigma": len(sigma), "alpha": len(alpha)},
            "warnings": warn}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    for m in warn:
        print("WARNING", m, file=sys.stderr)
    print(f"wrote {len(files)} .tex files, figure, summary.csv, provenance.json -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
