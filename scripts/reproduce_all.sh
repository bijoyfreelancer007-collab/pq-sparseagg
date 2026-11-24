#!/usr/bin/env bash
# Reproduce every experiment, table and figure of the paper from the three public datasets.
# Usage (from the repository root):  bash scripts/reproduce_all.sh [workers]
# Every grid is resumable: if a run stops, start the same command again and finished runs are skipped.
# Total CPU time on an 8-vCPU VM with 4 workers is roughly 1.5-2 days.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
W="${1:-4}"
[ -f .venv/bin/activate ] && source .venv/bin/activate
REAL=experiments/real; IMP=experiments/improve; EXD=experiments/extraD
mkdir -p "$REAL" "$IMP" "$EXD" paper/generated
DS2="edge_iiotset unsw_nb15"

echo "[0] unit tests"
python tests/test_secagg.py
python tests/test_improve.py

echo "[1] preprocess raw CSVs (see data/README.md) -> data/*_processed.npz"
for d in edge_iiotset ton_iot unsw_nb15; do
  [ -f "data/${d}_processed.npz" ] || python -m src.data.prepare_datasets --raw-dir data/raw --out-dir data --only "$d"
done

echo "[2] original grid at C = 5 (Edge-IIoTset, UNSW-NB15): all methods, 4 attacks + no attack, 5 seeds"
python -m src.exp.run_experiments --grid main  --datasets $DS2 --workers "$W" --out "$REAL/main.jsonl"
python -m src.exp.run_experiments --grid alpha --datasets $DS2 --workers "$W" --out "$REAL/alpha.jsonl"
python -m src.exp.run_experiments --grid timing --datasets $DS2 --workers 1 --out "$REAL/timing.jsonl"

echo "[3] same grid on ToN-IoT"
python -m src.exp.run_experiments --grid main --datasets ton_iot --workers "$W" \
  --variants fedavg median trimmed_mean krum multi_krum fltrust pq_no_filter pq_full --out "$IMP/ton_main.jsonl"
python -m src.exp.run_experiments --grid alpha --datasets ton_iot --workers "$W" --out "$IMP/ton_alpha.jsonl"

echo "[4] C = 1 variants (PQ-Med_1, PQ-Root, PQ-GRep), Clip-only baseline and alpha sweep, all datasets"
python -m src.exp.run_experiments --grid main2    --workers "$W" --out "$IMP/main2.jsonl"
python -m src.exp.run_experiments --grid clipbase --workers "$W" --out "$IMP/clipbase.jsonl"
python -m src.exp.run_experiments --grid alpha2   --workers "$W" --out "$IMP/alpha2.jsonl"

echo "[5] privacy-robustness frontier (C = 1, eps_T in {32,16,8,4}, 5 seeds)"
python -m src.exp.run_experiments --grid privfront_c1 --seeds 1 2 3 4 5 --workers "$W" --out "$IMP/privfront.jsonl"

echo "[6] dominant-class leakage of sketches (C = 5)"
python -m src.exp.leakage_attack --datasets $DS2 --seeds 1 2 3 4 5 --rounds 10 --clip-C 5.0 --out "$IMP/leakage.jsonl"

echo "[7] secure-aggregation scaling, real model sizes"
python -m src.exp.bench_secagg --clients 20 50 100 200 --dim 13455 --reps 3 --out "$IMP/bench_edge.jsonl" --tex "$IMP/tab_scaling_edge.tex"
python -m src.exp.bench_secagg --clients 20 50 100 200 --dim 21322 --reps 3 --out "$IMP/bench_unsw.jsonl" --tex "$IMP/tab_scaling_unsw.tex"

echo "[8] same-session timing (FedAvg, FLTrust, PQ-Med back to back, 1 worker) and PQ-GRep timing"
python -m src.exp.run_experiments --grid timing --datasets $DS2 --variants fedavg fltrust pq_full --workers 1 --out "$IMP/timing_same.jsonl"
python -m src.exp.run_experiments --grid timing2 --workers 1 --out "$IMP/timing2.jsonl"

echo "[9] ten-seed Min-Max significance, extra attacks (IPM, label flip, ALIE z = 1.5), ten-seed frontier"
S6="6 7 8 9 10"; S1="1 2 3 4 5"
R() { python -m src.exp.run_experiments --datasets $DS2 --workers "$W" "$@"; }
R --grid main     --attacks minmax --variants fedavg fltrust  --seeds $S6 --out $EXD/seeds10.jsonl
R --grid main2    --attacks minmax --variants pq_grep pq_root --seeds $S6 --out $EXD/seeds10.jsonl
R --grid clipbase --attacks minmax                            --seeds $S6 --out $EXD/seeds10.jsonl
R --grid alpha    --variants fedavg fltrust                   --seeds $S6 --out $EXD/seeds10_alpha.jsonl
R --grid alpha2   --variants pq_no_filter pq_grep pq_root     --seeds $S6 --out $EXD/seeds10_alpha.jsonl
R --grid main     --attacks ipm label_flip --variants fedavg multi_krum fltrust --seeds $S1 --out $EXD/attacks.jsonl
R --grid main2    --attacks ipm label_flip --variants pq_grep pq_root           --seeds $S1 --out $EXD/attacks.jsonl
R --grid clipbase --attacks ipm label_flip                                      --seeds $S1 --out $EXD/attacks.jsonl
R --grid main     --attacks alie --variants fedavg multi_krum fltrust --set alie_z=1.5 --seeds $S1 --out $EXD/alie15.jsonl
R --grid main2    --attacks alie --variants pq_grep pq_root           --set alie_z=1.5 --seeds $S1 --out $EXD/alie15.jsonl
R --grid clipbase --attacks alie                                      --set alie_z=1.5 --seeds $S1 --out $EXD/alie15.jsonl
R --grid privfront_c1 --variants pq_grep --seeds $S6 --out $EXD/privfront10.jsonl

echo "[10] tables, figures and statistics"
python -m src.exp.make_tables_v2 --real "$REAL" --improve "$IMP" --extra "$EXD" --out paper/generated
python scripts/stats_paired5.py
python scripts/stats_seeds10.py "$EXD"
python scripts/stats_extra_attacks.py "$EXD"
echo "Done: paper/generated/ holds every table and figure of the paper."
