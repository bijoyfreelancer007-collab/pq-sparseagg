# PQ-SparseAgg

Code for the paper **"PQ-SparseAgg: Sparse Post-Quantum Secure Aggregation with Reference-Guided Sketch Filtering for
Byzantine-Robust Federated Intrusion Detection"** (B. K. Chowdhury, submitted to IEEE Transactions on Dependable and
Secure Computing).

PQ-SparseAgg derives pairwise masks over a logarithmic-degree sparse graph with ML-KEM-768, authenticates every message
with ML-DSA-65, double-masks updates in Z_q with Shamir-shared self-masks, and screens clients through clipped,
Gaussian-noised sketches that an aggregate consistency check binds to the masked updates. The PQ-GRep filter scores each
sketch against the projected previous global update and keeps a multi-round reputation.

## Layout

| Path | Content |
|---|---|
| `src/secagg/protocol.py` | Secure aggregation: ML-KEM-768 seeds, ML-DSA-65 signatures, sparse graph, double masking in Z_{2^32}, Shamir sharing, commit-then-project, consistency check, unmasking with dropouts/rejections |
| `src/fl/byzantine_filter.py` | Sketch scoring (median / global-update / root reference), reputation, rank rejection |
| `src/exp/fl_core.py` | MLP, Dirichlet partitions, local training, attacks (sign flip, scaling, ALIE, Min-Max, IPM, label flip), baseline aggregators (FedAvg, Median, Trimmed-Mean, Krum, Multi-Krum, FLTrust) |
| `src/exp/run_experiments.py` | Experiment grids; one JSON line per run with config hash, environment and full history |
| `src/exp/leakage_attack.py` | Dominant-class inference attack on sketches |
| `src/exp/bench_secagg.py` | Sparse vs. dense secure-aggregation scaling benchmark |
| `src/exp/make_tables_v2.py` | Writes every table, figure and macro of the paper from the JSONL results (no hand-typed numbers) |
| `src/data/prepare_datasets.py` | Preprocessing of the raw CSVs |
| `scripts/` | `reproduce_all.sh` and the paired significance tests |
| `tests/` | Unit tests (exact unmasking, Shamir, PRG, filter) |

## Setup

Python 3.10 or newer on Linux, CPU only.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export PYTHONPATH=$PWD
python tests/test_secagg.py && python tests/test_improve.py
```

## Data

See [`data/README.md`](data/README.md). The datasets are public but not redistributed here.

## Reproduce the paper

```bash
bash scripts/reproduce_all.sh 4      # 4 parallel workers
```

This runs every grid of the paper (5 or 10 seeds, N = 20 clients, T = 30 rounds, 25% Byzantine clients) and then writes
the tables and figures to `paper/generated/`. Every grid resumes after an interruption. The paper's runs used one
Ubuntu VM with 8 vCPUs (Intel Core i5-8350U), 11 GiB RAM, no GPU, PyTorch 2.x CPU and liboqs 0.16.0; absolute
timings depend on the machine, and the overhead table compares methods only within one timing session.

Single run example:

```bash
python -m src.exp.run_experiments --grid main2 --datasets edge_iiotset --attacks minmax \
  --variants pq_grep --seeds 1 --workers 1 --out experiments/try.jsonl
```

Variant names used in the code: `pq_full` = PQ-Med (median reference), `pq_grep` = PQ-GRep (main design),
`pq_root` = PQ-Root (root-set reference), `pq_no_filter` = Clip-only.

## Scope

The protocol runs end to end in a single process (real ML-KEM/ML-DSA, Shamir sharing, unmasking); network transfer is
not emulated, and message withholding or corrupted shares are not simulated. See the paper's limitations section.

## Citation

If you use this code, please cite the paper (citation details will be added after publication).

## License

MIT (see `LICENSE`).
