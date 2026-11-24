"""
Preprocess the REAL raw CSV files of Edge-IIoTset, ToN-IoT and UNSW-NB15 into
`data/<name>_processed.npz` files used by the experiments.

Nothing here generates data. If a raw file is missing the script stops with an
error that says which file to download.

Expected raw layout (search is recursive, so the unzipped Kaggle folders work):
  data/raw/edge_iiotset/**/DNN-EdgeIIoT-dataset.csv   (or ML-EdgeIIoT-dataset.csv)
  data/raw/ton_iot/**/train_test_network.csv          (ToN-IoT "Train_Test_Network")
  data/raw/unsw_nb15/**/UNSW_NB15_training-set.csv
  data/raw/unsw_nb15/**/UNSW_NB15_testing-set.csv     (official train/test split)

Pipeline per dataset (all choices are recorded in the npz `meta` field):
  1. drop identifier / free-text columns (IPs, timestamps, ports, payloads, URIs)
  2. drop exact duplicate rows (before the split, so no train/test leakage)
  3. multi-class label = attack type column; classes sorted alphabetically
  4. stratified train/test split (UNSW-NB15 uses its official split)
  5. columns that are >=95% numeric -> numeric; other columns -> one-hot of the
     top-K categories + "other"
  6. signed log1p on numeric columns, then z-score with train-only statistics
  7. optional stratified caps on train/test size (--max-train / --max-test)

Usage:
  python -I -m src.data.prepare_datasets --raw-dir data/raw --out-dir data
  python -m src.data.prepare_datasets --only unsw_nb15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 0  # fixed: the test set never changes across experiment seeds

DATASETS = {
    "edge_iiotset": {
        "patterns": ["DNN-EdgeIIoT-dataset.csv", "ML-EdgeIIoT-dataset.csv"],
        "label": "Attack_type",
        "drop_extra_labels": ["Attack_label"],
        # Drop list used by the dataset authors (Ferrag et al., 2022)
        "drop": [
            "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
            "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
            "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
            "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
        ],
        "download": "https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
    },
    "ton_iot": {
        "patterns": ["train_test_network.csv", "Train_Test_Network.csv"],
        "label": "type",
        "drop_extra_labels": ["label"],
        "drop": [
            "ts", "src_ip", "dst_ip", "src_port", "dst_port", "dns_query",
            "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent",
        ],
        "download": "https://research.unsw.edu.au/projects/toniot-datasets (Train_Test_datasets/Train_Test_Network_dataset)",
    },
    "unsw_nb15": {
        "patterns": ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
        "official_split": True,
        "label": "attack_cat",
        "drop_extra_labels": ["label"],
        "drop": ["id"],
        "download": "https://research.unsw.edu.au/projects/unsw-nb15-dataset (or Kaggle mrwellsdavid/unsw-nb15)",
    },
}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_file(raw_dir: Path, name: str) -> Path | None:
    hits = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.name.lower() == name.lower())
    return hits[0] if hits else None


def read_csv_dropping(path: Path, drop: list[str]) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    header_stripped = [c.strip() for c in header]
    keep = [c for c, s in zip(header, header_stripped) if s not in set(drop)]
    dropped = [s for s in header_stripped if s in set(drop)]
    df = pd.read_csv(path, usecols=keep, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df, dropped


def stratified_cap(X: np.ndarray, y: np.ndarray, cap: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if cap is None or cap <= 0 or len(y) <= cap:
        return X, y
    # classes with a single sample cannot be stratified; keep them in full
    vals, counts = np.unique(y, return_counts=True)
    single = np.isin(y, vals[counts < 2])
    idx_all = np.arange(len(y))
    idx_multi = idx_all[~single]
    n_take = max(1, cap - int(single.sum()))
    take, _ = train_test_split(idx_multi, train_size=n_take, stratify=y[idx_multi], random_state=seed)
    idx = np.sort(np.concatenate([take, idx_all[single]]))
    return X[idx], y[idx]


class TabularEncoder:
    """Fit on train only; numeric = signed log1p + z-score; categorical = top-K one-hot."""

    def __init__(self, top_k: int = 32, numeric_threshold: float = 0.95, log_transform: bool = True):
        self.top_k = top_k
        self.numeric_threshold = numeric_threshold
        self.log_transform = log_transform

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        self.numeric_cols: list[str] = []
        self.cat_cols: dict[str, list[str]] = {}
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                self.numeric_cols.append(c)
                continue
            parsed = pd.to_numeric(s, errors="coerce")
            frac_numeric = float(parsed.notna().mean()) if len(s) else 0.0
            if frac_numeric >= self.numeric_threshold:
                self.numeric_cols.append(c)
            else:
                top = s.astype(str).value_counts().index[: self.top_k].tolist()
                self.cat_cols[c] = top
        num = self._numeric_block(df)
        self.mean_ = num.mean(axis=0)
        std = num.std(axis=0)
        # constant columns on train are dropped
        self.keep_num_ = std > 1e-12
        self.std_ = np.where(self.keep_num_, std, 1.0)
        return self

    def _numeric_block(self, df: pd.DataFrame) -> np.ndarray:
        if not self.numeric_cols:
            return np.zeros((len(df), 0), dtype=np.float64)
        block = df[self.numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        if self.log_transform:
            block = np.sign(block) * np.log1p(np.abs(block))
        return block

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = (self._numeric_block(df) - self.mean_) / self.std_
        parts = [num[:, self.keep_num_]]
        for c, cats in self.cat_cols.items():
            sv = df[c].astype(str)
            codes = pd.Categorical(sv.where(sv.isin(cats)), categories=cats).codes.astype(np.int64)
            codes[codes < 0] = len(cats)  # unseen / rare -> "other"
            onehot = np.zeros((len(df), len(cats) + 1), dtype=np.float32)
            onehot[np.arange(len(df)), codes] = 1.0
            parts.append(onehot)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def feature_names(self) -> list[str]:
        names = [c for c, k in zip(self.numeric_cols, self.keep_num_) if k]
        for c, cats in self.cat_cols.items():
            names += [f"{c}={v}" for v in cats] + [f"{c}=<other>"]
        return names


def prepare(name: str, raw_dir: Path, out_dir: Path, max_train: int | None, max_test: int | None,
            test_size: float, top_k: int, dedup: bool) -> dict:
    spec = DATASETS[name]
    t0 = time.time()
    ds_raw = raw_dir / name
    if not ds_raw.exists():
        ds_raw = raw_dir
    drop = spec["drop"]

    if spec.get("official_split"):
        f_train = find_file(ds_raw, spec["patterns"][0])
        f_test = find_file(ds_raw, spec["patterns"][1])
        if f_train is None or f_test is None:
            raise FileNotFoundError(f"[{name}] need {spec['patterns']} under {ds_raw}. Download: {spec['download']}")
        sources = [f_train, f_test]
        df_tr, dropped = read_csv_dropping(f_train, drop)
        df_te, _ = read_csv_dropping(f_test, drop)
        split_swapped = False
        if len(df_tr) < len(df_te):
            # Official UNSW-NB15 split: 175,341 train / 82,332 test. Some mirrors ship the two
            # files with the names swapped; use the larger file as the training split.
            print(f"WARNING [{name}]: '{f_train.name}' has {len(df_tr)} rows < '{f_test.name}' "
                  f"{len(df_te)} rows; swapping so the larger file is the training split.", file=sys.stderr)
            df_tr, df_te = df_te, df_tr
            sources = [f_test, f_train]
            split_swapped = True
        n_raw = len(df_tr) + len(df_te)
        n_dup = 0
        if dedup:
            before = len(df_tr) + len(df_te)
            df_tr = df_tr.drop_duplicates()
            df_te = df_te.drop_duplicates()
            n_dup = before - len(df_tr) - len(df_te)
    else:
        src = None
        for pat in spec["patterns"]:
            src = find_file(ds_raw, pat)
            if src is not None:
                break
        if src is None:
            raise FileNotFoundError(f"[{name}] none of {spec['patterns']} found under {ds_raw}. Download: {spec['download']}")
        sources = [src]
        df, dropped = read_csv_dropping(src, drop)
        n_raw = len(df)
        n_dup = 0
        if dedup:
            before = len(df)
            df = df.drop_duplicates()
            n_dup = before - len(df)
        df_tr = df_te = None
        split_swapped = False

    label = spec["label"]

    def clean_labels(d: pd.DataFrame) -> pd.DataFrame:
        if label not in d.columns:
            raise KeyError(f"[{name}] label column '{label}' not in CSV columns: {list(d.columns)[:10]}...")
        d = d[d[label].notna()].copy()
        lab = d[label].astype(str).str.strip()
        lab = lab.replace({"": "Normal", "-": "Normal"})
        d[label] = lab
        return d.drop(columns=[c for c in spec["drop_extra_labels"] if c in d.columns])

    if df_tr is None:
        df = clean_labels(df)
        class_names = sorted(df[label].unique().tolist())
        y_all = df[label].map({c: i for i, c in enumerate(class_names)}).to_numpy(np.int64)
        idx = np.arange(len(df))
        vals, counts = np.unique(y_all, return_counts=True)
        if counts.min() < 2:
            raise ValueError(f"[{name}] a class has <2 samples after dedup: {dict(zip(vals, counts))}")
        i_tr, i_te = train_test_split(idx, test_size=test_size, stratify=y_all, random_state=SPLIT_SEED)
        df_tr, df_te = df.iloc[i_tr], df.iloc[i_te]
    else:
        df_tr, df_te = clean_labels(df_tr), clean_labels(df_te)
        class_names = sorted(set(df_tr[label]) | set(df_te[label]))

    cmap = {c: i for i, c in enumerate(class_names)}
    y_tr = df_tr[label].map(cmap).to_numpy(np.int64)
    y_te = df_te[label].map(cmap).to_numpy(np.int64)
    feat_tr = df_tr.drop(columns=[label])
    feat_te = df_te.drop(columns=[label])
    raw_feature_count = feat_tr.shape[1]

    enc = TabularEncoder(top_k=top_k).fit(feat_tr)
    X_tr = enc.transform(feat_tr)
    X_te = enc.transform(feat_te)
    n_tr_full, n_te_full = len(y_tr), len(y_te)
    X_tr, y_tr = stratified_cap(X_tr, y_tr, max_train, SPLIT_SEED)
    X_te, y_te = stratified_cap(X_te, y_te, max_test, SPLIT_SEED + 1)

    meta = {
        "dataset": name,
        "source_files": [str(p) for p in sources],
        "source_sha256": [sha256_file(p) for p in sources],
        "rows_raw": int(n_raw),
        "duplicates_removed": int(n_dup),
        "dropped_columns": dropped,
        "label_column": label,
        "class_names": class_names,
        "raw_feature_columns": int(raw_feature_count),
        "encoded_features": int(X_tr.shape[1]),
        "numeric_columns": len([k for k in enc.keep_num_ if k]),
        "categorical_columns": list(enc.cat_cols.keys()),
        "train_rows_before_cap": int(n_tr_full),
        "test_rows_before_cap": int(n_te_full),
        "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)),
        "train_class_counts": {class_names[i]: int(c) for i, c in zip(*np.unique(y_tr, return_counts=True))},
        "test_class_counts": {class_names[i]: int(c) for i, c in zip(*np.unique(y_te, return_counts=True))},
        "split": ("official" + (" (file names swapped, larger file used as train)" if split_swapped else ""))
                 if spec.get("official_split") else f"stratified test_size={test_size} seed={SPLIT_SEED}",
        "train_file": str(sources[0]),
        "max_train": max_train,
        "max_test": max_test,
        "top_k_categories": top_k,
        "dedup": dedup,
        "prep_seconds": round(time.time() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}_processed.npz"
    np.savez_compressed(
        out, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        feature_names=np.array(enc.feature_names()), class_names=np.array(class_names),
        meta=np.array(json.dumps(meta)),
    )
    meta["output"] = str(out)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--only", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--max-train", type=int, default=200000, help="stratified cap on train rows (0 = no cap)")
    ap.add_argument("--max-test", type=int, default=50000, help="stratified cap on test rows (0 = no cap)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args(argv)

    failed = 0
    for name in args.only:
        try:
            meta = prepare(name, Path(args.raw_dir), Path(args.out_dir), args.max_train or None,
                           args.max_test or None, args.test_size, args.top_k, not args.no_dedup)
            print(json.dumps({k: meta[k] for k in (
                "dataset", "rows_raw", "duplicates_removed", "raw_feature_columns", "encoded_features",
                "train_rows", "test_rows", "class_names", "output")}, ensure_ascii=False))
            print("  train classes:", meta["train_class_counts"])
        except Exception as e:  # report and continue with the other datasets
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
