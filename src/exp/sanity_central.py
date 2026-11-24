"""
Sanity check (not a paper result): train the same MLP centrally on the whole
training split and report test accuracy / macro-F1 per epoch, plus the
majority-class baseline and per-class F1. If FL numbers are far below this,
the gap is due to FL (rounds, non-IID), not the data pipeline.

  python -m src.exp.sanity_central --dataset unsw_nb15 --epochs 5
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from src.exp import fl_core as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unsw_nb15")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    torch.manual_seed(0)
    d = F.load_processed(a.dataset, a.data_dir)
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    C = len(d["class_names"])
    print(f"{a.dataset}: train {Xtr.shape}, test {Xte.shape}, classes {C}, "
          f"feature range [{Xtr.min():.2f}, {Xtr.max():.2f}], NaN={np.isnan(Xtr).sum()}")
    maj = np.bincount(ytr).argmax()
    print(f"majority class '{d['class_names'][maj]}': test acc if always predicted = {np.mean(yte == maj):.4f}")
    model = F.MLP(Xtr.shape[1], C)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    g = torch.Generator().manual_seed(0)
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(yt), generator=g)
        tot = 0.0
        for s in range(0, len(yt), a.batch):
            b = perm[s:s + a.batch]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        ev = F.evaluate(model, F.get_flat(model), Xte, yte, C)
        print(f"epoch {ep}: train loss {tot/len(yt):.4f}  test acc {ev['acc']:.4f}  macro-F1 {ev['macro_f1']:.4f}")
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(1).numpy()
    for c, name in enumerate(d["class_names"]):
        tp = np.sum((pred == c) & (yte == c)); fp = np.sum((pred == c) & (yte != c)); fn = np.sum((pred != c) & (yte == c))
        print(f"  {name:15s} n_test={int(np.sum(yte == c)):6d}  F1={2*tp/max(1, 2*tp+fp+fn):.3f}")


if __name__ == "__main__":
    main()
