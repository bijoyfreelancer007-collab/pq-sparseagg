"""Paired t-tests (5 seeds) of PQ-GRep / PQ-Root against FLTrust and FedAvg under Min-Max."""
import json,sys,numpy as np
from scipy import stats
def load(p): return [json.loads(l) for l in open(p)]
import os
old=load("experiments/real/main.jsonl")+load("experiments/real/alpha.jsonl")
for p in ("experiments/improve/ton_main.jsonl","experiments/improve/ton_alpha.jsonl"):
    if os.path.exists(p): old+=load(p)
new=load("experiments/improve/main2.jsonl")+load("experiments/improve/alpha2.jsonl")
def get(rows,ds,atk,var,alpha,C=None):
    out={}
    for r in rows:
        if "error" in r: continue
        if r["dataset"]!=ds or r["attack"]!=atk or r["variant"]!=var: continue
        if abs(r["cfg"]["alpha"]-alpha)>1e-9: continue
        if C is not None and abs(r["cfg"]["clip_C"]-C)>1e-9: continue
        out[r["seed"]]=r["final_macro_f1"]
    return out
for ds in ["edge_iiotset","ton_iot","unsw_nb15"]:
  for a in [0.1,0.3,0.5,1.0]:
    fl=get(old,ds,"minmax","fltrust",a); fa=get(old,ds,"minmax","fedavg",a)
    for v in ["pq_grep","pq_root"]:
      g=get(new,ds,"minmax",v,a,1.0)
      if not g: continue
      s=sorted(set(g)&set(fl)&set(fa))
      x=np.array([g[k] for k in s]); y=np.array([fl[k] for k in s]); z=np.array([fa[k] for k in s])
      p1=stats.ttest_rel(x,y).pvalue; p2=stats.ttest_rel(x,z).pvalue
      print(ds,a,v,len(s),f"{x.mean():.3f} fl {y.mean():.3f} d {x.mean()-y.mean():+.3f} p {p1:.3f} | fedavg {z.mean():.3f} p {p2:.4f}")
