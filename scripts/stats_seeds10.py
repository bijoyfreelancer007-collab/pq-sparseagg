"""Ten-seed Min-Max significance (Table seeds10): paired t-tests against FLTrust and FedAvg."""
import json,numpy as np,collections,sys
from scipy import stats

L=lambda p:[json.loads(l) for l in open(p)]
old=L("experiments/real/main.jsonl")+L("experiments/real/alpha.jsonl")
new=L("experiments/improve/main2.jsonl")+L("experiments/improve/alpha2.jsonl")+L("experiments/improve/clipbase.jsonl")
D=(sys.argv[1] if len(sys.argv)>1 else "experiments/extraD").rstrip("/")+"/"
d_old=L(D+"seeds10.jsonl")+L(D+"seeds10_alpha.jsonl")
# hash check
def hashes(rows,var,ds,a,C=None):
    return {r["cfg_hash"] for r in rows if r["variant"]==var and r["dataset"]==ds and r["attack"]=="minmax" and abs(r["cfg"]["alpha"]-a)<1e-9 and (C is None or abs(r["cfg"]["clip_C"]-C)<1e-9)}
for var,C in [("fedavg",None),("fltrust",None),("pq_grep",1.0),("pq_root",1.0),("pq_no_filter",1.0)]:
    for a in [0.1,0.5]:
        print("hash",var,a,hashes(old+new,var,"edge_iiotset",a,C),hashes(d_old,var,"edge_iiotset",a,C))
allr=old+new+d_old
def get(var,ds,a,C=None):
    out={}
    for r in allr:
        if r["variant"]==var and r["dataset"]==ds and r["attack"]=="minmax" and abs(r["cfg"]["alpha"]-a)<1e-9 and (C is None or abs(r["cfg"]["clip_C"]-C)<1e-9) and "error" not in r:
            if r["seed"] in out and abs(out[r["seed"]]-r["final_macro_f1"])>1e-12: print("DUP",var,ds,a,r["seed"])
            out[r["seed"]]=r["final_macro_f1"]
    return out
print()
for ds in ["edge_iiotset","unsw_nb15"]:
  for a in [0.1,0.3,0.5,1.0]:
    fl=get("fltrust",ds,a); fa=get("fedavg",ds,a); cl=get("pq_no_filter",ds,a,1.0)
    for v in ["pq_grep","pq_root"]:
      g=get(v,ds,a,1.0); s=sorted(set(g)&set(fl)&set(fa))
      x=np.array([g[k] for k in s]);y=np.array([fl[k] for k in s]);z=np.array([fa[k] for k in s])
      s5=[k for k in s if k>5]; x5=np.array([g[k] for k in s5]); y5=np.array([fl[k] for k in s5])
      w=stats.wilcoxon(x,y).pvalue
      print(f"{ds[:4]} a={a} {v} n={len(s)} {x.mean():.3f}±{x.std(ddof=1):.3f} FL {y.mean():.3f}±{y.std(ddof=1):.3f} d={x.mean()-y.mean():+.3f} p_t={stats.ttest_rel(x,y).pvalue:.4f} p_w={w:.4f} wins={int((x>y).sum())}/{len(s)} | FA {z.mean():.3f} p={stats.ttest_rel(x,z).pvalue:.2e} | seeds6-10 d={x5.mean()-y5.mean():+.3f} | clip {np.mean([cl[k] for k in s if k in cl]):.3f} n={sum(k in cl for k in s)}")
