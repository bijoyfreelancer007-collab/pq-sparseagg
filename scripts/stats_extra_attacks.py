"""Extra attacks (IPM, label flip, ALIE z=1.5) and ten-seed privacy frontier: F1, TPR/FPR, paired tests."""
import json,numpy as np,collections,sys
from scipy import stats
D=(sys.argv[1] if len(sys.argv)>1 else "experiments/extraD").rstrip("/")+"/"
L=lambda p:[json.loads(l) for l in open(p)]
def rates(r):
    h=r["history"]; R=len(h["byz_rejected"]); f=r["byzantine_count"]; N=r["cfg"]["clients"]
    return (sum(h["byz_rejected"])/(f*R) if f and R else float('nan'), sum(h["honest_rejected"])/((N-f)*R) if R else float('nan'))
def name(r):
    v=r["variant"]
    if v in("pq_no_filter","pq_grep","pq_root","pq_full"): v+=f"_C{r['cfg']['clip_C']:g}"
    return v
for f in ["attacks.jsonl","alie15.jsonl"]:
    print("=====",f)
    g=collections.defaultdict(list)
    for r in L(D+f): g[(r["dataset"][:4],r["attack"],name(r))].append((np.nan if r.get("diverged") else r["final_macro_f1"],)+rates(r)+(r["seed"],))
    for k in sorted(g):
        v=np.array([x[:3] for x in g[k]])
        print(f"{k[0]} {k[1]:10s} {k[2]:18s} n={len(v)} f1={np.nanmean(v[:,0]):.3f}±{np.nanstd(v[:,0],ddof=1):.3f} tpr={np.nanmean(v[:,1]):.2f} fpr={np.nanmean(v[:,2]):.2f}")
    # paired GRep vs others
    for ds in ["edge","unsw"]:
      for atk in sorted({k[1] for k in g}):
        gr={x[3]:x[0] for x in g[(ds,atk,"pq_grep_C1")]}
        for other in ["fedavg","fltrust","multi_krum","pq_no_filter_C1"]:
            o={x[3]:x[0] for x in g.get((ds,atk,other),[])}
            s=sorted(set(gr)&set(o))
            if len(s)<3: continue
            x=np.array([gr[k] for k in s]); y=np.array([o[k] for k in s])
            print(f"  {ds} {atk} GRep-{other}: {x.mean()-y.mean():+.3f} p={stats.ttest_rel(x,y).pvalue:.3f}")
print("===== privfront 10 seeds, pq_grep C=1")
pf=[r for r in L("experiments/improve/privfront.jsonl") if r["variant"]=="pq_grep" and abs(r["cfg"]["clip_C"]-1)<1e-9 and r["dataset"]!="ton_iot"]+L(D+"privfront10.jsonl")
cb=[r for r in L("experiments/improve/clipbase.jsonl")+L(D+"seeds10.jsonl") if r["variant"]=="pq_no_filter" and abs(r["cfg"]["clip_C"]-1)<1e-9 and abs(r["cfg"]["alpha"]-0.5)<1e-9 and r["dataset"]!="ton_iot"]
print("hashes", collections.Counter((r["cfg"].get("dp_eps_target"),r["cfg_hash"]) for r in pf if r["dataset"]=="edge_iiotset" and r["attack"]=="minmax"))
for ds in ["edge_iiotset","unsw_nb15"]:
  for atk in ["sign_flip","minmax"]:
    c={r["seed"]:r["final_macro_f1"] for r in cb if r["dataset"]==ds and r["attack"]==atk}
    for e in [None,32.0,16.0,8.0,4.0]:
      rs=[r for r in pf if r["dataset"]==ds and r["attack"]==atk and r["cfg"].get("dp_eps_target")==e]
      g={r["seed"]:r["final_macro_f1"] for r in rs}
      s=sorted(set(g)&set(c)); x=np.array([g[k] for k in s]); y=np.array([c[k] for k in s])
      tp=np.mean([rates(r)[0] for r in rs])
      print(f"{ds[:4]} {atk:9s} eps={e} n={len(rs)} f1={np.mean(list(g.values())):.3f}±{np.std(list(g.values()),ddof=1):.3f} tpr={tp:.2f} | paired vs clip n={len(s)} d={x.mean()-y.mean():+.3f} p={stats.ttest_rel(x,y).pvalue:.3f}")
