import argparse, pandas as pd, numpy as np
from scipy.stats import ks_2samp

def psi(a, b, bins=10):
    a, b = np.asarray(a), np.asarray(b)
    qs = np.quantile(np.concatenate([a,b]), np.linspace(0,1,bins+1))
    qs[0]-=1e-9; qs[-1]+=1e-9
    ca,_ = np.histogram(a, qs); cb,_ = np.histogram(b, qs)
    pa = ca/ca.sum(); pb = cb/cb.sum()
    pa = np.clip(pa, 1e-6, 1); pb = np.clip(pb, 1e-6, 1)
    return float(np.sum((pa-pb)*np.log(pa/pb)))

def expected_value(tp, fp, fn, benefit_tp=5000, cost_fp=5, cost_fn=5000):
    return tp*benefit_tp - fp*cost_fp - fn*cost_fn

def monitor(ref_scores_csv, cur_scores_csv, out_report="reports/drift.md"):
    ref = pd.read_csv(ref_scores_csv)
    cur = pd.read_csv(cur_scores_csv)
    ref_r, cur_r = ref["risk"].values, cur["risk"].values
    ks_p = ks_2samp(ref_r, cur_r).pvalue
    psi_score = psi(ref_r, cur_r)

    with open(out_report,"w") as f:
        f.write(f"# Drift Report\n\nKS p-value: {ks_p:.4g}\n\nPSI: {psi_score:.4f}\n")
        f.write("\nThresholds: PSI>0.2 warn, >0.3 alert; KS p<0.01 alert.\n")
    print("Drift ->", {"KS_p": ks_p, "PSI": psi_score})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cur", required=True)
    ap.add_argument("--out", default="reports/drift.md")
    args = ap.parse_args()
    monitor(args.ref, args.cur, args.out)
