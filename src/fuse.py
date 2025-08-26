# src/fuse.py (Correct and complete version)
import argparse, joblib, pandas as pd, numpy as np, os
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def minmax(x):
    x = np.asarray(x, dtype=float);
    return (x - x.min())/(x.max()-x.min()+1e-8)

def fuse_multi(scores, weights):
    s = np.zeros_like(scores[0])
    for w,sc in zip(weights, scores):
        s += w * minmax(sc)
    return np.clip(s, 0, 1)

def evaluate(models_dir, eval_path, report_dir, graph_csv=None):
    df = pd.read_parquet(eval_path)
    y = df["is_fraud"].values
    X = df.drop(columns=["is_fraud"])

    sup = joblib.load(f"{models_dir}/supervised.pkl")
    uns = joblib.load(f"{models_dir}/unsupervised.pkl")

    p_sup = sup.predict_proba(X)[:,1]
    s_uns = -uns.decision_function(X)

    # optional graph & seq
    graph_score = pd.read_csv(graph_csv)["graph_score"].values if graph_csv else np.zeros(len(X))
    seq_score = df[["seq_amt_z","seq_cat_surprise"]].mean(axis=1).fillna(0).values if "seq_amt_z" in df else np.zeros(len(X))

    risk = fuse_multi([p_sup, s_uns, graph_score, seq_score], weights=[0.55, 0.2, 0.15, 0.10])

    # The original script was missing this line, which is why we created the folder manually before.
    # It's good practice to have it in the script itself.
    os.makedirs(report_dir, exist_ok=True)
    pd.DataFrame({"risk": risk, "y": y}).to_csv(f"{report_dir}/scores.csv", index=False)

    print("ROC-AUC:", roc_auc_score(y, risk))
    print("PR-AUC :", average_precision_score(y, risk))
    prec, rec, th = precision_recall_curve(y, risk)
    best = np.argmax(2*prec*rec/(prec+rec+1e-8))
    print("Best-F1 threshold:", float(th[max(best-1,0)]))

# This is the part that was missing/incorrect before
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--report", required=True)
    # Add the missing argument so the script knows what --graph_csv is
    ap.add_argument("--graph_csv", required=True)
    args = ap.parse_args()
    evaluate(args.models_dir, args.eval, args.report, args.graph_csv)