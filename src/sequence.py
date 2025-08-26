import argparse, pandas as pd, numpy as np, os

CARD, TSTAMP, AMT, CAT = "cc_num","trans_date_trans_time","amt","category"
TARGET="is_fraud"

def robust_z(x, window=20):
    # rolling median & MAD for robustness
    med = x.rolling(window, min_periods=5).median()
    mad = (x - med).abs().rolling(window, min_periods=5).median()
    z = (x - med) / (mad*1.4826 + 1e-6)
    return z.abs().clip(0, 20)

def cat_surprise(series, window=20):
    # probability of current category in last W observations (lower = more surprising)
    probs = []
    hist = []
    for c in series:
        if len(hist)==0: probs.append(1.0)
        else:
            p = (hist.count(c) / len(hist))
            probs.append(p if p>0 else 1.0/len(hist))
        hist.append(c)
        if len(hist)>window: hist.pop(0)
    # convert to surprise score
    s = 1 - np.array(probs, dtype=float)
    return pd.Series(s).clip(0,1)

def build_seq_scores(df: pd.DataFrame):
    df = df.sort_values([CARD, TSTAMP])
    df["seq_amt_z"] = df.groupby(CARD)[AMT].transform(lambda s: robust_z(s, 20))
    df["seq_cat_surprise"] = df.groupby(CARD)[CAT].transform(lambda s: cat_surprise(s, 20))
    # normalize to [0,1]
    for c in ["seq_amt_z","seq_cat_surprise"]:
        v = df[c].fillna(0).values
        v = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-8)
        df[c] = v
    return df[["seq_amt_z","seq_cat_surprise"]]

def main(in_parquet, out_parquet):
    df = pd.read_parquet(in_parquet)
    if TSTAMP in df.columns:
        df[TSTAMP] = pd.to_datetime(df[TSTAMP], errors="coerce")
    seq = build_seq_scores(df)
    out = pd.concat([df.reset_index(drop=True), seq.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print("saved with sequence features ->", out_parquet)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    args = ap.parse_args()
    main(args.in_parquet, args.out_parquet)
