import argparse, pandas as pd, numpy as np, os

DROP_COLS = ["first","last","street","dob","job"]  # PII / leakage-y
TARGET = "is_fraud"

def main(in_path, out_path):
    df = pd.read_csv(in_path)
    for c in DROP_COLS:
        if c in df.columns: df = df.drop(columns=c)

    # basic cleaning
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df = df.dropna(subset=["amt"])  # must have amount

    # save as parquet (fast)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"saved {len(df):,} rows -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()
    main(args.in_path, args.out_path)
