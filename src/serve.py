# src/serve.py (Corrected and more robust version)
from fastapi import FastAPI, UploadFile
import pandas as pd
import joblib
import numpy as np
from src.schema import Transaction
from src.explain import reason_codes, build_globals
# --- ADD THIS IMPORT ---
from src.features import NUMERIC, CATEG

app = FastAPI(title="Fraud Risk API")
sup = joblib.load("models/supervised.pkl")
uns = joblib.load("models/unsupervised.pkl")

def fuse(sup_p, uns_s, w_sup=0.7, w_uns=0.3):
    return float(np.clip(w_sup*sup_p + w_uns*uns_s, 0, 1))

# --- THIS FUNCTION IS NOW FIXED ---
@app.post("/score")
def score_tx(tx: Transaction):
    # Create a full DataFrame with all expected columns, initially filled with None
    all_cols = list(set(NUMERIC + CATEG))
    row_dict = {c: None for c in all_cols}
    
    # Overwrite the None values with the data we received from the request
    for key, val in tx.dict().items():
        if key in row_dict:
            row_dict[key] = val
    
    # Now create the DataFrame from the complete dictionary
    row = pd.DataFrame([row_dict])

    # The rest of the scoring logic can now run without crashing
    p_sup = sup.predict_proba(row)[:,1][0]
    s_uns = -uns.decision_function(row)[0]
    # normalize per-request (ok for demo)
    s_uns = (s_uns - (-1.0)) / (2.0)
    risk = fuse(p_sup, s_uns)

    globals_ = {"amt_p95": 500.0, "rare_cats": set(), "geo_km_cut": 100.0}
    reasons = reason_codes(row.iloc[0], globals_)
    label = "FRAUD" if risk >= 0.5 else "SAFE"
    return {"risk": risk, "label": label, "p_supervised": float(p_sup), "s_unsupervised": float(s_uns), "reasons": reasons}

@app.post("/score_csv")
async def score_csv(file: UploadFile):
    df = pd.read_csv(file.file)
    globals_ = build_globals(df)
    p_sup = sup.predict_proba(df)[:,1]
    s_uns = -uns.decision_function(df)
    s_uns = (s_uns - s_uns.min()) / (s_uns.max()-s_uns.min() + 1e-8)

    risks = []
    reasons_list = []
    for i, row in df.iterrows():
        risk = fuse(p_sup[i], s_uns[i])
        risks.append(risk)
        reasons_list.append(reason_codes(row, globals_))

    out = df.copy()
    out["risk"] = risks
    out["reasons"] = [", ".join(r[:3]) for r in reasons_list]
    return {"n": len(out), "preview": out.head(10).to_dict(orient="records")}