import numpy as np, pandas as pd

# Simple reason codes (fast + human-friendly)
def reason_codes(row: pd.Series, globals_: dict):
    reasons = []
    amt = row.get("amt", 0.0)
    hour = row.get("hour", None)
    cat = row.get("category", None)

    # 1) High amount vs global 95th
    p95 = globals_.get("amt_p95", 500.0)
    if amt > p95:
        reasons.append(f"Amount above 95th percentile ({p95:.0f})")

    # 2) Odd hour
    if hour is not None and (hour <= 4 or hour >= 23):
        reasons.append("Odd hour transaction")

    # 3) Rare category frequency
    rare_cats = globals_.get("rare_cats", set())
    if cat in rare_cats:
        reasons.append(f"Rare category: {cat}")

    # 4) Large geo gap (if lat/long + merch_lat/long exist)
    try:
        d = haversine(row.get("lat"), row.get("long"), row.get("merch_lat"), row.get("merch_long"))
        if d is not None and d > globals_.get("geo_km_cut", 100.0):
            reasons.append(f"Large customer→merchant distance ~{int(d)} km")
    except:
        pass

    return reasons[:3]  # top-3

def haversine(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2): return None
    from math import radians, sin, cos, asin, sqrt
    R = 6371.0
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

def build_globals(df: pd.DataFrame):
    amt_p95 = float(df["amt"].quantile(0.95)) if "amt" in df else 500.0
    cat_counts = df["category"].value_counts(normalize=True) if "category" in df else pd.Series(dtype=float)
    rare_cats = set(cat_counts[cat_counts < 0.01].index)
    return {"amt_p95": amt_p95, "rare_cats": rare_cats, "geo_km_cut": 100.0}

# Simple counterfactual: reduce amount until risk < cut
def counterfactual_amount(row_dict, model, preprocessor, cut=0.5, step=0.9, min_amt=1.0):
    import copy, pandas as pd, numpy as np
    row = copy.deepcopy(row_dict)
    base_amt = row.get("amt", 0.0)
    for _ in range(12):
        X = pd.DataFrame([row])
        p = model.predict_proba(X)[:,1][0]
        if p < cut:
            return {"new_amt": row["amt"], "risk": float(p)}
        row["amt"] = max(min_amt, row["amt"]*step)
    return {"new_amt": row["amt"], "risk": float(p)}
