import argparse, joblib, pandas as pd, numpy as np, os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from src.features import basic_preprocessor, train_test, TARGET

def train_models(train_path, models_dir):
    df = pd.read_parquet(train_path)
    X_train, X_test, y_train, y_test = train_test(df, test_size=0.2)

    pre = basic_preprocessor()
    # Supervised
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    from sklearn.pipeline import Pipeline
    pipe_sup = Pipeline([("pre", pre), ("clf", clf)])
    pipe_sup.fit(X_train, y_train)
    p_sup = pipe_sup.predict_proba(X_test)[:,1]
    print("Supervised  ROC-AUC:", roc_auc_score(y_test, p_sup))
    print("Supervised  PR-AUC :", average_precision_score(y_test, p_sup))

    # Unsupervised (on X only)
    pre2 = basic_preprocessor()
    pipe_uns = Pipeline([("pre", pre2), ("if", IsolationForest(n_estimators=200, contamination=0.02, random_state=42))])
    pipe_uns.fit(X_train)
    # IsolationForest returns 1 (inlier) / -1 (outlier); convert to [0,1] score
    s_uns = -pipe_uns.decision_function(X_test)
    s_uns = (s_uns - s_uns.min()) / (s_uns.max()-s_uns.min() + 1e-8)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipe_sup, f"{models_dir}/supervised.pkl")
    joblib.dump(pipe_uns, f"{models_dir}/unsupervised.pkl")
    print("models saved ->", models_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--models_dir", required=True)
    args = ap.parse_args()
    train_models(args.train, args.models_dir)
