import argparse, pandas as pd, numpy as np, joblib, os, networkx as nx
from node2vec import Node2Vec
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

CARD, MERCH = "cc_num", "merchant"   # adapt if your columns differ
TARGET = "is_fraud"

def build_graph(df: pd.DataFrame):
    G = nx.Graph()
    for _, r in df[[CARD, MERCH]].dropna().drop_duplicates().iterrows():
        G.add_node(f"C:{r[CARD]}", bipartite=0)
        G.add_node(f"M:{r[MERCH]}", bipartite=1)
        G.add_edge(f"C:{r[CARD]}", f"M:{r[MERCH]}")
    return G

def node2vec_embed(G: nx.Graph, dim=32, seed=42):
    n2v = Node2Vec(G, dimensions=dim, walk_length=20, num_walks=200, workers=1, seed=seed, quiet=True)
    model = n2v.fit(window=10, min_count=1, batch_words=64)
    return model

def edge_features(df, model, G):
    def get_vec(node):
        try: return model.wv.get_vector(node)
        except KeyError: return np.zeros(model.wv.vector_size, dtype=np.float32)

    feats = []
    for _, r in df.iterrows():
        c, m = f"C:{r.get(CARD)}", f"M:{r.get(MERCH)}"
        vc, vm = get_vec(c), get_vec(m)
        if np.all(vc==0) or np.all(vm==0): cos = 0.0
        else: cos = float(cosine_similarity([vc],[vm])[0,0])
        deg_c = float(G.degree(c)) if G.has_node(c) else 0.0
        deg_m = float(G.degree(m)) if G.has_node(m) else 0.0
        feats.append([cos, deg_c, deg_m])
    X = np.array(feats, dtype=np.float32)
    return pd.DataFrame(X, columns=["cos_card_merchant","deg_card","deg_merchant"])

def train_graph(train_path, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_parquet(train_path)
    # build graph on train only (avoid leakage)
    G = build_graph(df)
    emb = node2vec_embed(G, dim=32)
    Xg = edge_features(df, emb, G)
    clf = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    clf.fit(Xg)

    joblib.dump({"embed": emb, "graph": G, "clf": clf}, f"{models_dir}/graph.pkl")
    print("saved graph model ->", f"{models_dir}/graph.pkl")

def score_graph(models_dir, eval_path, out_csv=None):
    bundle = joblib.load(f"{models_dir}/graph.pkl")
    emb, G, clf = bundle["embed"], bundle["graph"], bundle["clf"]
    df = pd.read_parquet(eval_path)
    Xg = edge_features(df, emb, G)
    s = -clf.decision_function(Xg)  # higher = more anomalous
    s = (s - s.min())/(s.max()-s.min()+1e-8)
    if out_csv:
        pd.DataFrame({"graph_score": s}).to_csv(out_csv, index=False)
    return s

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", help="parquet with train rows")
    ap.add_argument("--eval")
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out_csv")
    args = ap.parse_args()
    if args.train: train_graph(args.train, args.models_dir)
    if args.eval: score_graph(args.models_dir, args.eval, args.out_csv)
