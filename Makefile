# .PHONY: setup features train eval serve ui demo

# setup:
# 	pip install -r requirements.txt

# features:
# 	python3 -m src.data_prep --in data/raw/transactions.csv --out data/features/train.parquet

# train:
# 	python3 -m src.detectors.tabular --train data/features/train.parquet --models_dir models

# eval:
# 	python3 -m src.fuse --eval data/features/train.parquet --models_dir models --report reports

# serve:
# 	uvicorn src.serve:app --host 0.0.0.0 --port 8000 --workers 1

# ui:
# 	streamlit run app/streamlit_app.py

# demo:
# 	docker compose -f docker/compose.yml up --build

# graph-train:
# 	python3 -m src.graph --train data/features/train.parquet --models_dir models

# graph-score:
# 	python3 -m src.graph --eval data/features/train.parquet --models_dir models --out_csv reports/graph_scores.csv

# seq-features:
# 	python3 -m src.sequence --in_parquet data/features/train.parquet --out_parquet data/features/train_seq.parquet

# fuse-all:
# 	python3 -m src.fuse --eval data/features/train_seq.parquet --models_dir models --report reports --graph_csv reports/graph_scores.csv

# monitor:
# 	python3 -m src.monitoring --ref reports/scores_ref.csv --cur reports/scores.csv --out reports/drift.md

# Makefile - using the smaller, cleaned dataset for a fast test run
.PHONY: setup features train eval serve ui demo graph-train graph-score seq-features fuse-all monitor

setup:
	pip install -r requirements.txt

features:
	# This line now points to your reduced dataset
	python3 -m src.data_prep --in data/raw/ReducedFraudDatset.csv --out data/features/train.parquet

train:
	python3 -m src.detectors.tabular --train data/features/train.parquet --models_dir models

eval:
	python3 -m src.fuse --eval data/features/train.parquet --models_dir models --report reports

serve:
	uvicorn src.serve:app --host 0.0.0.0 --port 8000 --workers 1

ui:
	streamlit run app/streamlit_app.py

demo:
	docker compose -f docker/compose.yml up --build

# --- NEW COMMANDS ---
graph-train:
	python3 -m src.graph --train data/features/train.parquet --models_dir models

graph-score:
	python3 -m src.graph --eval data/features/train.parquet --models_dir models --out_csv reports/graph_scores.csv

seq-features:
	python3 -m src.sequence --in_parquet data/features/train.parquet --out_parquet data/features/train_seq.parquet

fuse-all:
	python3 -m src.fuse --eval data/features/train_seq.parquet --models_dir models --report reports --graph_csv reports/graph_scores.csv

monitor:
	python3 -m src.monitoring --ref reports/scores_ref.csv --cur reports/scores.csv --out reports/drift.md