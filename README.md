# Multi-View Fraud Detection System

### A Powerful Classical Baseline with a Clear Roadmap for Quantum ML Integration
### AQVH 2025 Hackathon (ID: AQVH918)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](YOUR_GITHUB_REPO_LINK)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](YOUR_LINKEDIN_PROFILE_LINK)

## 🚀 Live Demo

**Experience the live application here:**
[**Fraud Detection Dashboard**](YOUR_STREAMLIT_APP_LINK)

*(You can add a screenshot of your running app here)*

---

## 🎯 The Problem

Financial fraud is evolving rapidly, with fraudsters using complex, coordinated schemes that are difficult for traditional, single-view models to detect. This results in billions of dollars in losses and erodes consumer trust. Our project for the **AQVH 2025 Hackathon** tackles this challenge by building a system capable of identifying anomalies in transaction data.

## ✨ Our Solution: A Powerful Multi-View Classical System

We have developed a powerful, real-time, multi-view fraud detection engine. It intelligently combines three robust classical machine learning models to create a highly accurate risk score. This system serves as a powerful **classical baseline**, proving the effectiveness of our architecture.

Our system analyzes transactions from three critical angles:
1.  **Tabular View:** What are the intrinsic properties of the transaction? (Amount, location, time, etc.)
2.  **Graph View:** Who is transacting with whom? (Analyzing the network of cards and merchants to spot fraud rings).
3.  **Sequence View:** Is this behavior normal for *this specific card*? (Detecting sudden spikes in spending or unusual purchase categories).

### 🔮 Future Work: The Roadmap to Quantum Advantage
Our **clear roadmap** involves integrating a **Quantum Machine Learning (QML) classifier** as a final decision layer. This will allow the system to tackle the most sophisticated fraud patterns that are invisible to even advanced classical methods, directly addressing the core theme of the hackathon.

---

## 🏛️ Project Architecture

Our system is built on a modular, production-ready architecture, ensuring scalability and maintainability.

```
fraudhack/
├── app/
│   └── streamlit_app.py     # The interactive Streamlit UI
├── configs/
│   └── model.yaml           # Central configuration for models & parameters
├── data/
│   ├── raw/                 # Original, unprocessed transaction data
│   └── features/            # Processed data ready for modeling
├── docker/
│   ├── Dockerfile.api       # Docker instructions for the backend API
│   ├── Dockerfile.ui        # Docker instructions for the frontend UI
│   └── compose.yml          # Docker Compose to orchestrate services
├── models/
│   ├── supervised.pkl       # Trained Random Forest classifier
│   ├── unsupervised.pkl     # Trained Isolation Forest anomaly detector
│   └── graph.pkl            # Trained Node2Vec and graph model bundle
├── reports/
│   ├── graph_scores.csv     # Anomaly scores from the graph model
│   └── scores.csv           # Final fused risk scores for evaluation
├── src/
│   ├── detectors/
│   │   └── tabular.py       # Trains the supervised and unsupervised models
│   ├── data_prep.py         # Cleans and prepares raw data
│   ├── features.py          # Defines feature preprocessing pipelines
│   ├── graph.py             # Builds the graph, learns embeddings, scores nodes
│   ├── sequence.py          # Generates behavioral "surprise" features
│   ├── fuse.py              # Fuses scores from all models into a final risk score
│   ├── serve.py             # FastAPI backend to serve the models via a REST API
│   └── monitoring.py        # Scripts to detect data drift (PSI/KS-test)
├── Makefile                 # Automation for setup, training, and deployment
├── requirements.txt         # All Python dependencies for the project
└── README.md                # You are here!
```

---

## 🛠️ How to Run Locally

Get the entire system running on your machine in just a few steps.

**1. Clone the Repository:**
```bash
git clone YOUR_GITHUB_REPO_LINK
cd fraudhack
```

**2. Set Up the Environment:**
This will install all necessary Python packages.
```bash
make setup
```

**3. Run the Full ML Pipeline:**
This sequence of commands will prepare the data, train all the models (tabular, graph, sequence), and fuse the results.
```bash
# 1. Prepare features from raw data
make features

# 2. Train the core tabular models
make train

# 3. Train the graph-based model
make graph-train
make graph-score

# 4. Generate sequential behavioral features
make seq-features

# 5. Fuse all scores into a final risk assessment
make fuse-all
```

**4. Launch the Application:**
You'll need two separate terminal windows for this.

*   **In Terminal 1 (Start the Backend API):**
    ```bash
    make serve
    ```
    *The API will be live at `http://localhost:8000`.*

*   **In Terminal 2 (Start the Frontend UI):**
    ```bash
    make ui
    ```
    *The Streamlit Dashboard will open automatically at `http://localhost:8501`.*

---

## 💡 Innovation & Key Features

-   **Multi-View Fusion:** Our system's core strength is its ability to combine tabular, graph, and sequence data to create a holistic risk profile, a significant improvement over single-view models.
-   **Roadmap to Quantum Advantage:** This project serves as the foundation for a next-generation hybrid system. Our modular design allows for the future integration of a QML model (e.g., using IBM Qiskit) to enhance detection capabilities for the most complex fraud cases.
-   **Production-Grade Code:** With a modular structure, Docker support, and Makefile automation, this project isn't just a script—it's a scalable, deployable application.
-   **Real-time & Interactive:** The FastAPI backend and Streamlit frontend provide a responsive, user-friendly experience for real-time scoring and analysis.