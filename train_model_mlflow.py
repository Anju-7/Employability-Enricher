"""
Same training pipeline as train_model.py, instrumented with MLflow so every
run is tracked (params, per-epoch loss, final model) and the resulting
model is registered in the MLflow Model Registry.

Run the tracking UI in a separate terminal to browse past runs:
    mlflow ui --backend-store-uri ./mlruns
Then open http://127.0.0.1:5000

Run a training job:
    python train_model_mlflow.py
Optional overrides:
    python train_model_mlflow.py --hidden-dim 256 --epochs 150 --lr 0.003
"""

import os

# Set Hugging Face cache location BEFORE importing transformers or sentence_transformers
os.environ["HF_HOME"] = "D:/PBL_MLOPS/hf_cache"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import argparse
import xml.etree.ElementTree as ET

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import mlflow
from mlflow.tracking import MlflowClient

def promote_model_if_qualified(run_id, current_loss, threshold=0.05):
    client = MlflowClient()
    model_name = "employability-deep-scorer"
    
    if current_loss <= threshold:
        print(f"[+] Loss benchmark met ({current_loss:.4f}). Registering model...")
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        
        # Promote newly registered version to Production
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"[SUCCESS] Model v{mv.version} promoted to PRODUCTION stage.")

EXPERIMENT_NAME = "employability-deep-scorer"
REGISTERED_MODEL_NAME = "employability-deep-scorer"

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)



# ==========================================
# 1. DEEP LEARNING MODEL DEFINITION (PyTorch)
# ==========================================
class EmployabilityDeepScorer(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super(EmployabilityDeepScorer, self).__init__()

        self.drift_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.history_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, resume_vec, drift_vec, history_vec):
        drift_features = self.drift_encoder(drift_vec)
        history_features = self.history_encoder(history_vec)
        fused_context = torch.cat((drift_features, history_features), dim=1)
        score = self.fusion_network(fused_context)
        return score

def promote_model_if_qualified(run_id, current_loss, threshold=0.05):
    client = MlflowClient()
    model_name = "employability-deep-scorer"
    
    if current_loss <= threshold:
        print(f"[+] Loss benchmark met ({current_loss:.4f}). Registering model...")
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        
        # Promote newly registered version to Production
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"[SUCCESS] Model v{mv.version} promoted to PRODUCTION stage.")

# ==========================================
# 2. DATA UTILITY FUNCTIONS FOR LOG FILES
# ==========================================
def load_xml_text_payload(xml_path):
    """Parses and returns a flat text string from raw xml tags."""
    if not os.path.exists(xml_path):
        return "Foundational data engineering architecture deployment pipeline metadata systems."

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        text_nodes = []
        for item in root.findall(".//TrendItem"):
            title = item.find("Title").text if item.find("Title") is not None else ""
            summary = item.find("SummaryPayload").text if item.find("SummaryPayload") is not None else ""
            text_nodes.append(f"{title} {summary}")
        return " ".join(text_nodes) if text_nodes else "Empty corpus placeholder context."
    except Exception:
        return "Backup standard engineering parameters."


# ==========================================
# 3. CORE PROCESSING AND EXECUTION LOOP
# ==========================================
def train_deep_alignment_model(hidden_dim=128, epochs=100, lr=0.005, log_every=20):
    print("[*] Launching MLOps Deep Learning Training Matrix...")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print(f"[*] MLflow run id: {run.info.run_id}")

        # --- Log the hyperparameters for this run up front ---
        mlflow.log_params(
            {
                "input_dim": 384,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "learning_rate": lr,
                "optimizer": "Adam",
                "loss_fn": "MSELoss",
            }
        )

        drift_text = load_xml_text_payload("todays_market_drift.xml")
        history_text = load_xml_text_payload("historical_market_intelligence.xml")

        # Log which data snapshot this run trained on (traceability)
        mlflow.log_text(drift_text, "inputs/drift_text.txt")
        mlflow.log_text(history_text, "inputs/history_text.txt")

        sample_resumes = [
            "Experienced ML Engineer python pytorch docker model deployment pipeline scikit-learn",
            "Frontend designer developer html css javascript responsive react bootstrap web UI layouts",
            "Cloud systems administrator aws infrastructure networking terraform kubernetes linux bash",
        ]
        ground_truth_targets = torch.tensor([[0.88], [0.35], [0.72]], dtype=torch.float32)

        print("[*] Initializing local base text embedding matrix layer...")
        base_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=r"D:\PBL_MLOPS\hf_cache")

        # Optimized NumPy to PyTorch conversion to bypass slow conversion warnings
        drift_emb = torch.from_numpy(base_model.encode(drift_text)).unsqueeze(0).float()
        history_emb = torch.from_numpy(base_model.encode(history_text)).unsqueeze(0).float()

        drift_batch = drift_emb.repeat(len(sample_resumes), 1)
        history_batch = history_emb.repeat(len(sample_resumes), 1)
        resume_batch = torch.tensor(
            np.array([base_model.encode(res) for res in sample_resumes]), dtype=torch.float32
        )

        model = EmployabilityDeepScorer(input_dim=384, hidden_dim=hidden_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        print("[*] Commencing Gradient Descent weight optimizations...")
        model.train()
        final_loss = None
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            predictions = model(resume_batch, drift_batch, history_batch)
            loss = criterion(predictions, ground_truth_targets)
            loss.backward()
            optimizer.step()

            final_loss = loss.item()
            # Log every epoch so the MLflow UI can chart the full curve
            mlflow.log_metric("train_loss", final_loss, step=epoch)

            if epoch % log_every == 0:
                print(f"    -> Epoch [{epoch}/{epochs}] | Current Feature Processing Loss: {final_loss:.6f}")

        mlflow.log_metric("final_loss", final_loss)

        print("\n[+] Training Complete! Saving weights matrix to disk...")
        weights_path = "deep_scorer_weights.pth"
        torch.save(model.state_dict(), weights_path)
        print(f"[SUCCESS] Model file weights saved cleanly as: {weights_path}")

        # Log the raw .pth artifact for downstream evaluation scripts
        mlflow.log_artifact(weights_path, artifact_path="weights")

        # Log & register the PyTorch model using pickle serialization format
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )

        print(f"[+] Run '{run.info.run_id}' logged under experiment '{EXPERIMENT_NAME}'.")
        print("[+] Launch `mlflow ui` to compare this run against previous ones.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the employability deep scorer with MLflow tracking.")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    train_deep_alignment_model(hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr)
