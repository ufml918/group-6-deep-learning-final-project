"""
Minimal Local Link Prediction Pipeline (Torch-only)
- Multi-hot feature encoding
- Simple message passing (mean aggregation)
- Dot-product decoder
- Train/test split with negative sampling
- Metrics: Accuracy, ROC-AUC, AP
"""

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score
import os


# =========================
# CONFIG (Set your local data folder here)
# =========================
DATA_DIR = r"./data"    # <-- Put your CSVs here

DRUGS_CSV    = os.path.join(DATA_DIR, "drugsInfo.csv")
DISEASES_CSV = os.path.join(DATA_DIR, "diseasesInfo.csv")
MAPPING_CSV  = os.path.join(DATA_DIR, "mapping.csv")


# =========================
# Device (CPU or GPU)
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")


# =========================
# Feature Engineering
# =========================
def parse_listish(x):
    """Convert NaN/'[a,b]'/'foo' → python list."""
    if x is None or pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            try:
                v = ast.literal_eval(x)
                return v if isinstance(v, list) else [v]
            except:
                return [x]
        return [x]
    return []


def encode_multi_hot(df, columns):
    """Multi-hot encode selected columns."""
    all_labels = set()
    for c in columns:
        df[c] = df[c].apply(parse_listish)
        for lst in df[c]:
            all_labels.update(lst)

    classes = sorted(all_labels)
    mlb = MultiLabelBinarizer(classes=classes)

    mats = [mlb.fit_transform(df[c]) for c in columns]
    X = np.concatenate(mats, axis=1).astype(np.float32)
    return torch.from_numpy(X)


# =========================
# Mean Aggregation Layer
# =========================
class MeanAggLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, neighbors):
        # neighbors: list[list[int]]
        N = x.size(0)
        neigh_mean = torch.zeros_like(x)

        for i in range(N):
            neigh = neighbors[i]
            if neigh:
                neigh_mean[i] = x[neigh].mean(dim=0)

        h = x + neigh_mean
        return F.relu(self.lin(h))


# =========================
# Encoder & Decoder
# =========================
class MPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = MeanAggLayer(in_dim, hidden_dim)
        self.l2 = MeanAggLayer(hidden_dim, out_dim)

    def forward(self, x, neighbors):
        h = self.l1(x, neighbors)
        z = self.l2(h, neighbors)
        return z


class DotDecoder(nn.Module):
    def forward(self, z, edges):
        u, v = edges
        score = (z[u] * z[v]).sum(dim=1)
        return torch.sigmoid(score)


# =========================
# Negative sampling
# =========================
def sample_negatives(num, N_drugs, N_nodes, pos_set, rng):
    neg = []
    while len(neg) < num:
        u = int(rng.integers(0, N_drugs))
        v = int(rng.integers(N_drugs, N_nodes))
        if (u, v) not in pos_set:
            neg.append((u, v))
    return torch.tensor(neg, dtype=torch.long).T


def build_adj(N, edges):
    neighbors = [[] for _ in range(N)]
    u, v = edges
    for ui, vi in zip(u.tolist(), v.tolist()):
        neighbors[ui].append(vi)
        neighbors[vi].append(ui)
    return neighbors


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Load CSVs
    # ---------------------------------------------------------------
    print("Loading CSV files...")
    df_drugs = pd.read_csv(DRUGS_CSV)
    df_diseases = pd.read_csv(DISEASES_CSV)
    df_map = pd.read_csv(MAPPING_CSV)

    # ---------------------------------------------------------------
    # Feature encoding
    # ---------------------------------------------------------------
    X_drug    = encode_multi_hot(df_drugs,    ["DrugTarget", "DrugCategories"])
    X_disease = encode_multi_hot(df_diseases, ["SlimMapping", "PathwayNames"])

    print(X_drug)

    N_drugs, D1 = X_drug.shape
    N_diseases, D2 = X_disease.shape

    IN_DIM = max(D1, D2)

    X_drug = F.pad(X_drug, (0, IN_DIM - D1))
    X_disease = F.pad(X_disease, (0, IN_DIM - D2))

    X = torch.cat([X_drug, X_disease], dim=0).to(DEVICE)
    N_nodes = X.size(0)

    # ---------------------------------------------------------------
    # Build node index maps
    # ---------------------------------------------------------------
    drug_to_idx = {d: i for i, d in enumerate(df_drugs["DrugID"])}
    disease_to_idx = {d: i + N_drugs for i, d in enumerate(df_diseases["DiseaseID"])}

    valid = df_map[
        df_map["DrugID"].isin(drug_to_idx)
        & df_map["DiseaseID"].isin(disease_to_idx)
    ]

    src = [drug_to_idx[d] for d in valid["DrugID"]]
    dst = [disease_to_idx[d] for d in valid["DiseaseID"]]

    pos_edges = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)
    M = pos_edges.size(1)

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(M), test_size=0.2, random_state=42
    )

    train_pos = pos_edges[:, train_idx]
    test_pos = pos_edges[:, test_idx]

    # Build adjacency only from training positives
    train_adj = build_adj(N_nodes, train_pos.cpu())

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    HIDDEN_DIM = 64
    OUT_DIM = 32

    encoder = MPEncoder(IN_DIM, HIDDEN_DIM, OUT_DIM).to(DEVICE)
    decoder = DotDecoder()

    opt = torch.optim.Adam(encoder.parameters(), lr=1e-2)
    loss_fn = nn.BCELoss()
    rng = np.random.default_rng(42)

    pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    print("Training...")
    EPOCHS = 50
    TH = 0.5

    for ep in range(1, EPOCHS + 1):
        encoder.train()
        opt.zero_grad()

        # fresh negatives each epoch
        train_neg = sample_negatives(
            len(train_idx), N_drugs, N_nodes, pos_set, rng
        ).to(DEVICE)

        Z = encoder(X, train_adj)

        edges = torch.cat([train_pos, train_neg], dim=1)
        labels = torch.cat([
            torch.ones(train_pos.size(1)),
            torch.zeros(train_neg.size(1)),
        ]).to(DEVICE)

        preds = decoder(Z, edges)
        loss = loss_fn(preds, labels)
        loss.backward()
        opt.step()

        if ep % 10 == 0 or ep == 1:
            acc = ((preds >= TH).float() == labels).float().mean().item()
            print(f"Epoch {ep:03d} | Loss={loss.item():.4f} | Train Acc={acc:.3f}")

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    encoder.eval()
    with torch.no_grad():
        Z = encoder(X, train_adj)

        test_neg = sample_negatives(
            len(test_idx), N_drugs, N_nodes, pos_set, rng
        ).to(DEVICE)

        test_edges = torch.cat([test_pos, test_neg], dim=1)
        y_true = torch.cat([
            torch.ones(test_pos.size(1)),
            torch.zeros(test_neg.size(1)),
        ]).cpu().numpy()

        p = decoder(Z, test_edges).cpu().numpy()

        pred_bin = (p >= TH).astype(int)
        acc = (pred_bin == y_true).mean()
        roc = roc_auc_score(y_true, p)
        ap = average_precision_score(y_true, p)

        print("\n====== TEST RESULTS ======")
        print(f"Accuracy : {acc:.4f}")
        print(f"ROC-AUC  : {roc:.4f}")
        print(f"AP Score : {ap:.4f}\n")

        print("First 10 predictions:")
        print("Prob :", np.round(p[:10], 4).tolist())
        print("Pred :", pred_bin[:10].tolist())
        print("True :", y_true[:10].tolist())
