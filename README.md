# group-6-deep-learning-final-project
## **📌 Overview**

This project implements a **minimal link-prediction pipeline** for a heterogeneous graph consisting of:

- **Drug nodes**
- **Disease nodes**
- **Edges indicating known Drug ↔ Disease associations**

The pipeline uses **PyTorch only** (no PyTorch Geometric) and includes:

- Multi-hot feature encoding for each drug/disease
- A simple 2-layer **message passing encoder (mean aggregation)**
- **Dot-product decoder** to score links
- **Negative sampling**
- Train/Test split + evaluation metrics (Accuracy, ROC-AUC, Average Precision)

This serves as a clean and easily extensible baseline for drug-disease prediction tasks.

---

## **📁 Project Structure & Data Requirements**

Place your CSV files under:

./data/
    drugsInfo.csv
    diseasesInfo.csv
    mapping.csv

Where:

### **drugsInfo.csv**
Required columns:
- DrugID
- DrugTarget (list-like string)
- DrugCategories (list-like string)

### **diseasesInfo.csv**
Required columns:
- DiseaseID
- SlimMapping (list-like string)
- PathwayNames (list-like string)

### **mapping.csv**
Defines positive links:
- DrugID
- DiseaseID

All list-like strings can be:
"[A, B]"   "[X]"   "Y"   or empty
The code automatically parses these into Python lists.

---

## **🧠 Model Architecture**

### **1. Multi-Hot Feature Encoding**
Each selected column is converted to a multi-hot vector using a MultiLabelBinarizer.

Example:
If "DrugCategories" = ["antibiotic", "antifungal"]
→ the vector activates those two indices.

Both drugs and diseases are padded to the same dimension and concatenated into a unified **node-feature matrix**.

---

### **2. Graph Construction**
Nodes:
- 0 … N_drugs-1 → drug nodes
- N_drugs … N_nodes-1 → disease nodes

Edges:
- Only **training positive edges** are used to build adjacency.

---

### **3. Message Passing Encoder**
Two layers of **Mean Aggregation**:

h = x + mean(neighbors)
h = ReLU(Linear(h))

This encoder learns latent embeddings for each node based solely on its features and neighborhood.

---

### **4. Dot-Product Decoder**
To score an edge:

score = sigmoid( z_u · z_v )

Links with higher values are more likely to represent a true drug–disease association.

---

### **5. Negative Sampling**
For each positive edge, a random **Drug → Disease** pair is sampled that is *not* in the known mapping.
Used both in training and testing.

---

## **📈 Training & Evaluation**

The script trains for **50 epochs** using:
- Loss: **Binary Cross-Entropy**
- Optimizer: **Adam**
- Threshold = 0.5 for binary accuracy

Evaluation metrics include:
- **Accuracy**
- **ROC-AUC**
- **Average Precision (AP)**

First 10 prediction scores are also printed for quick inspection.

---

## **🚀 How to Run**

### **1. Install dependencies**
pip install torch pandas scikit-learn numpy

### **2. Put all CSVs into** ./data/

### **3. Run the script**
python gnn.py

If a CUDA GPU is available, it will be used automatically:
Running on device: cuda

---

## **📊 Example Output**
Typical console output:

Epoch 010 | Loss=0.52 | Train Acc=0.74
Epoch 020 | Loss=0.41 | Train Acc=0.80
...

====== TEST RESULTS ======
Accuracy : 0.78
ROC-AUC  : 0.85
AP Score : 0.83

---

## **🧩 Customization**
You can easily modify:
- Feature columns
- Aggregation strategy
- Hidden / output dimensions
- Sampling ratio
- Decoder type (MLP, bilinear, etc.)

This script is intentionally small and interpretable, making it ideal for experimentation or educational purposes.

---

## **📘 Why This Baseline is Useful**
This project demonstrates:
- How to run link prediction **without** heavy frameworks (e.g., PyG, DGL)
- How to handle heterogeneous drug/disease data
- How a graph encoder + decoder pipeline works
- How negative sampling operates in practice
- How to evaluate link prediction models properly

Perfect for research prototypes or classroom demonstration.