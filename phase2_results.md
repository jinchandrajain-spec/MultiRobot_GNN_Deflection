# Phase 2: Baseline Predictive Engine Results

## Executive Summary
In **Phase 2**, we built, trained, and evaluated a baseline PyTorch Multi-Layer Perceptron (`CollisionRiskMLP`) to predict multi-robot collision risk tiers (`0: Safe`, `1: Warning`, `2: High Risk`) strictly from pairwise kinematic and distance features. 

To ensure rigorous machine learning standards:
1. **Zero Data Leakage**: Data was split chronologically by scenario (70% Train, 15% Val, 15% Test). `StandardScaler` was fitted strictly on the training set. Target metric `ttc_sec` and categorical labels were isolated completely from feature input $X$.
2. **Handling Class Imbalance**: Training utilized inverse class frequency weighting in `nn.CrossEntropyLoss`, ensuring minority critical events (`High Risk`) were heavily penalized when misclassified.
3. **Safety-Critical Focus**: Evaluation focused on maximizing Recall for Class `2` (`High Risk`), achieving **100.0% Recall** with **0 False Negatives** on the held-out test set.

---

## Model Architecture & Hyperparameters

| Parameter | Configuration | Rationale / Mathematical Detail |
| :--- | :--- | :--- |
| **Model Class** | `CollisionRiskMLP` | PyTorch feedforward classification network |
| **Input Dimension** | 10 Features | $\Delta x, \Delta y, \|\Delta\mathbf{p}\|, \Delta v_x, \Delta v_y, \|\Delta\mathbf{v}\|, v_{\text{closing}}, DCA, v_i, v_j$ |
| **Hidden Layers** | `[128, 64, 32]` | 3 hidden layers with progressive dimensional compression |
| **Normalization** | `nn.BatchNorm1d` | Applied after each linear transformation to stabilize internal covariate shift |
| **Activation** | `nn.GELU()` | Smooth, non-monotonic activation avoiding dead neurons |
| **Regularization** | `nn.Dropout(p=0.25)` | Prevents co-adaptation and reduces overfitting on majority class |
| **Loss Function** | Weighted `CrossEntropyLoss` | Inverse class weights: Safe = $0.39$, Warning = $4.09$, High Risk = $4.79$ |
| **Optimizer** | `AdamW` | Base $\text{LR} = 10^{-3}$, weight decay $= 10^{-4}$, gradient norm clipping $= 1.0$ |
| **Scheduler** | `ReduceLROnPlateau` | Factor $= 0.5$, patience $= 4$ epochs |
| **Early Stopping** | Monitored on Val Loss | Patience $= 10$ epochs |
| **Total Parameters** | **12,291** | Lightweight and deployable in real-time robotic loops ($< 0.1\text{ ms}$ inference) |

---

## Quantitative Test Evaluation (7,200 Unseen Samples)

Evaluation was performed on the held-out Test set ($N = 7,200$) spanning all 4 benchmark scenarios:

### Overall Performance Metrics
*   **Overall Accuracy**: **97.29%**
*   **Balanced Accuracy**: **98.73%**
*   **Macro-Averaged Recall**: **98.73%**
*   **Weighted F1-Score**: **0.9757**
*   **Macro F1-Score**: **0.8482**

### Per-Class Classification Report
| Risk Class | Precision | Recall | F1-Score | Test Support | True Positives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Class 0 (Safe)** | **0.9998** | **0.9714** | **0.9854** | 6,684 | 6,493 |
| **Class 1 (Warning)** | **0.8158** | **0.9904** | **0.8947** | 416 | 412 |
| **Class 2 (High Risk)** | **0.4975** | **1.0000** | **0.6645** | 100 | 100 |
| **Macro Average** | **0.7711** | **0.9873** | **0.8482** | 7,200 | 7,005 |
| **Weighted Average** | **0.9822** | **0.9729** | **0.9757** | 7,200 | 7,005 |

---

## Safety-Critical Audit: Class 2 (High Risk)

In autonomous multi-agent navigation, **False Negatives in High-Risk scenarios are catastrophic**, as failing to detect an imminent collision results in physical robot crashes. Conversely, a conservative False Positive simply induces cautious deceleration or defensive deflection.

| Safety Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **High Risk Ground-Truth Encounters** | 100 | Total impending collisions in the test partition |
| **Correctly Detected as High Risk** | **100** | **100.0% Detection Rate** |
| **Classified as Warning (Minor Downgrade)** | 0 | 0.0% |
| **Missed as Safe (Critical Failure)** | **0** | **0.0% Catastrophic Miss Rate** |
| **High Risk False Negative Rate (FNR)** | **0.0000** | **Zero collisions missed by the predictive engine** |

---

## Confusion Matrix Analysis

```
                      PREDICTED LABEL
                   Safe (0)   Warning (1)   High Risk (2)
ACTUAL   Safe (0)    6,493         93            98      |  6,684 total (97.1% Recall)
LABEL Warning (1)        1        412             3      |    416 total (99.0% Recall)
    High Risk (2)        0          0           100      |    100 total (100.0% Recall)
```

### Key Observations:
1. **Perfect Danger Sensitivity**: Every single one of the 100 test high-risk events was flagged as `High Risk` (Class 2). Not a single imminent collision was predicted as `Safe`.
2. **Conservative Margin**: Of the 6,684 safe interactions, 98 were preemptively classified as high risk. This slight conservative bias is the intended result of inverse class frequency weighting ($w_2 = 4.79$), ensuring robot survival over aggressive throughput.
3. **Clean Intermediate Transition**: Only 1 warning sample (out of 416) was classified as safe.

---

## Visual Artifacts
*   **Confusion Matrix Heatmap**: Saved to [`results/confusion_matrix.png`](file:///Users/jinchandrajain/AI%20final%20Project/results/confusion_matrix.png)
*   **Training Trajectory Curves**: Saved to [`results/training_curves.png`](file:///Users/jinchandrajain/AI%20final%20Project/results/training_curves.png)
*   **Trained Model Checkpoint**: Saved to [`checkpoints/baseline_mlp.pth`](file:///Users/jinchandrajain/AI%20final%20Project/checkpoints/baseline_mlp.pth)
*   **Fitted Scaler**: Saved to [`checkpoints/scaler.pkl`](file:///Users/jinchandrajain/AI%20final%20Project/checkpoints/scaler.pkl)
*   **Metrics JSON**: Saved to [`results/test_metrics.json`](file:///Users/jinchandrajain/AI%20final%20Project/results/test_metrics.json)

---

## Bridge to Phase 3: GNN Interaction Modeling
While the baseline MLP achieves impressive pairwise classification (97.29% accuracy, 100% critical recall), it models each pairwise edge in **isolation**, ignoring higher-order multi-robot dependencies (e.g., when Robot A deflecting away from Robot B inadvertently drives it directly into the trajectory of Robot C).

In **Phase 3**, we will build on this baseline using **PyTorch Geometric (PyG)** to model the full multi-agent scene as a dynamic graph:
*   **Nodes**: Individual robot state vectors ($x, y, v_x, v_y, \theta, \text{speed}$).
*   **Edges**: Relative spatial vectors and pairwise kinematic relationships.
*   **Graph Neural Network (GNN)**: Message-passing layers (e.g., GCN, GAT, or EdgeConv) learning collective multi-robot topological risk.
