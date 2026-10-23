# Multi-Robot Collision-Risk Prediction & Adaptive Deflection (`MultiRobot_GNN_Deflection`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-torch__geometric-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A full-stack, closed-loop AI navigation framework combining **analytical Time-to-Collision (TTC) forecasting**, **Attentive Graph Neural Networks ($\text{GATv2}$)**, and **real-time adaptive path deflection control** for autonomous multi-robot fleets in shared environments.

---

## 🌟 Key Features

1. **Analytical Ground-Truth Kinematics (Phase 1):**
   - Closed-form quadratic Time-to-Collision (TTC) derivation ($a t^2 + b t + c \le 0$) based on relative kinematics.
   - Closest Point of Approach ($t_{\text{CPA}}$), Distance of Closest Approach ($DCA$), and scalar closing velocity ($v_{\text{closing}}$).
   - 3-tier discrete risk categorization: `Safe (0)`, `Warning (1)`, and `High Risk (2)`.
   - 4 standardized benchmark conflict scenarios: *4-Way Cross Intersection*, *Head-On Converge Conflict*, *Chokepoint Bottleneck Transit*, and *Continuous Dynamic Roam*.

2. **Baseline Predictive Engine (Phase 2):**
   - 10D tabular kinematic feature classifier (`CollisionRiskMLP`, 12,291 parameters).
   - Weighted Cross-Entropy Loss to counter heavy class imbalance ($w_{\text{safe}}=0.39, w_{\text{warn}}=4.09, w_{\text{high}}=4.79$).
   - **100.0% Recall (0 False Negatives)** on critical high-risk collision events ($N=7,200$ test set).

3. **Relational Graph Attention Modeling (Phase 3):**
   - Dynamic multi-robot scene graphs: Nodes = robots (6D state), Edges = pairwise interactions (8D relative kinematics).
   - Permutation-equivariant dual-layer $\text{GATv2Conv}$ network (69,027 parameters) with multi-head attention conditioned on closing velocity and distance.
   - Eliminates multi-agent blindness by evaluating global topological risk ($0.42\text{ ms}$ inference).

4. **Closed-Loop Real-Time Adaptive Deflection Controller (Phase 4):**
   - **Anti-Chatter State Machine:** Evaluates lateral clearance and commits to a persistent avoidance side (starboard-passing convention) with a $+1200.0$ side-switching penalty (**strictly 0 direction flips**).
   - **Forward-Biased Candidate Generation:** Generates bypass steering vectors ($22^\circ, 32^\circ, 45^\circ, 60^\circ, 75^\circ, 90^\circ$ offsets) to preserve forward mission pace.
   - **Multi-Objective Cost Rollout:** Evaluates static obstacles, arena wall margins, surrounding robots, and goal alignment over a $1.5\text{ s}$ horizon.
   - **Hysteresis & Recovery:** $0.40\text{ s}$ hysteresis persistence with $0.35\text{ s}$ smooth exponential recovery decay.

5. **Cyber-Dark Visualization & HUD UI:**
   - Real-time Pygame dashboard with dynamic glowing cyan deflected trajectories, dimmed intended route overlays, mini-radar, velocity vectors, compass headings, and live avoidance telemetry.

---

## 📊 Benchmark Evaluation (1,000 Paired Episodes)

In a paired 500 Baseline vs. 500 Adaptive evaluation under identical deterministic seeds, initial spawns, velocities, target waypoints, and obstacle layouts:

| Metric | Baseline (Unassisted) | Adaptive (GNN Active) | Performance Differential |
| :--- | :---: | :---: | :---: |
| **Collision Episodes** | **75.00%** (375/500) | **0.40%** (2/500) | **-99.47% Relative Reduction** |
| **Total Physical Collision Events** | 8,267 impacts | **2 edge grazes** | **-99.98% Impact Elimination** |
| **Mean Collisions per Episode** | 16.53 ± 18.79 | **0.004 ± 0.06** | **Near-Zero Collision Rate** |
| **Goal Navigation Success Rate** | 23.40% (117/500) | **39.40%** (197/500) | **+68.38% Improvement** |
| **Median Min Inter-Robot Clearance** | 33.80 px *(penetration)* | **52.40 px** *(safe buffer)* | **+55.03% Safety Margin** |
| **Mean Cumulative Path Length** | 648.39 px | 661.99 px | **+2.10% Negligible Overhead** |
| **Closed-Loop Processing Speed** | 119,416 FPS (raw) | **309.3 FPS** | **5.15× Faster than 60 Hz** |

*Note: In 4-Way Cross and Head-On scenarios, the Adaptive mode paths were actually **shorter** (-7.1% and -8.0%) than Baseline, because baseline robots ricocheted violently off each other, whereas adaptive robots executed smooth bypass trajectories.*

---

## 🏗️ Repository Architecture

```
MultiRobot_GNN_Deflection/
├── checkpoints/                          # Pretrained neural network weights & scalers
│   ├── baseline_mlp.pth                 # Phase 2 trained MLP weights (166 KB)
│   ├── gnn_interaction_model.pth        # Phase 3 trained GATv2 GNN weights (852 KB)
│   ├── gnn_scalers.pkl                  # Fitted feature standardizer for GNN
│   └── scaler.pkl                       # Fitted feature standardizer for MLP
├── dataset/                              # Simulation dataset directory
│   └── .gitkeep
├── results/                              # Quantitative benchmark logs & visual artifacts
│   ├── confusion_matrix.png             # Phase 2 baseline confusion matrix
│   ├── gnn_confusion_matrix.png         # Phase 3 GNN confusion matrix
│   ├── gnn_training_curves.png          # GNN loss & accuracy trajectory
│   ├── phase4_benchmark.csv             # Full 1,000-run raw benchmark dataset
│   └── training_curves.png              # Baseline MLP training curves
├── src/                                  # Core source library
│   ├── adaptive_deflection.py           # Closed-loop candidate deflection controller & state machine
│   ├── benchmark_phase4.py              # Headless 1,000-episode paired benchmark runner
│   ├── data_loader.py                   # Anti-leakage chronological split & DataLoader pipeline
│   ├── evaluate.py                      # Baseline MLP test evaluation suite
│   ├── evaluate_gnn.py                  # GNN test evaluation suite
│   ├── generate_dataset.py              # Multi-scenario automated data generation engine
│   ├── gnn_inference_controller.py      # Singleton real-time GNN inference interface
│   ├── graph_builder.py                 # PyTorch Geometric scene graph construction
│   ├── model_baseline.py                # PyTorch CollisionRiskMLP architecture
│   ├── model_gnn.py                     # PyTorch Geometric CollisionRiskGNN (GATv2) architecture
│   ├── train_gnn.py                     # GNN training script with class-weighted CrossEntropy
│   └── train_model.py                   # Baseline MLP training script
├── tests/                                # Comprehensive unit & regression test suite (43 tests)
│   ├── test_adaptive_deflection.py      # Evasion candidate & anti-chatter unit tests
│   ├── test_gnn_inference_controller.py # Real-time controller unit tests
│   ├── test_phase2_model.py             # MLP architecture tests
│   ├── test_phase3_gnn.py               # Graph builder & GATv2Conv tests
│   ├── test_phase4_stage_d.py           # Visual overlay & HUD event tests
│   └── test_simulation.py               # Kinematics & collision resolution tests
├── config.py                             # Global simulation constants & physical parameters
├── data_collector.py                     # Analytical quadratic TTC extraction & logging
├── main.py                               # Interactive visual application entry point
├── robot_state.py                        # Robot physics, obstacle geometries & scenarios
├── ui_engine.py                          # High-fidelity Pygame rendering & HUD engine
└── phase4_results.md                     # Comprehensive experimental findings & analysis
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/jinchandrajain-spec/MultiRobot_GNN_Deflection.git
cd MultiRobot_GNN_Deflection
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pygame-ce numpy pandas scikit-learn
```

### 2. Launch Interactive Simulation
Launch the real-time simulation with active GNN inference and adaptive deflection:
```bash
# Launch interactive visualizer (Scenario 1: 4-Way Cross Intersection)
python3 main.py

# Launch Head-On Converge Conflict
python3 main.py --scenario 2

# Launch with 4 robots
python3 main.py --robots 4
```

**Interactive Controls:**
- `[SPACE]` : Pause / Resume simulation.
- `[D]`     : Toggle AI Adaptive Deflection ON / OFF in real time.
- `[R]`     : Toggle data recording.
- `[1 - 4]` : Switch scenarios in real time.
- `[ESC]`   : Quit.

### 3. Run Test Suite
Verify all 43 targeted unit and integration tests:
```bash
python3 -m unittest discover tests
```

### 4. Run Headless Benchmark
Reproduce the paired 1,000-episode benchmark:
```bash
# Rapid 10-episode smoke verification
python3 src/benchmark_phase4.py --episodes 5 --smoke

# Full 1,000-episode benchmark (500 Baseline vs 500 Adaptive)
python3 src/benchmark_phase4.py --episodes 500 --output results/phase4_benchmark.csv
```

---

## 👥 Authors
- **Jinchandra Jain** (PRN: 125BTAI1032)
- **Kavya Hindocha** (PRN: 125BTAI1053)
- **Guide:** Dr. Manish Rana

*Department of Artificial Intelligence and Data Science, Shah & Anchor Kutchhi Engineering College, Chembur, Mumbai, Maharashtra, India.*

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
