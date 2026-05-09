# ⚡ FP-DQL for Islanded Microgrid Voltage Regulation

### **Model-Less Forecast-Proactive Intelligent Control for Inverter in Islanded Microgrid**

📄 **IEEE ICICCS 2026 Paper:**  
👉 https://ieeexplore.ieee.org/document/11502612

📘 **Permanent Research Archive (Zenodo):**  
👉 [![DOI](https://zenodo.org/badge/1143300566.svg)](https://doi.org/10.5281/zenodo.18388844)

---

## 📌 Overview

This repository provides the **complete implementation** of the proposed:

> **Forecast-Proactive Deep Q-Learning (FP-DQL)**

for intelligent voltage regulation in **islanded inverter-dominated AC microgrids**.

The proposed controller directly regulates the inverter voltage reference using:

- ✅ Local measurements only  
- ✅ Model-less reinforcement learning  
- ✅ Forecast-proactive load awareness  
- ✅ No explicit network model  

The framework is specifically designed for:

- ⚡ Dynamic and unbalanced loads  
- 🔋 Renewable-energy-powered microgrids  
- 🧠 Intelligent inverter voltage control  
- 📉 Robust voltage regulation under uncertainty  

---

## 🏗️ System Architecture

![System Architecture](./Proposed_System.pdf)

The microgrid consists of:

- 800 V renewable DC source
- Grid-forming inverter
- 4-bus islanded AC microgrid
- Dynamic and unbalanced load profiles
- Forecast-proactive RL controller

---

## 🧠 Key Idea: FP-DQL

Traditional controllers depend on:
- Explicit system models
- Offline tuning
- Fixed operating assumptions

The proposed **FP-DQL** framework instead learns voltage regulation behavior directly from measurements.

### 🔹 State Vector

The RL agent observes:

```math
[V_1,V_2,V_3,V_4,I_{inv},P_{tot},h,\hat{P}_{t+1}]
```

including:
- Bus voltages
- Inverter current
- Total active power
- Time index
- Next-hour load forecast

---

### 🔹 Forecast-Proactive Feature

Unlike complex forecasting networks, FP-DQL uses:

✅ Lightweight next-hour scheduled load information

to improve anticipatory voltage regulation during:
- Peak demand periods
- Rapid load transitions
- Dynamic operating conditions

---

### 🔹 Control Action

The controller updates the inverter voltage reference:

```math
V_{ref,t+1}=V_{ref,t}+\Delta V_t
```

while maintaining:

```math
0.98 \leq V_{ref} \leq 1.02
```

---

## 🚀 Main Contributions

✔ Model-less RL voltage controller for islanded microgrids  
✔ Forecast-proactive reinforcement learning framework  
✔ No explicit electrical network model required  
✔ Voltage regulation under unbalanced dynamic loads  
✔ Stable operation within strict voltage limits  
✔ Practical alternative to conventional PID control  

---

## 🧪 Simulation Environment

### Microgrid Configuration

| Component | Description |
|---|---|
| DC Source | 800 V renewable interface |
| Inverter | Grid-forming VSI |
| Network | 4-bus islanded AC microgrid |
| Bus-1 Load | 10 kW fixed |
| Bus-2 Load | 12 kW fixed |
| Bus-3 Load | Dynamic 24-hour profile |
| Bus-4 Load | Dynamic 24-hour profile |

---

### Training Setup

| Parameter | Value |
|---|---|
| RL Method | Deep Q-Learning |
| Hidden Layers | 2 × 64 neurons |
| Activation | ReLU |
| Discount Factor | 0.98 |
| Exploration | ε-greedy |
| Time Step | 0.1 s |
| Training Type | Online randomized episodes |

---

## 📊 Compared Controllers

The repository includes comparison against:

| Controller | Type |
|---|---|
| PID | Conventional model-based |
| LSTM | Prediction-based learning |
| FP-DQL | Proposed model-less RL |

---

## 📈 Results Summary

The proposed FP-DQL controller achieves:

✅ Voltage regulation comparable to optimally tuned PID  
✅ Superior robustness under dynamic load changes  
✅ Stable operation without explicit system modeling  
✅ Strict voltage constraint satisfaction (0.98–1.02 pu)

### Quantitative Results

| Metric | PID | LSTM | Proposed FP-DQL |
|---|---|---|---|
| Max Voltage Deviation (pu) | 0.0113 | 0.0793 | **0.0124** |
| Mean Voltage Deviation (pu) | 0.0028 | 0.0541 | **0.0024** |
| Overcurrent Events | 0 | 0 | 0 |

---

## 📂 Repository Structure

```bash
├── notebooks/
│   └── main.ipynb
│
├── src/
│   ├── environment/
│   ├── controllers/
│   ├── training/
│   └── utils/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── sim_data/
│
├── Proposed_System.png
├── FP-DQL.pdf
└── README.md
```

---

## ▶ Quick Start

### Run Jupyter Notebook

```bash
notebooks/main.ipynb
```

### Run from Terminal

```bash
python -m src.run_experiment
```

---

## 📁 Outputs

Simulation outputs include:

- 📊 Voltage profiles
- 📉 Training convergence curves
- ⚡ Voltage deviation statistics
- 📂 Raw simulation data
- 📈 Performance comparison figures

Generated automatically in:

```bash
results/
```

---

## 🔬 Implementation Highlights

✔ Model-less reinforcement learning control  
✔ Forecast-proactive load awareness  
✔ Replay buffer + target network  
✔ Online learning with randomized daily profiles  
✔ Voltage constraint enforcement  
✔ Lightweight surrogate microgrid simulator  

---

## 🧩 Applications

This framework is suitable for:

- Renewable-energy microgrids
- Islanded AC systems
- Smart inverter control
- Autonomous voltage regulation
- AI-driven power electronics research

---

## ⭐ Research Significance

This work demonstrates that:

> Intelligent model-less RL controllers can achieve voltage regulation performance comparable to classical optimized PID control while avoiding explicit electrical network modeling.

The proposed FP-DQL framework bridges:

**Reinforcement Learning ⚡ Power Electronics ⚡ Smart Microgrids**

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{kamoona2026fpdql,
  title={Model-Less Forecast-Proactive Intelligent Control for Inverter in Islanded Microgrid},
  author={Kamoona, Mustafa A. and Mauricio, Juan Manuel},
  booktitle={2026 9th International Conference on Intelligent Computing and Control Systems (ICICCS)},
  year={2026},
  doi={10.1109/ICICCS67901.2026.11502612}
}
```

---

## 📄 Paper

📘 IEEE Xplore:  
https://ieeexplore.ieee.org/document/11502612

---

## 🔗 Data and Code Availability

The source code and simulation data are openly available through:

- GitHub Repository
- Zenodo Permanent Archive

Zenodo DOI:  
https://doi.org/10.5281/zenodo.18388844

---

## 👨‍💻 Authors

**Mustafa A. Kamoona**  
Department of Electrical Engineering  
Universidad de Sevilla, Spain

**Juan Manuel Mauricio**  
Department of Electrical Engineering  
Universidad de Sevilla, Spain
