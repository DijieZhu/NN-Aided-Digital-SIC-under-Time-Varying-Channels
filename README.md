# NN-Aided-Digital-SIC-under-Time-Varying-Channels

# NN-Aided Digital Self-Interference Cancellation under Time-Varying Channels

**Timeline:** Sep 2024 – Jan 2025  
**Instructor:** Prof. Ian Roberts, UCLA  
**Domains:** Neural Networks · Signal Processing · Communication Systems  
**Languages:** MATLAB · Python  

---

## 📖 Overview

In in-band full-duplex (IBFD) radio, self-interference (SI) from the transmitter can overwhelm the receiver chain. Traditional digital SIC methods assume a static channel and either incur high complexity (memory-polynomial) or require frame-wise retraining (NN-based). This project develops an MLP-based neural network with **linear preprocessing features** that:

- **Encodes time-varying channel dynamics** without per-frame retraining  
- **Cuts computational cost by ∼ 75%** vs. full memory-polynomial  
- Achieves **mean SIC gain of 6.93 dB** (σ² = 1.21 dB²) under Jakes’ fading  
- Outperforms adaptive MP and residual-NN baselines (variance ↓ 30×)  
- Runs in real time on MATLAB/Python simulation platforms  

---

## ✨ Features

- **MLP-based SIC**  
  - Linear-feature preprocessing fuses channel estimates and delayed samples  
  - No per-frame retraining required  
- **Efficient channel encoding**  
  - First-order memory polynomial for coefficient extraction  
  - Real/imaginary splits as NN inputs  
- **Physics-based evaluation**  
  - Jakes’ fading model for realistic mobile channels  
  - MATLAB physics simulations + Python result analysis  
- **Benchmarking**  
  - Adaptive memory-polynomial (MP SIC)  
  - Simple MLP and Residual NN baselines  
  - Metrics: mean SIC gain, variance, compute cost

---
## ⚙️ Prerequisites

- MATLAB R2022a (or later) with Signal Processing Toolbox  
- Python 3.8+  
  - NumPy  
  - Matplotlib  
  - SciPy (optional, for additional stats)  

---

## 🚀 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/nn-digital-sic.git
   cd nn-digital-sic

