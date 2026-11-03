# DNN Project 2 — Group 1: Attention Mechanism

**Course:** Deep Neural Networks  
**Semester:** 1, 2026–2027

---

## Group Members

| Student ID | Name | GitHub |
|------------|------|--------|
| 11247371 | Phạm Bá Việt | [@davvph](https://github.com/davvph) |
| 11247291 | Lê Huy Hoàng | [@helian16](https://github.com/helian16) |
| 11247289 | Lương Thế Hiếu | [@adcc17](https://github.com/adcc17) |
| — | Bé Thành Tiên | [@Tienachilles](https://github.com/Tienachilles) |
| 11247287 | Chử Vũ Thảo Hiền | [@dqchien](https://github.com/dqchien) |

---

## Project Overview

This repository contains our midterm work for the Deep Neural Networks course:

1. **RNN Exercise**: time series forecasting with recurrent neural networks on historical Google stock data.
2. **Research Assignment (Topic 2)**: the Attention Mechanism in deep learning.

---

## Part I: RNN Exercise — Stock Price Forecasting

### Objective
Develop an RNN-based model to predict the next-day closing price of Alphabet Inc. (GOOGL) stock using the previous 30 trading days as input.

### Dataset
[Google Stock Data on Kaggle](https://www.kaggle.com/datasets/varpit94/google-stock-data) — historical daily OHLCV data for Alphabet Inc.

### Problem Formulation
Given an input sequence of 30 consecutive trading days:

$$X_t = [x_{t-29}, \ldots, x_t]$$

the model predicts the next-day closing price:

$$\hat{y}_{t+1} = f(X_t;\, \theta)$$

### Preprocessing
- Parse and sort by `Date` in chronological order
- Handle missing values
- Use `Close` price as the primary feature (optionally include `Open`, `High`, `Low`, `Volume`)
- Apply Min-Max normalization fitted on training data only
- Construct sliding-window sequences of length 30

### Data Split (chronological)
| Split | Proportion |
|-------|-----------|
| Training | 70% (earliest) |
| Validation | 15% |
| Test | 15% (most recent) |

### Model Architecture (SimpleRNN)
```
Input → SimpleRNN(64, tanh) → Dropout(0.2) → Dense(1)
```
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Early stopping on validation loss

### Optional Extensions
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)

### Evaluation Metrics
| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

> All metrics are computed after inverse-transforming predictions back to the original price scale.

---

## Part II: Research Assignment — Attention Mechanism

### Key Topics Covered

1. **Introduction** — What is the Attention Mechanism and what problems does it solve?
2. **Attention in Encoder–Decoder** — How Attention improves handling of long input sequences
3. **Core Components** — Query (Q), Key (K), Value (V) and their roles
4. **Mathematical Formulation** — Attention scores, weights, and context vectors
5. **Types of Attention** — Additive, Multiplicative, Dot-Product, Scaled Dot-Product
6. **Self-Attention** — Capturing intra-sequence relationships
7. **Multi-Head Attention** — Parallel attention heads and their advantages
8. **Attention in Transformers** — Integration in Encoder and Decoder layers
9. **Computational Complexity** — Memory and time complexity vs. sequence length
10. **Applications** — NLP, computer vision, speech, time-series, multimodal learning
11. **Advantages & Limitations** — Comparison with RNNs and LSTMs
12. **MCQs** — 5 multiple-choice questions (2 easy, 2 intermediate, 1 difficult)

### Presentation
- **Format:** 15–20 slides
- **Duration:** 15–20 minutes

---

## Repository Structure

```
DNN-project2-G1-Attention-Mechanism/
├── data/
│   ├── GOOGL.csv                  # raw dataset
│   └── processed/                 # sequences and scalers (output of 02)
├── code/
│   ├── 01_data_explore.ipynb      # inspect data, split, baseline
│   ├── 02_preprocessing.ipynb     # scaling, 30-day windows, saved arrays
│   └── 03_model_training.ipynb    # SimpleRNN / GRU / LSTM, evaluation
├── models/                        # trained models (output of 03)
├── results/                       # one folder per notebook
│   ├── 01_data_explore/
│   ├── 02_preprocessing/
│   └── 03_model_training/
├── slides/                        # research assignment presentation
└── README.md
```

---

## Setup

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib joblib jupyter
```

Run the notebooks in order, from inside the `code` folder, because they read data with relative paths such as `../data/GOOGL.csv`:

```bash
cd code
jupyter notebook
```

Run `01_data_explore.ipynb`, then `02_preprocessing.ipynb`, then `03_model_training.ipynb`.

---

## Results

*To be filled in from `results/03_model_training/test_metrics.csv` after training.*

| Model | MAE | RMSE | MSE | R² |
|-------|-----|------|-----|----|
| Persistence baseline | — | — | — | — |
| SimpleRNN | — | — | — | — |
| GRU | — | — | — | — |
| LSTM | — | — | — | — |
