# Lightweight Multimodal Ensemble for Gold-Price Forecasting

A multimodal deep learning ensemble for next-day gold price forecasting, leveraging financial time-series data such as gold futures, oil prices, USD index, and CPI. Developed as a final project for CSCI 567 (Machine Learning) at USC.

## 📈 Problem Statement

Traditional financial models often rely on isolated indicators or historical prices, missing out on early signals from macroeconomic factors. Our goal is to build a lightweight ensemble model that integrates diverse financial inputs to forecast gold prices more accurately and in a timely fashion, improving decision-making in trading and risk management contexts.

## 🗃️ Data Sources

We use four primary financial data streams spanning 2010–2015, preprocessed into 30-day windows:

- **Gold Futures (GC=F)** — Daily COMEX gold prices
- **U.S. Dollar Index (DX-Y.NYB)** — USD value relative to basket of global currencies
- **Crude Oil Futures (CL=F)** — WTI oil prices as inflation and economic activity indicators
- **Consumer Price Index (CPI)** — Official measure of inflation from U.S. Bureau of Labor Statistics

Each 30-day window is used to predict the next day’s gold price.

## 🧠 Model Architecture

We build an ensemble model that fuses predictions from three deep learning architectures:

### 🔁 Bi-GRU

- Captures sequential trends in financial data
- 28k parameters | Best single-model RMSE: **0.067**
- Directional accuracy: ~49%

### 🔍 Transformer

- Models global temporal dependencies with self-attention
- 389k parameters | RMSE: **0.148**
- Best directional accuracy: **~53%**

### 📊 Temporal Convolutional Network (TCN)

- Fast and causal pattern extraction with dilated convolutions
- 36k parameters | RMSE: **0.158**
- Directional accuracy: ~50%

### ➕ Ridge Regression Ensemble

- Combines Bi-GRU, Transformer, and TCN predictions
- Achieved best overall RMSE: **0.062**
- Maintains balanced directional accuracy: **~49%**


## 📊 Results

| Model        | RMSE (scaled) | Directional Accuracy |
|--------------|----------------|-----------------------|
| Bi-GRU       | 0.067          | 49.3%                |
| Transformer  | 0.148          | **52.7%**            |
| TCN          | 0.158          | 50.1%                |
| **Ensemble** | **0.062**      | 49.0%                |

### vs Benchmarks

| Baseline / Benchmark         | Metric         | Our Result       | Notes                           |
|-----------------------------|----------------|------------------|---------------------------------|
| Naive (Tomorrow = Today)    | RMSE ~0.09–0.10 | 0.067 → 30% ↓    | Stronger than trivial baseline |
| Academic Papers             | RMSE ~0.07–0.12 | 0.067            | On par with GRU/LSTM literature |
| State-of-the-art (TFT)      | RMSE = 0.050    | 0.062            | Within ~20%                    |
| Quant Funds (Daily Horizon) | Direction Hit ~54% | 53%          | Competitive directional edge   |

## 🚀 Future Work

- Incorporate real-time sentiment from financial news
- Explore Temporal Fusion Transformers (TFT)
- Use hybrid loss functions (RMSE + directional)
- Improve ensemble weight learning strategies

## 🛠️ Tech Stack

- Python, PyTorch
- Hugging Face Transformers
- NumPy, pandas
- Matplotlib (visualizations)

## 📚 References

- [Cho et al., 2014 - GRU](https://arxiv.org/abs/1406.1078)
- [Vaswani et al., 2017 - Transformer](https://arxiv.org/abs/1706.03762)
- [Bai et al., 2018 - TCNs](https://arxiv.org/abs/1803.01271)
- [Kaggle - Gold Price](https://www.kaggle.com/datasets/ritwikb3/daily-gold-price-1996-2023-time-series)
- [Kaggle - USD Index](https://www.kaggle.com/datasets/balabaskar/us-dollar-index-data)
- [Kaggle - Oil Futures](https://www.kaggle.com/datasets/tarique7/daily-crude-price-dataset)
- [FRED - CPI](https://fred.stlouisfed.org/series/CPIAUCSL)

---
