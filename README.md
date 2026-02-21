# Event-Aware Ensemble for Stock Prediction 📈
**Phase 2 Status:** Hybrid Ensemble (LSTM + XGBoost) Complete

A quantitative finance pipeline investigating if "Ensemble Learning" (combining multiple AI models) can predict stock movements better than individual models during market volatility.

---

## 🎯 Current Progress: Phase 2 (Hybrid Architecture)
We have successfully implemented a **Weighted Ensemble** that combines two distinct modeling philosophies:
1.  **The "Time" Expert (LSTM):** A Deep Learning model that captures sequential patterns and long-term dependencies in price history.
2.  **The "Logic" Expert (XGBoost):** A Gradient Boosting model that captures non-linear relationships between technical indicators (RSI, Bollinger Bands, Volume).

### Key Features
✅ **Multi-Model Fusion:** Averaging predictions from LSTM and XGBoost to reduce variance.
✅ **Lag Reduction:** XGBoost helps correct the "lag" often seen in pure LSTM models.
✅ **Dynamic Data Pipeline:** Automated download (Yahoo Finance), processing, and feature engineering.
✅ **Comparative Evaluation:** Side-by-side metrics (RMSE, Accuracy) for LSTM vs. XGBoost vs. Ensemble.

---

## 🏗️ Architecture (Current)

```mermaid
graph TD
    Data[Input Data: OHLCV] --> TI[Tech Indicators]
    TI --> LSTM[Model A: LSTM]
    TI --> XGB[Model B: XGBoost]
    LSTM -->|Prediction A| Avg(Weighted Average)
    XGB -->|Prediction B| Avg
    Avg --> Final[Final Prediction]