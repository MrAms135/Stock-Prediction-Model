# 📄 Phase 3 Model Card: Tri-Model Attention Ensemble

**Date:** February 2026
**Author:** Adarsh M S
**Status:** Phase 3 Complete (v3.0)

## 1. Project Overview
**Objective:** Break the 52% directional accuracy ceiling by introducing a Time-Series Transformer, and resolve magnitude instability using a Majority Vote logic.
**Target Variable:** Daily Returns (Percentage change in Close price).

## 2. Methodology & Architecture
**The Tri-Model Ensemble:**
* **Model A (LSTM):** 2-Layer LSTM with Dropout (0.2). Sequence Length: 60 days.
* **Model B (XGBoost):** Gradient Boosting Regressor (1000 estimators).
* **Model C (Transformer):** Multi-Head Attention architecture (4 Heads, Size 256).

**The "Majority Vote" Integration Strategy:**
Initial testing revealed that the Transformer model suffered from severe magnitude instability (RMSE 0.4419), rendering traditional weighted averaging mathematically unviable. To solve this, a **Signal-Based Majority Vote** was implemented:
1. Predictions are converted to pure signals: `np.sign(prediction)`.
2. The signals are summed: `(+1) + (-1) + (+1)`.
3. The winning direction dictates the Ensemble's final prediction.

## 3. Results Summary (Test Set)

| Model | RMSE | MAE | R² Score | Direction Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM Only** | 0.0271 | 0.0190 | -0.0158 | 50.87% |
| **XGBoost Only** | 0.0308 | 0.0232 | -0.3133 | 52.61% |
| **Transformer Only** | 0.4419 | 0.3501 | -268.89 | 53.04% |
| **Majority Vote Ensemble**| **0.0278** | **0.0200** | **-0.0694** | **53.04%** |

## 4. Key Quantitative Findings
1. **Transformer's Directional Edge:** The Transformer successfully identified market direction better than legacy models, likely due to the Self-Attention mechanism isolating distant but relevant technical events.
2. **The "Outlier" Neutralization:** The Majority Vote logic successfully insulated the Ensemble from the Transformer's numerical explosions. The Ensemble's RMSE was pulled back down to 0.0278, maintaining structural stability.
3. **The 53% Benchmark:** Achieving a stable >53% directional accuracy across an automated test window proves a statistical edge over a random walk (50%) or baseline momentum strategies.

## 5. Next Steps
To validate the robustness of this 53.04% edge, the pipeline must be abstracted to run dynamically across a diversified portfolio of tickers (AI Winners vs. SaaS Victims) rather than relying solely on NVDA.