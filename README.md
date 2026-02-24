# 📈 Tri-Model AI Trading Ensemble: Navigating the AI Boom & SaaS Crash

## 📝 Executive Summary
This project is an advanced quantitative trading algorithm designed to navigate complex, split-market environments. By combining three distinct Machine Learning architectures into a voting ensemble, the system identifies trend exhaustion and acts as an automated hedge against sector-specific market crashes.

Instead of relying on rigid, binary "Buy/Sell" signals, the algorithm utilizes **Dynamic Position Sizing (Scaled Voting)**. It adjusts portfolio capital allocation based on the mathematical consensus of the three models, preserving capital during uncertainty and aggressively shorting during high-confidence downturns.

## 🧠 System Architecture
1. **Long Short-Term Memory (LSTM):** Captures sequential, time-series momentum and long-term price dependencies.
2. **Transformer (Self-Attention):** Analyzes non-linear, distant technical indicators without the vanishing gradient problems of traditional RNNs.
3. **XGBoost (Gradient Boosting):** Extracts feature importance from tabular technical data (RSI, MACD, Bollinger Bands) to identify immediate overbought/oversold conditions.

## ⚙️ The "Scaled Voting" Mechanism
The ensemble aggregates the daily predictions (Long=1, Short=-1) of all three models to calculate a daily confidence threshold, dynamically scaling the portfolio:
* **Unanimous Buy (+3):** 100% Long Allocation
* **Majority Buy (+1):** 33% Long Allocation (Risk Reduction during market chop)
* **Majority Sell (-1):** 33% Short Allocation (Early Hedging)
* **Unanimous Sell (-3):** 100% Short Allocation (Aggressive Crash Shorting)

## 📊 Performance: The 2023-2026 "Split Market"
The algorithm was backtested on the highly volatile "Era 3" market (2023-2026). During this period, mega-cap AI stocks rallied violently while enterprise SaaS equities suffered a severe, prolonged sector rotation (the "SaaSapocalypse"). 

### Key Performance Highlights:
* **Masterful Crash Shorting:** A standard Buy & Hold strategy on ServiceNow (NOW) suffered a brutal **-42.84%** loss. The AI ensemble recognized the sector rotation, aggressively shorted the crash, and generated a **+10.93% net profit**.
* **Flipping Losses to Gains:** On Adobe (ADBE) and Snowflake (SNOW), the algorithm turned massive **-26.80%** and **-29.17%** benchmark crashes into **+19.18%** and **+13.29%** profits, respectively.
* **Protecting the AI Winners:** Even within the AI sector, the algorithm successfully hedged against sharp corrections. When Microsoft (MSFT) dropped **-23.31%**, the dynamic position sizing flipped the trade into a **+8.57%** gain.

## 📉 Elite Risk Management (Maximum Drawdown)
In quantitative finance, preserving capital is more important than chasing maximum upside. The algorithm's crowning achievement is its drastic reduction of Maximum Drawdown (MDD)—the peak-to-trough panic an investor feels during a crash.
* **NOW:** Reduced MDD from a catastrophic **-46.86%** down to just **-5.62%**.
* **MSFT:** Reduced MDD from **-27.24%** down to a highly defensive **-2.71%**.
* **CRM:** Reduced MDD from **-30.78%** down to **-9.70%** while maintaining a net positive return (+1.84%) against a -27.23% market crash.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras (LSTM, Transformers)
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy, yfinance
* **Evaluation:** Custom Vectorized Backtesting Engine with Dynamic Position Sizing