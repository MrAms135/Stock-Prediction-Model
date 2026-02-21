# 📄 Phase 4 Model Card: Multi-Stock Portfolio & Risk Analysis

**Date:** February 2026
**Author:** Adarsh M S
**Status:** Phase 4 Complete (v4.0)

## 1. Phase 4 Objective
To transition the Tri-Model Majority Vote Ensemble from a single-asset environment (NVDA) into a multi-asset quantitative pipeline. The goal is to stress-test the model's predictive edge across differing market regimes, specifically analyzing the divergence between AI infrastructure companies and traditional SaaS companies during the "SaaSapocalypse."

## 2. Methodology: The Independent Loop (Local Models)
Instead of building a single global model, the architecture utilizes an isolated loop approach. For each ticker:
1. The pipeline downloads the specific historical data.
2. The 3 base models (LSTM, XGBoost, Transformer) are trained entirely on that asset's unique volatility and price scale.
3. The Majority Vote logic executes.
4. The Backtester calculates returns and risk metrics.
5. Keras memory sessions are cleared (`tf.keras.backend.clear_session()`) to prevent resource exhaustion before moving to the next ticker.

## 3. Quantitative Risk Integration: Maximum Drawdown

Absolute return is an insufficient metric for evaluating algorithmic strategies. Phase 4 introduces **Maximum Drawdown (MDD)** to the vectorized backtester to quantify the model's ability to protect capital. 

## 4. Key Findings & Market Analysis

**1. The SaaS Crash Shield (CRM & ADBE):**
During the analyzed period, traditional SaaS companies faced severe algorithmic sell-offs. The benchmark Buy & Hold strategy for Salesforce (CRM) suffered a -29.51% return. The Ensemble model successfully detected this macroeconomic shift, achieving a peak 54.78% directional accuracy and mitigating the loss to only -14.37%. By shifting to cash during critical downward volatility, the model successfully acted as a capital preservation tool.

**2. Alpha Generation in Sideways Markets (MSFT):**
Microsoft exhibited choppy, sideways momentum, yielding a benchmark return of only +9.30%. The Ensemble strategy effectively traded this volatility, more than doubling the benchmark to achieve a +23.92% return.

**3. The Bull Market Opportunity Cost (NVDA & GOOGL):**
In purely vertical, irrational bull runs (such as GOOGL's +98.25% run), the risk-averse nature of the Ensemble resulted in underperformance. The algorithm's propensity to lock in profits and exit the market to avoid pullbacks caused it to miss extended upside continuation. This is a known and acceptable trade-off in conservative quantitative systems.

## 5. Next Steps
The pipeline is now mathematically sound and logically robust. Phase 5 will expand the portfolio to 10 total stocks and execute the models across three distinct historical timeframes to find the optimal data lookback window. Finally, a unified Matplotlib dashboard will be constructed to visualize the entire portfolio's risk-adjusted performance on a single screen.