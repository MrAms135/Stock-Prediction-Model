# Event-Aware Ensemble for Stock Prediction
## Analyzing AI-Driven Market Disruptions with Deep Learning

**A final year project investigating ensemble deep learning models for stock prediction during the 2026 SaaSapocalypse event**

---

## 🎯 Project Overview

This project develops an **adaptive ensemble architecture** that combines deep learning and machine learning models to predict IT sector stock prices, with a specific focus on detecting and responding to market disruptions caused by AI automation fears (the February 2026 SaaSapocalypse).

### Key Features

✅ **Multi-Model Ensemble**: LSTM + Transformer + LightGBM  
✅ **Event Detection**: Anomaly detection layer for market regime identification  
✅ **Adaptive Weighting**: Dynamic model combination based on market conditions  
✅ **Real-World Validation**: Tested on the Feb 3, 2026 SaaSapocalypse event  
✅ **Comparative Analysis**: AI Winners (NVDA, MSFT, GOOGL) vs SaaS Victims (CRM, ADBE, NOW)

---

## 🏗️ Architecture

```
Input Data (OHLCV + Technical Indicators)
           ↓
    ┌──────┴──────┐
    │   Anomaly   │  ← Detects market regime (Normal/Crisis)
    │   Detector  │
    └──────┬──────┘
           ↓
    ┌──────┴──────────────────┐
    │    Base Models          │
    │  • LSTM (temporal)      │
    │  • Transformer (attn)   │
    │  • LightGBM (features)  │
    └──────┬──────────────────┘
           ↓
    ┌──────┴──────────────┐
    │  Adaptive Fusion    │
    │  Normal → Weights A │
    │  Crisis → Weights B │
    └──────┬──────────────┘
           ↓
    Final Prediction
```

---

## 📁 Project Structure

```
stock-ensemble-project/
│
├── data/                          # Data storage
│   ├── raw/                       # Downloaded stock data
│   └── processed/                 # Preprocessed data with indicators
│
├── models/                        # Saved model files
│   ├── lstm_model/
│   ├── transformer_model/
│   ├── lightgbm_model/
│   └── ensemble_model/
│
├── results/                       # Outputs and analysis
│   ├── figures/                   # Plots and visualizations
│   ├── predictions/               # Model predictions
│   └── metrics/                   # Performance metrics
│
├── logs/                          # Training logs
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_saasapocalypse_analysis.ipynb
│
├── src/                           # Source code (Phase 2)
│   ├── models/
│   ├── preprocessing/
│   ├── ensemble/
│   └── evaluation/
│
├── data_collection.py             # Download stock data
├── technical_indicators.py        # Calculate technical indicators
├── config.py                      # Configuration and hyperparameters
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum (16GB recommended)
- GPU optional (CPU works fine for this scale)

### Installation

1. **Clone or create project directory**
   ```bash
   mkdir stock-ensemble-project
   cd stock-ensemble-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn
   pip install yfinance pandas-ta
   pip install tensorflow  # or: pip install torch
   pip install lightgbm scikit-learn
   pip install plotly jupyter
   ```

4. **Create requirements.txt**
   ```bash
   pip freeze > requirements.txt
   ```

### Quick Start (Week 1)

**Step 1: Download Data**
```bash
python data_collection.py
```
This will download historical data for all 6 stocks (2019-2026).

**Step 2: View Configuration**
```bash
python config.py
```
Review all hyperparameters and settings.

**Step 3: Explore Data** (optional)
```bash
jupyter notebook
# Open notebooks/01_data_exploration.ipynb
```

---

## 📊 Dataset

### Stocks Analyzed

**AI Winners** (benefited from AI boom):
- NVDA (NVIDIA) - GPU leader
- MSFT (Microsoft) - Azure, OpenAI partnership
- GOOGL (Google) - Search, Cloud, AI

**SaaS Victims** (hurt by SaaSapocalypse):
- CRM (Salesforce) - SaaS leader
- ADBE (Adobe) - Creative cloud
- NOW (ServiceNow) - Enterprise automation

### Time Period

- **Training**: 2019-01-01 to 2024-12-31 (6 years)
- **Event Date**: 2026-02-03 (SaaSapocalypse)
- **Test Period**: Includes the event and aftermath

### Features

**Price Data**: Open, High, Low, Close, Volume

**Technical Indicators**:
- Trend: SMA (10,20,50,200), EMA (12,26)
- Momentum: RSI, MACD, Stochastic, ROC
- Volatility: Bollinger Bands, ATR, Historical Volatility
- Volume: OBV, Volume MA, CMF
- Lagged: Returns and volumes from past days
- Statistical: Rolling mean, std, skew, kurtosis

---

## 🎓 Implementation Phases

### **Phase 1: Foundation (Weeks 1-3)** ✅ YOU ARE HERE

**Week 1**: Environment setup and data collection
- [x] Install dependencies
- [x] Download stock data
- [x] Create project structure

**Week 2**: Feature engineering
- [ ] Calculate technical indicators
- [ ] Create sequences for LSTM/Transformer
- [ ] Prepare tabular data for LightGBM
- [ ] Split data (train/val/test)

**Week 3**: Build individual models
- [ ] Implement LSTM model
- [ ] Implement Transformer model (optional: start with LSTM+LightGBM only)
- [ ] Implement LightGBM model
- [ ] Train and evaluate each model

### **Phase 2: Novel Components (Weeks 4-7)**

**Week 4**: Anomaly detection
- [ ] Implement Isolation Forest detector
- [ ] Test on historical volatility events
- [ ] Validate on SaaSapocalypse date

**Week 5**: Basic ensemble
- [ ] Simple averaging ensemble
- [ ] Weighted averaging ensemble
- [ ] Compare with individual models

**Week 6**: Adaptive ensemble
- [ ] Implement regime detection
- [ ] Adaptive weight switching
- [ ] Meta-learner (optional)

**Week 7**: Testing and refinement
- [ ] Walk-forward validation
- [ ] Hyperparameter tuning
- [ ] Error analysis

### **Phase 3: Event Analysis (Weeks 8-9)**

**Week 8**: SaaSapocalypse deep dive
- [ ] Pre-event pattern analysis
- [ ] Event detection evaluation
- [ ] Post-event recovery prediction

**Week 9**: Comparative analysis
- [ ] AI Winners vs SaaS Victims performance
- [ ] Feature importance analysis
- [ ] Create visualizations

### **Phase 4: Documentation (Weeks 10-12)**

**Week 10**: Results compilation
- [ ] Generate all metrics
- [ ] Create comparison tables
- [ ] Statistical significance tests

**Week 11**: Report writing
- [ ] Introduction and literature review
- [ ] Methodology section
- [ ] Results and discussion
- [ ] Conclusion

**Week 12**: Presentation
- [ ] Create slides
- [ ] Prepare demo
- [ ] Practice presentation

---

## 📈 Expected Results

### Performance Targets

- **RMSE**: 15-25% improvement over individual models
- **Direction Accuracy**: 55-65% (vs 50% random baseline)
- **Sharpe Ratio**: > 1.0 (vs buy-and-hold)
- **Event Detection**: Identify anomaly on Feb 3, 2026

### Key Findings to Demonstrate

1. Ensemble outperforms individual models
2. Adaptive weighting improves crisis performance
3. Clear difference between AI Winners and SaaS Victims
4. Model detected unusual patterns on event day

---

## 🔧 Troubleshooting

### Common Issues

**"ModuleNotFoundError: pandas_ta"**
```bash
pip install pandas-ta
```

**"yfinance download fails"**
- Check internet connection
- Try downloading one stock at a time
- Use alternative date ranges

**"Out of memory error"**
- Reduce SEQUENCE_LENGTH in config.py
- Reduce batch_size in model configs
- Process one stock at a time

**"Model training is too slow"**
- Reduce number of epochs
- Use smaller model architectures
- Consider using CPU for LightGBM (it's fast anyway)

---

## 📚 Resources

### Documentation
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Pandas-TA Indicators](https://github.com/twopirllc/pandas-ta)

### Tutorials
- Time Series Forecasting with LSTM
- Transformer Models for Sequential Data
- Ensemble Learning Techniques
- Stock Market Technical Analysis

### Papers (for literature review)
- "Attention Is All You Need" (Transformer architecture)
- "Ensemble Methods in Machine Learning" (Dietterich, 2000)
- "Financial Time Series Forecasting with Deep Learning" (surveys)

---

## 🎯 Problem Statement

**Title**: Event-Aware Ensemble Deep Learning for Stock Prediction Under AI-Driven Market Disruption

**Abstract**: The February 2026 SaaSapocalypse demonstrated that traditional ML models fail during AI-driven market disruptions. This project proposes an event-aware ensemble architecture that combines LSTM, Transformer, and gradient boosting models with an adaptive weighting mechanism that detects regime changes and adjusts prediction strategy during high-volatility events, evaluated on IT stocks during the AI disruption period.

**Research Questions**:
1. Do ensemble models outperform individual models for IT stock prediction?
2. Can anomaly detection identify market regime changes before crashes?
3. Does adaptive weighting improve performance during crisis vs normal periods?
4. What features differentiate AI Winners from SaaS Victims during disruption?

**Novel Contributions**:
- Adaptive ensemble with regime-based weighting
- Real-time validation on recent market event (Feb 2026)
- Comparative analysis of AI disruption impact across sectors

---

## 📞 Next Steps

**Immediate Actions**:
1. ✅ Run `python data_collection.py` to download data
2. ⏳ Calculate technical indicators (Week 2)
3. ⏳ Build first model (LSTM) in Week 3

**Need Help?**
- Check logs in `./logs/` directory
- Review configuration in `config.py`
- Adjust hyperparameters as needed

---

## 📄 License

This is an academic project for educational purposes.

---

## 👤 Author

[Your Name]  
Final Year Project - [Your University]  
[Degree Program]  
[Year]

---

**Version**: 1.0.0  
**Last Updated**: February 2026  
**Status**: Phase 1 - Data Collection ✅
