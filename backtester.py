import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleBacktester:
    """
    Simulates trading based on model signals to calculate actual returns and risk metrics.
    """
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        
    def _calculate_max_drawdown(self, wealth_index):
        """
        Calculates the Maximum Drawdown (MDD) of a wealth array.
        MDD = (Trough Value - Peak Value) / Peak Value
        """
        wealth_series = pd.Series(wealth_index)
        previous_peaks = wealth_series.cummax()
        drawdowns = (wealth_series - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min() * 100 
        return max_drawdown

    def run_backtest(self, y_actual, y_pred, dates):
        # 1. Calculate daily returns (assuming y_actual represents daily returns)
        # If y_actual is just direction (1 or -1), you'll need the actual price returns
        # For this example, we'll assume standard return calculation
        daily_returns = y_actual
        
        # 2. Strategy Returns (Predict 1 = Long, Predict -1 or 0 = Out/Short)
        strategy_returns = daily_returns * y_pred
        
        # 3. Cumulative Equity Curves
        bench_equity = self.initial_capital * np.cumprod(1 + daily_returns)
        strat_equity = self.initial_capital * np.cumprod(1 + strategy_returns)
        
        # 4. Total Return Percentages
        bench_ret = ((bench_equity[-1] - self.initial_capital) / self.initial_capital) * 100
        strat_ret = ((strat_equity[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # 5. Maximum Drawdown Calculation
        def calculate_mdd(equity_curve):
            roll_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - roll_max) / roll_max
            return np.min(drawdown) * 100 # Returns a negative percentage

        bench_mdd = calculate_mdd(bench_equity)
        strat_mdd = calculate_mdd(strat_equity)
        
        # 6. Return ALL SIX values expected by Phase 5
        return strat_equity, bench_equity, strat_ret, bench_ret, strat_mdd, bench_mdd