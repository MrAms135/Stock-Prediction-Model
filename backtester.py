import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleBacktester:
    """
    Simulates trading based on model signals to calculate actual returns.
    """
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        
    def run_backtest(self, y_actual, y_pred, dates):
        """
        y_actual: The real daily returns of the stock
        y_pred: The model's predictions
        """
        # 1. Generate Trading Signals
        # +1 means Buy/Hold, -1 means Sell/Stay in Cash
        signals = np.where(y_pred > 0, 1, 0) 
        
        # 2. Calculate Strategy Returns
        # If signal is 1, we get the stock's return that day. If 0, we get 0% (cash).
        # Note: We shift signals by 1 because yesterday's prediction dictates today's position
        strategy_returns = np.roll(signals, 1) * y_actual
        strategy_returns[0] = 0 # First day has no previous signal
        
        # 3. Calculate Cumulative Wealth
        # (1 + return) cumulative product * initial capital
        benchmark_wealth = self.initial_capital * np.cumprod(1 + y_actual)
        strategy_wealth = self.initial_capital * np.cumprod(1 + strategy_returns)
        
        # 4. Calculate Final Metrics
        total_bench_return = (benchmark_wealth[-1] - self.initial_capital) / self.initial_capital * 100
        total_strat_return = (strategy_wealth[-1] - self.initial_capital) / self.initial_capital * 100
        
        print("\n" + "="*50)
        print("💰 BACKTESTING RESULTS (Starting Capital: $10k)")
        print("="*50)
        print(f"Buy & Hold Final Value:  ${benchmark_wealth[-1]:,.2f} ({total_bench_return:+.2f}%)")
        print(f"Ensemble Final Value:    ${strategy_wealth[-1]:,.2f} ({total_strat_return:+.2f}%)")
        print("="*50)
        
        # 5. Plot the Equity Curve
        plt.figure(figsize=(14, 6))
        plt.plot(dates, benchmark_wealth, label='Buy & Hold (Benchmark)', color='gray', alpha=0.7)
        plt.plot(dates, strategy_wealth, label='AI Ensemble Strategy', color='green', linewidth=2)
        
        plt.title('Backtest: AI Strategy vs. Buy & Hold')
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/figures/backtest_results.png')
        print("✓ Saved equity curve to results/figures/backtest_results.png")
        
        return strategy_wealth, benchmark_wealth