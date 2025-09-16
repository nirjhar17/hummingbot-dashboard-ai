"""
SOL Scalping Strategy Backtesting Script
========================================

This script backtests the SOL Scalping Strategy on historical data
"""

import pandas as pd
import numpy as np
try:
    import talib as _ta
except Exception:  # Fallback if TA-Lib is unavailable
    _ta = None
from datetime import datetime, timedelta
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # Graceful fallback when matplotlib is unavailable
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None  # Optional; not required
from typing import Dict, List, Tuple, Optional
import logging

# Import our strategy and config
from .sol_scalping_config import get_config, validate_config

class SolScalpingBacktester:
    """
    Backtester for SOL Scalping Strategy
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        self.strategy_config = self.config["strategy"]
        self.risk_rules = self.config["risk_rules"]
        
        # Initialize tracking variables
        self.trades = []
        self.daily_pnl = []
        self.equity_curve = []
        self.current_position = None
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.trades_today = 0
        self.daily_pnl_today = 0.0
        self.consecutive_losses = 0
        self.last_trade_date = None
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, data_file: str) -> pd.DataFrame:
        """Load historical data from CSV file"""
        try:
            df = pd.read_csv(data_file)
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Sort by timestamp
            df = df.sort_index()
            
            self.logger.info(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Helper implementations with TA-Lib fallback
            def _ema(series: pd.Series, period: int) -> pd.Series:
                if _ta is not None:
                    return _ta.EMA(series.astype(float), timeperiod=int(period))
                # Pandas EWM fallback
                return series.astype(float).ewm(span=int(period), adjust=False).mean()

            def _rsi(series: pd.Series, period: int) -> pd.Series:
                if _ta is not None:
                    return _ta.RSI(series.astype(float), timeperiod=int(period))
                # Vectorized RSI fallback
                delta = series.astype(float).diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=int(period), min_periods=int(period)).mean()
                avg_loss = loss.rolling(window=int(period), min_periods=int(period)).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                return rsi

            # Calculate EMAs
            df['ema_fast'] = _ema(df['close'], self.strategy_config['ema_fast'])
            df['ema_slow'] = _ema(df['close'], self.strategy_config['ema_slow'])

            # Calculate RSI
            df['rsi'] = _rsi(df['close'], self.strategy_config['rsi_period'])
            
            # Calculate volume average
            df['volume_avg'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_avg']
            
            # Calculate price change
            df['price_change'] = df['close'].pct_change()
            
            self.logger.info("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise
    
    def check_entry_conditions(self, row: pd.Series) -> Optional[str]:
        """Check for entry conditions"""
        try:
            rsi_high = float(self.strategy_config.get('rsi_neutral_high', 50))
            rsi_low = float(self.strategy_config.get('rsi_neutral_low', 50))
            vol_thresh = float(self.strategy_config.get('volume_threshold', 1.2))
            # Skip if we don't have enough data
            if pd.isna(row['ema_fast']) or pd.isna(row['ema_slow']) or pd.isna(row['rsi']):
                return None
            
            # Long entry conditions
            if (row['ema_fast'] > row['ema_slow'] and  # Uptrend
                row['rsi'] > rsi_high and  # Momentum
                row['volume_ratio'] > vol_thresh and  # Volume confirmation
                row['close'] > row['ema_slow']):  # Price above slow EMA
                
                return 'LONG'
                
            # Short entry conditions
            elif (row['ema_fast'] < row['ema_slow'] and  # Downtrend
                  row['rsi'] < rsi_low and  # Momentum
                  row['volume_ratio'] > vol_thresh and  # Volume confirmation
                  row['close'] < row['ema_slow']):  # Price below slow EMA
                
                return 'SHORT'
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking entry conditions: {e}")
            return None
    
    def check_exit_conditions(self, row: pd.Series) -> Optional[str]:
        """Check for exit conditions"""
        try:
            if self.current_position is None:
                return None
            
            # Check stop loss
            if self.current_position == 'LONG' and row['close'] <= self.stop_loss_price:
                return 'Stop Loss'
            elif self.current_position == 'SHORT' and row['close'] >= self.stop_loss_price:
                return 'Stop Loss'
                
            # Check take profit
            elif self.current_position == 'LONG' and row['close'] >= self.take_profit_price:
                return 'Take Profit'
            elif self.current_position == 'SHORT' and row['close'] <= self.take_profit_price:
                return 'Take Profit'
                
            # Check RSI exit conditions
            elif self.current_position == 'LONG' and row['rsi'] > self.strategy_config['rsi_overbought']:
                return 'RSI Overbought'
            elif self.current_position == 'SHORT' and row['rsi'] < self.strategy_config['rsi_oversold']:
                return 'RSI Oversold'
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def should_stop_trading(self, current_date: datetime.date) -> bool:
        """Check if we should stop trading based on risk limits"""
        # Reset daily counters if new day
        if self.last_trade_date != current_date:
            self.trades_today = 0
            self.daily_pnl_today = 0.0
            self.consecutive_losses = 0
            self.last_trade_date = current_date
        
        # Check daily trade limit
        if self.trades_today >= self.strategy_config['max_trades_per_day']:
            return True
            
        # Check daily loss limit
        if self.daily_pnl_today <= -self.strategy_config['max_daily_loss_pct']:
            return True
            
        # Check daily profit limit
        if self.daily_pnl_today >= self.strategy_config['max_daily_profit_pct']:
            return True
            
        # Check consecutive losses
        if self.consecutive_losses >= self.strategy_config['max_consecutive_losses']:
            return True
            
        return False
    
    def enter_position(self, row: pd.Series, position_type: str):
        """Enter a position"""
        try:
            price = row['close']
            
            # Calculate position size
            position_size = self.strategy_config['backtest_initial_balance'] * self.strategy_config['position_size_pct'] * self.strategy_config['leverage']
            
            # Calculate stop loss and take profit prices
            if position_type == 'LONG':
                self.stop_loss_price = price * (1 - self.strategy_config['stop_loss_pct'])
                self.take_profit_price = price * (1 + self.strategy_config['take_profit_pct'])
            else:  # SHORT
                self.stop_loss_price = price * (1 + self.strategy_config['stop_loss_pct'])
                self.take_profit_price = price * (1 - self.strategy_config['take_profit_pct'])
            
            self.current_position = position_type
            self.entry_price = price
            self.trades_today += 1
            
            self.logger.info(f"{position_type} ENTRY: Price={price:.4f}, Stop={self.stop_loss_price:.4f}, Target={self.take_profit_price:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error entering position: {e}")
    
    def exit_position(self, row: pd.Series, exit_reason: str):
        """Exit current position"""
        try:
            if self.current_position is None:
                return
                
            exit_price = row['close']
            
            # Calculate P&L
            if self.current_position == 'LONG':
                pnl_pct = (exit_price - self.entry_price) / self.entry_price
            else:  # SHORT
                pnl_pct = (self.entry_price - exit_price) / self.entry_price
            
            # Apply leverage
            pnl_pct *= self.strategy_config['leverage']
            
            # Subtract commission
            pnl_pct -= self.strategy_config['backtest_commission'] * 2  # Entry and exit
            
            # Update daily P&L
            self.daily_pnl_today += pnl_pct
            
            # Update consecutive losses
            if pnl_pct < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Record trade
            trade = {
                'entry_time': self.entry_time,
                'exit_time': row.name,
                'position': self.current_position,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'daily_pnl': self.daily_pnl_today,
                'consecutive_losses': self.consecutive_losses
            }
            self.trades.append(trade)
            
            # Update performance metrics
            self.total_trades += 1
            if pnl_pct > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            self.total_pnl += pnl_pct
            
            # Update equity curve
            current_equity = self.strategy_config['backtest_initial_balance'] * (1 + self.total_pnl)
            self.equity_curve.append({
                'timestamp': row.name,
                'equity': current_equity,
                'pnl_pct': self.total_pnl
            })
            
            # Update drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
            
            self.logger.info(f"EXIT {self.current_position}: Price={exit_price:.4f}, P&L={pnl_pct:.2%}, Reason={exit_reason}")
            
            # Reset position
            self.current_position = None
            self.entry_price = None
            self.stop_loss_price = None
            self.take_profit_price = None
            
        except Exception as e:
            self.logger.error(f"Error exiting position: {e}")
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run the backtest"""
        try:
            self.logger.info("Starting backtest...")
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Run through each row
            for timestamp, row in df.iterrows():
                current_date = timestamp.date()
                
                # Check if we should stop trading
                if self.should_stop_trading(current_date):
                    continue
                
                # Check for exit conditions first
                if self.current_position is not None:
                    exit_reason = self.check_exit_conditions(row)
                    if exit_reason:
                        self.exit_position(row, exit_reason)
                        continue
                
                # Check for entry conditions
                if self.current_position is None:
                    entry_signal = self.check_entry_conditions(row)
                    if entry_signal:
                        self.entry_time = timestamp
                        self.enter_position(row, entry_signal)
            
            # Close any open position at the end
            if self.current_position is not None:
                last_row = df.iloc[-1]
                self.exit_position(last_row, "End of Data")
            
            # Calculate final metrics
            results = self.calculate_performance_metrics()
            
            self.logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        try:
            if not self.trades:
                return {"error": "No trades executed"}
            
            trades_df = pd.DataFrame(self.trades)
            
            # Basic metrics
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if self.winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if self.losing_trades > 0 else 0
            profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if self.losing_trades > 0 else float('inf')
            
            # Risk metrics
            sharpe_ratio = self.calculate_sharpe_ratio()
            max_consecutive_losses = trades_df['consecutive_losses'].max()
            
            # Time-based metrics
            total_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
            trades_per_day = self.total_trades / total_days if total_days > 0 else 0
            
            results = {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'total_pnl': self.total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'max_consecutive_losses': max_consecutive_losses,
                'trades_per_day': trades_per_day,
                'total_days': total_days
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.equity_curve) < 2:
                return 0.0
            
            equity_df = pd.DataFrame(self.equity_curve)
            returns = equity_df['pnl_pct'].pct_change().dropna()
            
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def plot_results(self, df: pd.DataFrame):
        """Plot backtest results and return the matplotlib figure (or None)."""
        try:
            if plt is None:
                self.logger.warning("matplotlib not available; skipping plots")
                return None
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # Plot 1: Price and EMAs
            axes[0].plot(df.index, df['close'], label='Close Price', alpha=0.7)
            axes[0].plot(df.index, df['ema_fast'], label=f'EMA {self.strategy_config["ema_fast"]}', alpha=0.7)
            axes[0].plot(df.index, df['ema_slow'], label=f'EMA {self.strategy_config["ema_slow"]}', alpha=0.7)
            axes[0].set_title('SOL/USDT Price and Moving Averages')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: RSI
            axes[1].plot(df.index, df['rsi'], label='RSI', color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            axes[1].axhline(y=50, color='k', linestyle='-', alpha=0.5, label='Neutral')
            axes[1].set_title('RSI Indicator')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Equity Curve
            if self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve)
                axes[2].plot(equity_df['timestamp'], equity_df['equity'], label='Equity Curve', color='green')
                axes[2].set_title('Equity Curve')
                axes[2].set_ylabel('Equity ($)')
                axes[2].set_xlabel('Date')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            # Caller (Streamlit page) will render this figure with st.pyplot
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {e}")
            return None
    
    def print_results(self, results: Dict):
        """Print backtest results"""
        print("\n" + "="*60)
        print("SOL SCALPING STRATEGY BACKTEST RESULTS")
        print("="*60)
        
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total P&L: {results['total_pnl']:.2%}")
        print(f"Average Win: {results['avg_win']:.2%}")
        print(f"Average Loss: {results['avg_loss']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Consecutive Losses: {results['max_consecutive_losses']}")
        print(f"Trades per Day: {results['trades_per_day']:.2f}")
        print(f"Total Days: {results['total_days']}")
        
        print("\n" + "="*60)
        print("RISK MANAGEMENT EVALUATION")
        print("="*60)
        
        # Check against risk rules
        risk_checks = {
            "Win Rate >= 60%": results['win_rate'] >= 0.60,
            "Profit Factor >= 1.5": results['profit_factor'] >= 1.5,
            "Max Drawdown <= 10%": results['max_drawdown'] <= 0.10,
            "Max Consecutive Losses <= 3": results['max_consecutive_losses'] <= 3,
        }
        
        for check, passed in risk_checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{check}: {status}")
        
        print("\n" + "="*60)

def main():
    """Main function to run backtest"""
    try:
        # Load configuration
        config = get_config()
        if not validate_config(config):
            print("❌ Configuration validation failed!")
            return
        
        # Initialize backtester
        backtester = SolScalpingBacktester(config)
        
        # Load data (you'll need to provide the data file)
        print("Please provide the path to your SOL/USDT historical data CSV file:")
        print("Required columns: timestamp, open, high, low, close, volume")
        print("Example: data/sol_usdt_15m.csv")
        
        # For now, we'll create a placeholder
        print("\nTo run the backtest, you need to:")
        print("1. Download SOL/USDT 15-minute data from Binance")
        print("2. Save it as a CSV file with the required columns")
        print("3. Update the data_file path in this script")
        print("4. Run the backtest")
        
        # Example of how to run the backtest:
        # data_file = "data/sol_usdt_15m.csv"
        # df = backtester.load_data(data_file)
        # results = backtester.run_backtest(df)
        # backtester.print_results(results)
        # backtester.plot_results(df)
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()

