"""
SOL Scalping Strategy Configuration
==================================

Configuration file for the SOL Scalping Strategy
"""

from decimal import Decimal
from typing import Dict, Any

# Strategy Configuration
STRATEGY_CONFIG = {
    # Technical Analysis Parameters
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_period": 14,
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "volume_threshold": 1.2,
    
    # Risk Management Parameters
    "stop_loss_pct": 0.015,  # 1.5%
    "take_profit_pct": 0.025,  # 2.5%
    "max_trades_per_day": 5,
    "max_daily_loss_pct": 0.02,  # 2%
    "max_daily_profit_pct": 0.03,  # 3%
    "max_consecutive_losses": 3,
    "position_size_pct": 0.15,  # 15% of account
    "leverage": 2.0,
    
    # Trading Parameters
    "trading_pair": "SOL-USDT",
    "timeframe": "15m",
    "min_order_size": 10.0,  # Minimum order size in USDT
    
    # Exchange Configuration
    "exchange": "bitget",
    "api_key": "",  # To be filled by user
    "api_secret": "",  # To be filled by user
    "passphrase": "",  # To be filled by user (if required)
    
    # Logging Configuration
    "log_level": "INFO",
    "log_file": "sol_scalping.log",
    
    # Backtesting Configuration
    "backtest_start_date": "2024-07-01",
    "backtest_end_date": "2024-12-31",
    "backtest_initial_balance": 1000.0,
    "backtest_commission": 0.001,  # 0.1% commission
}

# Risk Management Rules
RISK_RULES = {
    "max_position_size": 0.20,  # Maximum 20% of account per trade
    "max_daily_risk": 0.05,  # Maximum 5% daily risk
    "max_drawdown": 0.10,  # Maximum 10% drawdown
    "min_win_rate": 0.60,  # Minimum 60% win rate
    "min_profit_factor": 1.5,  # Minimum 1.5 profit factor
}

# Technical Indicator Thresholds
INDICATOR_THRESHOLDS = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "rsi_neutral_low": 45,
    "rsi_neutral_high": 55,
    "volume_spike": 1.5,  # 1.5x average volume
    "ema_crossover_threshold": 0.001,  # 0.1% difference for crossover
}

# Trading Hours (UTC)
TRADING_HOURS = {
    "start_hour": 0,  # 00:00 UTC
    "end_hour": 23,  # 23:59 UTC
    "timezone": "UTC",
}

# Notification Settings
NOTIFICATIONS = {
    "enable_slack": False,
    "slack_webhook": "",
    "enable_email": False,
    "email_recipients": [],
    "notify_on_trade": True,
    "notify_on_daily_summary": True,
    "notify_on_risk_limits": True,
}

# Performance Tracking
PERFORMANCE_TRACKING = {
    "track_metrics": True,
    "save_trades": True,
    "generate_reports": True,
    "report_frequency": "daily",  # daily, weekly, monthly
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration"""
    return {
        "strategy": STRATEGY_CONFIG,
        "risk_rules": RISK_RULES,
        "indicators": INDICATOR_THRESHOLDS,
        "trading_hours": TRADING_HOURS,
        "notifications": NOTIFICATIONS,
        "performance": PERFORMANCE_TRACKING,
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
    try:
        strategy = config["strategy"]
        
        # Validate risk parameters
        assert 0 < strategy["stop_loss_pct"] < 0.1, "Stop loss must be between 0% and 10%"
        assert 0 < strategy["take_profit_pct"] < 0.1, "Take profit must be between 0% and 10%"
        assert 1 <= strategy["max_trades_per_day"] <= 20, "Max trades per day must be between 1 and 20"
        assert 0 < strategy["position_size_pct"] <= 0.5, "Position size must be between 0% and 50%"
        assert 1.0 <= strategy["leverage"] <= 5.0, "Leverage must be between 1x and 5x"
        
        # Validate technical parameters
        assert strategy["ema_fast"] < strategy["ema_slow"], "Fast EMA must be less than slow EMA"
        assert 5 <= strategy["rsi_period"] <= 50, "RSI period must be between 5 and 50"
        assert 0 < strategy["rsi_oversold"] < 50, "RSI oversold must be between 0 and 50"
        assert 50 < strategy["rsi_overbought"] < 100, "RSI overbought must be between 50 and 100"
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test configuration
    config = get_config()
    if validate_config(config):
        print("✅ Configuration is valid!")
        print(f"Strategy: {config['strategy']['trading_pair']}")
        print(f"Risk Management: {config['strategy']['max_trades_per_day']} trades/day, {config['strategy']['max_daily_loss_pct']:.1%} max loss")
        print(f"Technical: EMA({config['strategy']['ema_fast']},{config['strategy']['ema_slow']}), RSI({config['strategy']['rsi_period']})")
    else:
        print("❌ Configuration is invalid!")

