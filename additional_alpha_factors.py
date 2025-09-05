

"""
Additional Alpha Factors for Crypto Trading
Based on research from quantitative trading literature and strategies

These factors can be added to your existing 101 WorldQuant alphas to enrich the alpha pool.
All factors are designed to work with your existing data structure (Date, Ticker, Open, High, Low, Close, Volume).
"""

import pandas as pd
import numpy as np

# Enhanced Alpha Factors (alpha_121 to alpha_140)
# Focus: Simple, robust, volatility-adjusted strategies

def alpha_121(df):
    """
    Volatility-Adjusted Momentum (inspired by alpha_101's success)
    Formula: (Close - Open) / (High - Low + 0.001) * Volume / ADV20
    Rationale: Combines alpha_101's normalized momentum with volume confirmation
    """
    daily_momentum = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 0.001)
    volume_ratio = df['Volume'] / (df.groupby('Ticker')['Volume'].rolling(20, min_periods=1).mean().values)
    return daily_momentum * volume_ratio

def alpha_122(df):
    """
    Range Position Indicator
    Formula: (Close - Low) / (High - Low + 0.001) - 0.5
    Rationale: Simple position within daily range, normalized around 0
    Values near +0.5 = close near high, -0.5 = close near low
    """
    range_position = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001)
    return range_position - 0.5

def alpha_123(df):
    """
    Volatility-Normalized Price Change
    Formula: (Close / Open - 1) / ((High / Low - 1) + 0.001)
    Rationale: Body-to-range ratio, similar to alpha_101 but with relative changes
    """
    body_ratio = (df['Close'] / df['Open'] - 1)
    range_ratio = (df['High'] / df['Low'] - 1) + 0.001
    return body_ratio / range_ratio

def alpha_124(df):
    """
    Intraday Momentum Strength
    Formula: (High - Open) * (Close - Low) / ((High - Low)^2 + 0.001)
    Rationale: Captures sustained upward momentum throughout the day
    """
    upward_strength = (df['High'] - df['Open']) * (df['Close'] - df['Low'])
    daily_range_sq = (df['High'] - df['Low'])**2 + 0.001
    return upward_strength / daily_range_sq

def alpha_125(df):
    """
    Opening Gap Momentum
    Formula: (Open - Close_lag1) / Close_lag1 if abs(gap) > 0.01 else 0
    Rationale: Captures overnight momentum in crypto 24/7 markets
    """
    close_lag1 = df.groupby('Ticker')['Close'].shift(1)
    opening_gap = (df['Open'] - close_lag1) / (close_lag1 + 0.001)
    # Only consider significant gaps
    return np.where(np.abs(opening_gap) > 0.01, opening_gap, 0)

def alpha_126(df):
    """
    Volume-Weighted Range Position
    Formula: ((Close - Low) / (High - Low + 0.001) - 0.5) * (Volume / ADV10)
    Rationale: Range position weighted by volume anomaly
    """
    range_pos = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001) - 0.5
    vol_ratio = df['Volume'] / (df.groupby('Ticker')['Volume'].rolling(10, min_periods=1).mean().values)
    return range_pos * vol_ratio

def alpha_127(df):
    """
    Candlestick Body Ratio
    Formula: abs(Close - Open) / (High - Low + 0.001)
    Rationale: Measures decisiveness of price movement (strong body vs. long wicks)
    """
    body_size = np.abs(df['Close'] - df['Open'])
    daily_range = df['High'] - df['Low'] + 0.001
    return body_size / daily_range

def alpha_128(df):
    """
    Price Acceleration
    Formula: (Close - Close_lag1) / Close_lag1 - (Close_lag1 - Close_lag2) / Close_lag2
    Rationale: Second derivative of price - captures acceleration/deceleration
    """
    close_lag1 = df.groupby('Ticker')['Close'].shift(1)
    close_lag2 = df.groupby('Ticker')['Close'].shift(2)
    current_change = (df['Close'] - close_lag1) / (close_lag1 + 0.001)
    previous_change = (close_lag1 - close_lag2) / (close_lag2 + 0.001)
    return current_change - previous_change

def alpha_129(df):
    """
    Intraday Volatility Ratio
    Formula: (High - Low) / (Open + Close) / 2
    Rationale: Daily volatility relative to average price level
    """
    daily_range = df['High'] - df['Low']
    avg_price = (df['Open'] + df['Close']) / 2 + 0.001
    return daily_range / avg_price

def alpha_130(df):
    """
    Volume-Price Momentum
    Formula: (Close / Open - 1) * log(Volume / ADV5 + 1)
    Rationale: Price momentum confirmed by volume surge
    """
    price_change = (df['Close'] / df['Open'] - 1)
    vol_surge = np.log(df['Volume'] / (df.groupby('Ticker')['Volume'].rolling(5, min_periods=1).mean().values) + 1)
    return price_change * vol_surge

def alpha_131(df):
    """
    Range Breakout Signal
    Formula: (Close - High_lag1) / High_lag1 if Close > High_lag1 else (Close - Low_lag1) / Low_lag1 if Close < Low_lag1 else 0
    Rationale: Captures breakouts from previous day's range
    """
    high_lag1 = df.groupby('Ticker')['High'].shift(1)
    low_lag1 = df.groupby('Ticker')['Low'].shift(1)
    upward_breakout = np.where(df['Close'] > high_lag1, 
                              (df['Close'] - high_lag1) / (high_lag1 + 0.001), 0)
    downward_breakout = np.where(df['Close'] < low_lag1,
                                (df['Close'] - low_lag1) / (low_lag1 + 0.001), 0)
    return upward_breakout + downward_breakout

def alpha_132(df):
    """
    Wick-to-Body Ratio
    Formula: (High + Low - Open - Close) / (abs(Close - Open) + 0.001)
    Rationale: Measures market indecision (long wicks) vs. conviction (strong body)
    """
    total_wick = df['High'] + df['Low'] - df['Open'] - df['Close']
    body_size = np.abs(df['Close'] - df['Open']) + 0.001
    return total_wick / body_size

def alpha_133(df):
    """
    Momentum Persistence
    Formula: sign(Close - Close_lag1) * (Close - Open) / (High - Low + 0.001)
    Rationale: Stronger signal when intraday and inter-day momentum align
    """
    close_lag1 = df.groupby('Ticker')['Close'].shift(1)
    momentum_direction = np.sign(df['Close'] - close_lag1)
    intraday_momentum = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 0.001)
    return momentum_direction * intraday_momentum

def alpha_134(df):
    """
    Volume-Adjusted True Range
    Formula: True_Range / (Volume / ADV20 + 0.1)
    Rationale: True range normalized by volume activity
    """
    # Calculate True Range
    high_low = df['High'] - df['Low']
    high_close_prev = np.abs(df['High'] - df.groupby('Ticker')['Close'].shift(1))
    low_close_prev = np.abs(df['Low'] - df.groupby('Ticker')['Close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    vol_ratio = df['Volume'] / (df.groupby('Ticker')['Volume'].rolling(20, min_periods=1).mean().values) + 0.1
    return true_range / vol_ratio

def alpha_135(df):
    """
    Price Level Mean Reversion
    Formula: (MA5 - Close) / Close * (High - Low) / Close
    Rationale: Mean reversion strength weighted by relative volatility
    """
    ma5 = df.groupby('Ticker')['Close'].rolling(5, min_periods=1).mean().values
    reversion_signal = (ma5 - df['Close']) / (df['Close'] + 0.001)
    volatility_weight = (df['High'] - df['Low']) / (df['Close'] + 0.001)
    return reversion_signal * volatility_weight

def alpha_136(df):
    """
    Gap Fade Strategy
    Formula: -(Open - Close_lag1) / (High - Low + 0.001) if abs(gap) > 0.02 else 0
    Rationale: Fade large overnight gaps (mean reversion)
    """
    close_lag1 = df.groupby('Ticker')['Close'].shift(1)
    gap = (df['Open'] - close_lag1) / (close_lag1 + 0.001)
    gap_normalized = -gap / (df['High'] - df['Low'] + 0.001)
    # Only consider significant gaps
    return np.where(np.abs(gap) > 0.02, gap_normalized, 0)

def alpha_137(df):
    """
    Intraday Trend Strength
    Formula: (Close - Open) / abs(Close - Open) * sqrt(High - Low) / sqrt(Close)
    Rationale: Directional momentum weighted by volatility measure
    """
    direction = np.sign(df['Close'] - df['Open'])
    volatility_weight = np.sqrt(df['High'] - df['Low']) / np.sqrt(df['Close'] + 0.001)
    return direction * volatility_weight

def alpha_138(df):
    """
    Price-Volume Divergence
    Formula: rank(Close_change) - rank(Volume_change) over 5 days
    Rationale: Identify divergence between price and volume momentum
    """
    close_change = df['Close'] / df.groupby('Ticker')['Close'].shift(1) - 1
    volume_change = df['Volume'] / df.groupby('Ticker')['Volume'].shift(1) - 1
    # Rolling rank over 5 periods
    price_rank = df.groupby('Ticker')['Close'].rolling(5, min_periods=1).rank(pct=True).values
    volume_rank = df.groupby('Ticker')['Volume'].rolling(5, min_periods=1).rank(pct=True).values
    return price_rank - volume_rank

def alpha_139(df):
    """
    Volatility Breakout
    Formula: (High - Low) / MA20(High - Low) - 1 if > 0.5 else 0
    Rationale: Identify volatility expansion days
    """
    daily_range = df['High'] - df['Low']
    avg_range = df.groupby('Ticker')[daily_range.name if hasattr(daily_range, 'name') else 'range'].rolling(20, min_periods=1).mean().values
    # Create a temporary series for rolling calculation
    temp_df = df.copy()
    temp_df['daily_range'] = daily_range
    avg_range = temp_df.groupby('Ticker')['daily_range'].rolling(20, min_periods=1).mean().values
    vol_expansion = daily_range / (avg_range + 0.001) - 1
    return np.where(vol_expansion > 0.5, vol_expansion, 0)

def alpha_140(df):
    """
    Composite Momentum Signal
    Formula: Simple average of normalized momentum indicators
    Rationale: Ensemble of simple momentum signals (inspired by alpha_101's simplicity)
    """
    # Component 1: Normalized intraday momentum (alpha_101 style)
    momentum1 = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 0.001)
    # Component 2: Range position
    momentum2 = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001) - 0.5
    # Component 3: Inter-day momentum
    close_lag1 = df.groupby('Ticker')['Close'].shift(1)
    momentum3 = (df['Close'] - close_lag1) / (df['High'] - df['Low'] + 0.001)
    # Simple average (equal weights)
    return (momentum1 + momentum2 + momentum3) / 3

def add_enhanced_alphas(df):
    """
    Add enhanced alpha factors (alpha_121 to alpha_140) to the DataFrame
    
    These factors are designed based on alpha_101's success pattern:
    - Simple, robust formulations
    - Volatility normalization
    - Focus on intraday patterns
    - Minimal parameter complexity
    """
    print("Adding enhanced alpha factors (alpha_121 to alpha_140)...")
    print("Focus: Simple, robust, volatility-adjusted momentum strategies")
    
    # Add each enhanced alpha factor
    enhanced_alphas = [
        alpha_121, alpha_122, alpha_123, alpha_124, alpha_125,
        alpha_126, alpha_127, alpha_128, alpha_129, alpha_130,
        alpha_131, alpha_132, alpha_133, alpha_134, alpha_135,
        alpha_136, alpha_137, alpha_138, alpha_139, alpha_140
    ]
    
    for i, alpha_func in enumerate(enhanced_alphas, start=121):
        try:
            alpha_name = f'alpha_{i}'
            print(f"  Computing {alpha_name}...")
            df[alpha_name] = alpha_func(df)
            
            # Handle any infinite or NaN values
            df[alpha_name] = df[alpha_name].replace([np.inf, -np.inf], np.nan)
            df[alpha_name] = df[alpha_name].fillna(0)
            
            print(f"  ✓ {alpha_name} computed successfully")
            
        except Exception as e:
            print(f"  ✗ Error computing {alpha_name}: {str(e)}")
            df[alpha_name] = 0
    
    print(f"Enhanced alpha factors complete: {len(enhanced_alphas)} new factors added!")
    print("\nKey innovations in this enhanced set:")
    print("1. Volatility normalization (following alpha_101's success)")
    print("2. Simple, robust formulations")
    print("3. Intraday pattern focus")
    print("4. Volume confirmation signals")
    print("5. Range-based momentum measures")
    
    return df

# Additional research-based alpha factors
import pandas as pd
import numpy as np
try:
    import talib
except ImportError:
    talib = None

def alpha_102(df):
    """
    Volatility-Volume Momentum Alpha
    High volume during low volatility periods may predict future momentum
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate 10-day volatility
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['volatility_10'] = df.groupby('Ticker')['returns'].rolling(10).std()
    
    # Calculate volume momentum
    df['volume_ma_5'] = df.groupby('Ticker')['Volume'].rolling(5).mean()
    df['volume_ma_20'] = df.groupby('Ticker')['Volume'].rolling(20).mean()
    df['volume_momentum'] = df['volume_ma_5'] / df['volume_ma_20']
    
    # Combine: High volume momentum during low volatility
    result = -df['volatility_10'] * df['volume_momentum']
    
    return result.fillna(0)

def alpha_103(df):
    """
    Gap Reversion Alpha
    Overnight gaps tend to revert during the day
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate overnight gap
    df['prev_close'] = df.groupby('Ticker')['Close'].shift(1)
    df['gap'] = (df['Open'] - df['prev_close']) / df['prev_close']
    
    # Calculate intraday move
    df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
    
    # Gap reversion signal: negative correlation between gap and expected return
    result = -df['gap']  # Simple reversion
    
    return result.fillna(0)

def alpha_104(df):
    """
    Volume-Price Trend (VPT) Momentum
    Combines price and volume momentum
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate price change
    df['price_change'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Calculate VPT
    df['vpt'] = (df['Volume'] * df['price_change']).fillna(0)
    df['vpt_cumsum'] = df.groupby('Ticker')['vpt'].cumsum()
    
    # VPT momentum
    df['vpt_ma_5'] = df.groupby('Ticker')['vpt_cumsum'].rolling(5).mean()
    df['vpt_ma_20'] = df.groupby('Ticker')['vpt_cumsum'].rolling(20).mean()
    
    result = df['vpt_ma_5'] - df['vpt_ma_20']
    
    return result.fillna(0)

def alpha_105(df):
    """
    Price-Volume Divergence Alpha
    When price moves but volume doesn't confirm, expect reversion
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Price momentum
    df['price_momentum'] = df.groupby('Ticker')['Close'].pct_change(5)
    
    # Volume momentum
    df['volume_ma_5'] = df.groupby('Ticker')['Volume'].rolling(5).mean()
    df['volume_ma_20'] = df.groupby('Ticker')['Volume'].rolling(20).mean()
    df['volume_momentum'] = (df['volume_ma_5'] / df['volume_ma_20'] - 1)
    
    # Divergence signal: price up but volume down (or vice versa)
    result = -df['price_momentum'] * np.sign(df['volume_momentum'])
    
    return result.fillna(0)

def alpha_106(df):
    """
    Intraday Range Expansion Alpha
    When today's range expands vs yesterday, expect momentum continuation
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate true range
    df['high_low'] = df['High'] - df['Low']
    df['high_close'] = np.abs(df['High'] - df.groupby('Ticker')['Close'].shift(1))
    df['low_close'] = np.abs(df['Low'] - df.groupby('Ticker')['Close'].shift(1))
    df['true_range'] = np.maximum(df['high_low'], np.maximum(df['high_close'], df['low_close']))
    
    # Range expansion
    df['prev_tr'] = df.groupby('Ticker')['true_range'].shift(1)
    df['range_expansion'] = df['true_range'] / df['prev_tr'] - 1
    
    # Return direction
    df['return_direction'] = np.sign(df.groupby('Ticker')['Close'].pct_change())
    
    result = df['range_expansion'] * df['return_direction']
    
    return result.fillna(0)

def alpha_107(df):
    """
    Volume Spike Reversion Alpha
    Unusual volume spikes often lead to price reversion
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Volume z-score
    df['volume_ma_20'] = df.groupby('Ticker')['Volume'].rolling(20).mean()
    df['volume_std_20'] = df.groupby('Ticker')['Volume'].rolling(20).std()
    df['volume_zscore'] = (df['Volume'] - df['volume_ma_20']) / df['volume_std_20']
    
    # Price change
    df['price_change'] = df.groupby('Ticker')['Close'].pct_change()
    
    # When volume spike is high, expect reversion
    result = -df['volume_zscore'] * df['price_change']
    
    return result.fillna(0)

def alpha_108(df):
    """
    Momentum Persistence Alpha
    Measures if momentum is accelerating or decelerating
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate momentum at different horizons
    df['mom_3'] = df.groupby('Ticker')['Close'].pct_change(3)
    df['mom_6'] = df.groupby('Ticker')['Close'].pct_change(6)
    df['mom_12'] = df.groupby('Ticker')['Close'].pct_change(12)
    
    # Momentum acceleration
    df['mom_accel_1'] = df['mom_3'] - df['mom_6']/2
    df['mom_accel_2'] = df['mom_6'] - df['mom_12']/2
    
    result = df['mom_accel_1'] + 0.5 * df['mom_accel_2']
    
    return result.fillna(0)

def alpha_109(df):
    """
    Bollinger Band Position Alpha
    Position within Bollinger Bands predicts mean reversion
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df.groupby('Ticker')['Close'].rolling(20).mean()
    df['bb_std'] = df.groupby('Ticker')['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Position within bands (-1 to 1)
    df['bb_position'] = (df['Close'] - df['bb_middle']) / (df['bb_upper'] - df['bb_middle'])
    
    # Mean reversion signal
    result = -df['bb_position']
    
    return result.fillna(0)

def alpha_110(df):
    """
    Price-Volume Correlation Alpha
    Rolling correlation between price and volume changes
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns
    df['price_change'] = df.groupby('Ticker')['Close'].pct_change()
    df['volume_change'] = df.groupby('Ticker')['Volume'].pct_change()
    
    # Rolling correlation
    def rolling_corr(group):
        return group['price_change'].rolling(10).corr(group['volume_change'])
    
    df['pv_corr'] = df.groupby('Ticker').apply(rolling_corr).reset_index(level=0, drop=True)
    
    # Current price momentum
    df['price_momentum'] = df.groupby('Ticker')['Close'].pct_change(5)
    
    # Signal: when correlation is high, momentum is more reliable
    result = df['pv_corr'] * df['price_momentum']
    
    return result.fillna(0)

def alpha_111(df):
    """
    Liquidity-Adjusted Momentum Alpha
    Adjusts momentum by trading liquidity (volume/volatility)
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate momentum
    df['momentum'] = df.groupby('Ticker')['Close'].pct_change(10)
    
    # Calculate liquidity proxy
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['volatility'] = df.groupby('Ticker')['returns'].rolling(10).std()
    df['liquidity'] = df['Volume'] / (df['volatility'] * df['Close'])
    
    # Liquidity-adjusted momentum
    result = df['momentum'] * np.log(1 + df['liquidity'])
    
    return result.fillna(0)

def alpha_112(df):
    """
    Overnight-Intraday Return Spread Alpha
    Exploits difference between overnight and intraday returns
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Overnight return
    df['prev_close'] = df.groupby('Ticker')['Close'].shift(1)
    df['overnight_return'] = (df['Open'] - df['prev_close']) / df['prev_close']
    
    # Intraday return
    df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
    
    # Return spread
    df['return_spread'] = df['intraday_return'] - df['overnight_return']
    
    # Moving average of spread
    result = df.groupby('Ticker')['return_spread'].rolling(5).mean().reset_index(level=0, drop=True)
    
    return result.fillna(0)

def alpha_113(df):
    """
    Volume-Weighted Price Momentum Alpha
    Momentum weighted by volume distribution
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate VWAP
    df['dollar_volume'] = df['Close'] * df['Volume']
    df['vwap_5'] = df.groupby('Ticker')['dollar_volume'].rolling(5).sum() / df.groupby('Ticker')['Volume'].rolling(5).sum()
    df['vwap_20'] = df.groupby('Ticker')['dollar_volume'].rolling(20).sum() / df.groupby('Ticker')['Volume'].rolling(20).sum()
    
    # VWAP momentum
    df['vwap_momentum'] = (df['vwap_5'] - df['vwap_20']) / df['vwap_20']
    
    # Current price vs VWAP
    df['price_vs_vwap'] = (df['Close'] - df['vwap_5']) / df['vwap_5']
    
    result = df['vwap_momentum'] + 0.5 * df['price_vs_vwap']
    
    return result.fillna(0)

def alpha_114(df):
    """
    Volatility Regime Change Alpha
    Detects changes in volatility regime
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Short and long term volatility
    df['vol_short'] = df.groupby('Ticker')['returns'].rolling(5).std()
    df['vol_long'] = df.groupby('Ticker')['returns'].rolling(20).std()
    
    # Volatility ratio
    df['vol_ratio'] = df['vol_short'] / df['vol_long']
    
    # Change in volatility ratio
    df['vol_ratio_change'] = df.groupby('Ticker')['vol_ratio'].pct_change()
    
    # Price momentum
    df['price_momentum'] = df.groupby('Ticker')['Close'].pct_change(3)
    
    # When volatility increases, momentum may continue
    result = df['vol_ratio_change'] * df['price_momentum']
    
    return result.fillna(0)

def alpha_115(df):
    """
    Support-Resistance Breakout Alpha
    Detects breakouts from recent price ranges
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate support and resistance levels
    df['resistance'] = df.groupby('Ticker')['High'].rolling(20).max()
    df['support'] = df.groupby('Ticker')['Low'].rolling(20).min()
    df['range_size'] = (df['resistance'] - df['support']) / df['Close']
    
    # Breakout signals
    df['resistance_breakout'] = np.where(df['Close'] > df['resistance'], 1, 0)
    df['support_breakout'] = np.where(df['Close'] < df['support'], -1, 0)
    
    # Volume confirmation
    df['volume_ma'] = df.groupby('Ticker')['Volume'].rolling(20).mean()
    df['volume_multiplier'] = np.minimum(df['Volume'] / df['volume_ma'], 3)  # Cap at 3x
    
    result = (df['resistance_breakout'] + df['support_breakout']) * df['volume_multiplier']
    
    return result.fillna(0)

def alpha_116(df):
    """
    Price Clustering Alpha
    Exploits tendency of prices to cluster at round numbers
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate distance from round numbers
    df['close_rounded'] = np.round(df['Close'])
    df['distance_from_round'] = np.abs(df['Close'] - df['close_rounded']) / df['Close']
    
    # Price momentum
    df['momentum'] = df.groupby('Ticker')['Close'].pct_change(5)
    
    # When price is near round number, momentum may stall
    result = -df['distance_from_round'] * df['momentum']
    
    return result.fillna(0)

def alpha_117(df):
    """
    Volatility-Adjusted Price Alpha
    Price momentum adjusted for volatility risk
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns and volatility
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['volatility'] = df.groupby('Ticker')['returns'].rolling(20).std()
    
    # Risk-adjusted momentum
    df['momentum'] = df.groupby('Ticker')['Close'].pct_change(10)
    df['risk_adjusted_momentum'] = df['momentum'] / df['volatility']
    
    result = df['risk_adjusted_momentum']
    
    return result.fillna(0)

def alpha_118(df):
    """
    Market Microstructure Alpha
    Uses bid-ask spread proxy from high-low range
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Proxy for bid-ask spread
    df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
    df['spread_ma'] = df.groupby('Ticker')['spread_proxy'].rolling(20).mean()
    df['spread_ratio'] = df['spread_proxy'] / df['spread_ma']
    
    # Price momentum
    df['momentum'] = df.groupby('Ticker')['Close'].pct_change(5)
    
    # When spread is narrow (high liquidity), momentum more reliable
    result = df['momentum'] / df['spread_ratio']
    
    return result.fillna(0)

def alpha_119(df):
    """
    Trend Strength Alpha
    Measures the consistency of price trends
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate daily returns
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Count positive and negative days in rolling window
    df['positive_days'] = df.groupby('Ticker')['returns'].rolling(10).apply(lambda x: (x > 0).sum())
    df['trend_strength'] = (df['positive_days'] - 5) / 5  # Normalize to -1 to 1
    
    # Current momentum
    df['momentum'] = df.groupby('Ticker')['Close'].pct_change(3)
    
    # Strong trend suggests momentum continuation
    result = df['trend_strength'] * df['momentum']
    
    return result.fillna(0)

def alpha_120(df):
    """
    Volume Momentum Divergence Alpha
    Compares price and volume momentum directions
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Price momentum
    df['price_momentum'] = df.groupby('Ticker')['Close'].pct_change(10)
    
    # Volume momentum
    df['volume_momentum'] = df.groupby('Ticker')['Volume'].pct_change(10)
    
    # When volume momentum confirms price momentum, signal is stronger
    df['momentum_alignment'] = np.sign(df['price_momentum']) * np.sign(df['volume_momentum'])
    
    result = df['price_momentum'] * df['momentum_alignment']
    
    return result.fillna(0)

# Function to add all new alphas to existing dataframe
def add_additional_alphas(df):
    """
    Add all additional alpha factors to the dataframe
    """
    print("Adding additional alpha factors...")
    
    additional_alphas = [
        alpha_102, alpha_103, alpha_104, alpha_105, alpha_106,
        alpha_107, alpha_108, alpha_109, alpha_110, alpha_111,
        alpha_112, alpha_113, alpha_114, alpha_115, alpha_116,
        alpha_117, alpha_118, alpha_119, alpha_120
    ]
    
    for i, alpha_func in enumerate(additional_alphas, 102):
        try:
            print(f"Processing alpha_{i}...")
            df[f"alpha_{i}"] = alpha_func(df)
            print(f"Successfully added alpha_{i}")
        except Exception as e:
            print(f"Error processing alpha_{i}: {e}")
            df[f"alpha_{i}"] = 0
            
    return df

if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        'Date': pd.date_range('2023-01-01', periods=100),
        'Ticker': ['BTC'] * 100,
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }
    
    test_df = pd.DataFrame(sample_data)
    result = add_additional_alphas(test_df)
    print(f"Added {len([col for col in result.columns if col.startswith('alpha_') and int(col.split('_')[1]) >= 102])} new alpha factors")
    print("New alpha columns:", [col for col in result.columns if col.startswith('alpha_') and int(col.split('_')[1]) >= 102])
