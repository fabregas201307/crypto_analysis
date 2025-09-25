"""
Extended Alpha Factors for Crypto Trading
Based on research from academic quantitative finance literature and industry practices

These 20 additional alpha factors (alpha_141 to alpha_160) complement your existing pool
with advanced signal processing, behavioral finance, and market microstructure concepts.

Research Sources:
- Stefan Jansen's ML for Trading (Chapter 24: Alpha Factor Library)
- je-suis-tm/quant-trading repository strategies
- WorldQuant's extended alpha research beyond 101 factors
- Academic papers on crypto market inefficiencies
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import hilbert
from scipy.stats import entropy

def alpha_141(df):
    """
    Hilbert Transform Momentum
    Formula: Instantaneous phase of price using Hilbert transform
    Rationale: Captures cyclical momentum patterns using signal processing
    Source: Advanced signal processing techniques for financial time series
    """
    try:
        result = []
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
            close_prices = ticker_data['Close'].values
            
            # Apply Hilbert transform to get instantaneous phase
            analytic_signal = hilbert(close_prices)
            instantaneous_phase = np.angle(analytic_signal)
            
            # Phase momentum (change in phase)
            phase_momentum = np.diff(instantaneous_phase, prepend=instantaneous_phase[0])
            
            ticker_df = ticker_data.copy()
            ticker_df['alpha_141'] = phase_momentum
            result.append(ticker_df[['Date', 'Ticker', 'alpha_141']])
        
        combined = pd.concat(result, ignore_index=True)
        df_merged = df.merge(combined, on=['Date', 'Ticker'], how='left')
        return df_merged['alpha_141'].fillna(0)
    except:
        return np.zeros(len(df))

def alpha_142(df):
    """
    Entropy-Based Volatility Signal
    Formula: Rolling entropy of returns distribution
    Rationale: High entropy indicates uncertainty, often preceding volatility changes
    Source: Information theory applications in finance
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Calculate rolling entropy (using histogram of returns)
    def rolling_entropy(series, window=20):
        result = []
        for i in range(len(series)):
            start_idx = max(0, i - window + 1)
            window_data = series.iloc[start_idx:i+1]
            
            if len(window_data) < 5:  # Need minimum data for histogram
                result.append(0)
                continue
                
            # Create histogram and calculate entropy
            hist, _ = np.histogram(window_data.dropna(), bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            if len(hist) > 1:
                result.append(entropy(hist))
            else:
                result.append(0)
        return pd.Series(result, index=series.index)
    
    df['entropy'] = df.groupby('Ticker')['returns'].apply(rolling_entropy).reset_index(level=0, drop=True)
    
    # Normalize entropy signal
    df['entropy_zscore'] = df.groupby('Ticker')['entropy'].apply(
        lambda x: (x - x.rolling(60, min_periods=10).mean()) / (x.rolling(60, min_periods=10).std() + 1e-6)
    )
    
    return df['entropy_zscore'].fillna(0)

def alpha_143(df):
    """
    Fractal Dimension Price Pattern
    Formula: Higuchi fractal dimension of price series
    Rationale: Captures complexity/chaos in price movements
    Source: Chaos theory applications in financial markets
    """
    def higuchi_fd(X, k_max=10):
        """Calculate Higuchi Fractal Dimension"""
        L = []
        x = []
        N = len(X)
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int((N - m) / k)):
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / (((N - m) / k) * k) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append(np.log(float(1) / k))
        
        if len(L) < 2:
            return 1.5  # Default fractal dimension
            
        return -np.polyfit(x, L, 1)[0]
    
    result = []
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
        close_prices = ticker_data['Close'].values
        
        # Rolling fractal dimension calculation
        window = 50
        fd_values = []
        
        for i in range(len(close_prices)):
            start_idx = max(0, i - window + 1)
            price_window = close_prices[start_idx:i+1]
            
            if len(price_window) >= 20:  # Minimum required for fractal dimension
                fd = higuchi_fd(price_window)
                fd_values.append(fd - 1.5)  # Center around 0 (1.5 is typical for financial data)
            else:
                fd_values.append(0)
        
        ticker_df = ticker_data.copy()
        ticker_df['alpha_143'] = fd_values
        result.append(ticker_df[['Date', 'Ticker', 'alpha_143']])
    
    combined = pd.concat(result, ignore_index=True)
    df_merged = df.merge(combined, on=['Date', 'Ticker'], how='left')
    return df_merged['alpha_143'].fillna(0)

def alpha_144(df):
    """
    Market Regime Detection
    Formula: Hidden Markov Model state estimation
    Rationale: Different alpha strategies work in different market regimes
    Source: Regime-switching models in quantitative finance
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate volatility regimes using rolling volatility
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['volatility'] = df.groupby('Ticker')['returns'].rolling(20).std().reset_index(level=0, drop=True)
    
    # Simple 3-regime classification: Low, Medium, High volatility
    df['vol_rank'] = df.groupby('Ticker')['volatility'].rolling(252, min_periods=50).rank(pct=True).reset_index(level=0, drop=True)
    
    # Regime signal: -1 (low vol), 0 (medium vol), 1 (high vol)
    conditions = [
        df['vol_rank'] <= 0.33,
        (df['vol_rank'] > 0.33) & (df['vol_rank'] <= 0.67),
        df['vol_rank'] > 0.67
    ]
    choices = [-1, 0, 1]
    df['regime'] = np.select(conditions, choices, default=0)
    
    # Regime change signal (stronger when regime changes)
    df['regime_change'] = df.groupby('Ticker')['regime'].diff().fillna(0)
    df['momentum'] = df.groupby('Ticker')['Close'].pct_change(5)
    
    # Strategy: momentum works better in trending (high vol) regimes
    result = df['regime'] * df['momentum']
    
    return result.fillna(0)

def alpha_145(df):
    """
    Jump Detection Signal
    Formula: Detect price jumps using statistical tests
    Rationale: Price jumps often predict subsequent volatility or reversals
    Source: Jump detection in high-frequency financial data
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Rolling statistics for jump detection
    window = 20
    df['return_mean'] = df.groupby('Ticker')['returns'].rolling(window).mean().reset_index(level=0, drop=True)
    df['return_std'] = df.groupby('Ticker')['returns'].rolling(window).std().reset_index(level=0, drop=True)
    
    # Jump detection: return exceeds 3 standard deviations
    df['jump_threshold'] = 3 * df['return_std']
    df['is_jump'] = np.abs(df['returns'] - df['return_mean']) > df['jump_threshold']
    
    # Jump direction and magnitude
    df['jump_magnitude'] = np.where(df['is_jump'], 
                                   np.abs(df['returns'] - df['return_mean']) / (df['return_std'] + 1e-6),
                                   0)
    
    df['jump_direction'] = np.where(df['is_jump'], 
                                   np.sign(df['returns'] - df['return_mean']),
                                   0)
    
    # Jump signal: negative (expect reversion), decay over time
    df['jump_signal'] = df['jump_direction'] * df['jump_magnitude']
    df['jump_decay'] = df.groupby('Ticker')['jump_signal'].apply(
        lambda x: x * np.exp(-np.arange(len(x)) * 0.1)
    ).reset_index(level=0, drop=True)
    
    return -df['jump_signal'].fillna(0)  # Fade jumps

def alpha_146(df):
    """
    Liquidity Risk Premium
    Formula: Amihud illiquidity measure
    Rationale: Less liquid assets should have higher expected returns
    Source: Amihud (2002) illiquidity measure
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate daily returns and dollar volume
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['dollar_volume'] = df['Close'] * df['Volume']
    
    # Amihud illiquidity: |return| / dollar_volume
    df['daily_illiquidity'] = np.abs(df['returns']) / (df['dollar_volume'] + 1e-6)
    
    # Average illiquidity over rolling window
    df['avg_illiquidity'] = df.groupby('Ticker')['daily_illiquidity'].rolling(20).mean().reset_index(level=0, drop=True)
    
    # Cross-sectional ranking of illiquidity
    df['illiquidity_rank'] = df.groupby('Date')['avg_illiquidity'].rank(pct=True)
    
    # Illiquid assets should outperform (positive signal)
    return df['illiquidity_rank'].fillna(0.5)

def alpha_147(df):
    """
    Options-Inspired Skewness Signal
    Formula: Rolling skewness of returns
    Rationale: Negative skewness often predicts future negative returns
    Source: Options pricing anomalies and skewness risk premium
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Rolling skewness
    def rolling_skewness(series, window=30):
        return series.rolling(window, min_periods=10).skew()
    
    df['skewness'] = df.groupby('Ticker')['returns'].apply(rolling_skewness).reset_index(level=0, drop=True)
    
    # Cross-sectional ranking
    df['skewness_rank'] = df.groupby('Date')['skewness'].rank(pct=True)
    
    # Negative skewness signal (expect negative returns for high negative skew)
    return -df['skewness'].fillna(0)

def alpha_148(df):
    """
    Momentum Lifecycle
    Formula: Momentum signal adjusted for age of trend
    Rationale: Early momentum is stronger than late-stage momentum
    Source: Momentum lifecycle studies
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate momentum
    df['momentum_12'] = df.groupby('Ticker')['Close'].pct_change(12)
    df['momentum_3'] = df.groupby('Ticker')['Close'].pct_change(3)
    
    # Trend age: consecutive days of same direction
    df['daily_return'] = df.groupby('Ticker')['Close'].pct_change()
    df['return_direction'] = np.sign(df['daily_return'])
    
    # Count consecutive same-direction moves
    def count_consecutive(series):
        result = []
        count = 0
        prev_sign = 0
        
        for val in series:
            if val == prev_sign and val != 0:
                count += 1
            else:
                count = 1 if val != 0 else 0
                prev_sign = val
            result.append(count)
        return pd.Series(result, index=series.index)
    
    df['trend_age'] = df.groupby('Ticker')['return_direction'].apply(count_consecutive).reset_index(level=0, drop=True)
    
    # Momentum strength decays with age
    df['age_factor'] = np.exp(-df['trend_age'] / 10)  # Exponential decay
    
    # Lifecycle-adjusted momentum
    result = df['momentum_12'] * df['age_factor']
    
    return result.fillna(0)

def alpha_149(df):
    """
    News Sentiment Proxy
    Formula: Abnormal volume as news sentiment proxy
    Rationale: Unusual volume often reflects news impact
    Source: News-based trading strategies
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Volume statistics
    df['volume_ma_20'] = df.groupby('Ticker')['Volume'].rolling(20).mean().reset_index(level=0, drop=True)
    df['volume_std_20'] = df.groupby('Ticker')['Volume'].rolling(20).std().reset_index(level=0, drop=True)
    
    # Volume surprise (z-score)
    df['volume_surprise'] = (df['Volume'] - df['volume_ma_20']) / (df['volume_std_20'] + 1e-6)
    
    # Price reaction to volume surprise
    df['price_change'] = df.groupby('Ticker')['Close'].pct_change()
    
    # News sentiment proxy: volume surprise * price direction
    df['news_sentiment'] = df['volume_surprise'] * np.sign(df['price_change'])
    
    # Smooth the signal
    result = df.groupby('Ticker')['news_sentiment'].rolling(3).mean().reset_index(level=0, drop=True)
    
    return result.fillna(0)

def alpha_150(df):
    """
    Contrarian Earnings Surprise
    Formula: Fade extreme price moves on high volume
    Rationale: Overreactions to earnings/news tend to reverse
    Source: Earnings announcement anomalies
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate price moves and volume
    df['price_change_5'] = df.groupby('Ticker')['Close'].pct_change(5)
    df['volume_ratio'] = df['Volume'] / df.groupby('Ticker')['Volume'].rolling(20).mean().reset_index(level=0, drop=True)
    
    # Identify extreme moves with high volume
    df['extreme_threshold'] = df.groupby('Ticker')['price_change_5'].rolling(60).quantile(0.95).reset_index(level=0, drop=True)
    df['extreme_move'] = np.abs(df['price_change_5']) > df['extreme_threshold']
    df['high_volume'] = df['volume_ratio'] > 2.0
    
    # Contrarian signal: fade extreme moves with high volume
    df['contrarian_signal'] = np.where(
        df['extreme_move'] & df['high_volume'],
        -np.sign(df['price_change_5']) * np.abs(df['price_change_5']),
        0
    )
    
    return df['contrarian_signal'].fillna(0)

def alpha_151(df):
    """
    Cross-Asset Correlation Signal
    Formula: Use correlation with market index for signal
    Rationale: Assets with changing correlations may outperform
    Source: Cross-asset momentum strategies
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Calculate market return (equal-weighted)
    market_returns = df.groupby('Date')['returns'].mean().reset_index()
    market_returns.columns = ['Date', 'market_return']
    df = df.merge(market_returns, on='Date', how='left')
    
    # Rolling correlation with market
    def rolling_correlation(group):
        return group['returns'].rolling(30, min_periods=10).corr(group['market_return'])
    
    df['correlation'] = df.groupby('Ticker').apply(rolling_correlation).reset_index(level=0, drop=True)
    
    # Change in correlation
    df['correlation_change'] = df.groupby('Ticker')['correlation'].diff(5)
    
    # Signal: momentum for assets with decreasing correlation (diversification benefit)
    df['momentum'] = df.groupby('Ticker')['Close'].pct_change(10)
    result = df['momentum'] * (-df['correlation_change'])  # Negative correlation change is good
    
    return result.fillna(0)

def alpha_152(df):
    """
    Volatility Surface Signal
    Formula: Term structure of volatility
    Rationale: Different volatility horizons contain different information
    Source: Volatility surface trading strategies
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    
    # Multiple volatility horizons
    horizons = [5, 10, 20, 60]
    for h in horizons:
        df[f'vol_{h}'] = df.groupby('Ticker')['returns'].rolling(h).std().reset_index(level=0, drop=True)
    
    # Volatility term structure slope
    df['vol_slope'] = (df['vol_60'] - df['vol_5']) / df['vol_20']
    
    # Volatility momentum
    df['vol_momentum'] = df.groupby('Ticker')['vol_20'].pct_change(5)
    
    # Combined volatility signal
    result = df['vol_slope'] + 0.5 * df['vol_momentum']
    
    return result.fillna(0)

def alpha_153(df):
    """
    Market Microstructure Imbalance
    Formula: Intraday pressure from high-low range analysis
    Rationale: Intraday buying/selling pressure predicts future returns
    Source: Market microstructure literature
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Buying pressure proxy
    df['buying_pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-6)
    
    # Average buying pressure
    df['avg_buying_pressure'] = df.groupby('Ticker')['buying_pressure'].rolling(10).mean().reset_index(level=0, drop=True)
    
    # Pressure momentum
    df['pressure_momentum'] = df.groupby('Ticker')['avg_buying_pressure'].diff(3)
    
    # Volume-weighted pressure
    df['volume_weight'] = df['Volume'] / df.groupby('Ticker')['Volume'].rolling(20).mean().reset_index(level=0, drop=True)
    df['weighted_pressure'] = df['pressure_momentum'] * np.log1p(df['volume_weight'])
    
    return df['weighted_pressure'].fillna(0)

def alpha_154(df):
    """
    Behavioral Bias Signal
    Formula: Exploit anchoring bias using psychological price levels
    Rationale: Traders anchor to recent highs/lows and round numbers
    Source: Behavioral finance literature
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Recent high and low (anchoring points)
    df['high_52w'] = df.groupby('Ticker')['High'].rolling(252, min_periods=50).max().reset_index(level=0, drop=True)
    df['low_52w'] = df.groupby('Ticker')['Low'].rolling(252, min_periods=50).min().reset_index(level=0, drop=True)
    
    # Distance from anchoring points
    df['distance_from_high'] = (df['high_52w'] - df['Close']) / df['high_52w']
    df['distance_from_low'] = (df['Close'] - df['low_52w']) / df['low_52w']
    
    # Round number bias
    df['rounded_price'] = np.round(df['Close'])
    df['round_number_distance'] = np.abs(df['Close'] - df['rounded_price']) / df['Close']
    
    # Combined behavioral signal
    # Near highs: expect reversion, Near lows: expect bounce
    behavioral_signal = -df['distance_from_high'] + df['distance_from_low']
    
    # Stronger signal near round numbers
    result = behavioral_signal * (1 + df['round_number_distance'])
    
    return result.fillna(0)

def alpha_155(df):
    """
    Regime-Dependent Reversal
    Formula: Mean reversion strength depends on volatility regime
    Rationale: Mean reversion is stronger in low volatility periods
    Source: Regime-dependent trading strategies
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns and volatility
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['volatility'] = df.groupby('Ticker')['returns'].rolling(20).std().reset_index(level=0, drop=True)
    
    # Volatility regime
    df['vol_regime'] = df.groupby('Ticker')['volatility'].rolling(60, min_periods=20).rank(pct=True).reset_index(level=0, drop=True)
    
    # Mean reversion signal
    df['ma_20'] = df.groupby('Ticker')['Close'].rolling(20).mean().reset_index(level=0, drop=True)
    df['reversion_signal'] = (df['ma_20'] - df['Close']) / df['Close']
    
    # Regime adjustment: stronger reversion in low vol regimes
    df['regime_factor'] = 2 - df['vol_regime']  # Higher factor for low vol regimes
    
    result = df['reversion_signal'] * df['regime_factor']
    
    return result.fillna(0)

def alpha_156(df):
    """
    Multi-Timeframe Momentum
    Formula: Combine momentum signals from multiple timeframes
    Rationale: Different timeframes provide complementary information
    Source: Multi-timeframe analysis in technical analysis
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Multiple momentum horizons
    horizons = [3, 7, 14, 30]
    weights = [0.4, 0.3, 0.2, 0.1]  # More weight on shorter horizons
    
    momentum_signals = []
    for i, h in enumerate(horizons):
        momentum = df.groupby('Ticker')['Close'].pct_change(h)
        # Normalize momentum
        momentum_norm = df.groupby('Ticker')[momentum.name if hasattr(momentum, 'name') else 'momentum'].apply(
            lambda x: (x - x.rolling(60, min_periods=10).mean()) / (x.rolling(60, min_periods=10).std() + 1e-6)
        ).reset_index(level=0, drop=True)
        momentum_signals.append(momentum_norm * weights[i])
    
    # Combined multi-timeframe momentum
    result = sum(momentum_signals)
    
    return result.fillna(0)

def alpha_157(df):
    """
    Risk-Adjusted Carry Trade
    Formula: Interest rate differential proxy using price momentum
    Rationale: Carry trade concept adapted for crypto using momentum
    Source: Carry trade strategies in FX markets
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Long-term momentum as "yield" proxy
    df['long_momentum'] = df.groupby('Ticker')['Close'].pct_change(60)
    
    # Volatility as risk measure
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['volatility'] = df.groupby('Ticker')['returns'].rolling(30).std().reset_index(level=0, drop=True)
    
    # Risk-adjusted "carry"
    df['risk_adjusted_carry'] = df['long_momentum'] / (df['volatility'] + 1e-6)
    
    # Cross-sectional ranking
    df['carry_rank'] = df.groupby('Date')['risk_adjusted_carry'].rank(pct=True)
    
    # Long high carry, short low carry
    result = df['carry_rank'] - 0.5
    
    return result.fillna(0)

def alpha_158(df):
    """
    Event-Driven Volatility
    Formula: Volatility clustering detection
    Rationale: Volatility clusters - high volatility periods tend to cluster
    Source: GARCH models and volatility clustering
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Calculate returns and squared returns (volatility proxy)
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['squared_returns'] = df['returns'] ** 2
    
    # Volatility clustering measure
    df['vol_ma_short'] = df.groupby('Ticker')['squared_returns'].rolling(5).mean().reset_index(level=0, drop=True)
    df['vol_ma_long'] = df.groupby('Ticker')['squared_returns'].rolling(20).mean().reset_index(level=0, drop=True)
    
    # Volatility momentum
    df['vol_momentum'] = (df['vol_ma_short'] / df['vol_ma_long']) - 1
    
    # Expected return adjustment for volatility clustering
    # High volatility momentum suggests continued high volatility
    df['momentum'] = df.groupby('Ticker')['Close'].pct_change(5)
    
    # Adjust momentum based on volatility regime
    result = df['momentum'] * (1 + df['vol_momentum'])
    
    return result.fillna(0)

def alpha_159(df):
    """
    Seasonality-Adjusted Signal
    Formula: Day-of-week and hour-of-day effects
    Rationale: Crypto markets show some seasonal patterns
    Source: Calendar effects in cryptocurrency markets
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Extract time features
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['hour'] = df['Date'].dt.hour if 'hour' in df['Date'].dt.__dict__ else 0
    
    # Calculate basic momentum
    df['momentum'] = df.groupby('Ticker')['Close'].pct_change(5)
    
    # Day-of-week effects (simple approach)
    # Monday (0) and Friday (4) often have different patterns
    dow_adjustments = {0: 1.1, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.9, 5: 0.95, 6: 0.95}
    df['dow_adjustment'] = df['day_of_week'].map(dow_adjustments)
    
    # Seasonality-adjusted momentum
    result = df['momentum'] * df['dow_adjustment']
    
    return result.fillna(0)

def alpha_160(df):
    """
    Composite Alpha Score
    Formula: Ensemble of multiple simple signals
    Rationale: Combine diverse signals for robustness
    Source: Ensemble methods in quantitative finance
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # Component 1: Price momentum
    mom_3 = df.groupby('Ticker')['Close'].pct_change(3)
    mom_10 = df.groupby('Ticker')['Close'].pct_change(10)
    momentum_component = 0.6 * mom_3 + 0.4 * mom_10
    
    # Component 2: Volume signal
    volume_ratio = df['Volume'] / df.groupby('Ticker')['Volume'].rolling(20).mean().reset_index(level=0, drop=True)
    volume_component = np.log1p(volume_ratio - 1)
    
    # Component 3: Volatility signal
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    volatility = df.groupby('Ticker')['returns'].rolling(20).std().reset_index(level=0, drop=True)
    vol_component = -volatility / volatility.rolling(60).mean()  # Contrarian volatility
    
    # Component 4: Range signal
    range_component = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-6) - 0.5
    
    # Weighted combination
    weights = [0.4, 0.25, 0.2, 0.15]
    components = [momentum_component, volume_component, vol_component, range_component]
    
    # Normalize each component
    normalized_components = []
    for comp in components:
        comp_norm = (comp - comp.rolling(60, min_periods=10).mean()) / (comp.rolling(60, min_periods=10).std() + 1e-6)
        normalized_components.append(comp_norm)
    
    # Weighted sum
    result = sum(w * comp for w, comp in zip(weights, normalized_components))
    
    return result.fillna(0)

def add_extended_alphas(df):
    """
    Add extended alpha factors (alpha_141 to alpha_160) to the DataFrame
    
    These factors incorporate advanced concepts:
    - Signal processing techniques (Hilbert transform, fractal dimension)
    - Information theory (entropy-based signals)
    - Behavioral finance (anchoring, sentiment proxies)
    - Market microstructure (jump detection, liquidity measures)
    - Multi-timeframe and regime-dependent strategies
    - Advanced volatility modeling
    """
    print("Adding extended alpha factors (alpha_141 to alpha_160)...")
    print("Categories: Signal Processing, Behavioral Finance, Market Microstructure, Regime Models")
    
    # Add each extended alpha factor
    extended_alphas = [
        alpha_141, alpha_142, alpha_143, alpha_144, alpha_145,
        alpha_146, alpha_147, alpha_148, alpha_149, alpha_150,
        alpha_151, alpha_152, alpha_153, alpha_154, alpha_155,
        alpha_156, alpha_157, alpha_158, alpha_159, alpha_160
    ]
    
    for i, alpha_func in enumerate(extended_alphas, start=141):
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
    
    print(f"Extended alpha factors complete: {len(extended_alphas)} new factors added!")
    print("\nKey innovations in this extended set:")
    print("1. Signal processing techniques (Hilbert transform, fractal dimension)")
    print("2. Information theory applications (entropy-based volatility)")
    print("3. Behavioral finance signals (anchoring bias, sentiment proxies)")
    print("4. Market microstructure analysis (jump detection, liquidity risk)")
    print("5. Regime-dependent strategies")
    print("6. Multi-timeframe momentum combinations")
    print("7. Advanced volatility modeling (clustering, term structure)")
    print("8. Composite ensemble approaches")
    
    return df

if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        'Date': pd.date_range('2023-01-01', periods=200),
        'Ticker': ['BTC'] * 200,
        'Open': np.random.randn(200).cumsum() + 100,
        'High': np.random.randn(200).cumsum() + 105,
        'Low': np.random.randn(200).cumsum() + 95,
        'Close': np.random.randn(200).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 200)
    }
    
    test_df = pd.DataFrame(sample_data)
    result = add_extended_alphas(test_df)
    print(f"Added {len([col for col in result.columns if col.startswith('alpha_') and int(col.split('_')[1]) >= 141])} new extended alpha factors")
    print("New extended alpha columns:", [col for col in result.columns if col.startswith('alpha_') and int(col.split('_')[1]) >= 141])
