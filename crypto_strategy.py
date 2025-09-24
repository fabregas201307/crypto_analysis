"""
Crypto Strategy Analysis with Extended Alpha Factors
Following the golden copy approach from enhanced_crypto_analysis.py

This implementation:
- Loads Binance data from "data/Binance_AllCrypto_d.csv" 
- Extends beyond original 101 alpha factors using additional modules
- Builds combined alpha strategy 
- Compares performance against BTC buy-and-hold benchmark
- Creates comprehensive visualizations and performance metrics

Extended alpha factors include:
- Original WorldQuant alphas (1-101)
- Research-based factors (102-120) 
- Enhanced factors (121-140)
- Extended factors (141-160)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scipy for statistical functions
from scipy import stats

# Import alpha factor modules (following enhanced_crypto_analysis.py)
from additional_alpha_factors import add_additional_alphas, add_enhanced_alphas
from extended_alpha_factors import add_extended_alphas

def load_binance_data():
    """Load cryptocurrency data from Binance CSV file (following golden copy)"""
    print("Loading Binance cryptocurrency data from local files...")
    
    try:
        # Load the combined dataframe (exactly as in enhanced_crypto_analysis.py)
        df = pd.read_csv("data/Binance_AllCrypto_d.csv")
        df.index = range(df.shape[0])
        
        # Convert Date column to proper datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date'])
        
        # Show data range and assets
        print(f"Data loaded: {len(df)} records")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Assets: {sorted(df['Ticker'].unique())}")
        
        return df
        
    except FileNotFoundError:
        print("Error: data/Binance_AllCrypto_d.csv not found!")
        print("Please run crypto_data_download.py first to download the data.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def calculate_extended_alpha_factors(df):
    """Calculate all extended alpha factors (beyond original 101)"""
    print("\n" + "="*60)
    print("CALCULATING EXTENDED ALPHA FACTOR STRATEGY")
    print("="*60)
    
    # Import the prepare_df function and original alphas from crypto_data_analysis_new.py
    sys.path.append('/Users/kaiwen/Desktop/projects/crypto_fund')
    
    try:
        from crypto_data_analysis_new import prepare_df
        # Import the original alpha functions (1-101)
        from crypto_data_analysis_new import alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8, alpha9, alpha10
        from crypto_data_analysis_new import alpha11, alpha12, alpha13, alpha14, alpha15, alpha16, alpha17, alpha18, alpha19, alpha20
        from crypto_data_analysis_new import alpha21, alpha22, alpha23, alpha24, alpha25, alpha26, alpha27, alpha28, alpha29, alpha30
        from crypto_data_analysis_new import alpha31, alpha32, alpha33, alpha34, alpha35, alpha36, alpha37, alpha38, alpha39, alpha40
        from crypto_data_analysis_new import alpha41, alpha42, alpha43, alpha44, alpha45, alpha46, alpha47, alpha48, alpha49, alpha50
        from crypto_data_analysis_new import alpha51, alpha52, alpha53, alpha54, alpha55, alpha56, alpha57, alpha58, alpha59, alpha60
        from crypto_data_analysis_new import alpha61, alpha62, alpha63, alpha64, alpha65, alpha66, alpha67, alpha68, alpha69, alpha70
        from crypto_data_analysis_new import alpha71, alpha72, alpha73, alpha74, alpha75, alpha76, alpha77, alpha78, alpha79, alpha80
        from crypto_data_analysis_new import alpha81, alpha82, alpha83, alpha84, alpha85, alpha86, alpha87, alpha88, alpha89, alpha90
        from crypto_data_analysis_new import alpha91, alpha92, alpha93, alpha94, alpha95, alpha96, alpha97, alpha98, alpha99, alpha100, alpha101
        
        # Prepare common columns (following enhanced_crypto_analysis.py approach)
        df = prepare_df(df)
        
        # Compute original alphas (1-101)
        print("Computing original WorldQuant alpha factors (1-101)...")
        alpha_functions = [
            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8, alpha9, alpha10,
            alpha11, alpha12, alpha13, alpha14, alpha15, alpha16, alpha17, alpha18, alpha19, alpha20,
            alpha21, alpha22, alpha23, alpha24, alpha25, alpha26, alpha27, alpha28, alpha29, alpha30,
            alpha31, alpha32, alpha33, alpha34, alpha35, alpha36, alpha37, alpha38, alpha39, alpha40,
            alpha41, alpha42, alpha43, alpha44, alpha45, alpha46, alpha47, alpha48, alpha49, alpha50,
            alpha51, alpha52, alpha53, alpha54, alpha55, alpha56, alpha57, alpha58, alpha59, alpha60,
            alpha61, alpha62, alpha63, alpha64, alpha65, alpha66, alpha67, alpha68, alpha69, alpha70,
            alpha71, alpha72, alpha73, alpha74, alpha75, alpha76, alpha77, alpha78, alpha79, alpha80,
            alpha81, alpha82, alpha83, alpha84, alpha85, alpha86, alpha87, alpha88, alpha89, alpha90,
            alpha91, alpha92, alpha93, alpha94, alpha95, alpha96, alpha97, alpha98, alpha99, alpha100, alpha101
        ]

        for i, alpha_func in enumerate(alpha_functions, 1):
            try:
                print(f"  Computing alpha_{i}...")
                df[f'alpha_{i}'] = alpha_func(df)
            except Exception as e:
                print(f"  Error computing alpha_{i}: {str(e)}")
        
        print("Original alpha factors (1-101) complete!")
        
    except ImportError as e:
        print(f"Could not import original alpha functions: {str(e)}")
        print("Skipping original alpha factors...")
    
    # Add additional alpha factors (alpha_102 to alpha_120) - following enhanced approach
    try:
        print("Adding additional alpha factors (102-120)...")
        df = add_additional_alphas(df)
    except Exception as e:
        print(f"Error adding additional alphas: {str(e)}")
    
    # Add enhanced alpha factors (alpha_121 to alpha_140) - following enhanced approach
    try:
        print("Adding enhanced alpha factors (121-140)...")
        df = add_enhanced_alphas(df)
    except Exception as e:
        print(f"Error adding enhanced alphas: {str(e)}")
    
    # Add extended alpha factors (alpha_141 to alpha_160) - following enhanced approach
    try:
        print("Adding extended alpha factors (141-160)...")
        df = add_extended_alphas(df)
    except Exception as e:
        print(f"Error adding extended alphas: {str(e)}")
    
    # Get list of all alpha columns
    alpha_columns = [col for col in df.columns if col.startswith('alpha_')]
    alpha_columns.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"\nTotal extended alpha factors computed: {len(alpha_columns)}")
    print("Factor categories:")
    print("  • alpha_1 to alpha_101: Original WorldQuant alpha factors")
    print("  • alpha_102 to alpha_120: Research-based momentum/mean reversion factors")
    print("  • alpha_121 to alpha_140: Enhanced versions of top-performing factors")
    print("  • alpha_141 to alpha_160: Extended factors with advanced techniques")
    
    return df, alpha_columns

def evaluate_alpha_performance(df, alpha_columns, forward_return_period=5):
    """Evaluate performance of all alpha factors"""
    print(f"\nEvaluating alpha factor performance with {forward_return_period}-day forward returns...")
    
    # Calculate forward returns (following golden copy approach)
    df = df.sort_values(['Ticker', 'Date'])
    df['forward_return'] = df.groupby('Ticker')['Close'].pct_change(forward_return_period).shift(-forward_return_period)
    
    # Remove rows with NaN forward returns
    df_valid = df.dropna(subset=['forward_return'])
    
    alpha_performance = []
    
    print("Computing performance metrics for all alpha factors...")
    for i, alpha in enumerate(alpha_columns):
        try:
            # Skip if alpha column doesn't exist or is all zeros/NaN
            if alpha not in df_valid.columns:
                continue
                
            if df_valid[alpha].abs().sum() == 0 or df_valid[alpha].isna().all():
                continue
            
            # Calculate information coefficient (IC) - correlation with forward returns
            correlation = df_valid[alpha].corr(df_valid['forward_return'])
            
            if pd.isna(correlation):
                continue
            
            # Calculate quintile performance for long-short return estimation
            try:
                df_valid[f'{alpha}_quintile'] = pd.qcut(df_valid[alpha], 5, labels=False, duplicates='drop')
                quintile_returns = df_valid.groupby(f'{alpha}_quintile')['forward_return'].mean()
                long_short_return = (quintile_returns.iloc[-1] - quintile_returns.iloc[0]) * 100  # Top - Bottom quintile
            except Exception as quintile_error:
                long_short_return = 0
            
            # Information ratio approximation (IC stability)
            ir = abs(correlation)
            
            alpha_performance.append({
                'alpha': alpha,
                'ic': correlation,
                'abs_ic': abs(correlation),
                'long_short_return': long_short_return,
                'information_ratio': ir
            })
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(alpha_columns)} factors...")
                
        except Exception as e:
            print(f"  Error processing {alpha}: {str(e)}")
            continue
    
    if len(alpha_performance) == 0:
        print("No valid alpha factors found!")
        return pd.DataFrame(), df_valid
    
    performance_df = pd.DataFrame(alpha_performance)
    performance_df = performance_df.sort_values('abs_ic', ascending=False)
    
    print(f"Successfully evaluated {len(performance_df)} alpha factors")
    return performance_df, df_valid

def build_combined_alpha_strategy(df, performance_df, top_n=20,
                                  mode='equal_mean', long_frac=0.3, short_frac=0.3):
    """Build combined alpha strategy using top-performing factors.

    Weighting modes for fair benchmark comparison:
    - 'equal_mean' (legacy): mean of position returns across all assets (variable exposure)
    - 'long_only': use only the long side; re-normalize long weights to sum to 1 each day
    - 'long_short_gross1': dollar-neutral; allocate 0.5 to longs and 0.5 to shorts daily

    Args:
        df: Data with prices and alpha columns
        performance_df: Ranked factors with column 'alpha'
        top_n: number of factors to combine
        mode: weighting mode as described above
        long_frac/short_frac: fraction of universe to select for long/short buckets
    """
    print(f"\nBuilding combined alpha strategy with top {top_n} alpha factors...")

    # Select top alphas based on absolute IC
    top_alphas = performance_df.head(top_n)['alpha'].tolist()
    print(f"Selected top alphas: {top_alphas[:5]}... (showing first 5)")

    # Calculate combined alpha score (equal-weight across factors)
    df = df.copy()
    df['combined_alpha'] = df[top_alphas].mean(axis=1)

    # Build daily weights
    portfolio_data = []

    long_threshold = 1 - long_frac
    short_threshold = short_frac

    for date, date_data in df.groupby('Date'):
        date_data = date_data.copy()
        if len(date_data) < 2:
            continue

        # Cross-sectional rank of combined alpha
        date_data['alpha_rank'] = date_data['combined_alpha'].rank(pct=True, method='first')

        # Raw selection masks
        long_mask = date_data['alpha_rank'] >= long_threshold
        short_mask = date_data['alpha_rank'] <= short_threshold

        # Initialize weight column
        date_data['weight'] = 0.0

        if mode == 'equal_mean':
            # Legacy: +/-1 positions; return computed later as mean across all assets
            date_data.loc[long_mask, 'weight'] = 1.0
            date_data.loc[short_mask, 'weight'] = -1.0

        elif mode == 'long_only':
            # Only use longs; re-normalize to sum to 1 each day
            n_long = int(long_mask.sum())
            if n_long > 0:
                # Equal-weight the long bucket to sum to 1.0
                eq_w = 1.0 / n_long
                date_data.loc[long_mask, 'weight'] = eq_w
            # shorts remain 0

        elif mode == 'long_short_gross1':
            # Dollar-neutral with gross exposure = 1 (0.5 long / 0.5 short)
            n_long = int(long_mask.sum())
            n_short = int(short_mask.sum())
            if n_long > 0:
                date_data.loc[long_mask, 'weight'] = 0.5 / n_long
            if n_short > 0:
                date_data.loc[short_mask, 'weight'] = -(0.5 / n_short)

        else:
            raise ValueError(f"Unknown weighting mode: {mode}")

        portfolio_data.append(date_data)

    portfolio_df = pd.concat(portfolio_data, ignore_index=True)

    # Compute asset returns
    portfolio_df = portfolio_df.sort_values(['Ticker', 'Date'])
    portfolio_df['asset_return'] = portfolio_df.groupby('Ticker')['Close'].pct_change()

    # Compute daily PnL using weights
    portfolio_df['weighted_return'] = portfolio_df['weight'] * portfolio_df['asset_return']

    agg = {
        'weighted_return': 'sum',
        'weight': ['sum', lambda x: x.abs().sum(),
                   lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]
    }
    daily = portfolio_df.groupby('Date').agg(agg)
    # Flatten columns
    daily.columns = ['strategy_return', 'net_exposure', 'gross_exposure', 'n_longs', 'n_shorts']
    daily = daily.reset_index()

    # Legacy mode parity: keep num_positions similar for downstream
    if mode == 'equal_mean':
        # Reproduce earlier averaging across all assets (variable exposure)
        # Note: This branch keeps behavior for backward compatibility.
        portfolio_df['position_return'] = portfolio_df['weight'] * portfolio_df['asset_return']
        legacy = portfolio_df.groupby('Date').agg({
            'position_return': 'mean',
            'weight': lambda x: (x != 0).sum()
        }).reset_index().rename(columns={'position_return': 'strategy_return', 'weight': 'num_positions'})
        legacy = legacy[legacy['num_positions'] > 0]
        print(f"Combined alpha strategy created with {len(legacy)} trading days (mode={mode})")
        return legacy[['Date', 'strategy_return', 'num_positions']], top_alphas, portfolio_df

    # For normalized modes, add diagnostics
    daily['num_positions'] = daily['n_longs'] + daily['n_shorts']
    print(
        f"Combined alpha strategy created with {len(daily)} trading days (mode={mode}).\n"
        f"Avg gross: {daily['gross_exposure'].mean():.3f}, Avg net: {daily['net_exposure'].mean():.3f}, "
        f"Avg longs: {daily['n_longs'].mean():.1f}, Avg shorts: {daily['n_shorts'].mean():.1f}"
    )

    # Align return schema with downstream consumers
    daily_returns = daily[['Date', 'strategy_return', 'num_positions']]
    return daily_returns, top_alphas, portfolio_df

def calculate_btc_benchmark(df):
    """Calculate BTC buy-and-hold benchmark from Binance data"""
    print("Calculating BTC buy-and-hold benchmark...")
    
    # Extract BTC data from the dataset (following golden copy)
    print(f"Looking for BTCUSDT in dataset with tickers: {sorted(df['Ticker'].unique())}")
    btc_data = df[df['Ticker'] == 'BTCUSDT'].copy()
    
    if len(btc_data) == 0:
        print("Warning: BTCUSDT not found in dataset")
        return pd.DataFrame(columns=['Date', 'btc_return'])
    
    print(f"Found {len(btc_data)} BTC data points before processing")
    btc_data = btc_data.sort_values('Date')
    btc_data['btc_return'] = btc_data['Close'].pct_change()
    
    # Only drop NaNs from the btc_return column, not all columns
    btc_clean = btc_data.dropna(subset=['btc_return'])
    
    print(f"BTC benchmark calculated: {len(btc_clean)} data points after processing")
    if len(btc_clean) > 0:
        print(f"BTC date range: {btc_clean['Date'].min()} to {btc_clean['Date'].max()}")
    return btc_clean[['Date', 'btc_return']]

def calculate_performance_metrics(returns):
    """Calculate comprehensive performance metrics"""
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # Basic performance metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown calculation
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}"
    }

def comprehensive_factor_analysis(df, alpha_columns, performance_df, top_n=15, periods=(1, 5, 10, 21)):
    """
    Comprehensive factor analysis similar to Alphalens but custom-built for crypto data
    Includes: IC analysis, quintile analysis, turnover analysis, and factor reports
    
    Args:
        df: DataFrame with price data and alpha factors
        alpha_columns: List of alpha factor column names
        performance_df: Performance DataFrame with factor rankings
        top_n: Number of top factors to analyze in detail
        periods: Forward return periods to analyze (in days)
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE FACTOR ANALYSIS (Custom Implementation)")
    print("="*60)
    
    # Select top factors for detailed analysis
    top_factors = performance_df.head(top_n)['alpha'].tolist()
    print(f"Analyzing top {len(top_factors)} factors:")
    for i, factor in enumerate(top_factors, 1):
        ic_score = performance_df[performance_df['alpha'] == factor]['ic'].iloc[0]
        print(f"  {i:2d}. {factor:<12} (IC: {ic_score:7.4f})")
    
    # Prepare data for analysis
    print("\nPreparing data for comprehensive analysis...")
    
    df_sorted = df.copy()
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    df_sorted = df_sorted.sort_values(['Date', 'Ticker'])
    
    analysis_results = {}
    
    for factor_name in top_factors:
        print(f"\n📊 Analyzing factor: {factor_name}")
        
        try:
            if factor_name not in df_sorted.columns:
                print(f"  ❌ Factor {factor_name} not found")
                continue
            
            factor_analysis = analyze_single_factor(df_sorted, factor_name, periods)
            if factor_analysis:
                analysis_results[factor_name] = factor_analysis
                print(f"  ✅ {factor_name} analysis completed")
            else:
                print(f"  ❌ {factor_name} analysis failed")
                
        except Exception as e:
            print(f"  ❌ Error analyzing {factor_name}: {str(e)}")
            continue
    
    print(f"\n✅ Successfully analyzed {len(analysis_results)} factors")
    return analysis_results

def analyze_single_factor(df, factor_name, periods=(1, 5, 10, 21)):
    """
    Analyze a single factor with comprehensive metrics
    """
    
    # Calculate forward returns for all periods
    for period in periods:
        df[f'forward_return_{period}d'] = df.groupby('Ticker')['Close'].pct_change(period).shift(-period)
    
    # Remove rows with missing factor or forward return data
    df_clean = df.dropna(subset=[factor_name] + [f'forward_return_{period}d' for period in periods])
    
    if len(df_clean) < 100:  # Need minimum data for analysis
        return None
    
    factor_results = {
        'factor_name': factor_name,
        'total_observations': len(df_clean),
        'date_range': (df_clean['Date'].min(), df_clean['Date'].max()),
        'ic_analysis': {},
        'quintile_analysis': {},
        'turnover_analysis': {},
        'performance_metrics': {}
    }
    
    # 1. Information Coefficient Analysis
    for period in periods:
        return_col = f'forward_return_{period}d'
        ic_series = df_clean.groupby('Date').apply(
            lambda x: x[factor_name].corr(x[return_col]) if len(x) > 1 and x[factor_name].std() > 0 else np.nan
        ).dropna()
        
        factor_results['ic_analysis'][f'{period}d'] = {
            'mean_ic': ic_series.mean(),
            'std_ic': ic_series.std(),
            'ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            'ic_series': ic_series
        }
    
    # 2. Quintile Analysis (Bucket Analysis)
    for period in periods:
        return_col = f'forward_return_{period}d'
        
        # Assign quintiles by factor score within each date
        df_clean[f'quintile_{period}d'] = df_clean.groupby('Date')[factor_name].transform(
            lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5], duplicates='drop') if len(x) >= 5 else np.nan
        )
        
        # Calculate mean return by quintile
        quintile_returns = df_clean.groupby(f'quintile_{period}d')[return_col].agg(['mean', 'std', 'count'])
        
        # Calculate long-short return (Q5 - Q1)
        if 5 in quintile_returns.index and 1 in quintile_returns.index:
            long_short = quintile_returns.loc[5, 'mean'] - quintile_returns.loc[1, 'mean']
        else:
            long_short = np.nan
        
        factor_results['quintile_analysis'][f'{period}d'] = {
            'quintile_returns': quintile_returns,
            'long_short_return': long_short,
            'monotonicity': check_monotonicity(quintile_returns['mean']) if len(quintile_returns) == 5 else np.nan
        }
    
    # 3. Turnover Analysis (Factor Stability)
    factor_pivot = df_clean.pivot_table(
        index='Date', columns='Ticker', values=factor_name, aggfunc='last'
    )
    
    # Calculate rank correlation between consecutive periods
    rank_correlations = []
    for i in range(1, len(factor_pivot)):
        prev_ranks = factor_pivot.iloc[i-1].rank()
        curr_ranks = factor_pivot.iloc[i].rank()
        corr = prev_ranks.corr(curr_ranks)
        if not pd.isna(corr):
            rank_correlations.append(corr)
    
    if rank_correlations:
        factor_results['turnover_analysis'] = {
            'rank_autocorr': np.mean(rank_correlations),
            'turnover': 1 - np.mean(rank_correlations),  # Higher turnover = less stable rankings
            'turnover_series': rank_correlations
        }
    else:
        factor_results['turnover_analysis'] = {
            'rank_autocorr': np.nan,
            'turnover': np.nan,
            'turnover_series': []
        }
    
    # 4. Overall Performance Metrics
    primary_period = 5  # Use 5-day returns as primary metric
    if f'{primary_period}d' in factor_results['ic_analysis']:
        ic_data = factor_results['ic_analysis'][f'{primary_period}d']
        quintile_data = factor_results['quintile_analysis'][f'{primary_period}d']
        
        factor_results['performance_metrics'] = {
            'primary_ic': ic_data['mean_ic'],
            'primary_ir': ic_data['ir'],
            'primary_long_short': quintile_data['long_short_return'],
            'turnover': factor_results['turnover_analysis']['turnover'],
            'factor_strength': abs(ic_data['mean_ic']) * (1 - factor_results['turnover_analysis']['turnover']),
            'consistency': ic_data['ir']  # Information Ratio as consistency measure
        }
    
    return factor_results

def check_monotonicity(quintile_means):
    """
    Check if quintile returns show monotonic relationship with factor
    Returns: monotonicity score (-1 to 1, where 1 is perfectly monotonic)
    """
    if len(quintile_means) < 5:
        return np.nan
    
    # Check both increasing and decreasing monotonicity
    increasing = all(quintile_means.iloc[i] >= quintile_means.iloc[i-1] for i in range(1, len(quintile_means)))
    decreasing = all(quintile_means.iloc[i] <= quintile_means.iloc[i-1] for i in range(1, len(quintile_means)))
    
    if increasing:
        return 1.0
    elif decreasing:
        return -1.0
    else:
        # Calculate Spearman correlation as monotonicity measure
        corr, _ = stats.spearmanr(list(range(1, 6)), quintile_means)
        return corr if not pd.isna(corr) else 0

def create_factor_analysis_reports(analysis_results, output_dir="data/factor_reports"):
    """
    Generate comprehensive factor analysis reports and visualizations
    """
    print(f"\n📊 Creating factor analysis reports in {output_dir}/")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary report across all factors
    summary_data = []
    
    for factor_name, results in analysis_results.items():
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            summary_data.append({
                'factor': factor_name,
                'ic_5d': metrics.get('primary_ic', np.nan),
                'ir_5d': metrics.get('primary_ir', np.nan),
                'long_short_5d': metrics.get('primary_long_short', np.nan),
                'turnover': metrics.get('turnover', np.nan),
                'factor_strength': metrics.get('factor_strength', np.nan),
                'consistency': metrics.get('consistency', np.nan),
                'observations': results['total_observations']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('factor_strength', ascending=False)
        summary_df.to_csv(f"{output_dir}/factor_summary.csv", index=False)
        
        print("\n📋 FACTOR ANALYSIS SUMMARY")
        print("="*90)
        print(f"{'Factor':<15} {'IC_5D':<8} {'IR_5D':<8} {'LongShort':<10} {'Turnover':<9} {'Strength':<9} {'Obs':<8}")
        print("="*90)
        
        for _, row in summary_df.head(10).iterrows():
            print(f"{row['factor']:<15} "
                  f"{row['ic_5d']:<8.4f} "
                  f"{row['ir_5d']:<8.2f} "
                  f"{row['long_short_5d']:<10.4f} "
                  f"{row['turnover']:<9.3f} "
                  f"{row['factor_strength']:<9.4f} "
                  f"{int(row['observations']):<8}")
        
        # Create comprehensive visualization
        create_factor_summary_plots(analysis_results, summary_df, output_dir)
    
    # Individual factor reports
    for factor_name, results in analysis_results.items():
        create_individual_factor_report(factor_name, results, output_dir)
    
    print(f"\n✅ All factor analysis reports generated in {output_dir}/")
    return summary_data

def create_factor_summary_plots(analysis_results, summary_df, output_dir):
    """
    Create summary plots comparing all factors
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Factor Analysis Summary', fontsize=16, fontweight='bold')
    
    # 1. IC vs Information Ratio scatter
    ax1 = axes[0, 0]
    valid_data = summary_df.dropna(subset=['ic_5d', 'ir_5d', 'long_short_5d'])
    if len(valid_data) > 0:
        scatter = ax1.scatter(valid_data['ic_5d'], valid_data['ir_5d'], 
                             c=valid_data['long_short_5d'], cmap='RdYlBu', s=60, alpha=0.7)
        ax1.set_xlabel('IC Mean (5D)')
        ax1.set_ylabel('Information Ratio (5D)')
        ax1.set_title('IC vs Information Ratio')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Long-Short Return')
        
        # Annotate best factors
        for i, row in valid_data.head(5).iterrows():
            ax1.annotate(row['factor'].replace('alpha_', 'α'), 
                        (row['ic_5d'], row['ir_5d']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    # 2. Long-Short Returns ranking
    ax2 = axes[0, 1]
    top_long_short = summary_df.dropna(subset=['long_short_5d']).head(12)
    if len(top_long_short) > 0:
        bars = ax2.barh(range(len(top_long_short)), top_long_short['long_short_5d'], 
                        color='green', alpha=0.7)
        ax2.set_yticks(range(len(top_long_short)))
        ax2.set_yticklabels([f.replace('alpha_', 'α') for f in top_long_short['factor']])
        ax2.set_xlabel('Long-Short Return (5D)')
        ax2.set_title('Top Factors by Long-Short Return')
        ax2.grid(True, alpha=0.3)
    
    # 3. Factor Strength vs Turnover trade-off
    ax3 = axes[0, 2]
    trade_off_data = summary_df.dropna(subset=['turnover', 'factor_strength'])
    if len(trade_off_data) > 0:
        scatter2 = ax3.scatter(trade_off_data['turnover'], trade_off_data['factor_strength'],
                              c=trade_off_data['ic_5d'], cmap='viridis', s=60, alpha=0.7)
        ax3.set_xlabel('Turnover')
        ax3.set_ylabel('Factor Strength')
        ax3.set_title('Factor Strength vs Turnover')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax3, label='IC (5D)')
    
    # 4. IC Distribution across top factors
    ax4 = axes[1, 0]
    ic_data = []
    factor_names = []
    for factor_name in summary_df.head(8)['factor']:
        if factor_name in analysis_results:
            ic_series = analysis_results[factor_name]['ic_analysis'].get('5d', {}).get('ic_series', pd.Series())
            if len(ic_series) > 10:
                ic_data.append(ic_series.values)
                factor_names.append(factor_name.replace('alpha_', 'α'))
    
    if ic_data:
        ax4.boxplot(ic_data, labels=factor_names)
        ax4.set_ylabel('Information Coefficient')
        ax4.set_title('IC Distribution (Top 8 Factors)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 5. Quintile Returns Heatmap
    ax5 = axes[1, 1]
    quintile_matrix = []
    factor_labels = []
    
    for factor_name in summary_df.head(8)['factor']:
        if factor_name in analysis_results:
            quintile_data = analysis_results[factor_name]['quintile_analysis'].get('5d', {})
            if 'quintile_returns' in quintile_data:
                returns = quintile_data['quintile_returns']['mean']
                if len(returns) == 5:
                    quintile_matrix.append(returns.values)
                    factor_labels.append(factor_name.replace('alpha_', 'α'))
    
    if quintile_matrix:
        im = ax5.imshow(quintile_matrix, cmap='RdYlBu', aspect='auto')
        ax5.set_xticks(range(5))
        ax5.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax5.set_yticks(range(len(factor_labels)))
        ax5.set_yticklabels(factor_labels)
        ax5.set_title('Quintile Returns Heatmap')
        plt.colorbar(im, ax=ax5, label='Mean Return')
    
    # 6. Consistency Analysis
    ax6 = axes[1, 2]
    consistency_data = summary_df.dropna(subset=['consistency', 'factor_strength'])
    if len(consistency_data) > 0:
        ax6.scatter(consistency_data['consistency'], consistency_data['factor_strength'],
                   alpha=0.7, s=60)
        ax6.set_xlabel('Consistency (IR)')
        ax6.set_ylabel('Factor Strength')
        ax6.set_title('Consistency vs Strength')
        ax6.grid(True, alpha=0.3)
        
        # Add factor labels for top performers
        for i, row in consistency_data.head(5).iterrows():
            ax6.annotate(row['factor'].replace('alpha_', 'α'),
                        (row['consistency'], row['factor_strength']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/factor_analysis_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Summary visualization saved to {output_dir}/factor_analysis_summary.png")

def create_individual_factor_report(factor_name, results, output_dir):
    """
    Create individual factor analysis report with detailed plots
    """
    factor_dir = f"{output_dir}/{factor_name}"
    os.makedirs(factor_dir, exist_ok=True)
    
    # Create detailed plots for this factor
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{factor_name} - Detailed Factor Analysis', fontsize=14, fontweight='bold')
    
    # 1. IC Time Series
    ax1 = axes[0, 0]
    if '5d' in results['ic_analysis']:
        ic_series = results['ic_analysis']['5d']['ic_series']
        if len(ic_series) > 0:
            ax1.plot(ic_series.index, ic_series.values, alpha=0.7)
            ax1.axhline(y=ic_series.mean(), color='red', linestyle='--', 
                       label=f'Mean IC: {ic_series.mean():.4f}')
            ax1.set_title('Information Coefficient Time Series (5D)')
            ax1.set_ylabel('IC')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
    
    # 2. Quintile Returns Bar Chart
    ax2 = axes[0, 1]
    if '5d' in results['quintile_analysis']:
        quintile_data = results['quintile_analysis']['5d']['quintile_returns']
        if len(quintile_data) > 0:
            bars = ax2.bar(quintile_data.index.astype(str), quintile_data['mean'], 
                          yerr=quintile_data['std'], capsize=5, alpha=0.7, color='green')
            ax2.set_title('Mean Return by Quintile (5D)')
            ax2.set_xlabel('Quintile')
            ax2.set_ylabel('Mean Return')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, quintile_data['mean']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{mean_val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. IC across different periods
    ax3 = axes[1, 0]
    periods = ['1d', '5d', '10d', '21d']
    ic_means = []
    ic_stds = []
    for period in periods:
        if period in results['ic_analysis']:
            ic_means.append(results['ic_analysis'][period]['mean_ic'])
            ic_stds.append(results['ic_analysis'][period]['std_ic'])
        else:
            ic_means.append(np.nan)
            ic_stds.append(np.nan)
    
    ax3.bar(periods, ic_means, yerr=ic_stds, capsize=5, alpha=0.7)
    ax3.set_title('IC Across Different Periods')
    ax3.set_xlabel('Forward Return Period')
    ax3.set_ylabel('Mean IC')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 4. Performance Summary Text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        summary_text = f"""
PERFORMANCE SUMMARY

Primary Metrics (5D):
• IC: {metrics.get('primary_ic', 'N/A'):.4f}
• Information Ratio: {metrics.get('primary_ir', 'N/A'):.3f}
• Long-Short Return: {metrics.get('primary_long_short', 'N/A'):.4f}

Stability:
• Turnover: {metrics.get('turnover', 'N/A'):.3f}
• Factor Strength: {metrics.get('factor_strength', 'N/A'):.4f}

Data:
• Total Observations: {results['total_observations']:,}
• Date Range: {results['date_range'][0].strftime('%Y-%m-%d')} to 
  {results['date_range'][1].strftime('%Y-%m-%d')}
        """
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{factor_dir}/detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed data
    if '5d' in results['ic_analysis']:
        results['ic_analysis']['5d']['ic_series'].to_csv(f"{factor_dir}/ic_timeseries.csv")
    
    if '5d' in results['quintile_analysis']:
        results['quintile_analysis']['5d']['quintile_returns'].to_csv(f"{factor_dir}/quintile_returns.csv")

def alphalens_comprehensive_analysis(df, alpha_columns, performance_df, top_n=15, periods=(1, 5, 10, 21)):
    """
    Comprehensive factor analysis using Alphalens package
    
    Args:
        df: DataFrame with price data and alpha factors
        alpha_columns: List of alpha factor column names
        performance_df: Performance DataFrame with factor rankings
        top_n: Number of top factors to analyze in detail
        periods: Forward return periods to analyze (in days)
    """
    print("\n" + "="*60)
    print("ALPHALENS COMPREHENSIVE FACTOR ANALYSIS")
    print("="*60)
    
    # Select top factors for detailed analysis
    top_factors = performance_df.head(top_n)['alpha'].tolist()
    print(f"Analyzing top {len(top_factors)} factors with Alphalens:")
    for i, factor in enumerate(top_factors, 1):
        print(f"  {i:2d}. {factor}")
    
    # Prepare data for Alphalens
    print("\nPreparing data for Alphalens analysis...")
    
    # Ensure Date column is datetime and sort data
    df_sorted = df.copy()
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    df_sorted = df_sorted.sort_values(['Date', 'Ticker'])
    
    # Create price data in the format Alphalens expects (MultiIndex: date, asset)
    price_data = df_sorted.pivot_table(
        index='Date', 
        columns='Ticker', 
        values='Close',
        aggfunc='last'  # Take last value if multiple entries per day
    )
    
    # Forward fill and backward fill to handle missing data
    price_data = price_data.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
    
    print(f"Price data shape: {price_data.shape}")
    print(f"Date range: {price_data.index.min()} to {price_data.index.max()}")
    print(f"Assets: {list(price_data.columns)}")
    
    alphalens_results = {}
    
    # Analyze each top factor
    for factor_name in top_factors:
        print(f"\n🔍 Analyzing factor: {factor_name}")
        
        try:
            # Check if factor exists in dataframe
            if factor_name not in df_sorted.columns:
                print(f"  ❌ Factor {factor_name} not found in dataframe")
                continue
            
            # Prepare factor data (MultiIndex: date, asset)
            factor_data = df_sorted.pivot_table(
                index='Date',
                columns='Ticker', 
                values=factor_name,
                aggfunc='last'  # Take last value if multiple entries per day
            )
            
            # Forward fill and backward fill to handle missing data
            factor_data = factor_data.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
            
            # Stack to create MultiIndex Series (date, asset)
            factor_series = factor_data.stack()
            factor_series.index.names = ['date', 'asset']
            
            # Clean data - remove NaN and infinite values
            factor_series = factor_series.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(factor_series) == 0:
                print(f"  ❌ No valid data for {factor_name}")
                continue
                
            # Check if factor has sufficient variation
            if factor_series.std() == 0:
                print(f"  ❌ Factor {factor_name} has no variation")
                continue
            
            print(f"  ✓ Factor data prepared: {len(factor_series)} observations")
            
            # Align price data to factor data dates
            factor_dates = factor_series.index.get_level_values('date').unique()
            price_data_aligned = price_data.loc[price_data.index.isin(factor_dates)]
            
            if len(price_data_aligned) == 0:
                print(f"  ❌ No price data available for {factor_name} dates")
                continue
            
            # Get factor data with forward returns and quantiles
            try:
                factor_data_al = utils.get_clean_factor_and_forward_returns(
                    factor=factor_series,
                    prices=price_data_aligned,
                    quantiles=5,  # Quintile analysis
                    periods=periods,
                    max_loss=0.5  # Allow up to 50% data loss for crypto data
                )
                
                print(f"  ✓ Alphalens data created: {len(factor_data_al)} clean observations")
                
                # Store results for this factor
                alphalens_results[factor_name] = {
                    'factor_data': factor_data_al,
                    'factor_series': factor_series
                }
                
            except Exception as alphalens_error:
                print(f"  ❌ Error in Alphalens processing for {factor_name}: {str(alphalens_error)}")
                continue
            
        except Exception as e:
            print(f"  ❌ Error processing {factor_name}: {str(e)}")
            continue
    
    print(f"\n✅ Successfully prepared {len(alphalens_results)} factors for Alphalens analysis")
    
    return alphalens_results, price_data

def create_alphalens_reports(alphalens_results, output_dir="data/alphalens_reports"):
    """
    Generate comprehensive Alphalens reports for all factors
    """
    print(f"\n📊 Creating Alphalens reports in {output_dir}/")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    factor_summaries = []
    
    for factor_name, data in alphalens_results.items():
        print(f"\n📈 Generating reports for {factor_name}...")
        
        try:
            factor_data = data['factor_data']
            
            # Create individual factor report directory
            factor_dir = os.path.join(output_dir, factor_name)
            os.makedirs(factor_dir, exist_ok=True)
            
            # 1. Information Coefficient Analysis
            print("  • IC Analysis...")
            ic_summary = perf.factor_information_coefficient(factor_data)
            ic_summary.to_csv(f"{factor_dir}/ic_summary.csv")
            
            # Plot IC time series
            fig, ax = plt.subplots(figsize=(12, 6))
            plotting.plot_ic_ts(factor_data, ax=ax)
            plt.title(f'{factor_name} - Information Coefficient Time Series')
            plt.tight_layout()
            plt.savefig(f"{factor_dir}/ic_timeseries.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot IC heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            plotting.plot_ic_hist(factor_data, ax=ax)
            plt.title(f'{factor_name} - IC Distribution')
            plt.tight_layout()
            plt.savefig(f"{factor_dir}/ic_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Quintile Analysis (Bucket Analysis)
            print("  • Quintile Analysis...")
            mean_return_by_q, std_err = perf.mean_return_by_quantile(factor_data)
            mean_return_by_q.to_csv(f"{factor_dir}/quintile_returns.csv")
            
            # Plot quintile bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            plotting.plot_quantile_returns_bar(mean_return_by_q, ax=ax)
            plt.title(f'{factor_name} - Mean Return by Quintile')
            plt.tight_layout()
            plt.savefig(f"{factor_dir}/quintile_bar.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot quintile violin plot
            fig, ax = plt.subplots(figsize=(12, 8))
            plotting.plot_quantile_returns_violin(factor_data, ax=ax)
            plt.title(f'{factor_name} - Return Distribution by Quintile')
            plt.tight_layout()
            plt.savefig(f"{factor_dir}/quintile_violin.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Turnover Analysis
            print("  • Turnover Analysis...")
            turnover_summary = perf.factor_rank_autocorrelation(factor_data)
            turnover_summary.to_csv(f"{factor_dir}/turnover_summary.csv")
            
            # Plot turnover
            fig, ax = plt.subplots(figsize=(10, 6))
            plotting.plot_factor_rank_auto_correlation(factor_data, ax=ax)
            plt.title(f'{factor_name} - Factor Rank Autocorrelation (Turnover)')
            plt.tight_layout()
            plt.savefig(f"{factor_dir}/turnover.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Cumulative Returns by Quintile
            print("  • Cumulative Returns...")
            fig, ax = plt.subplots(figsize=(12, 8))
            plotting.plot_cumulative_returns_by_quantile(factor_data, period=1, ax=ax)
            plt.title(f'{factor_name} - Cumulative Returns by Quintile (1D)')
            plt.tight_layout()
            plt.savefig(f"{factor_dir}/cumulative_returns_1d.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Factor Summary Statistics
            print("  • Summary Statistics...")
            
            # Calculate key metrics
            ic_mean = ic_summary.mean()
            ic_std = ic_summary.std()
            ic_ir = ic_mean / ic_std  # Information Ratio
            
            # Top-bottom quintile spread
            q_returns_1d = mean_return_by_q[1]  # 1-day forward returns
            if len(q_returns_1d) >= 5:
                top_bottom_spread = q_returns_1d.iloc[-1] - q_returns_1d.iloc[0]  # Q5 - Q1
            else:
                top_bottom_spread = np.nan
            
            # Turnover (1 - autocorrelation)
            turnover_1d = 1 - turnover_summary[1] if 1 in turnover_summary else np.nan
            
            factor_summary = {
                'factor': factor_name,
                'ic_mean_1d': ic_mean[1] if 1 in ic_mean else np.nan,
                'ic_std_1d': ic_std[1] if 1 in ic_std else np.nan,
                'ic_ir_1d': ic_ir[1] if 1 in ic_ir else np.nan,
                'ic_mean_5d': ic_mean[5] if 5 in ic_mean else np.nan,
                'ic_mean_21d': ic_mean[21] if 21 in ic_mean else np.nan,
                'top_bottom_spread_1d': top_bottom_spread,
                'turnover_1d': turnover_1d,
                'num_observations': len(factor_data)
            }
            
            factor_summaries.append(factor_summary)
            
            print(f"  ✅ {factor_name} reports completed")
            
        except Exception as e:
            print(f"  ❌ Error generating reports for {factor_name}: {str(e)}")
            continue
    
    # Create consolidated summary report
    if factor_summaries:
        summary_df = pd.DataFrame(factor_summaries)
        summary_df = summary_df.sort_values('ic_ir_1d', ascending=False, na_last=True)
        summary_df.to_csv(f"{output_dir}/factor_summary_report.csv", index=False)
        
        print(f"\n📋 Factor Summary Report:")
        print("="*80)
        print(f"{'Factor':<15} {'IC_1D':<8} {'IC_IR_1D':<8} {'Spread_1D':<10} {'Turnover':<9} {'Obs':<8}")
        print("="*80)
        
        for _, row in summary_df.head(10).iterrows():
            print(f"{row['factor']:<15} "
                  f"{row['ic_mean_1d']:<8.4f} "
                  f"{row['ic_ir_1d']:<8.2f} "
                  f"{row['top_bottom_spread_1d']:<10.4f} "
                  f"{row['turnover_1d']:<9.3f} "
                  f"{int(row['num_observations']):<8}")
    
    print(f"\n✅ All Alphalens reports generated in {output_dir}/")
    return factor_summaries

def create_alphalens_summary_visualization(factor_summaries, output_dir="data/alphalens_reports"):
    """
    Create summary visualizations comparing all factors
    """
    if not factor_summaries:
        print("No factor summaries available for visualization")
        return
    
    print("\n📊 Creating Alphalens summary visualizations...")
    
    df = pd.DataFrame(factor_summaries).dropna()
    
    if len(df) == 0:
        print("No valid factor data for visualization")
        return
    
    # Create comprehensive summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Alphalens Factor Analysis Summary', fontsize=16, fontweight='bold')
    
    # 1. IC vs Information Ratio scatter plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['ic_mean_1d'], df['ic_ir_1d'], 
                         c=df['top_bottom_spread_1d'], cmap='RdYlBu',
                         s=60, alpha=0.7)
    ax1.set_xlabel('IC Mean (1D)')
    ax1.set_ylabel('Information Ratio (1D)')
    ax1.set_title('IC vs Information Ratio')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Top-Bottom Spread (1D)')
    
    # Annotate best factors
    for i, row in df.head(5).iterrows():
        ax1.annotate(row['factor'].replace('alpha_', 'α'), 
                    (row['ic_mean_1d'], row['ic_ir_1d']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    # 2. Top-Bottom Spread comparison
    ax2 = axes[0, 1]
    top_factors = df.nlargest(15, 'top_bottom_spread_1d')
    bars = ax2.barh(range(len(top_factors)), top_factors['top_bottom_spread_1d'], 
                    color='green', alpha=0.7)
    ax2.set_yticks(range(len(top_factors)))
    ax2.set_yticklabels([f.replace('alpha_', 'α') for f in top_factors['factor']])
    ax2.set_xlabel('Top-Bottom Quintile Spread (1D)')
    ax2.set_title('Top 15 Factors by Quintile Spread')
    ax2.grid(True, alpha=0.3)
    
    # 3. IC across different periods
    ax3 = axes[1, 0]
    periods = ['ic_mean_1d', 'ic_mean_5d', 'ic_mean_21d']
    period_labels = ['1D', '5D', '21D']
    
    ic_data = []
    for period in periods:
        if period in df.columns:
            ic_data.append(df[period].dropna())
        else:
            ic_data.append(pd.Series(dtype=float))
    
    ax3.boxplot(ic_data, labels=period_labels)
    ax3.set_ylabel('Information Coefficient')
    ax3.set_xlabel('Forward Return Period')
    ax3.set_title('IC Distribution Across Periods')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 4. Turnover vs Performance trade-off
    ax4 = axes[1, 1]
    scatter2 = ax4.scatter(df['turnover_1d'], df['ic_ir_1d'],
                          c=df['ic_mean_1d'], cmap='viridis',
                          s=60, alpha=0.7)
    ax4.set_xlabel('Turnover (1D)')
    ax4.set_ylabel('Information Ratio (1D)')
    ax4.set_title('Turnover vs Performance Trade-off')
    ax4.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax4)
    cbar2.set_label('IC Mean (1D)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/alphalens_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Summary visualization saved to {output_dir}/alphalens_summary.png")

def create_strategy_visualization(strategy_returns, btc_data, performance_df, top_alphas):
    """Create comprehensive visualization comparing strategy vs BTC benchmark"""
    plt.close('all')  # Clear any existing plots
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Extended Alpha Strategy vs BTC Buy-and-Hold Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.3, wspace=0.3)
    
    # Plot 1: Strategy vs BTC Cumulative Performance (Full Width)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Merge strategy and BTC data for comparison
    print(f"Strategy returns: {len(strategy_returns)} days")
    print(f"BTC benchmark: {len(btc_data)} days")
    merged_data = strategy_returns.merge(btc_data, on='Date', how='inner')
    print(f"Merged data: {len(merged_data)} overlapping days")
    
    # Always plot strategy returns with proper timestamps
    strategy_clean = strategy_returns.dropna(subset=['strategy_return'])
    if len(strategy_clean) > 0:
        strategy_cumulative = (1 + strategy_clean['strategy_return']).cumprod()
        ax1.plot(pd.to_datetime(strategy_clean['Date']), strategy_cumulative, 
                label='Extended Alpha Strategy', linewidth=3, color='red', alpha=0.9)
        print(f"✓ Strategy performance plotted: {len(strategy_cumulative)} data points")
        
        # Set proper date formatting for x-axis
        import matplotlib.dates as mdates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))  # January and July
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    else:
        print("✗ No valid strategy returns data to plot")
    
    # Plot BTC benchmark if available and overlapping
    if len(merged_data) > 0:
        btc_cumulative = (1 + merged_data['btc_return']).cumprod()
        ax1.plot(pd.to_datetime(merged_data['Date']), btc_cumulative, 
                label='BTC Buy-and-Hold', linewidth=3, color='orange', alpha=0.8)
        print(f"✓ BTC benchmark plotted: {len(btc_cumulative)} data points")
    else:
        # If no overlapping data, try to plot BTC separately if available
        if len(btc_data) > 0:
            btc_clean = btc_data.dropna(subset=['btc_return'])
            if len(btc_clean) > 0:
                btc_cumulative = (1 + btc_clean['btc_return']).cumprod()
                ax1.plot(pd.to_datetime(btc_clean['Date']), btc_cumulative, 
                        label='BTC Buy-and-Hold', linewidth=3, color='orange', alpha=0.8)
                print(f"✓ BTC benchmark plotted separately: {len(btc_cumulative)} data points")
        print("✗ No overlapping data between strategy and BTC benchmark")
    
    ax1.set_title('Extended Alpha Strategy vs BTC Buy-and-Hold - Cumulative Returns', 
                  fontweight='bold', fontsize=14)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to show performance differences better
    
    # Plot 2: Top Alpha Factors Performance
    ax2 = fig.add_subplot(gs[1, 0])
    top_15 = performance_df.head(15)
    bars = ax2.barh(range(len(top_15)), top_15['abs_ic'], color='green', alpha=0.7)
    ax2.set_yticks(range(len(top_15)))
    ax2.set_yticklabels([f"{alpha}" for alpha in top_15['alpha']], fontsize=10)
    ax2.set_title('Top 15 Alpha Factors (by |IC|)', fontweight='bold')
    ax2.set_xlabel('Absolute Information Coefficient')
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=8)
    
    # Plot 3: Return Distribution Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    if len(merged_data) > 0:
        ax3.hist(merged_data['strategy_return'], bins=50, alpha=0.7, color='red', 
                label='Alpha Strategy', density=True)
        ax3.hist(merged_data['btc_return'], bins=50, alpha=0.7, color='orange', 
                label='BTC Returns', density=True)
        ax3.set_title('Daily Return Distributions', fontweight='bold')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling Performance Comparison
    ax4 = fig.add_subplot(gs[2, 0])
    if len(merged_data) > 0 and len(merged_data) >= 252:  # At least 1 year of data
        # Calculate rolling 252-day (1-year) returns
        strategy_rolling = merged_data['strategy_return'].rolling(252).apply(
            lambda x: (1 + x).prod() - 1)
        btc_rolling = merged_data['btc_return'].rolling(252).apply(
            lambda x: (1 + x).prod() - 1)
        
        ax4.plot(merged_data['Date'], strategy_rolling, 
                label='Alpha Strategy (1Y Rolling)', color='red', alpha=0.8)
        ax4.plot(merged_data['Date'], btc_rolling, 
                label='BTC (1Y Rolling)', color='orange', alpha=0.8)
        ax4.set_title('Rolling 1-Year Returns', fontweight='bold')
        ax4.set_ylabel('1-Year Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for rolling analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    # Plot 5: Factor Category Analysis
    ax5 = fig.add_subplot(gs[2, 1])
    categories = {
        'Original (1-101)': [f'alpha_{i}' for i in range(1, 102)],
        'Research (102-120)': [f'alpha_{i}' for i in range(102, 121)],
        'Enhanced (121-140)': [f'alpha_{i}' for i in range(121, 141)],
        'Extended (141-160)': [f'alpha_{i}' for i in range(141, 161)]
    }
    
    category_performance = {}
    for category, alpha_list in categories.items():
        category_alphas = performance_df[performance_df['alpha'].isin(alpha_list)]
        if len(category_alphas) > 0:
            category_performance[category] = category_alphas['abs_ic'].mean()
    
    if category_performance:
        category_names = list(category_performance.keys())
        category_scores = list(category_performance.values())
        
        bars = ax5.bar(category_names, category_scores, 
                      color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        ax5.set_title('Average |IC| by Factor Category', fontweight='bold')
        ax5.set_ylabel('Average |IC|')
        ax5.tick_params(axis='x', rotation=15)
        
        # Add value labels on bars
        for bar, score in zip(bars, category_scores):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = "data/crypto_strategy_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Strategy analysis plot saved to: {plot_filename}")
    
    plt.show()
    
    return merged_data


def main():
    """Main execution function for crypto strategy analysis"""
    print("="*70)
    print("CRYPTO EXTENDED ALPHA STRATEGY ANALYSIS")
    print("Following Golden Copy Approach - Extended Alpha Factors vs BTC Benchmark")
    print("="*70)
    
    try:
        # 1. Load Binance data (following golden copy)
        df = load_binance_data()
        if df is None:
            return
        
        # 2. Prepare data with technical indicators
        print("\nPreparing technical indicators...")
        tickers = df["Ticker"].unique()
        for ticker in tickers:
            mask = df["Ticker"] == ticker
            try:
                df.loc[mask, 'returns'] = df.loc[mask, 'Close'].pct_change()
                df.loc[mask, 'volume_ratio'] = df.loc[mask, 'Volume'] / df.loc[mask, 'Volume'].rolling(20, min_periods=1).mean()
            except Exception as e:
                print(f"Error preparing indicators for {ticker}: {str(e)}")
        
        # 3. Calculate extended alpha factors (beyond original 101)
        df, alpha_columns = calculate_extended_alpha_factors(df)
        
        # 4. Evaluate alpha performance
        performance_df, df_valid = evaluate_alpha_performance(df, alpha_columns)
        
        if len(performance_df) == 0:
            print("No alpha factors could be evaluated. Check data quality.")
            return
            
        print(f"\nSuccessfully evaluated {len(performance_df)} extended alpha factors")
        
        # 6. Comprehensive Factor Analysis (Custom Implementation)
        print("\n" + "="*50)
        print("🔬 COMPREHENSIVE FACTOR ANALYSIS")
        print("="*50)
        
        # Prepare data for comprehensive analysis
        analysis_results = comprehensive_factor_analysis(
            df_valid, alpha_columns, performance_df, top_n=15
        )
        
        # Generate comprehensive factor reports
        if analysis_results:
            factor_summaries = create_factor_analysis_reports(analysis_results)
        else:
            print("❌ No valid factors for comprehensive analysis")
        
        # 7. Build combined alpha strategy
        # To compare fairly with a long-only BTC benchmark, default to long-only normalized weights.
        strategy_mode = 'long_only'  # options: 'long_only', 'long_short_gross1', 'equal_mean'
        strategy_returns, top_alphas, portfolio_df = build_combined_alpha_strategy(
            df_valid, performance_df, top_n=20, mode=strategy_mode, long_frac=0.3, short_frac=0.3
        )
        print(f"Using strategy weighting mode: {strategy_mode}")

        # Export per-asset daily weights and daily exposure diagnostics
        try:
            weights_out = "data/extended_strategy_weights.csv"
            exposure_out = "data/extended_strategy_daily_exposure.csv"

            # Minimal per-asset weight table
            cols = ['Date', 'Ticker', 'weight', 'alpha_rank', 'combined_alpha']
            export_cols = [c for c in cols if c in portfolio_df.columns]
            portfolio_df.sort_values(['Date', 'Ticker'])[export_cols].to_csv(weights_out, index=False)

            # Daily exposure summary
            daily_exposure = portfolio_df.groupby('Date').agg(
                net_exposure=('weight', 'sum'),
                gross_exposure=('weight', lambda x: x.abs().sum()),
                n_longs=('weight', lambda x: (x > 0).sum()),
                n_shorts=('weight', lambda x: (x < 0).sum())
            ).reset_index()
            daily_exposure.to_csv(exposure_out, index=False)
            print(f"Weights exported to: {weights_out}")
            print(f"Daily exposure exported to: {exposure_out}")
        except Exception as e:
            print(f"Warning: failed to export weights/exposure CSVs: {e}")
        
        # 8. Calculate BTC buy-and-hold benchmark using the same validated data
        btc_data = calculate_btc_benchmark(df_valid)
        
        # 9. Create comprehensive visualization
        merged_data = create_strategy_visualization(strategy_returns, btc_data, performance_df, top_alphas)
        
        # 10. Performance comparison analysis
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON: EXTENDED ALPHA STRATEGY vs BTC BUY-AND-HOLD")
        print("="*70)
        
        # Calculate strategy performance metrics (use all available strategy data)
        strategy_clean = strategy_returns.dropna(subset=['strategy_return'])
        if len(strategy_clean) > 0:
            strategy_metrics = calculate_performance_metrics(strategy_clean['strategy_return'])
            print("📈 Extended Alpha Strategy Performance:")
            for key, value in strategy_metrics.items():
                print(f"  {key}: {value}")
        
        # Calculate BTC benchmark metrics (use all available BTC data)
        if len(btc_data) > 0:
            btc_clean = btc_data.dropna(subset=['btc_return'])
            if len(btc_clean) > 0:
                btc_metrics = calculate_performance_metrics(btc_clean['btc_return'])
                print("\n🟡 BTC Buy-and-Hold Benchmark Performance:")
                for key, value in btc_metrics.items():
                    print(f"  {key}: {value}")
        
        # Strategy vs Benchmark comparison (only if overlapping data exists)
        if len(merged_data) > 0:
            strategy_final = (1 + merged_data['strategy_return']).prod()
            btc_final = (1 + merged_data['btc_return']).prod()
            outperformance = (strategy_final / btc_final - 1) * 100
            
            print(f"\n🚀 Strategy Outperformance vs BTC (overlapping period): {outperformance:.2f}%")
            print(f"📊 Strategy Final Value (vs $1): ${strategy_final:.2f}")
            print(f"🟡 BTC Final Value (vs $1): ${btc_final:.2f}")
        else:
            print(f"\n📊 Strategy Final Value: ${(1 + strategy_clean['strategy_return']).prod():.2f}")
            if len(btc_data) > 0 and len(btc_clean) > 0:
                print(f"🟡 BTC Final Value: ${(1 + btc_clean['btc_return']).prod():.2f}")
                print("⚠️  Note: Strategy and BTC have different time periods - direct comparison not available")
            
        # 11. Save results to CSV files
        results_filename = "data/crypto_strategy_performance.csv"
        performance_df.to_csv(results_filename, index=False)
        print(f"\n💾 Alpha performance results saved to: {results_filename}")
        
        strategy_filename = "data/crypto_strategy_returns.csv"
        strategy_returns.to_csv(strategy_filename, index=False)
        print(f"📊 Strategy daily returns saved to: {strategy_filename}")
        
        # 12. Alpha factor summary
        print("\n" + "="*60)
        print("EXTENDED ALPHA FACTOR SUMMARY")
        print("="*60)
        print(f"Total alpha factors evaluated: {len(performance_df)}")
        print(f"Factors with positive IC: {sum(performance_df['ic'] > 0)}")
        print(f"Factors with |IC| > 0.01: {sum(performance_df['abs_ic'] > 0.01)}")
        print(f"Factors with |IC| > 0.02: {sum(performance_df['abs_ic'] > 0.02)}")
        print(f"Best performing alpha: {performance_df.iloc[0]['alpha']} (IC: {performance_df.iloc[0]['ic']:.4f})")
        
        print(f"\nTop 10 Alpha Factors used in strategy:")
        for i, row in performance_df.head(10).iterrows():
            alpha_num = int(row['alpha'].split('_')[1])
            if alpha_num <= 101:
                category = "Original"
            elif alpha_num <= 120:
                category = "Research"
            elif alpha_num <= 140:
                category = "Enhanced"
            else:
                category = "Extended"
            print(f"  {i+1:2d}. {row['alpha']:>12} | IC: {row['ic']:>7.4f} | Category: {category}")
        
        print("\n" + "="*50)
        print("CRYPTO STRATEGY ANALYSIS COMPLETE!")
        print("="*50)
        print("✓ Binance data loaded successfully")
        print(f"✓ {len(performance_df)} extended alpha factors evaluated")
        print("✓ Comprehensive factor analysis completed")
        print("✓ Combined alpha strategy built and tested")
        print("✓ BTC benchmark comparison completed") 
        print("✓ Comprehensive visualizations created")
        print("✓ Performance metrics calculated")
        print("✓ Results exported to CSV files")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
