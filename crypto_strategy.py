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

def build_combined_alpha_strategy(df, performance_df, top_n=20):
    """Build combined alpha strategy using top-performing factors"""
    print(f"\nBuilding combined alpha strategy with top {top_n} alpha factors...")
    
    # Select top alphas based on absolute IC
    top_alphas = performance_df.head(top_n)['alpha'].tolist()
    print(f"Selected top alphas: {top_alphas[:5]}... (showing first 5)")
    
    # Calculate combined alpha score (equal weighted approach)
    df['combined_alpha'] = df[top_alphas].mean(axis=1)
    
    # Create portfolio returns using long-short strategy
    portfolio_data = []
    
    for date in df['Date'].unique():
        date_data = df[df['Date'] == date].copy()
        
        if len(date_data) < 2:  # Need at least 2 assets for ranking
            continue
            
        # Rank assets by combined alpha score
        date_data['alpha_rank'] = date_data['combined_alpha'].rank(pct=True, method='first')
        
        # Long top 30%, short bottom 30% (following golden copy strategy)
        long_threshold = 0.7
        short_threshold = 0.3
        
        date_data['position'] = 0
        date_data.loc[date_data['alpha_rank'] >= long_threshold, 'position'] = 1
        date_data.loc[date_data['alpha_rank'] <= short_threshold, 'position'] = -1
        
        portfolio_data.append(date_data)
    
    portfolio_df = pd.concat(portfolio_data, ignore_index=True)
    
    # Calculate strategy returns
    portfolio_df = portfolio_df.sort_values(['Ticker', 'Date'])
    portfolio_df['asset_return'] = portfolio_df.groupby('Ticker')['Close'].pct_change()
    portfolio_df['position_return'] = portfolio_df['position'] * portfolio_df['asset_return']
    
    # Daily strategy return (mean of all position returns)
    daily_returns = portfolio_df.groupby('Date').agg({
        'position_return': 'mean',
        'position': lambda x: (x != 0).sum()  # Number of active positions
    }).reset_index()
    
    daily_returns.columns = ['Date', 'strategy_return', 'num_positions']
    daily_returns = daily_returns[daily_returns['num_positions'] > 0]  # Only days with positions
    
    print(f"Combined alpha strategy created with {len(daily_returns)} trading days")
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
        
        # 5. Build combined alpha strategy
        strategy_returns, top_alphas, portfolio_df = build_combined_alpha_strategy(df_valid, performance_df)
        
        # 6. Calculate BTC buy-and-hold benchmark using the same validated data
        btc_data = calculate_btc_benchmark(df_valid)
        
        # 7. Create comprehensive visualization
        merged_data = create_strategy_visualization(strategy_returns, btc_data, performance_df, top_alphas)
        
        # 8. Performance comparison analysis
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
            
        # 9. Save results to CSV files
        results_filename = "data/crypto_strategy_performance.csv"
        performance_df.to_csv(results_filename, index=False)
        print(f"\n💾 Alpha performance results saved to: {results_filename}")
        
        strategy_filename = "data/crypto_strategy_returns.csv"
        strategy_returns.to_csv(strategy_filename, index=False)
        print(f"📊 Strategy daily returns saved to: {strategy_filename}")
        
        # 10. Alpha factor summary
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
