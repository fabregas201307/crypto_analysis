"""
Enhanced Crypto Alpha Analysis with Extended Factor Pool
Updated to include 160 total alpha factors (alpha_1 to alpha_160)

This enhanced version combines:
- Original alpha_101 factors (alpha_1 to alpha_101) 
- Research-based factors (alpha_102 to alpha_120)
- Enhanced factors (alpha_121 to alpha_140)
- Extended factors (alpha_141 to alpha_160)

New categories added:
- Signal Processing (Hilbert transform, fractal dimension)
- Information Theory (entropy-based signals)
- Behavioral Finance (anchoring bias, sentiment proxies)
- Market Microstructure (jump detection, liquidity measures)
- Regime-Dependent Strategies
- Multi-Timeframe Analysis
- Advanced Volatility Modeling
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

# Import all alpha factor modules
from additional_alpha_factors import add_additional_alphas, add_enhanced_alphas
from extended_alpha_factors import add_extended_alphas

def load_binance_data():
    """Load cryptocurrency data from Binance CSV file"""
    print("Loading Binance cryptocurrency data from local files...")
    
    try:
        # Load the combined dataframe
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

def calculate_all_alpha_factors(df):
    """Calculate all 160 alpha factors"""
    print("\n" + "="*50)
    print("CALCULATING COMPREHENSIVE ALPHA FACTOR POOL")
    print("="*50)
    
    # Import the prepare_df function and original alphas
    import sys
    sys.path.append('/Users/kaiwen/Desktop/projects/crypto_fund')
    
    try:
        from crypto_data_analysis_new import prepare_df
        # Import the original alpha functions
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
        
        # Prepare common columns
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
                df[f"alpha_{i}"] = alpha_func(df)
            except Exception as e:
                print(f"  Error computing alpha_{i}: {str(e)}")
                df[f"alpha_{i}"] = 0  # Fill with zeros if calculation fails
        
        print("Original alpha factors complete!")
        
    except ImportError as e:
        print(f"Could not import original alpha functions: {str(e)}")
        print("Skipping original alpha factors...")
    
    # Add additional alpha factors (alpha_102 to alpha_120)
    try:
        df = add_additional_alphas(df)
    except Exception as e:
        print(f"Error adding additional alphas: {str(e)}")
    
    # Add enhanced alpha factors (alpha_121 to alpha_140)
    try:
        df = add_enhanced_alphas(df)
    except Exception as e:
        print(f"Error adding enhanced alphas: {str(e)}")
    
    # Add new extended alpha factors (alpha_141 to alpha_160)
    try:
        df = add_extended_alphas(df)
    except Exception as e:
        print(f"Error adding extended alphas: {str(e)}")
    
    # Get list of all alpha columns
    alpha_columns = [col for col in df.columns if col.startswith('alpha_')]
    alpha_columns.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"\nTotal alpha factors computed: {len(alpha_columns)}")
    print("Factor categories:")
    print("  • alpha_1 to alpha_101: Original WorldQuant alpha factors")
    print("  • alpha_102 to alpha_120: Research-based momentum/mean reversion factors")
    print("  • alpha_121 to alpha_140: Enhanced versions of top-performing factors")
    print("  • alpha_141 to alpha_160: Extended factors with advanced techniques")
    
    return df, alpha_columns

def calculate_alpha_performance(df, alpha_columns, forward_return_period=5):
    """Calculate performance metrics for all alpha factors"""
    print(f"\nCalculating alpha performance with {forward_return_period}-day forward returns...")
    
    # Calculate forward returns
    df = df.sort_values(['Ticker', 'Date'])
    df['forward_return'] = df.groupby('Ticker')['Close'].pct_change(forward_return_period).shift(-forward_return_period)
    
    # Remove rows with NaN forward returns
    df_valid = df.dropna(subset=['forward_return'])
    
    alpha_performance = []
    
    print("Computing performance metrics...")
    for i, alpha in enumerate(alpha_columns):
        try:
            # Skip if alpha column doesn't exist or is all zeros or NaN
            if alpha not in df_valid.columns:
                print(f"  Skipping {alpha}: column not found")
                continue
                
            if df_valid[alpha].abs().sum() == 0 or df_valid[alpha].isna().all():
                print(f"  Skipping {alpha}: all zeros or NaN")
                continue
                
            # Calculate correlation with forward returns
            correlation = df_valid[alpha].corr(df_valid['forward_return'])
            
            if pd.isna(correlation):
                print(f"  Skipping {alpha}: correlation is NaN")
                continue
            
            # Calculate information coefficient (IC)
            ic = correlation
            
            # Calculate quintile performance
            try:
                df_valid_temp = df_valid[[alpha, 'forward_return']].dropna()
                if len(df_valid_temp) < 10:  # Need minimum observations
                    continue
                    
                df_valid_temp['quintile'] = pd.qcut(df_valid_temp[alpha].rank(method='first'), 
                                             q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
                
                quintile_returns = df_valid_temp.groupby('quintile')['forward_return'].mean()
                
                # Long-short return (Q5 - Q1)
                if 'Q5' in quintile_returns.index and 'Q1' in quintile_returns.index:
                    long_short_return = quintile_returns['Q5'] - quintile_returns['Q1']
                else:
                    long_short_return = 0
                    
            except Exception as quintile_error:
                print(f"  Error in quintile calculation for {alpha}: {str(quintile_error)}")
                long_short_return = 0
            
            # Information ratio (IC / std(IC))
            # For simplicity, we'll use absolute IC as a proxy for consistency
            ir = abs(ic)
            
            alpha_performance.append({
                'alpha': alpha,
                'ic': ic,
                'abs_ic': abs(ic),
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

def create_alpha_portfolio(df, performance_df, top_n=20):
    """Create portfolio using top-performing alpha factors"""
    print(f"\nCreating portfolio with top {top_n} alpha factors...")
    
    # Select top alphas
    top_alphas = performance_df.head(top_n)['alpha'].tolist()
    print(f"Selected alphas: {top_alphas[:5]}... (showing first 5)")
    
    # Calculate combined alpha score (equal weighted)
    df['combined_alpha'] = df[top_alphas].mean(axis=1)
    
    # Create portfolio returns
    portfolio_data = []
    
    for date in df['Date'].unique():
        date_data = df[df['Date'] == date].copy()
        
        if len(date_data) < 2:  # Need at least 2 assets
            continue
            
        # Rank assets by combined alpha score
        date_data['alpha_rank'] = date_data['combined_alpha'].rank(pct=True)
        
        # Long top 30%, short bottom 30%
        long_threshold = 0.7
        short_threshold = 0.3
        
        date_data['position'] = 0
        date_data.loc[date_data['alpha_rank'] >= long_threshold, 'position'] = 1
        date_data.loc[date_data['alpha_rank'] <= short_threshold, 'position'] = -1
        
        portfolio_data.append(date_data)
    
    portfolio_df = pd.concat(portfolio_data, ignore_index=True)
    
    # Calculate portfolio returns
    portfolio_df = portfolio_df.sort_values(['Ticker', 'Date'])
    portfolio_df['asset_return'] = portfolio_df.groupby('Ticker')['Close'].pct_change()
    portfolio_df['position_return'] = portfolio_df['position'] * portfolio_df['asset_return']
    
    # Daily portfolio return
    daily_returns = portfolio_df.groupby('Date').agg({
        'position_return': 'mean',
        'position': lambda x: (x != 0).sum()  # Number of positions
    }).reset_index()
    
    daily_returns.columns = ['Date', 'portfolio_return', 'num_positions']
    daily_returns = daily_returns[daily_returns['num_positions'] > 0]  # Only days with positions
    
    print(f"Portfolio created with {len(daily_returns)} trading days")
    return daily_returns, top_alphas, portfolio_df

def calculate_btc_benchmark(df):
    """Calculate BTC buy-and-hold benchmark from Binance data"""
    print("Calculating BTC benchmark from Binance data...")
    
    # Find BTC data in the existing dataset
    btc_data = df[df['Ticker'] == 'BTCUSDT'].copy()
    
    if len(btc_data) == 0:
        print("Warning: BTCUSDT not found in dataset")
        return pd.DataFrame(columns=['Date', 'btc_return'])
    
    btc_data = btc_data.sort_values('Date')
    btc_data['btc_return'] = btc_data['Close'].pct_change()
    btc_data = btc_data.dropna()
    
    print(f"BTC benchmark calculated: {len(btc_data)} data points")
    return btc_data[['Date', 'btc_return']]

def calculate_performance_metrics(returns):
    """Calculate comprehensive performance metrics"""
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown
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

def plot_comprehensive_results(daily_returns, btc_data, performance_df, top_alphas, df_valid):
    """Create comprehensive visualization of results including individual alpha performance"""
    plt.close('all')  # Clear any existing plots
    
    # Initialize individual_alpha_returns at the beginning
    individual_alpha_returns = {}
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 20))
    
    # Main title
    fig.suptitle('Comprehensive Crypto Alpha Strategy Analysis (160 Factors) - Extended History', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Create a 3x2 grid for plots
    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Plot 1: Top 10 Individual Alpha Strategies vs BTC (Full Width)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Always plot the Combined Alpha Portfolio first
    print("\nPlotting Combined Alpha Portfolio performance...")
    
    if len(daily_returns) > 0:
        portfolio_cumulative = (1 + daily_returns['portfolio_return']).cumprod()
        ax1.plot(daily_returns['Date'], portfolio_cumulative, 
                label='Combined Alpha Portfolio', linewidth=4, color='red', alpha=0.9)
        print(f"✓ Combined Alpha Portfolio plotted: {len(portfolio_cumulative)} data points")
    else:
        print("✗ No daily returns data to plot")
    
    # Try to add BTC benchmark if available
    merged_data = pd.DataFrame()
    if len(btc_data) > 0:
        merged_data = daily_returns.merge(btc_data, on='Date', how='inner')
        if len(merged_data) > 0:
            btc_cumulative = (1 + merged_data['btc_return']).cumprod()
            ax1.plot(merged_data['Date'], btc_cumulative, 
                    label='BTC Benchmark', linewidth=3, color='orange', alpha=0.8)
            print(f"✓ BTC benchmark plotted: {len(btc_cumulative)} data points")
        else:
            print("✗ No matching dates between portfolio and BTC data")
    else:
        print("✗ No BTC data available for comparison")
        # Use daily_returns as merged_data for individual alpha calculations
        merged_data = daily_returns.copy()
    
    # Calculate individual alpha performance for top alphas
    individual_alpha_returns = {}
    
    print("\nCalculating individual alpha strategy performance...")
    for alpha in top_alphas[:10]:  # Top 10 alphas
        try:
            # Create individual alpha strategy
            alpha_portfolio_data = []
            
            for date in df_valid['Date'].unique():
                date_data = df_valid[df_valid['Date'] == date].copy()
                
                if len(date_data) < 2 or alpha not in date_data.columns:
                    continue
                    
                # Handle NaN values in alpha scores
                date_data = date_data.dropna(subset=[alpha])
                if len(date_data) < 2:
                    continue
                    
                # Rank assets by this specific alpha
                date_data['alpha_rank'] = date_data[alpha].rank(pct=True, method='first')
                
                # Long top 30%, short bottom 30%
                date_data['position'] = 0
                date_data.loc[date_data['alpha_rank'] >= 0.7, 'position'] = 1
                date_data.loc[date_data['alpha_rank'] <= 0.3, 'position'] = -1
                
                alpha_portfolio_data.append(date_data)
            
            if alpha_portfolio_data:
                alpha_df = pd.concat(alpha_portfolio_data, ignore_index=True)
                alpha_df = alpha_df.sort_values(['Ticker', 'Date'])
                alpha_df['asset_return'] = alpha_df.groupby('Ticker')['Close'].pct_change()
                alpha_df['position_return'] = alpha_df['position'] * alpha_df['asset_return']
                
                # Calculate daily returns for this alpha strategy
                alpha_daily_returns = alpha_df.groupby('Date').agg({
                    'position_return': 'mean',
                    'position': lambda x: (x != 0).sum()
                }).reset_index()
                
                alpha_daily_returns.columns = ['Date', f'{alpha}_return', 'num_positions']
                alpha_daily_returns = alpha_daily_returns[alpha_daily_returns['num_positions'] > 0]
                
                if len(alpha_daily_returns) > 0:
                    individual_alpha_returns[alpha] = alpha_daily_returns[['Date', f'{alpha}_return']]
                    print(f"  ✓ {alpha}: {len(alpha_daily_returns)} trading days")
                else:
                    print(f"  ✗ {alpha}: No valid trading days")
                
        except Exception as e:
            print(f"  Error calculating {alpha}: {str(e)}")
            continue
    
    # Plot individual alphas
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    plotted_count = 0
    for i, (alpha, alpha_data) in enumerate(individual_alpha_returns.items()):
        try:
            # Merge with daily_returns to get matching dates
            merged_alpha = daily_returns.merge(alpha_data, on='Date', how='inner')
            if len(merged_alpha) > 0:
                alpha_cumulative = (1 + merged_alpha[f'{alpha}_return']).cumprod()
                
                # Get alpha category for legend
                alpha_num = int(alpha.split('_')[1])
                if alpha_num <= 101:
                    category = "Orig"
                elif alpha_num <= 120:
                    category = "Res"
                elif alpha_num <= 140:
                    category = "Enh"
                else:
                    category = "Ext"
                
                ax1.plot(merged_alpha['Date'], alpha_cumulative, 
                        label=f'{alpha} ({category})', 
                        linewidth=2, color=colors[i % len(colors)], alpha=0.7)
                plotted_count += 1
                print(f"  ✓ {alpha} plotted: {len(alpha_cumulative)} data points")
        except Exception as e:
            print(f"  Error plotting {alpha}: {str(e)}")
            continue
    
    print(f"Total individual alphas plotted: {plotted_count}")
        
    print(f"Total individual alphas plotted: {plotted_count}")
    
    ax1.set_title('Combined Alpha Portfolio & Individual Strategies - Cumulative Returns', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to better show performance differences
    
    # Calculate and display performance metrics
    if len(daily_returns) > 0:
        portfolio_metrics = calculate_performance_metrics(daily_returns['portfolio_return'])
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON (Extended History)")
        print("="*60)
        print("\nCombined Alpha Portfolio:")
        for key, value in portfolio_metrics.items():
            print(f"  {key}: {value}")
    
    if len(merged_data) > 0 and 'btc_return' in merged_data.columns:
        btc_metrics = calculate_performance_metrics(merged_data['btc_return'])
        print("\nBTC Benchmark:")
        for key, value in btc_metrics.items():
            print(f"  {key}: {value}")
    else:
        print("\nBTC Benchmark: No data available for comparison")
    
    # Plot 2: Alpha Factor IC Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    top_20 = performance_df.head(20)
    bars = ax2.bar(range(len(top_20)), top_20['abs_ic'], 
                   color='green', alpha=0.7)
    ax2.set_title('Top 20 Alpha Factors (by |IC|)', fontweight='bold')
    ax2.set_ylabel('Absolute Information Coefficient')
    ax2.set_xlabel('Alpha Factor Rank')
    
    # Add alpha names on bars (rotated)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{top_20.iloc[i]["alpha"]}', 
                ha='center', va='bottom', rotation=45, fontsize=8)
    
    # Plot 3: Sharpe Ratio Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate Sharpe ratios for individual alphas
    sharpe_ratios = []
    alpha_names = []
    
    if individual_alpha_returns:  # Only if we have individual alpha returns
        for alpha, alpha_data in individual_alpha_returns.items():
            merged_alpha = merged_data.merge(alpha_data, on='Date', how='inner')
            if len(merged_alpha) > 0:
                returns = merged_alpha[f'{alpha}_return'].dropna()
                if len(returns) > 0:
                    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
                    volatility = returns.std() * np.sqrt(252)
                    sharpe = annual_return / volatility if volatility > 0 else 0
                    sharpe_ratios.append(sharpe)
                    alpha_names.append(alpha)
    
    if sharpe_ratios:
        bars = ax3.bar(alpha_names, sharpe_ratios, alpha=0.7, color='blue')
        
        # Add BTC Sharpe ratio for comparison if available
        if len(merged_data) > 0 and 'btc_return' in merged_data.columns:
            btc_annual_return = (1 + merged_data['btc_return']).prod() ** (252 / len(merged_data)) - 1
            btc_volatility = merged_data['btc_return'].std() * np.sqrt(252)
            btc_sharpe = btc_annual_return / btc_volatility if btc_volatility > 0 else 0
            ax3.axhline(y=btc_sharpe, color='orange', linestyle='--', linewidth=2, 
                   label=f'BTC Benchmark ({btc_sharpe:.2f})')
            ax3.set_title('Sharpe Ratios: Alpha Strategies vs BTC', fontweight='bold')
        else:
            ax3.set_title('Sharpe Ratios: Alpha Strategies', fontweight='bold')
        
        ax3.set_ylabel('Sharpe Ratio')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No individual alpha returns available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Sharpe Ratios: Alpha Strategies', fontweight='bold')
    
    # Plot 4: IC Distribution Histogram
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(performance_df['ic'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_title('Distribution of Alpha Factor IC', fontweight='bold')
    ax4.set_xlabel('Information Coefficient')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    mean_ic = performance_df['ic'].mean()
    std_ic = performance_df['ic'].std()
    ax4.text(0.02, 0.95, f'Mean IC: {mean_ic:.4f}\nStd IC: {std_ic:.4f}', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 5: Risk-Return Scatter
    ax5 = fig.add_subplot(gs[2, 1])
    
    if sharpe_ratios:
        # Calculate risk-return for each alpha
        alpha_returns_annual = []
        alpha_volatilities = []
        
        for alpha, alpha_data in individual_alpha_returns.items():
            merged_alpha = merged_data.merge(alpha_data, on='Date', how='inner')
            if len(merged_alpha) > 0:
                returns = merged_alpha[f'{alpha}_return'].dropna()
                if len(returns) > 0:
                    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
                    volatility = returns.std() * np.sqrt(252)
                    alpha_returns_annual.append(annual_return * 100)  # Convert to percentage
                    alpha_volatilities.append(volatility * 100)  # Convert to percentage
        
        if alpha_returns_annual:
            # Color by Sharpe ratio
            scatter = ax5.scatter(alpha_volatilities, alpha_returns_annual, 
                                c=sharpe_ratios, cmap='RdYlGn', s=100, alpha=0.8)
            
            # Add BTC benchmark point if available
            if len(merged_data) > 0 and 'btc_return' in merged_data.columns:
                btc_annual_return = (1 + merged_data['btc_return']).prod() ** (252 / len(merged_data)) - 1
                btc_volatility = merged_data['btc_return'].std() * np.sqrt(252)
                ax5.scatter(btc_volatility * 100, btc_annual_return * 100, 
                           color='orange', s=200, marker='*', 
                           label='BTC Benchmark', edgecolors='black', linewidth=2)
            
            # Add combined portfolio point
            portfolio_annual = (1 + merged_data['portfolio_return']).prod() ** (252 / len(merged_data)) - 1
            portfolio_vol = merged_data['portfolio_return'].std() * np.sqrt(252)
            ax5.scatter(portfolio_vol * 100, portfolio_annual * 100, 
                       color='red', s=200, marker='*', 
                       label='Combined Portfolio', edgecolors='black', linewidth=2)
            
            plt.colorbar(scatter, ax=ax5, label='Sharpe Ratio')
            ax5.set_xlabel('Annualized Volatility (%)')
            ax5.set_ylabel('Annualized Return (%)')
            ax5.set_title('Risk-Return Profile: Alpha Strategies', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
    
    # Plot 6: Factor Category Performance
    ax6 = fig.add_subplot(gs[3, :])
    
    categories = {
        'Original (1-101)': [f'alpha_{i}' for i in range(1, 102)],
        'Research (102-120)': [f'alpha_{i}' for i in range(102, 121)],
        'Enhanced (121-140)': [f'alpha_{i}' for i in range(121, 141)],
        'Extended (141-160)': [f'alpha_{i}' for i in range(141, 161)]
    }
    
    category_performance = {}
    category_counts = {}
    for category, alpha_list in categories.items():
        category_alphas = performance_df[performance_df['alpha'].isin(alpha_list)]
        if len(category_alphas) > 0:
            category_performance[category] = category_alphas['abs_ic'].mean()
            category_counts[category] = len(category_alphas)
    
    if category_performance:
        category_names = list(category_performance.keys())
        category_scores = list(category_performance.values())
        
        bars = ax6.bar(category_names, category_scores, 
                       color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        ax6.set_title('Average Performance by Factor Category', fontweight='bold')
        ax6.set_ylabel('Average |IC|')
        ax6.tick_params(axis='x', rotation=15)
        
        # Add value labels and counts on bars
        for bar, score, category in zip(bars, category_scores, category_names):
            height = bar.get_height()
            count = category_counts[category]
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                     f'{score:.4f}\n({count} factors)', 
                     ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot to data folder
    plot_filename = "data/enhanced_alpha_analysis_extended_history.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Comprehensive analysis plot saved to: {plot_filename}")
    
    plt.show()
    
    # Individual alpha performance summary
    if individual_alpha_returns:
        print("\n" + "="*60)
        print("INDIVIDUAL ALPHA PERFORMANCE (Top 10)")
        print("="*60)
        
        for i, (alpha, alpha_data) in enumerate(individual_alpha_returns.items()):
            merged_alpha = merged_data.merge(alpha_data, on='Date', how='inner')
            if len(merged_alpha) > 0:
                returns = merged_alpha[f'{alpha}_return'].dropna()
                if len(returns) > 0:
                    metrics = calculate_performance_metrics(returns)
                    alpha_num = int(alpha.split('_')[1])
                    if alpha_num <= 101:
                        category = "Original"
                    elif alpha_num <= 120:
                        category = "Research"  
                    elif alpha_num <= 140:
                        category = "Enhanced"
                    else:
                        category = "Extended"
                    
                    print(f"\n{alpha} ({category}):")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
    
    return individual_alpha_returns
    
    # Summary statistics
    print("\n" + "="*50)
    print("ALPHA FACTOR SUMMARY")
    print("="*50)
    print(f"Total alpha factors evaluated: {len(performance_df)}")
    print(f"Factors with positive IC: {sum(performance_df['ic'] > 0)}")
    print(f"Factors with |IC| > 0.01: {sum(performance_df['abs_ic'] > 0.01)}")
    print(f"Factors with |IC| > 0.02: {sum(performance_df['abs_ic'] > 0.02)}")
    print(f"Best performing alpha: {performance_df.iloc[0]['alpha']} (IC: {performance_df.iloc[0]['ic']:.4f})")
    
    print("\nTop 10 Alpha Factors:")
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

def main():
    """Main execution function"""
    print("="*60)
    print("COMPREHENSIVE CRYPTO ALPHA FACTOR STRATEGY")
    print("Enhanced with 160 Alpha Factors - Using Historical Binance Data")
    print("="*60)
    
    try:
        # Load the existing Binance data
        df = load_binance_data()
        if df is None:
            return
        
        # Prepare data with technical indicators
        print("\nPreparing technical indicators...")
        tickers = df["Ticker"].unique()
        for ticker in tickers:
            mask = df["Ticker"] == ticker
            try:
                # Add basic technical indicators if not present
                if 'SMA_20' not in df.columns:
                    df.loc[mask, "SMA_20"] = talib.SMA(df.loc[mask, "Close"], timeperiod=20)
                    df.loc[mask, "EMA_20"] = talib.EMA(df.loc[mask, "Close"], timeperiod=20)
                    df.loc[mask, "RSI_14"] = talib.RSI(df.loc[mask, "Close"], timeperiod=14)
                    df.loc[mask, "MACD"], df.loc[mask, "MACD_signal"], df.loc[mask, "MACD_hist"] = talib.MACD(df.loc[mask, "Close"])
            except Exception as e:
                print(f"Warning: Could not compute technical indicators for {ticker}: {str(e)}")
        
        # Calculate all alpha factors
        df, alpha_columns = calculate_all_alpha_factors(df)
        
        # Evaluate alpha performance
        performance_df, df_valid = calculate_alpha_performance(df, alpha_columns)
        
        if len(performance_df) == 0:
            print("No alpha factors could be evaluated. Check data quality.")
            return
            
        print(f"\nSuccessfully evaluated {len(performance_df)} alpha factors")
        
        # Create portfolio
        daily_returns, top_alphas, portfolio_df = create_alpha_portfolio(df_valid, performance_df)
        
        # Calculate BTC benchmark using the same Binance data
        btc_data = calculate_btc_benchmark(df)
        
        # Create visualizations
        individual_alphas = plot_comprehensive_results(daily_returns, btc_data, performance_df, top_alphas, df_valid)
        
        # Calculate and display portfolio performance metrics
        print("\n" + "="*60)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("="*60)
        
        if len(daily_returns) > 0:
            portfolio_metrics = calculate_performance_metrics(daily_returns['portfolio_return'])
            print("📈 Combined Alpha Portfolio Performance:")
            for key, value in portfolio_metrics.items():
                print(f"  {key}: {value}")
        
        if len(btc_data) > 0:
            btc_metrics = calculate_performance_metrics(btc_data['btc_return'])
            print("\n🪙 BTC Benchmark Performance:")
            for key, value in btc_metrics.items():
                print(f"  {key}: {value}")
        
        # Save performance results to CSV
        results_filename = "data/enhanced_alpha_performance_results.csv"
        performance_df.to_csv(results_filename, index=False)
        print(f"\n💾 Alpha performance results saved to: {results_filename}")
        
        portfolio_filename = "data/enhanced_portfolio_returns.csv"
        daily_returns.to_csv(portfolio_filename, index=False)
        print(f"📊 Portfolio daily returns saved to: {portfolio_filename}")
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("✓ Binance data loaded from local files (2017-2025)")
        print(f"✓ {len(performance_df)} alpha factors evaluated")
        print("✓ Performance evaluation completed")
        print("✓ Portfolio strategy implemented")
        print("✓ Comprehensive visualizations created and saved")
        print("✓ Results exported to CSV files")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
