"""
Test Alphalens integration with simple synthetic data
"""
import pandas as pd
import numpy as np
import alphalens as al

# Create simple test data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
assets = ['ASSET_A', 'ASSET_B', 'ASSET_C']

# Create price data
price_data = pd.DataFrame(
    np.random.randn(100, 3).cumsum(axis=0) + 100,
    index=dates,
    columns=assets
)

# Create factor data
factor_data = pd.DataFrame(
    np.random.randn(100, 3),
    index=dates,
    columns=assets
)

# Stack factor data for Alphalens
factor_series = factor_data.stack()
factor_series.index.names = ['date', 'asset']

print("Testing Alphalens with synthetic data...")
print(f"Price data shape: {price_data.shape}")
print(f"Factor series length: {len(factor_series)}")

try:
    # Test Alphalens basic functionality
    factor_data_al = al.utils.get_clean_factor_and_forward_returns(
        factor=factor_series,
        prices=price_data,
        quantiles=5,
        periods=(1, 5),
        max_loss=0.5
    )
    
    print(f"✅ Alphalens test successful: {len(factor_data_al)} observations")
    print(factor_data_al.head())
    
except Exception as e:
    print(f"❌ Alphalens test failed: {str(e)}")
    import traceback
    traceback.print_exc()
