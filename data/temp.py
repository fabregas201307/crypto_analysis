import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 1. Create a Fake Messy Dataset (Like what CTC will give you)
data = {
    'timestamp': pd.to_datetime(['2026-02-27 10:00:00', '2026-02-27 10:00:01', '2026-02-27 10:00:02', '2026-02-27 09:00:00']), # Last one is stale
    'strike': [5000, 5100, 4900, 5000],
    'type': ['C', 'C', 'P', 'C'],
    'bid': [100.0, 50.0, 0.0, 105.0],   # 4900 Put has a 0 bid (illiquid)
    'ask': [102.0, 48.0, 55.0, 107.0],  # 5100 Call is crossed (bid 50 > ask 48)
    'spot': [5000, 5000, 5000, 4950],
    'dte': [30/365, 30/365, 30/365, 30/365] # Days to expiry in years
}
df = pd.DataFrame(data)
print("--- RAW DATA ---")
print(df)

# 2. Data Cleaning (The "Sanity Checks")
print("\n--- CLEANING DATA ---")
# A. Remove Zero Bids (Illiquid options)
df_clean = df[df['bid'] > 0].copy()

# B. Remove Crossed Markets (Bid >= Ask)
df_clean = df_clean[df_clean['bid'] < df_clean['ask']].copy()

# C. Remove Stale Quotes (e.g., older than 5 seconds from current time)
current_time = pd.to_datetime('2026-02-27 10:00:05')
df_clean = df_clean[(current_time - df_clean['timestamp']).dt.total_seconds() < 5].copy()

print(df_clean)

# 3. Calculate Mid Price
df_clean['mid_price'] = (df_clean['bid'] + df_clean['ask']) / 2

# 4. Calculate Implied Volatility (Using Black-Scholes and Brent's Method)
def bs_price(vol, S, K, T, r, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if option_type == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calc_iv(row):
    # Objective function: BS Price - Market Price = 0
    objective = lambda vol: bs_price(vol, row['spot'], row['strike'], row['dte'], 0.05, row['type']) - row['mid_price']
    try:
        # Search for IV between 1% and 300%
        return brentq(objective, 0.01, 3.0) 
    except ValueError:
        return np.nan # Fails if arbitrage exists in the quote

df_clean['market_iv'] = df_clean.apply(calc_iv, axis=1)

# 5. Build a Simple Signal (Vol Arbitrage)
# Assume our proprietary 'Otto' model says fair IV for this strike is 15% (0.15)
df_clean['fair_iv'] = 0.15 

# Signal: If Market IV > Fair IV, the option is expensive -> SELL (-1)
# If Market IV < Fair IV, the option is cheap -> BUY (1)
df_clean['edge'] = df_clean['market_iv'] - df_clean['fair_iv']
df_clean['signal'] = np.where(df_clean['edge'] > 0, 'SELL', 'BUY')

print("\n--- FINAL PROCESSED DATA WITH SIGNAL ---")
print(df_clean[['strike', 'type', 'mid_price', 'market_iv', 'fair_iv', 'edge', 'signal']])


# =========================
# 6. Visualization helpers
# =========================
def bs_d1(S, K, T, r, q, vol):
    vol = np.maximum(vol, 1e-8)
    return (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))


def bs_d2(S, K, T, r, q, vol):
    return bs_d1(S, K, T, r, q, vol) - vol * np.sqrt(T)


def bs_call_price(S, K, T, r, q, vol):
    d1 = bs_d1(S, K, T, r, q, vol)
    d2 = d1 - vol * np.sqrt(T)
    return np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)


def bs_put_price(S, K, T, r, q, vol):
    d1 = bs_d1(S, K, T, r, q, vol)
    d2 = d1 - vol * np.sqrt(T)
    return np.exp(-r * T) * K * norm.cdf(-d2) - np.exp(-q * T) * S * norm.cdf(-d1)


def bs_call_delta(S, K, T, r, q, vol):
    d1 = bs_d1(S, K, T, r, q, vol)
    return np.exp(-q * T) * norm.cdf(d1)


def bs_vanna(S, K, T, r, q, vol):
    # One common closed-form for spot vanna in BS:
    # vanna = exp(-qT) * phi(d1) * (-d2 / vol)
    d1 = bs_d1(S, K, T, r, q, vol)
    d2 = bs_d2(S, K, T, r, q, vol)
    return np.exp(-q * T) * norm.pdf(d1) * (-d2 / np.maximum(vol, 1e-8))


# ===============================================
# 7. Plot (1): Danger of "Smile Premium" selling
# ===============================================
S0 = 100.0
r = 0.00
q = 0.00
T = 30 / 365

m_grid = np.linspace(-0.25, 0.25, 121)  # log-moneyness ln(K/F)
K_grid = S0 * np.exp(m_grid)

# Fair surface vs market surface with wing premium
fair_iv_curve = 0.13 + 0.20 * m_grid**2
wing_premium = 0.03 * (np.abs(m_grid) / 0.25) ** 1.4
market_iv_curve = fair_iv_curve + wing_premium

fig1, ax = plt.subplots(1, 2, figsize=(13, 5))

ax[0].plot(m_grid, fair_iv_curve, label='Fair IV (internal surface)', linewidth=2)
ax[0].plot(m_grid, market_iv_curve, label='Market IV (broker)', linewidth=2)
ax[0].set_title('Smile Premium: wings richer than internal fair value')
ax[0].set_xlabel('Log-moneyness ln(K/F)')
ax[0].set_ylabel('Implied Vol')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(m_grid, market_iv_curve - fair_iv_curve, color='darkred', linewidth=2)
ax[1].axhline(0, color='black', linewidth=1)
ax[1].set_title('Wing premium (Market IV - Fair IV)')
ax[1].set_xlabel('Log-moneyness ln(K/F)')
ax[1].set_ylabel('Vol premium')
ax[1].grid(alpha=0.3)

fig1.tight_layout()
fig1.savefig('ctc_plot_1_smile_premium.png', dpi=150)


# Optional stress chart: short-wing carry vs crash risk
K_put = 95.0
K_call = 105.0

def local_market_vol(K, S_ref):
    m = np.log(K / S_ref)
    return 0.13 + 0.20 * m**2 + 0.03 * (abs(m) / 0.25) ** 1.4


def local_fair_vol(K, S_ref):
    m = np.log(K / S_ref)
    return 0.13 + 0.20 * m**2


entry_premium = (
    bs_put_price(S0, K_put, T, r, q, local_market_vol(K_put, S0))
    + bs_call_price(S0, K_call, T, r, q, local_market_vol(K_call, S0))
)

S_stress = np.linspace(60, 140, 321)
T_next = 25 / 365

# "Normal" next-day assumption for smile premium carry:
# no crash, and a mild mean reversion from rich market wing vols toward fair vols.
put_fair_vol = local_fair_vol(K_put, S0)
call_fair_vol = local_fair_vol(K_call, S0)
close_cost_normal = bs_put_price(S_stress, K_put, T_next, r, q, put_fair_vol) + bs_call_price(
    S_stress, K_call, T_next, r, q, call_fair_vol
)
pnl_short_wings_normal = entry_premium - close_cost_normal

# "Black-swan" next-day vol assumption: large vol jump
crash_vol = 0.65
close_cost_crash = bs_put_price(S_stress, K_put, T_next, r, q, crash_vol) + bs_call_price(
    S_stress, K_call, T_next, r, q, crash_vol
)
pnl_short_wings_crash = entry_premium - close_cost_crash

fig1b, axb = plt.subplots(figsize=(8, 5))
axb.plot(S_stress, pnl_short_wings_normal, label='Short wings PnL (normal day: carry + vol mean reversion)', linewidth=2)
axb.plot(S_stress, pnl_short_wings_crash, label='Short wings PnL (vol shock next day)', linewidth=2)
axb.axhline(0, color='black', linewidth=1)
axb.axvline(S0, color='gray', linestyle='--', linewidth=1)
axb.set_title('Danger of Smile Premium: short-wing carry vs tail loss')
axb.set_xlabel('Spot next day')
axb.set_ylabel('Mark-to-market PnL')
axb.legend()
axb.grid(alpha=0.3)
fig1b.tight_layout()
fig1b.savefig('ctc_plot_1b_short_wings_tail_risk.png', dpi=150)

center_idx = np.argmin(np.abs(S_stress - S0))
print(f"\nShort-wings center PnL at S={S0:.0f} (normal): {pnl_short_wings_normal[center_idx]:.4f}")
print(f"Short-wings center PnL at S={S0:.0f} (crash):  {pnl_short_wings_crash[center_idx]:.4f}")


# ==================================================
# 8. Plot (2): Buy ATM / Sell wings visualization
# ==================================================
# Mispricing shape chosen to reflect: ATM cheap, wings rich
mispricing = fair_iv_curve - market_iv_curve  # >0 buy, <0 sell

fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))

ax2[0].plot(m_grid, mispricing, linewidth=2)
ax2[0].axhline(0, color='black', linewidth=1)
ax2[0].fill_between(m_grid, mispricing, 0, where=(mispricing > 0), alpha=0.25, label='BUY region (ATM)')
ax2[0].fill_between(m_grid, mispricing, 0, where=(mispricing < 0), alpha=0.25, label='SELL region (wings)')
ax2[0].set_title('Signal map: Fair IV - Market IV')
ax2[0].set_xlabel('Log-moneyness ln(K/F)')
ax2[0].set_ylabel('IV edge')
ax2[0].legend()
ax2[0].grid(alpha=0.3)

# Stylized position weights: long center, short shoulders/wings
strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
weights = np.array([-0.4, -0.6, -0.9, -0.4, 2.6, -0.4, -0.9, -0.6, -0.4])

ax2[1].bar(strikes, weights, width=3.5)
ax2[1].axhline(0, color='black', linewidth=1)
ax2[1].set_title('Stylized portfolio: buy ATM, sell wings/shoulders')
ax2[1].set_xlabel('Strike')
ax2[1].set_ylabel('Position weight (+ long / - short)')
ax2[1].grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig('ctc_plot_2_buy_atm_sell_wings.png', dpi=150)


# =================================================================
# 9. Plot (3): ATM vanna intuition and Delta-vs-vol relationship
# =================================================================
T_v = 0.25
vol_grid = np.linspace(0.05, 1.0, 120)
K_list = [90, 100, 110]

fig3, ax3 = plt.subplots(1, 2, figsize=(13, 5))

for K in K_list:
    deltas = bs_call_delta(S0, K, T_v, r, q, vol_grid)
    ax3[0].plot(vol_grid, deltas, label=f'Call Delta, K={K}')

ax3[0].axhline(0.5, color='gray', linestyle='--', linewidth=1)
ax3[0].set_title('Call Delta vs Implied Vol (spot fixed)')
ax3[0].set_xlabel('Implied Vol')
ax3[0].set_ylabel('Call Delta')
ax3[0].legend()
ax3[0].grid(alpha=0.3)

K_vanna = np.linspace(70, 130, 200)
vanna_vals = bs_vanna(S0, K_vanna, T_v, r, q, 0.20)
ax3[1].plot(K_vanna, vanna_vals, linewidth=2)
ax3[1].axhline(0, color='black', linewidth=1)
ax3[1].axvline(S0, color='gray', linestyle='--', linewidth=1)
ax3[1].set_title('Vanna vs Strike (vol fixed at 20%)')
ax3[1].set_xlabel('Strike')
ax3[1].set_ylabel('Vanna')
ax3[1].grid(alpha=0.3)

fig3.tight_layout()
fig3.savefig('ctc_plot_3_vanna_and_delta_vs_vol.png', dpi=150)


# =====================================================================
# 10. 3D Vol Surface demo: quotes -> implied vols -> interpolated surface
# =====================================================================
np.random.seed(42)

S_ref = 100.0
r_s = 0.01
q_s = 0.00

tenors_days = np.array([7, 14, 30, 60, 90, 180, 365])
tenors = tenors_days / 365.0
strikes_3d = np.arange(70, 131, 5)


def synthetic_market_iv(K, T):
    # fair base with term structure + skew/smile
    m = np.log(K / S_ref)
    base = 0.12 + 0.06 * np.exp(-2.0 * T) + 0.12 * (m**2)
    skew = -0.04 * m
    # add wing premium to mimic market insurance demand
    wing = 0.025 * (np.abs(m) / 0.25) ** 1.3
    return np.clip(base + skew + wing, 0.05, 1.50)


quote_rows = []
for T_i in tenors:
    for K_i in strikes_3d:
        iv_true = synthetic_market_iv(K_i, T_i)
        theo = bs_call_price(S_ref, K_i, T_i, r_s, q_s, iv_true)
        # realistic micro-noise + spread
        spread = max(0.02, 0.01 * theo)
        noise = np.random.normal(0.0, 0.15 * spread)
        mid = max(theo + noise, 0.01)
        bid = max(mid - spread / 2, 0.0)
        ask = mid + spread / 2

        # inject a few bad quotes to demonstrate cleaning
        if np.random.rand() < 0.02:
            bid = 0.0
        if np.random.rand() < 0.02:
            bid, ask = ask + 0.01, bid

        quote_rows.append(
            {
                'T': T_i,
                'days': int(round(T_i * 365)),
                'strike': float(K_i),
                'bid': float(bid),
                'ask': float(ask),
            }
        )

quotes = pd.DataFrame(quote_rows)
quotes['mid'] = (quotes['bid'] + quotes['ask']) / 2.0

# clean quote data
quotes_clean = quotes[(quotes['bid'] > 0) & (quotes['bid'] < quotes['ask'])].copy()


def iv_from_call_mid(mid, S, K, T, r, q):
    f = lambda vol: bs_call_price(S, K, T, r, q, vol) - mid
    try:
        return brentq(f, 1e-4, 3.0)
    except ValueError:
        return np.nan


quotes_clean['iv'] = quotes_clean.apply(
    lambda row: iv_from_call_mid(row['mid'], S_ref, row['strike'], row['T'], r_s, q_s), axis=1
)
quotes_clean = quotes_clean.dropna(subset=['iv']).copy()

# interpolate to smooth grid for 3D plotting
K_mesh = np.linspace(quotes_clean['strike'].min(), quotes_clean['strike'].max(), 60)
T_mesh = np.linspace(quotes_clean['T'].min(), quotes_clean['T'].max(), 60)
KK, TT = np.meshgrid(K_mesh, T_mesh)

points = np.column_stack((quotes_clean['strike'].values, quotes_clean['T'].values))
values = quotes_clean['iv'].values
IV_grid = griddata(points, values, (KK, TT), method='cubic')

# fill any interpolation holes with nearest
IV_grid_nearest = griddata(points, values, (KK, TT), method='nearest')
IV_grid = np.where(np.isnan(IV_grid), IV_grid_nearest, IV_grid)

fig4 = plt.figure(figsize=(14, 6))
ax4a = fig4.add_subplot(1, 2, 1, projection='3d')
ax4b = fig4.add_subplot(1, 2, 2, projection='3d')

# raw recovered IV points
ax4a.scatter(
    quotes_clean['strike'].values,
    quotes_clean['days'].values,
    quotes_clean['iv'].values,
    s=10,
    alpha=0.8,
)
ax4a.set_title('Recovered IV points from cleaned quotes')
ax4a.set_xlabel('Strike')
ax4a.set_ylabel('Days to expiry')
ax4a.set_zlabel('Implied vol')

# interpolated smooth surface
surf = ax4b.plot_surface(KK, TT * 365, IV_grid, linewidth=0, antialiased=True, alpha=0.95)
ax4b.set_title('Interpolated 3D vol surface (quotes -> IV -> surface)')
ax4b.set_xlabel('Strike')
ax4b.set_ylabel('Days to expiry')
ax4b.set_zlabel('Implied vol')
fig4.colorbar(surf, ax=ax4b, shrink=0.6, pad=0.08)

fig4.tight_layout()
fig4.savefig('ctc_plot_4_3d_surface_from_quotes.png', dpi=150)

print(f"\n3D surface build stats: raw quotes={len(quotes)}, cleaned quotes={len(quotes_clean)}")

print("\nSaved charts:")
print("- ctc_plot_1_smile_premium.png")
print("- ctc_plot_1b_short_wings_tail_risk.png")
print("- ctc_plot_2_buy_atm_sell_wings.png")
print("- ctc_plot_3_vanna_and_delta_vs_vol.png")
print("- ctc_plot_4_3d_surface_from_quotes.png")
