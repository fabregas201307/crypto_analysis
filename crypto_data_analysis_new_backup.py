import pandas as pd
import numpy as np
from scipy.stats import rankdata
import talib

# Helper functions updated to handle both string column names and series

def ts_min(df, s, d):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        s = pd.Series(s, index=df.index)
    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].rolling(d, min_periods=1).min())
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def ts_max(df, s, d):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        s = pd.Series(s, index=df.index)
    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].rolling(d, min_periods=1).max())
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def ts_argmin(df, s, d):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s = pd.Series(s, index=df.index)

    def argmin_lags(window):
        if len(window) == 0:
            return np.nan
        return len(window) - 1 - np.argmin(window)
    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].rolling(d, min_periods=1).apply(argmin_lags, raw=True))
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def ts_argmax(df, s, d):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s = pd.Series(s, index=df.index)
    
    def argmax_lags(window):
        if len(window) == 0:
            return np.nan
        return len(window) - 1 - np.argmax(window)
    
    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].rolling(d, min_periods=1).apply(argmax_lags, raw=True))
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def ts_rank(df, s, d):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s = pd.Series(s, index=df.index)

    def rank_window(window):
        if len(window) == 0:
            return np.nan
        ranks = rankdata(window)
        return ranks[-1] / len(window)
    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].rolling(d, min_periods=1).apply(rank_window, raw=True))
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def decay_linear(df, s, d):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s = pd.Series(s, index=df.index)

    def wma(window):
        if len(window) == 0:
            return np.nan
        weights = np.arange(1, len(window) + 1) / np.sum(np.arange(1, len(window) + 1))
        return np.dot(window, weights)
    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].rolling(d, min_periods=1).apply(wma, raw=True))
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def stddev(df, s, d):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s = pd.Series(s, index=df.index)

    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].rolling(d, min_periods=1).std())
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def ts_sum(df, s, d):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        s = pd.Series(s, index=df.index)

    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].rolling(d, min_periods=1).sum())
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)


def ts_product(df, s, d):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s = pd.Series(s, index=df.index)

    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].rolling(d, min_periods=1).apply(np.prod))
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def ts_correlation(df, s1, s2, d):
    if isinstance(s1, str):
        s1 = df[s1]
    elif isinstance(s1, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s1 = pd.Series(s1, index=df.index)

    if isinstance(s2, str):
        s2 = df[s2]
    elif isinstance(s2, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s2 = pd.Series(s2, index=df.index)

    def corr_func(g):
        return s1.loc[g.index].rolling(d, min_periods=1).corr(s2.loc[g.index])
    result = df.groupby('Ticker').apply(corr_func)
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def ts_covariance(df, s1, s2, d):
    if isinstance(s1, str):
        s1 = df[s1]
    elif isinstance(s1, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s1 = pd.Series(s1, index=df.index)

    if isinstance(s2, str):
        s2 = df[s2]
    elif isinstance(s2, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s2 = pd.Series(s2, index=df.index)

    def cov_func(g):
        return s1.loc[g.index].rolling(d, min_periods=1).cov(s2.loc[g.index])
    result = df.groupby('Ticker').apply(cov_func)
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def ts_delta(df, s, d):
    if isinstance(s, str):
        s = df[s]
    result = df.groupby('Ticker').apply(lambda g: s.loc[g.index].diff(d))
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def signed_power(df, s, a):
    if isinstance(s, str):
        s = df[s]
    return np.sign(s) * np.abs(s) ** a

def cs_rank(df, s):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s = pd.Series(s, index=df.index)
    result = df.groupby('Date').apply(lambda g: s.loc[g.index].rank(pct=True))
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

def cs_scale(df, s, a=1):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s = pd.Series(s, index=df.index)
    abs_sum = df.groupby('Date').apply(lambda g: np.sum(np.abs(s.loc[g.index]))).reindex(df['Date']).values
    return a * s / np.clip(abs_sum, a_min=1e-12, a_max=None)

def indneutralize(df, s, level=None):
    if isinstance(s, str):
        s = df[s]
    elif isinstance(s, np.ndarray):
        # Convert NumPy array to Series, aligning with df index
        s = pd.Series(s, index=df.index)
    means = df.groupby('Date').apply(lambda g: np.mean(s.loc[g.index])).reindex(df['Date']).values
    return s - means

def adv(df, d):
    result = df.groupby('Ticker')['Volume USDT'].rolling(d, min_periods=1).mean()
    result.index = result.index.droplevel(0)
    return result.reindex(df.index)

# Prepare common columns
def prepare_df(df):
    df = df.copy()
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['vwap'] = df['Volume USDT'] / df['Volume'].clip(lower=1e-12)
    df['cap'] = 1  # Placeholder for market cap
    
    # Create all required adv (average daily volume) columns
    df['adv5'] = adv(df, 5)
    df['adv10'] = adv(df, 10)
    df['adv15'] = adv(df, 15)
    df['adv20'] = adv(df, 20)
    df['adv30'] = adv(df, 30)
    df['adv40'] = adv(df, 40)
    df['adv50'] = adv(df, 50)
    df['adv60'] = adv(df, 60)
    df['adv81'] = adv(df, 81)
    df['adv120'] = adv(df, 120)
    df['adv150'] = adv(df, 150)
    df['adv180'] = adv(df, 180)
    
    return df

# All 101 alpha functions based on the formulas

def alpha1(df):
    inner = np.where(df['returns'] < 0, stddev(df, 'returns', 20), df['Close'])
    powered = signed_power(df, inner, 2)
    argmax = ts_argmax(df, powered, 5)
    ranked = cs_rank(df, argmax)
    return ranked - 0.5

def alpha2(df):
    delta_log_vol = ts_delta(df, np.log(df['Volume']), 2)
    ranked_delta = cs_rank(df, delta_log_vol)
    inner = (df['Close'] - df['Open']) / df['Open']
    ranked_inner = cs_rank(df, inner)
    corr = ts_correlation(df, ranked_delta, ranked_inner, 6)
    return -1 * corr

def alpha3(df):
    ranked_open = cs_rank(df, df['Open'])
    ranked_vol = cs_rank(df, df['Volume'])
    corr = ts_correlation(df, ranked_open, ranked_vol, 10)
    return -1 * corr

def alpha4(df):
    ranked_low = cs_rank(df, df['Low'])
    tsrank = ts_rank(df, ranked_low, 9)
    return -1 * tsrank

def alpha5(df):
    sum_vwap = ts_sum(df, 'vwap', 10)
    inner = df['Open'] - (sum_vwap / 10)
    ranked_close_vwap = cs_rank(df, df['Close'] - df['vwap'])
    abs_ranked = np.abs(ranked_close_vwap)
    return cs_rank(df, inner) * (-1 * abs_ranked)

def alpha6(df):
    corr = ts_correlation(df, 'Open', 'Volume', 10)
    return -1 * corr

def alpha7(df):
    adv20 = adv(df, 20)
    condition = adv20 < df['Volume']
    inner = -1 * ts_rank(df, np.abs(ts_delta(df, 'Close', 7)), 60) * np.sign(ts_delta(df, 'Close', 7))
    return np.where(condition, inner, -1)

# def alpha8(df):
#     sum_open = ts_sum(df, 'Open', 5)
#     sum_returns = ts_sum(df, 'returns', 5)
#     inner = sum_open * sum_returns
#     delayed = df.groupby('Ticker')[inner].shift(10)
#     diff = inner - delayed
#     ranked = cs_rank(df, diff)
#     return -1 * ranked

def alpha8(df):
    sum_open = ts_sum(df, 'Open', 5)
    sum_returns = ts_sum(df, 'returns', 5)
    # Add inner as a temporary column
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['temp_inner'] = sum_open * sum_returns
    # Groupby and shift
    delayed = df.groupby('Ticker')['temp_inner'].shift(10)
    diff = df['temp_inner'] - delayed
    ranked = cs_rank(df, diff)
    # Drop the temporary column
    df.drop(columns=['temp_inner'], inplace=True)
    return -1 * ranked

def alpha9(df):
    delta_close = ts_delta(df, 'Close', 1)
    condition1 = ts_min(df, delta_close, 5) > 0
    condition2 = ts_max(df, delta_close, 5) < 0
    return np.where(condition1, delta_close, np.where(condition2, delta_close, -1 * delta_close))

def alpha10(df):
    delta_close = ts_delta(df, 'Close', 1)
    condition1 = ts_min(df, delta_close, 4) > 0
    condition2 = ts_max(df, delta_close, 4) < 0
    inner = np.where(condition1, delta_close, np.where(condition2, delta_close, -1 * delta_close))
    return cs_rank(df, inner)

def alpha11(df):
    vwap_close = df['vwap'] - df['Close']
    max_vwap_close = ts_max(df, vwap_close, 3)
    min_vwap_close = ts_min(df, vwap_close, 3)
    rank_max = cs_rank(df, max_vwap_close)
    rank_min = cs_rank(df, min_vwap_close)
    delta_vol = ts_delta(df, 'Volume', 3)
    rank_delta_vol = cs_rank(df, delta_vol)
    return (rank_max + rank_min) * rank_delta_vol

def alpha12(df):
    delta_vol = ts_delta(df, 'Volume', 1)
    sign_delta_vol = np.sign(delta_vol)
    delta_close = ts_delta(df, 'Close', 1)
    return sign_delta_vol * (-1 * delta_close)

def alpha13(df):
    rank_close = cs_rank(df, 'Close')
    rank_vol = cs_rank(df, 'Volume')
    cov = ts_covariance(df, rank_close, rank_vol, 5)
    ranked_cov = cs_rank(df, cov)
    return -1 * ranked_cov

def alpha14(df):
    delta_ret = ts_delta(df, 'returns', 3)
    rank_delta_ret = cs_rank(df, delta_ret)
    corr = ts_correlation(df, 'Open', 'Volume', 10)
    return -1 * cs_rank(df, rank_delta_ret) * corr

def alpha15(df):
    rank_high = cs_rank(df, 'High')
    rank_vol = cs_rank(df, 'Volume')
    corr = ts_correlation(df, rank_high, rank_vol, 3)
    sum_corr = ts_sum(df, corr, 3)
    return -1 * sum_corr

def alpha16(df):
    rank_high = cs_rank(df, 'High')
    rank_vol = cs_rank(df, 'Volume')
    cov = ts_covariance(df, rank_high, rank_vol, 5)
    ranked_cov = cs_rank(df, cov)
    return -1 * ranked_cov

def alpha17(df):
    ts_rank_close = ts_rank(df, 'Close', 10)
    rank_ts_close = cs_rank(df, ts_rank_close)
    delta_close = ts_delta(df, 'Close', 1)
    delta_delta_close = ts_delta(df, delta_close, 1)
    rank_delta_delta = cs_rank(df, delta_delta_close)
    vol_adv = df['Volume'] / adv(df, 20)
    ts_rank_vol_adv = ts_rank(df, vol_adv, 5)
    rank_ts_vol_adv = cs_rank(df, ts_rank_vol_adv)
    return -1 * rank_ts_close * rank_delta_delta * rank_ts_vol_adv

def alpha18(df):
    abs_close_open = np.abs(df['Close'] - df['Open'])
    std_abs_close_open = stddev(df, abs_close_open, 5)
    corr_close_open = ts_correlation(df, 'Close', 'Open', 10)
    inner = std_abs_close_open + (df['Close'] - df['Open']) + corr_close_open
    ranked_inner = cs_rank(df, inner)
    return -1 * ranked_inner

def alpha19(df):
    delta_close7 = ts_delta(df, 'Close', 7)
    sign_delta7 = np.sign(delta_close7 + delta_close7)
    sign_total = -1 * np.sign(delta_close7 + sign_delta7)
    sum_ret250 = ts_sum(df, 'returns', 250)
    rank_sum1 = cs_rank(df, 1 + sum_ret250)
    return sign_total * (1 + rank_sum1)

def alpha20(df):
    delay_high1 = df.groupby('Ticker')['High'].shift(1)
    delay_close1 = df.groupby('Ticker')['Close'].shift(1)
    delay_low1 = df.groupby('Ticker')['Low'].shift(1)
    rank_open_delay_high = cs_rank(df, df['Open'] - delay_high1)
    rank_open_delay_close = cs_rank(df, df['Open'] - delay_close1)
    rank_open_delay_low = cs_rank(df, df['Open'] - delay_low1)
    return -1 * rank_open_delay_high * rank_open_delay_close * rank_open_delay_low

def alpha21(df):
    sum_close8 = ts_sum(df, 'Close', 8)
    sum_close8_8 = sum_close8 / 8
    std_close8 = stddev(df, 'Close', 8)
    sum_close2 = ts_sum(df, 'Close', 2)
    sum_close2_2 = sum_close2 / 2
    adv20 = adv(df, 20)
    condition1 = (sum_close8_8 + std_close8) < sum_close2_2
    condition2 = sum_close2_2 < (sum_close8_8 - std_close8)
    condition3 = (df['Volume'] / adv20) >= 1
    return np.where(condition1, -1, np.where(condition2, 1, np.where(condition3, 1, -1)))

def alpha22(df):
    corr_high_vol = ts_correlation(df, 'High', 'Volume', 5)
    delta_corr = ts_delta(df, corr_high_vol, 5)
    std_close20 = stddev(df, 'Close', 20)
    rank_std_close20 = cs_rank(df, std_close20)
    return -1 * delta_corr * rank_std_close20

def alpha23(df):
    sum_high20 = ts_sum(df, 'High', 20)
    sum_high20_20 = sum_high20 / 20
    # sum_high20_20 = pd.Series(sum_high20 / 20, index=df.index)
    condition = sum_high20_20 < df['High']
    delta_high2 = ts_delta(df, 'High', 2)
    return np.where(condition, -1 * delta_high2, 0)

def alpha24(df):
    sum_close100 = ts_sum(df, 'Close', 100)
    sum_close100_100 = sum_close100 / 100
    delta_sum = ts_delta(df, sum_close100_100, 100)
    delay_close100 = df.groupby('Ticker')['Close'].shift(100)
    delta_delay = delta_sum / delay_close100
    condition = delta_delay <= 0.05
    ts_min_close100 = ts_min(df, 'Close', 100)
    delta_close3 = ts_delta(df, 'Close', 3)
    return np.where(condition, -1 * (df['Close'] - ts_min_close100), -1 * delta_close3)

def alpha25(df):
    inner = -1 * df['returns'] * adv(df, 20) * df['vwap'] * (df['High'] - df['Close'])
    return cs_rank(df, inner)

def alpha26(df):
    tsrank_vol5 = ts_rank(df, 'Volume', 5)
    tsrank_high5 = ts_rank(df, 'High', 5)
    corr = ts_correlation(df, tsrank_vol5, tsrank_high5, 5)
    ts_max_corr = ts_max(df, corr, 3)
    return -1 * ts_max_corr

def alpha27(df):
    rank_vol = cs_rank(df, 'Volume')
    rank_vwap = cs_rank(df, 'vwap')
    corr = ts_correlation(df, rank_vol, rank_vwap, 6)
    sum_corr2 = ts_sum(df, corr, 2) / 2.0
    rank_sum = cs_rank(df, sum_corr2)
    condition = 0.5 < rank_sum
    return np.where(condition, -1, 1)

def alpha28(df):
    corr_adv_low = ts_correlation(df, adv(df, 20), 'Low', 5)
    hl2 = (df['High'] + df['Low']) / 2
    inner = corr_adv_low + hl2 - df['Close']
    return cs_scale(df, inner)

def alpha29(df):
    close_minus1 = df['Close'] - 1
    delta_close5 = ts_delta(df, close_minus1, 5)
    rank_delta = cs_rank(df, delta_close5)
    rank_rank = cs_rank(df, rank_delta)
    scale_log = np.log(ts_sum(df, ts_min(df, rank_rank, 2), 1))
    rank_scale_log = cs_rank(df, scale_log)
    min_product = np.minimum(ts_product(df, rank_scale_log, 1), 5)
    delay_returns6 = df.groupby('Ticker')['returns'].shift(6)
    ts_rank_delay = ts_rank(df, delay_returns6, 5)
    return min_product + ts_rank_delay

def alpha30(df):
    delay_close1 = df.groupby('Ticker')['Close'].shift(1)
    delay_close2 = df.groupby('Ticker')['Close'].shift(2)
    delay_close3 = df.groupby('Ticker')['Close'].shift(3)
    sign_close_delay1 = np.sign(df['Close'] - delay_close1)
    sign_delay1_delay2 = np.sign(delay_close1 - delay_close2)
    sign_delay2_delay3 = np.sign(delay_close2 - delay_close3)
    sum_sign = sign_close_delay1 + sign_delay1_delay2 + sign_delay2_delay3
    rank_sum_sign = cs_rank(df, sum_sign)
    sum_vol5 = ts_sum(df, 'Volume', 5)
    sum_vol20 = ts_sum(df, 'Volume', 20)
    inner = (1.0 - rank_sum_sign) * sum_vol5 / sum_vol20
    return inner

def alpha31(df):
    delta_close3 = ts_delta(df, 'Close', 3)
    delta_delta_close = ts_delta(df, delta_close3, 1)
    rank_delta_delta = cs_rank(df, delta_delta_close)
    rank_rank_delta = cs_rank(df, rank_delta_delta)
    decay_delta10 = decay_linear(df, -1 * rank_rank_delta, 10)
    rank_decay = cs_rank(df, decay_delta10)
    rank_rank_rank = cs_rank(df, rank_decay)
    rank_delta_close3 = cs_rank(df, -1 * delta_close3)
    corr_adv_low = ts_correlation(df, adv(df, 20), 'Low', 12)
    scale_corr = cs_scale(df, corr_adv_low)
    sign_scale = np.sign(scale_corr)
    return rank_rank_rank + rank_delta_close3 + sign_scale

def alpha32(df):
    sum_close7 = ts_sum(df, 'Close', 7)
    sum_close7_7 = sum_close7 / 7
    inner1 = sum_close7_7 - df['Close']
    scale_inner1 = cs_scale(df, inner1)
    delay_close5 = df.groupby('Ticker')['Close'].shift(5)
    corr_vwap_delay = ts_correlation(df, 'vwap', delay_close5, 230)
    scale_corr = cs_scale(df, corr_vwap_delay)
    inner2 = 20 * scale_corr
    return scale_inner1 + inner2

def alpha33(df):
    inner = 1 - (df['Open'] / df['Close'])
    power = signed_power(df, inner, 1)
    return cs_rank(df, -1 * power)

def alpha34(df):
    std_ret2 = stddev(df, 'returns', 2)
    std_ret5 = stddev(df, 'returns', 5)
    ratio_std = std_ret2 / std_ret5
    rank_ratio = cs_rank(df, ratio_std)
    delta_close1 = ts_delta(df, 'Close', 1)
    rank_delta_close1 = cs_rank(df, delta_close1)
    inner = (1 - rank_ratio) + (1 - rank_delta_close1)
    return cs_rank(df, inner)

def alpha35(df):
    ts_rank_vol32 = ts_rank(df, 'Volume', 32)
    hl_low = (df['Close'] + df['High']) - df['Low']
    ts_rank_hl_low = ts_rank(df, hl_low, 16)
    ts_rank_ret32 = ts_rank(df, 'returns', 32)
    inner1 = ts_rank_vol32 * (1 - ts_rank_hl_low)
    inner2 = 1 - ts_rank_ret32
    return inner1 * inner2

def alpha36(df):
    delay_vol1 = df.groupby('Ticker')['Volume'].shift(1)
    close_open = df['Close'] - df['Open']
    corr_close_open_delay_vol = ts_correlation(df, close_open, delay_vol1, 15)
    rank_corr = cs_rank(df, corr_close_open_delay_vol)
    part1 = 2.21 * rank_corr
    open_close = df['Open'] - df['Close']
    rank_open_close = cs_rank(df, open_close)
    part2 = 0.7 * rank_open_close
    delay_returns6 = df.groupby('Ticker')['returns'].shift(6)
    ts_rank_delay = ts_rank(df, -1 * delay_returns6, 5)
    rank_ts = cs_rank(df, ts_rank_delay)
    part3 = 0.73 * rank_ts
    corr_vwap_adv = ts_correlation(df, 'vwap', adv(df, 20), 6)
    abs_corr = np.abs(corr_vwap_adv)
    rank_abs = cs_rank(df, abs_corr)
    part4 = rank_abs
    sum_close200 = ts_sum(df, 'Close', 200)
    sum_close200_200 = sum_close200 / 200
    inner_sum_open = (sum_close200_200 - df['Open']) * close_open
    rank_inner = cs_rank(df, inner_sum_open)
    part5 = 0.6 * rank_inner
    return part1 + part2 + part3 + part4 + part5

def alpha37(df):
    open_close = df['Open'] - df['Close']
    # Create delayed version using groupby shift
    delay_open_close1 = df.groupby('Ticker').apply(lambda g: open_close.loc[g.index].shift(1)).reset_index(level=0, drop=True).reindex(df.index)
    corr_delay_close = ts_correlation(df, delay_open_close1, 'Close', 200)
    rank_corr = cs_rank(df, corr_delay_close)
    rank_open_close = cs_rank(df, open_close)
    return rank_corr + rank_open_close

def alpha38(df):
    ts_rank_close10 = ts_rank(df, 'Close', 10)
    rank_ts = cs_rank(df, ts_rank_close10)
    close_open_ratio = df['Close'] / df['Open']
    rank_close_open = cs_rank(df, close_open_ratio)
    return -1 * rank_ts * rank_close_open

def alpha39(df):
    adv20 = adv(df, 20)
    vol_adv = df['Volume'] / adv20
    decay_vol_adv9 = decay_linear(df, vol_adv, 9)
    rank_decay = cs_rank(df, decay_vol_adv9)
    delta_close7 = ts_delta(df, 'Close', 7)
    inner = delta_close7 * (1 - rank_decay)
    rank_inner = cs_rank(df, inner)
    sum_ret250 = ts_sum(df, 'returns', 250)
    rank_sum = cs_rank(df, sum_ret250)
    return -1 * rank_inner * (1 + rank_sum)

def alpha40(df):
    std_high10 = stddev(df, 'High', 10)
    rank_std = cs_rank(df, std_high10)
    corr_high_vol = ts_correlation(df, 'High', 'Volume', 10)
    return -1 * rank_std * corr_high_vol

def alpha41(df):
    hl_pow = signed_power(df, df['High'] * df['Low'], 0.5)
    return hl_pow - df['vwap']

def alpha42(df):
    vwap_close = df['vwap'] - df['Close']
    rank_vwap_close = cs_rank(df, vwap_close)
    vwap_close_pos = df['vwap'] + df['Close']
    rank_vwap_close_pos = cs_rank(df, vwap_close_pos)
    return rank_vwap_close / rank_vwap_close_pos

def alpha43(df):
    adv20 = adv(df, 20)
    vol_adv = df['Volume'] / adv20
    ts_rank_vol_adv = ts_rank(df, vol_adv, 20)
    delta_close7 = ts_delta(df, 'Close', 7)
    ts_rank_delta = ts_rank(df, -1 * delta_close7, 8)
    return ts_rank_vol_adv * ts_rank_delta

def alpha44(df):
    rank_vol = cs_rank(df, 'Volume')
    corr_high_rank_vol = ts_correlation(df, 'High', rank_vol, 5)
    return -1 * corr_high_rank_vol

def alpha45(df):
    delay_close5 = df.groupby('Ticker')['Close'].shift(5)
    sum_delay20 = ts_sum(df, delay_close5, 20) / 20
    rank_sum = cs_rank(df, sum_delay20)
    corr_close_vol = ts_correlation(df, 'Close', 'Volume', 2)
    sum_close5 = ts_sum(df, 'Close', 5)
    sum_close20 = ts_sum(df, 'Close', 20)
    corr_sum = ts_correlation(df, sum_close5, sum_close20, 2)
    rank_corr_sum = cs_rank(df, corr_sum)
    inner = rank_sum * corr_close_vol * rank_corr_sum
    return -1 * inner

def alpha46(df):
    delay_close20 = df.groupby('Ticker')['Close'].shift(20)
    delay_close10 = df.groupby('Ticker')['Close'].shift(10)
    part1 = (delay_close20 - delay_close10) / 10
    part2 = (delay_close10 - df['Close']) / 10
    diff = part1 - part2
    condition1 = 0.25 < diff
    condition2 = diff < 0
    delta_close1 = ts_delta(df, 'Close', 1)
    return np.where(condition1, -1, np.where(condition2, 1, -1 * delta_close1))

def alpha47(df):
    adv20 = adv(df, 20)
    close_inv = 1 / df['Close']
    rank_close_inv = cs_rank(df, close_inv)
    vol_adv = rank_close_inv * df['Volume'] / adv20
    high_close = df['High'] - df['Close']
    rank_high_close = cs_rank(df, high_close)
    sum_high5 = ts_sum(df, 'High', 5)
    sum_high5_5 = sum_high5 / 5
    inner = df['High'] * rank_high_close / sum_high5_5
    delay_vwap5 = df.groupby('Ticker')['vwap'].shift(5)
    rank_vwap_delay = cs_rank(df, df['vwap'] - delay_vwap5)
    return vol_adv * inner - rank_vwap_delay

def alpha48(df):
    delta_close1 = ts_delta(df, 'Close', 1)
    delay_close1 = df.groupby('Ticker')['Close'].shift(1)
    delta_delay1 = ts_delta(df, delay_close1, 1)
    corr_delta = ts_correlation(df, delta_close1, delta_delay1, 250)
    inner = corr_delta * (delta_close1 / df['Close'])
    neutralized = indneutralize(df, inner)
    delay_close1_sq = (delta_close1 / delay_close1) ** 2
    sum_sq = ts_sum(df, delay_close1_sq, 250)
    return neutralized / sum_sq

def alpha49(df):
    delay_close20 = df.groupby('Ticker')['Close'].shift(20)
    delay_close10 = df.groupby('Ticker')['Close'].shift(10)
    part1 = (delay_close20 - delay_close10) / 10
    part2 = (delay_close10 - df['Close']) / 10
    diff = part1 - part2
    condition = diff < -0.1
    delta_close1 = ts_delta(df, 'Close', 1)
    return np.where(condition, 1, -1 * delta_close1)

def alpha50(df):
    rank_vol = cs_rank(df, 'Volume')
    rank_vwap = cs_rank(df, 'vwap')
    corr_rank = ts_correlation(df, rank_vol, rank_vwap, 5)
    rank_corr = cs_rank(df, corr_rank)
    ts_max_rank = ts_max(df, rank_corr, 5)
    return -1 * ts_max_rank

def alpha51(df):
    delay_close20 = df.groupby('Ticker')['Close'].shift(20)
    delay_close10 = df.groupby('Ticker')['Close'].shift(10)
    part1 = (delay_close20 - delay_close10) / 10
    part2 = (delay_close10 - df['Close']) / 10
    diff = part1 - part2
    condition = diff < -0.05
    delta_close1 = ts_delta(df, 'Close', 1)
    return np.where(condition, 1, -1 * delta_close1)

def alpha52(df):
    ts_min_low5 = ts_min(df, 'Low', 5)
    # Create delayed version using groupby shift with proper Series handling
    delay_ts_min5 = df.groupby('Ticker').apply(lambda g: ts_min_low5.loc[g.index].shift(5)).reset_index(level=0, drop=True).reindex(df.index)
    inner1 = -1 * ts_min_low5 + delay_ts_min5
    sum_ret240 = ts_sum(df, 'returns', 240)
    sum_ret20 = ts_sum(df, 'returns', 20)
    diff_sum = sum_ret240 - sum_ret20
    rank_diff = cs_rank(df, diff_sum / 220)
    ts_rank_vol5 = ts_rank(df, 'Volume', 5)
    return inner1 * rank_diff * ts_rank_vol5

def alpha53(df):
    hl_close = (df['Close'] - df['Low']) - (df['High'] - df['Close'])
    denom = df['Close'] - df['Low']
    inner = hl_close / denom
    return -1 * ts_delta(df, inner, 9)

def alpha54(df):
    low_close = df['Low'] - df['Close']
    open_pow5 = df['Open'] ** 5
    num = -1 * low_close * open_pow5
    low_high = df['Low'] - df['High']
    close_pow5 = df['Close'] ** 5
    denom = low_high * close_pow5
    return num / denom

def alpha55(df):
    ts_min_low12 = ts_min(df, 'Low', 12)
    ts_max_high12 = ts_max(df, 'High', 12)
    num = df['Close'] - ts_min_low12
    denom = ts_max_high12 - ts_min_low12
    ratio = num / denom
    rank_ratio = cs_rank(df, ratio)
    rank_vol = cs_rank(df, 'Volume')
    corr = ts_correlation(df, rank_ratio, rank_vol, 6)
    return -1 * corr

def alpha56(df):
    sum_ret10 = ts_sum(df, 'returns', 10)
    sum_ret2 = ts_sum(df, 'returns', 2)
    sum_sum_ret2_3 = ts_sum(df, sum_ret2, 3) / 3
    inner = sum_ret10 / sum_sum_ret2_3
    rank_inner = cs_rank(df, inner)
    ret_cap = df['returns'] * df['cap']
    rank_ret_cap = cs_rank(df, ret_cap)
    return 0 - (1 * rank_inner * rank_ret_cap)

def alpha57(df):
    close_vwap = df['Close'] - df['vwap']
    ts_argmax_close30 = ts_argmax(df, 'Close', 30)
    rank_ts_argmax_close30 = ts_rank(df, ts_argmax_close30, 2)
    decay_rank = decay_linear(df, rank_ts_argmax_close30, 2)
    inner = close_vwap / decay_rank
    return 0 - (1 * inner)

def alpha58(df):
    corr_vwap_vol = ts_correlation(df, 'vwap', 'Volume', int(3.92795))  # Convert to integer
    decay_corr = decay_linear(df, corr_vwap_vol, int(7.89291))  # Convert to integer
    rank_decay = cs_rank(df, decay_corr)
    neutralized_vwap = indneutralize(df, 'vwap')
    ts_rank_neut = ts_rank(df, rank_decay, int(5.50322))  # Convert to integer
    return -1 * ts_rank_neut

def alpha59(df):
    open_weight = df['Open'] * (1 - 0.318108)
    vwap_weight = df['vwap'] * 0.318108
    weighted = vwap_weight + open_weight
    corr_weighted_adv = ts_correlation(df, weighted, 'adv180', int(13.557))
    decay_corr = decay_linear(df, corr_weighted_adv, int(12.2883))
    rank_decay = cs_rank(df, decay_corr)
    neutralized_close = indneutralize(df, 'Close')
    delta_neut = ts_delta(df, neutralized_close, int(2.25164))
    decay_delta = decay_linear(df, delta_neut, int(8.22237))
    rank_decay_delta = cs_rank(df, decay_delta)
    inner = rank_decay_delta - rank_decay
    return inner * -1

def alpha60(df):
    hl_vol = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])) * df['Volume']
    rank_hl_vol = cs_rank(df, hl_vol)
    scale_rank = cs_scale(df, rank_hl_vol)
    inner1 = 2 * scale_rank
    ts_argmax_close10 = ts_argmax(df, 'Close', 10)
    rank_ts = cs_rank(df, ts_argmax_close10)
    scale_rank_ts = cs_scale(df, rank_ts)
    inner2 = inner1 - scale_rank_ts
    return 0 - (1 * inner2)

def alpha61(df):
    ts_min_vwap = ts_min(df, 'vwap', int(16.1219))
    vwap_min = df['vwap'] - ts_min_vwap
    rank_vwap_min = cs_rank(df, vwap_min)
    corr_vwap_adv = ts_correlation(df, 'vwap', 'adv180', int(17.9282))
    rank_corr = cs_rank(df, corr_vwap_adv)
    return rank_vwap_min < rank_corr

def alpha62(df):
    adv20 = adv(df, 20)
    sum_adv20 = ts_sum(df, adv20, int(22.4101))
    corr_vwap_sum = ts_correlation(df, 'vwap', sum_adv20, int(9.91009))
    rank_corr = cs_rank(df, corr_vwap_sum)
    rank_open1 = cs_rank(df, 'Open')
    rank_open2 = cs_rank(df, 'Open')
    sum_open = rank_open1 + rank_open2
    hl2 = (df['High'] + df['Low']) / 2
    rank_hl2 = cs_rank(df, hl2)
    rank_high = cs_rank(df, 'High')
    sum_right = rank_hl2 + rank_high
    condition = sum_open < sum_right
    rank_condition = cs_rank(df, condition)
    return rank_corr < rank_condition * -1

def alpha63(df):
    neutralized_close = indneutralize(df, 'Close')
    delta_neut = ts_delta(df, neutralized_close, int(2.25164))
    decay_delta = decay_linear(df, delta_neut, int(8.22237))
    rank_decay_delta = cs_rank(df, decay_delta)
    open_weight = df['Open'] * (1 - 0.318108)
    vwap_weight = df['vwap'] * 0.318108
    weighted = vwap_weight + open_weight
    sum_adv180 = ts_sum(df, 'adv180', int(37.2467))
    corr_weighted_sum = ts_correlation(df, weighted, sum_adv180, int(13.557))
    decay_corr = decay_linear(df, corr_weighted_sum, int(12.2883))
    rank_decay_corr = cs_rank(df, decay_corr)
    inner = rank_decay_delta - rank_decay_corr
    return inner * -1

def alpha64(df):
    open_weight = df['Open'] * (1 - 0.178404)
    low_weight = df['Low'] * 0.178404
    weighted = low_weight + open_weight
    sum_weighted = ts_sum(df, weighted, int(12.7054))
    sum_adv120 = ts_sum(df, 'adv120', int(12.7054))
    corr_sum = ts_correlation(df, sum_weighted, sum_adv120, int(16.6208))
    rank_corr = cs_rank(df, corr_sum)
    hl2 = (df['High'] + df['Low']) / 2
    hl2_weight = hl2 * 0.178404
    vwap_weight = df['vwap'] * (1 - 0.178404)
    weighted2 = hl2_weight + vwap_weight
    delta_weighted = ts_delta(df, weighted2, int(3.69741))
    rank_delta = cs_rank(df, delta_weighted)
    inner = rank_corr < rank_delta
    return inner * -1

def alpha65(df):
    open_weight = df['Open'] * (1 - 0.00817205)
    vwap_weight = df['vwap'] * 0.00817205
    weighted = vwap_weight + open_weight
    sum_adv60 = ts_sum(df, 'adv60', int(8.6911))
    corr_weighted_sum = ts_correlation(df, weighted, sum_adv60, int(6.40374))
    rank_corr = cs_rank(df, corr_weighted_sum)
    ts_min_open = ts_min(df, 'Open', int(13.635))
    open_min = df['Open'] - ts_min_open
    rank_open_min = cs_rank(df, open_min)
    inner = rank_corr < rank_open_min
    return inner * -1

def alpha66(df):
    delta_vwap = ts_delta(df, 'vwap', int(3.51013))
    decay_delta = decay_linear(df, delta_vwap, int(7.23052))
    rank_decay = cs_rank(df, decay_delta)
    low_weight = df['Low'] * 0.96633
    low_weight2 = df['Low'] * (1 - 0.96633)
    weighted_low = low_weight + low_weight2
    weighted_vwap = weighted_low - df['vwap']
    open_hl2 = df['Open'] - ((df['High'] + df['Low']) / 2)
    denom = open_hl2
    inner = weighted_vwap / denom
    corr_inner = ts_correlation(df, 'Low', 'adv81', int(19.569))
    decay_corr = decay_linear(df, corr_inner, int(17.1543))
    ts_rank_decay = ts_rank(df, decay_corr, int(6.72611))
    inner2 = rank_decay + ts_rank_decay
    return inner2 * -1

def alpha67(df):
    ts_min_high = ts_min(df, 'High', int(2.14593))
    high_min = df['High'] - ts_min_high
    rank_high_min = cs_rank(df, high_min)
    neutralized_vwap = indneutralize(df, 'vwap')
    corr_vwap_adv = ts_correlation(df, neutralized_vwap, 'adv20', int(6.02936))
    rank_corr = cs_rank(df, corr_vwap_adv)
    power = signed_power(df, rank_high_min, rank_corr)
    return power * -1

def alpha68(df):
    rank_high = cs_rank(df, 'High')
    rank_adv15 = cs_rank(df, 'adv15')
    corr_rank = ts_correlation(df, rank_high, rank_adv15, int(8.91644))
    ts_rank_corr = ts_rank(df, corr_rank, int(13.9333))
    close_weight = df['Close'] * (1 - 0.518371)
    low_weight = df['Low'] * 0.518371
    weighted = low_weight + close_weight
    delta_weighted = ts_delta(df, weighted, int(1.06157))
    rank_delta = cs_rank(df, delta_weighted)
    inner = ts_rank_corr < rank_delta
    return inner * -1

def alpha69(df):
    neutralized_vwap = indneutralize(df, 'vwap')
    delta_neut = ts_delta(df, neutralized_vwap, int(2.72412))
    ts_max_delta = ts_max(df, delta_neut, int(4.79344))
    rank_ts_max = cs_rank(df, ts_max_delta)
    close_weight = df['Close'] * (1 - 0.490655)
    vwap_weight = df['vwap'] * 0.490655
    weighted = vwap_weight + close_weight
    corr_weighted_adv = ts_correlation(df, weighted, 'adv20', int(4.92416))
    ts_rank_corr = ts_rank(df, corr_weighted_adv, int(9.0615))
    power = signed_power(df, rank_ts_max, ts_rank_corr)
    return power * -1

def alpha70(df):
    delta_vwap = ts_delta(df, 'vwap', int(1.29456))
    rank_delta = cs_rank(df, delta_vwap)
    neutralized_close = indneutralize(df, 'Close')
    corr_close_adv = ts_correlation(df, neutralized_close, 'adv50', int(17.8256))
    ts_rank_corr = ts_rank(df, corr_close_adv, int(17.9171))
    power = signed_power(df, rank_delta, ts_rank_corr)
    return power * -1

def alpha71(df):
    ts_rank_close = ts_rank(df, 'Close', int(3.43976))
    ts_rank_adv180 = ts_rank(df, 'adv180', int(12.0647))
    corr_ts = ts_correlation(df, ts_rank_close, ts_rank_adv180, int(18.0175))
    decay_corr = decay_linear(df, corr_ts, int(4.20501))
    ts_rank_decay1 = ts_rank(df, decay_corr, int(15.6948))
    hl_close = (df['Low'] + df['Open']) - (df['vwap'] + df['vwap'])
    power_hl = signed_power(df, hl_close, 2)
    rank_power = cs_rank(df, power_hl)
    decay_rank = decay_linear(df, rank_power, int(16.4662))
    ts_rank_decay2 = ts_rank(df, decay_rank, int(4.4388))
    return np.maximum(ts_rank_decay1, ts_rank_decay2)

def alpha72(df):
    hl2 = (df['High'] + df['Low']) / 2
    corr_hl_adv = ts_correlation(df, hl2, 'adv40', int(8.93345))
    decay_corr = decay_linear(df, corr_hl_adv, int(10.1519))
    rank_decay = cs_rank(df, decay_corr)
    ts_rank_vwap = ts_rank(df, 'vwap', int(3.72469))
    ts_rank_vol = ts_rank(df, 'Volume', int(18.5188))
    corr_ts = ts_correlation(df, ts_rank_vwap, ts_rank_vol, int(6.86671))
    decay_corr2 = decay_linear(df, corr_ts, int(2.95011))
    rank_decay2 = cs_rank(df, decay_corr2)
    return rank_decay / rank_decay2

def alpha73(df):
    delta_vwap = ts_delta(df, 'vwap', int(4.72775))
    decay_delta = decay_linear(df, delta_vwap, int(2.91864))
    rank_decay = cs_rank(df, decay_delta)
    open_weight = df['Open'] * (1 - 0.147155)
    low_weight = df['Low'] * 0.147155
    weighted = low_weight + open_weight
    denom = weighted
    delta_weighted = ts_delta(df, weighted, int(2.03608)) / denom * -1
    decay_delta_weight = decay_linear(df, delta_weighted, int(3.33829))
    ts_rank_decay = ts_rank(df, decay_delta_weight, int(16.7411))
    inner = np.maximum(rank_decay, ts_rank_decay)
    return inner * -1

def alpha74(df):
    sum_adv30 = ts_sum(df, 'adv30', int(37.4843))
    corr_close_sum = ts_correlation(df, 'Close', sum_adv30, int(15.1365))
    rank_corr = cs_rank(df, corr_close_sum)
    high_weight = df['High'] * 0.0261661
    vwap_weight = df['vwap'] * (1 - 0.0261661)
    weighted = high_weight + vwap_weight
    rank_weighted = cs_rank(df, weighted)
    rank_vol = cs_rank(df, 'Volume')
    corr_rank = ts_correlation(df, rank_weighted, rank_vol, int(11.4791))
    rank_corr2 = cs_rank(df, corr_rank)
    inner = rank_corr < rank_corr2
    return inner * -1

def alpha75(df):
    corr_vwap_vol = ts_correlation(df, 'vwap', 'Volume', int(4.24304))
    rank_corr = cs_rank(df, corr_vwap_vol)
    rank_low = cs_rank(df, 'Low')
    rank_adv50 = cs_rank(df, 'adv50')
    corr_rank_low_adv = ts_correlation(df, rank_low, rank_adv50, int(12.4413))
    rank_corr2 = cs_rank(df, corr_rank_low_adv)
    return rank_corr < rank_corr2

def alpha76(df):
    delta_vwap = ts_delta(df, 'vwap', int(1.24383))
    decay_delta = decay_linear(df, delta_vwap, int(11.8259))
    rank_decay = cs_rank(df, decay_delta)
    neutralized_low = indneutralize(df, 'Low')
    corr_low_adv = ts_correlation(df, neutralized_low, 'adv81', int(8.14941))
    ts_rank_corr = ts_rank(df, corr_low_adv, int(19.569))
    decay_ts_rank = decay_linear(df, ts_rank_corr, int(17.1543))
    ts_rank_decay2 = ts_rank(df, decay_ts_rank, int(19.383))
    inner = np.maximum(rank_decay, ts_rank_decay2)
    return inner * -1

def alpha77(df):
    hl2 = (df['High'] + df['Low']) / 2
    hl2_high = (hl2 + df['High']) - (df['vwap'] + df['High'])
    decay_hl = decay_linear(df, hl2_high, int(20.0451))
    rank_decay = cs_rank(df, decay_hl)
    corr_hl_adv = ts_correlation(df, hl2, 'adv40', int(3.1614))
    decay_corr = decay_linear(df, corr_hl_adv, int(5.64125))
    rank_decay2 = cs_rank(df, decay_corr)
    return np.minimum(rank_decay, rank_decay2)

def alpha78(df):
    low_weight = df['Low'] * (1 - 0.352233)
    vwap_weight = df['vwap'] * 0.352233
    weighted = low_weight + vwap_weight
    sum_weighted = ts_sum(df, weighted, int(19.7428))
    sum_adv40 = ts_sum(df, 'adv40', int(19.7428))
    corr_sum = ts_correlation(df, sum_weighted, sum_adv40, int(6.83313))
    rank_corr = cs_rank(df, corr_sum)
    rank_vwap = cs_rank(df, 'vwap')
    rank_vol = cs_rank(df, 'Volume')
    corr_rank_vwap_vol = ts_correlation(df, rank_vwap, rank_vol, int(5.77492))
    rank_corr2 = cs_rank(df, corr_rank_vwap_vol)
    power = signed_power(df, rank_corr, rank_corr2)
    return power

def alpha79(df):
    close_weight = df['Close'] * (1 - 0.60733)
    open_weight = df['Open'] * 0.60733
    weighted = open_weight + close_weight
    neutralized_weighted = indneutralize(df, weighted)
    delta_neut = ts_delta(df, neutralized_weighted, int(1.23438))
    rank_delta = cs_rank(df, delta_neut)
    ts_rank_vwap = ts_rank(df, 'vwap', int(3.60973))
    ts_rank_adv150 = ts_rank(df, 'adv150', int(9.18637))
    corr_ts = ts_correlation(df, ts_rank_vwap, ts_rank_adv150, int(14.6644))
    rank_corr = cs_rank(df, corr_ts)
    inner = rank_delta < rank_corr
    return inner

def alpha80(df):
    open_weight = df['Open'] * (1 - 0.868128)
    high_weight = df['High'] * 0.868128
    weighted = high_weight + open_weight
    neutralized_weighted = indneutralize(df, weighted)
    delta_neut = ts_delta(df, neutralized_weighted, int(4.04545))
    sign_delta = np.sign(delta_neut)
    rank_sign = cs_rank(df, sign_delta)
    corr_high_adv = ts_correlation(df, 'High', 'adv10', int(5.11456))
    ts_rank_corr = ts_rank(df, corr_high_adv, int(5.53756))
    power = signed_power(df, rank_sign, ts_rank_corr)
    return power * -1

def alpha81(df):
    sum_adv10 = ts_sum(df, 'adv10', int(49.6054))
    corr_vwap_sum = ts_correlation(df, 'vwap', sum_adv10, int(8.47743))
    rank_corr = cs_rank(df, corr_vwap_sum)
    power4 = signed_power(df, rank_corr, 4)
    product_rank = ts_product(df, power4, int(14.9655))
    log_product = np.log(product_rank)
    rank_log = cs_rank(df, log_product)
    rank_vwap = cs_rank(df, 'vwap')
    rank_vol = cs_rank(df, 'Volume')
    corr_rank_vwap_vol = ts_correlation(df, rank_vwap, rank_vol, int(5.07914))
    rank_corr2 = cs_rank(df, corr_rank_vwap_vol)
    inner = rank_log < rank_corr2
    return inner * -1

def alpha82(df):
    delta_open = ts_delta(df, 'Open', int(1.46063))
    decay_open = decay_linear(df, delta_open, int(14.8717))
    rank_decay = cs_rank(df, decay_open)
    neutralized_vol = indneutralize(df, 'Volume')
    open_weight = df['Open'] * (1 - 0.634196)
    open_weight2 = df['Open'] * 0.634196
    weighted = open_weight + open_weight2
    corr_weighted_open = ts_correlation(df, neutralized_vol, weighted, int(17.4842))
    decay_corr = decay_linear(df, corr_weighted_open, int(6.92131))
    ts_rank_decay2 = ts_rank(df, decay_corr, int(13.4283))
    return np.minimum(rank_decay, ts_rank_decay2) * -1

def alpha83(df):
    sum_close5 = ts_sum(df, 'Close', 5)
    sum_close5_5 = sum_close5 / 5
    hl_denom = (df['High'] - df['Low']) / sum_close5_5
    # Create a temporary DataFrame with the series and ticker for proper groupby shift
    temp_df = df[['Ticker']].copy()
    temp_df['hl_denom'] = hl_denom
    delay_hl_denom2 = temp_df.groupby('Ticker')['hl_denom'].shift(2)
    rank_delay = cs_rank(df, delay_hl_denom2)
    rank_vol = cs_rank(df, 'Volume')
    num = rank_delay * rank_vol
    vwap_close = df['vwap'] - df['Close']
    denom = hl_denom / vwap_close
    return num / denom

def alpha84(df):
    ts_max_vwap = ts_max(df, 'vwap', int(15.3217))
    vwap_ts_max = df['vwap'] - ts_max_vwap
    ts_rank_vwap = ts_rank(df, vwap_ts_max, int(20.7127))
    delta_close = ts_delta(df, 'Close', int(4.96796))
    return signed_power(df, ts_rank_vwap, delta_close)

def alpha85(df):
    high_weight = df['High'] * (1 - 0.876703)
    close_weight = df['Close'] * 0.876703
    weighted = close_weight + high_weight
    corr_weighted_adv = ts_correlation(df, weighted, 'adv30', int(9.61331))
    rank_corr = cs_rank(df, corr_weighted_adv)
    hl2 = (df['High'] + df['Low']) / 2
    ts_rank_hl2 = ts_rank(df, hl2, int(3.70596))
    ts_rank_vol = ts_rank(df, 'Volume', int(10.1595))
    corr_ts = ts_correlation(df, ts_rank_hl2, ts_rank_vol, int(7.11408))
    rank_corr2 = cs_rank(df, corr_ts)
    power = signed_power(df, rank_corr, rank_corr2)
    return power

def alpha86(df):
    sum_adv20 = ts_sum(df, 'adv20', int(14.7444))
    corr_close_sum = ts_correlation(df, 'Close', sum_adv20, int(6.00049))
    ts_rank_corr = ts_rank(df, corr_close_sum, int(20.4195))
    open_close = df['Open'] + df['Close']
    vwap_open = df['vwap'] + df['Open']
    diff = open_close - vwap_open
    rank_diff = cs_rank(df, diff)
    inner = ts_rank_corr < rank_diff
    return inner * -1

def alpha87(df):
    close_weight = df['Close'] * (1 - 0.369701)
    vwap_weight = df['vwap'] * 0.369701
    weighted = vwap_weight + close_weight
    delta_weighted = ts_delta(df, weighted, int(1.91233))
    decay_delta = decay_linear(df, delta_weighted, int(2.65461))
    rank_decay = cs_rank(df, decay_delta)
    neutralized_adv81 = indneutralize(df, 'adv81')
    corr_adv_close = ts_correlation(df, neutralized_adv81, 'Close', int(13.4132))
    abs_corr = np.abs(corr_adv_close)
    decay_abs = decay_linear(df, abs_corr, int(4.89768))
    ts_rank_decay = ts_rank(df, decay_abs, int(14.4535))
    inner = np.maximum(rank_decay, ts_rank_decay)
    return inner * -1

def alpha88(df):
    rank_open = cs_rank(df, 'Open')
    rank_low = cs_rank(df, 'Low')
    sum_left = rank_open + rank_low
    rank_high = cs_rank(df, 'High')
    rank_close = cs_rank(df, 'Close')
    sum_right = rank_high + rank_close
    diff = sum_left - sum_right
    decay_diff = decay_linear(df, diff, int(8.06882))
    rank_decay = cs_rank(df, decay_diff)
    ts_rank_close = ts_rank(df, 'Close', int(8.44728))
    ts_rank_adv60 = ts_rank(df, 'adv60', int(20.6966))
    corr_ts = ts_correlation(df, ts_rank_close, ts_rank_adv60, int(8.01266))
    decay_corr = decay_linear(df, corr_ts, int(6.65053))
    ts_rank_decay2 = ts_rank(df, decay_corr, int(2.61957))
    return np.minimum(rank_decay, ts_rank_decay2)

def alpha89(df):
    low_weight = df['Low'] * (1 - 0.967285)
    low_weight2 = df['Low'] * 0.967285
    weighted = low_weight + low_weight2
    corr_weighted_adv = ts_correlation(df, weighted, 'adv10', int(6.94279))
    decay_corr = decay_linear(df, corr_weighted_adv, int(5.51607))
    ts_rank_decay = ts_rank(df, decay_corr, int(3.79744))
    neutralized_vwap = indneutralize(df, 'vwap')
    delta_neut = ts_delta(df, neutralized_vwap, int(3.48158))
    decay_delta = decay_linear(df, delta_neut, int(10.1466))
    ts_rank_decay2 = ts_rank(df, decay_delta, int(15.3012))
    return ts_rank_decay - ts_rank_decay2

def alpha90(df):
    ts_max_close = ts_max(df, 'Close', int(4.66719))
    close_ts_max = df['Close'] - ts_max_close
    rank_close_ts = cs_rank(df, close_ts_max)
    neutralized_adv40 = indneutralize(df, 'adv40')
    corr_adv_low = ts_correlation(df, neutralized_adv40, 'Low', int(5.38375))
    ts_rank_corr = ts_rank(df, corr_adv_low, int(3.21856))
    power = signed_power(df, rank_close_ts, ts_rank_corr)
    return power * -1

def alpha91(df):
    neutralized_close = indneutralize(df, 'Close')
    corr_neut_vol = ts_correlation(df, neutralized_close, 'Volume', int(9.74928))
    decay_corr = decay_linear(df, corr_neut_vol, int(16.398))
    decay_decay = decay_linear(df, decay_corr, int(3.83219))
    ts_rank_decay = ts_rank(df, decay_decay, int(4.8667))
    corr_vwap_adv = ts_correlation(df, 'vwap', 'adv30', int(4.01303))
    decay_corr2 = decay_linear(df, corr_vwap_adv, int(2.6809))
    rank_decay2 = cs_rank(df, decay_corr2)
    inner = ts_rank_decay - rank_decay2
    return inner * -1

def alpha92(df):
    hl2 = (df['High'] + df['Low']) / 2
    hl2_close = hl2 + df['Close']
    low_open = df['Low'] + df['Open']
    condition = hl2_close < low_open
    decay_hl = decay_linear(df, condition, int(14.7221))
    ts_rank_decay = ts_rank(df, decay_hl, int(18.8683))
    neutralized_vol = indneutralize(df, 'Volume')
    open_weight = df['Open'] * (1 - 0.634196)
    open_weight2 = df['Open'] * 0.634196
    weighted = open_weight + open_weight2
    corr_weighted = ts_correlation(df, neutralized_vol, weighted, int(17.4842))
    decay_corr = decay_linear(df, corr_weighted, int(6.92131))
    ts_rank_decay2 = ts_rank(df, decay_corr, int(13.4283))
    inner = np.minimum(ts_rank_decay, ts_rank_decay2)
    return inner * -1

def alpha93(df):
    neutralized_vwap = indneutralize(df, 'vwap')
    corr_neut_adv = ts_correlation(df, neutralized_vwap, 'adv81', int(17.4193))
    decay_corr = decay_linear(df, corr_neut_adv, int(19.848))
    ts_rank_decay = ts_rank(df, decay_corr, int(7.54455))
    close_weight = df['Close'] * (1 - 0.524434)
    vwap_weight = df['vwap'] * 0.524434
    weighted = vwap_weight + close_weight
    delta_weighted = ts_delta(df, weighted, int(2.77377))
    decay_delta = decay_linear(df, delta_weighted, int(16.2664))
    rank_decay = cs_rank(df, decay_delta)
    inner = ts_rank_decay / rank_decay
    return inner

def alpha94(df):
    ts_min_vwap = ts_min(df, 'vwap', int(11.5783))
    vwap_ts_min = df['vwap'] - ts_min_vwap
    rank_vwap_ts = cs_rank(df, vwap_ts_min)
    ts_rank_vwap = ts_rank(df, 'vwap', int(19.6462))
    ts_rank_adv60 = ts_rank(df, 'adv60', int(4.02992))
    corr_ts = ts_correlation(df, ts_rank_vwap, ts_rank_adv60, int(18.0926))
    ts_rank_corr = ts_rank(df, corr_ts, int(2.70756))
    power = signed_power(df, rank_vwap_ts, ts_rank_corr)
    return power * -1

def alpha95(df):
    ts_min_open = ts_min(df, 'Open', int(12.4105))
    open_ts_min = df['Open'] - ts_min_open
    rank_open_ts = cs_rank(df, open_ts_min)
    hl2 = (df['High'] + df['Low']) / 2
    sum_hl2 = ts_sum(df, hl2, int(19.1351))
    sum_adv40 = ts_sum(df, 'adv40', int(19.1351))
    corr_sum = ts_correlation(df, sum_hl2, sum_adv40, int(12.8742))
    power5 = signed_power(df, corr_sum, 5)
    rank_power = cs_rank(df, power5)
    ts_rank_rank = ts_rank(df, rank_power, int(11.7584))
    inner = rank_open_ts < ts_rank_rank
    return inner

def alpha96(df):
    corr_vwap_vol = ts_correlation(df, 'vwap', 'Volume', int(3.83878))
    decay_corr = decay_linear(df, corr_vwap_vol, int(4.16783))
    ts_rank_decay1 = ts_rank(df, decay_corr, int(8.38151))
    ts_rank_close = ts_rank(df, 'Close', int(7.45404))
    ts_rank_adv60 = ts_rank(df, 'adv60', int(4.13242))
    corr_ts = ts_correlation(df, ts_rank_close, ts_rank_adv60, int(3.65459))
    ts_argmax_corr = ts_argmax(df, corr_ts, int(12.6556))
    decay_argmax = decay_linear(df, ts_argmax_corr, int(14.0365))
    ts_rank_decay2 = ts_rank(df, decay_argmax, int(13.4143))
    inner = np.maximum(ts_rank_decay1, ts_rank_decay2)
    return inner * -1

def alpha97(df):
    low_weight = df['Low'] * (1 - 0.721001)
    vwap_weight = df['vwap'] * 0.721001
    weighted = vwap_weight + low_weight
    neutralized_weighted = indneutralize(df, weighted)
    delta_neut = ts_delta(df, neutralized_weighted, int(3.3705))
    decay_delta = decay_linear(df, delta_neut, int(20.4523))
    rank_decay = cs_rank(df, decay_delta)
    ts_rank_low = ts_rank(df, 'Low', int(7.87871))
    ts_rank_adv60 = ts_rank(df, 'adv60', int(17.255))
    corr_ts = ts_correlation(df, ts_rank_low, ts_rank_adv60, int(4.97547))
    ts_rank_corr = ts_rank(df, corr_ts, int(18.5925))
    decay_ts_rank = decay_linear(df, ts_rank_corr, int(15.7152))
    ts_rank_decay2 = ts_rank(df, decay_ts_rank, int(6.71659))
    inner = rank_decay - ts_rank_decay2
    return inner * -1

def alpha98(df):
    sum_adv5 = ts_sum(df, 'adv5', int(26.4719))
    corr_vwap_sum = ts_correlation(df, 'vwap', sum_adv5, int(4.58418))
    decay_corr = decay_linear(df, corr_vwap_sum, int(7.18088))
    rank_decay = cs_rank(df, decay_corr)
    rank_open = cs_rank(df, 'Open')
    rank_adv15 = cs_rank(df, 'adv15')
    corr_rank = ts_correlation(df, rank_open, rank_adv15, int(20.8187))
    ts_argmin_corr = ts_argmin(df, corr_rank, int(8.62571))
    ts_rank_argmin = ts_rank(df, ts_argmin_corr, int(6.95668))
    decay_ts_rank = decay_linear(df, ts_rank_argmin, int(8.07206))
    rank_decay2 = cs_rank(df, decay_ts_rank)
    inner = rank_decay - rank_decay2
    return inner

def alpha99(df):
    hl2 = (df['High'] + df['Low']) / 2
    sum_hl2 = ts_sum(df, hl2, int(19.8975))
    sum_adv60 = ts_sum(df, 'adv60', int(19.8975))
    corr_sum = ts_correlation(df, sum_hl2, sum_adv60, int(8.8136))
    rank_corr = cs_rank(df, corr_sum)
    corr_low_vol = ts_correlation(df, 'Low', 'Volume', int(6.28259))
    rank_corr2 = cs_rank(df, corr_low_vol)
    inner = rank_corr < rank_corr2
    return inner * -1

def alpha100(df):
    hl_vol = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])) * df['Volume']
    rank_hl_vol = cs_rank(df, hl_vol)
    neutralized_rank = indneutralize(df, rank_hl_vol)
    scale_neut = cs_scale(df, neutralized_rank, 1.5)
    rank_adv20 = cs_rank(df, adv(df, 20))
    corr_close_rank_adv = ts_correlation(df, 'Close', rank_adv20, 5)
    ts_argmin_close30 = ts_argmin(df, 'Close', 30)
    rank_ts_argmin_close = cs_rank(df, ts_argmin_close30)
    inner_corr = corr_close_rank_adv - rank_ts_argmin_close
    neut_inner = indneutralize(df, inner_corr)
    scale_neut2 = cs_scale(df, neut_inner)
    inner = scale_neut - scale_neut2
    vol_adv = df['Volume'] / adv(df, 20)
    mult = inner * vol_adv
    return 0 - (1 * mult)

def alpha101(df):
    hl = df['High'] - df['Low'] + 0.001
    close_open = df['Close'] - df['Open']
    return close_open / hl

def backtest_alphas(df):
    """
    Backtest alpha factors by converting them to trading signals and calculating PnL
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get alpha columns
    alpha_cols = [col for col in df.columns if col.startswith('alpha_')]
    
    # Calculate daily returns for each crypto
    df_sorted = df.sort_values(['Ticker', 'Date']).copy()
    
    # Ensure Date is properly formatted as datetime
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    
    # Calculate next day returns (forward-looking returns for backtesting)
    df_sorted['next_close'] = df_sorted.groupby('Ticker')['Close'].shift(-1)
    df_sorted['daily_return'] = (df_sorted['next_close'] - df_sorted['Close']) / df_sorted['Close']
    
    # Remove last day for each ticker (no forward return available)
    df_sorted = df_sorted.dropna(subset=['daily_return'])
    
    backtest_results = {}
    
    for alpha_col in alpha_cols:
        print(f"Backtesting {alpha_col}...")
        
        # Create trading signals from alpha values
        # Rank-based long/short strategy: top 20% long, bottom 20% short
        df_sorted[f'{alpha_col}_rank'] = df_sorted.groupby('Date')[alpha_col].rank(pct=True)
        
        # Create positions: 1 for long (top 20%), -1 for short (bottom 20%), 0 for neutral
        df_sorted[f'{alpha_col}_position'] = 0
        df_sorted.loc[df_sorted[f'{alpha_col}_rank'] >= 0.8, f'{alpha_col}_position'] = 1
        df_sorted.loc[df_sorted[f'{alpha_col}_rank'] <= 0.2, f'{alpha_col}_position'] = -1
        
        # Calculate strategy returns
        df_sorted[f'{alpha_col}_strategy_return'] = df_sorted[f'{alpha_col}_position'] * df_sorted['daily_return']
        
        # Calculate daily portfolio returns (average across all positions)
        daily_returns = df_sorted.groupby('Date')[f'{alpha_col}_strategy_return'].mean()
        
        # Ensure the index is datetime
        daily_returns.index = pd.to_datetime(daily_returns.index)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # Calculate performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
        max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
        
        backtest_results[alpha_col] = {
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        print(f"{alpha_col} - Total Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2%}")
    
    return backtest_results

def plot_cumulative_pnl(backtest_results):
    """
    Plot cumulative PnL for all alpha strategies with quarterly x-axis labels
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates
    from datetime import datetime
    
    plt.figure(figsize=(20, 12))
    
    # Plot top 10 performing alphas
    performance_summary = {}
    for alpha, results in backtest_results.items():
        performance_summary[alpha] = results['sharpe_ratio']
    
    # Sort by Sharpe ratio
    top_alphas = sorted(performance_summary.items(), key=lambda x: x[1], reverse=True)[:10]
    
    plt.subplot(2, 2, 1)
    for alpha, _ in top_alphas:
        cumulative_returns = backtest_results[alpha]['cumulative_returns']
        
        # Ensure dates are properly formatted
        dates = pd.to_datetime(cumulative_returns.index)
        
        plt.plot(dates, cumulative_returns.values, label=alpha, linewidth=1.5)
    
    plt.title('Top 10 Alpha Strategies - Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show quarters
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months (quarterly)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot all alphas (lighter lines)
    plt.subplot(2, 2, 2)
    for alpha, results in backtest_results.items():
        cumulative_returns = results['cumulative_returns']
        dates = pd.to_datetime(cumulative_returns.index)
        plt.plot(dates, cumulative_returns.values, alpha=0.3, linewidth=0.5)
    
    plt.title('All 101 Alpha Strategies - Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show quarters
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Performance distribution
    plt.subplot(2, 2, 3)
    sharpe_ratios = [results['sharpe_ratio'] for results in backtest_results.values()]
    plt.hist(sharpe_ratios, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Sharpe Ratios')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Number of Strategies')
    plt.grid(True, alpha=0.3)
    
    # Return vs Risk scatter
    plt.subplot(2, 2, 4)
    returns = [results['total_return'] for results in backtest_results.values()]
    volatilities = [results['volatility'] for results in backtest_results.values()]
    colors = [results['sharpe_ratio'] for results in backtest_results.values()]
    
    scatter = plt.scatter(volatilities, returns, c=colors, cmap='RdYlGn', alpha=0.7)
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.title('Risk-Return Profile of Alpha Strategies')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Total Return')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/alpha_backtest_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("BACKTESTING SUMMARY")
    print("="*80)
    print(f"Number of strategies: {len(backtest_results)}")
    print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.3f}")
    print(f"Best Sharpe Ratio: {max(sharpe_ratios):.3f}")
    print(f"Worst Sharpe Ratio: {min(sharpe_ratios):.3f}")
    print(f"Average Total Return: {np.mean(returns):.2%}")
    print(f"Best Total Return: {max(returns):.2%}")
    print(f"Worst Total Return: {min(returns):.2%}")
    
    print(f"\nTop 10 Alpha Strategies by Sharpe Ratio:")
    for i, (alpha, sharpe) in enumerate(top_alphas, 1):
        results = backtest_results[alpha]
        print(f"{i:2d}. {alpha:8s} - Sharpe: {sharpe:6.3f}, Return: {results['total_return']:7.2%}, Vol: {results['volatility']:6.2%}, MaxDD: {results['max_drawdown']:7.2%}")
    
    return backtest_results


def alphalens_analysis(df, backtest_results=None):
    """
    Simplified alpha factor analysis without Alphalens for now
    """
    import matplotlib.pyplot as plt
    print("Performing simplified alpha factor analysis...")
    
    # Get alpha columns
    alpha_cols = [col for col in df.columns if col.startswith('alpha_')]
    
    # Basic correlation analysis between alpha factors and forward returns
    df_analysis = df.copy()
    df_analysis['Date'] = pd.to_datetime(df_analysis['Date'])
    df_analysis = df_analysis.sort_values(['Ticker', 'Date'])
    
    # Calculate 1-day forward returns
    df_analysis['forward_return_1d'] = df_analysis.groupby('Ticker')['Close'].pct_change().shift(-1)
    
    # Calculate Information Coefficient (IC) for each alpha
    ic_results = {}
    
    print("Calculating Information Coefficients...")
    for alpha_col in alpha_cols:
        # Calculate correlation between alpha and forward returns
        valid_data = df_analysis[[alpha_col, 'forward_return_1d', 'Date']].dropna()
        
        if len(valid_data) > 10:  # Need sufficient data
            # Calculate daily IC (correlation by date)
            daily_ic = valid_data.groupby('Date').apply(
                lambda x: x[alpha_col].corr(x['forward_return_1d']) if len(x) > 1 else np.nan
            ).dropna()
            
            if len(daily_ic) > 0:
                mean_ic = daily_ic.mean()
                ic_std = daily_ic.std()
                ic_ir = mean_ic / ic_std if ic_std > 0 else 0
                
                ic_results[alpha_col] = {
                    'mean_ic': mean_ic,
                    'ic_std': ic_std, 
                    'ic_ir': ic_ir,
                    'daily_ic': daily_ic
                }
    
    # Sort by Information Ratio
    sorted_alphas = sorted(ic_results.items(), key=lambda x: abs(x[1]['ic_ir']), reverse=True)
    
    # Plot IC analysis
    if ic_results:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: IC time series for top 5 alphas
        top_5_alphas = sorted_alphas[:5]
        for alpha_name, results in top_5_alphas:
            daily_ic = results['daily_ic']
            dates = pd.to_datetime(daily_ic.index)
            axes[0,0].plot(dates, daily_ic.values, label=alpha_name, alpha=0.7)
        
        axes[0,0].set_title('Information Coefficient Time Series (Top 5 Alphas)')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('IC')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Mean IC distribution
        mean_ics = [results['mean_ic'] for results in ic_results.values()]
        axes[0,1].hist(mean_ics, bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Distribution of Mean IC')
        axes[0,1].set_xlabel('Mean IC')
        axes[0,1].set_ylabel('Number of Alphas')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Plot 3: IC vs IR scatter
        irs = [results['ic_ir'] for results in ic_results.values()]
        axes[1,0].scatter(mean_ics, irs, alpha=0.6)
        axes[1,0].set_title('Mean IC vs Information Ratio')
        axes[1,0].set_xlabel('Mean IC')
        axes[1,0].set_ylabel('Information Ratio')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Top alphas bar chart
        top_10_names = [name for name, _ in sorted_alphas[:10]]
        top_10_irs = [results['ic_ir'] for _, results in sorted_alphas[:10]]
        
        bars = axes[1,1].bar(range(len(top_10_names)), top_10_irs)
        axes[1,1].set_title('Top 10 Alphas by Information Ratio')
        axes[1,1].set_xlabel('Alpha Factor')
        axes[1,1].set_ylabel('Information Ratio')
        axes[1,1].set_xticks(range(len(top_10_names)))
        axes[1,1].set_xticklabels(top_10_names, rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Color bars based on value
        for i, bar in enumerate(bars):
            if top_10_irs[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig('data/alpha_ic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n" + "="*80)
        print("ALPHA FACTOR ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total alpha factors analyzed: {len(ic_results)}")
        print(f"Average IC: {np.mean(mean_ics):.4f}")
        print(f"Average IR: {np.mean(irs):.4f}")
        print(f"Positive IC alphas: {sum(1 for ic in mean_ics if ic > 0)}")
        print(f"Negative IC alphas: {sum(1 for ic in mean_ics if ic < 0)}")
        
        print(f"\nTop 10 Alpha Factors by Information Ratio:")
        for i, (alpha_name, results) in enumerate(sorted_alphas[:10], 1):
            print(f"{i:2d}. {alpha_name:10s} - IC: {results['mean_ic']:7.4f}, IR: {results['ic_ir']:7.3f}, IC_Std: {results['ic_std']:6.4f}")
        
        # Save detailed results
        ic_summary = pd.DataFrame({
            'alpha': list(ic_results.keys()),
            'mean_ic': [r['mean_ic'] for r in ic_results.values()],
            'ic_std': [r['ic_std'] for r in ic_results.values()],
            'ic_ir': [r['ic_ir'] for r in ic_results.values()]
        }).sort_values('ic_ir', key=abs, ascending=False)
        
        ic_summary.to_csv('data/alpha_ic_analysis.csv', index=False)
        print(f"\nDetailed IC analysis saved to 'data/alpha_ic_analysis.csv'")
        
    else:
        print("No valid alpha factors found for analysis.")
def main():
    # Load the combined dataframe
    all_df = pd.read_csv("data/Binance_AllCrypto_d.csv")
    all_df.index = range(all_df.shape[0])
    # all_df = pd.read_csv("data/ccxt_Binance_AllCrypto_d.csv")

    # Convert Date column to proper datetime format
    all_df['Date'] = pd.to_datetime(all_df['Date'])
    
    all_df = all_df.sort_values(['Ticker', 'Date'])

    # Loop through each ticker for TA-Lib indicators (time-series per ticker)
    tickers = all_df["Ticker"].unique()
    for ticker in tickers:
        mask = all_df["Ticker"] == ticker
        all_df.loc[mask, "SMA_20"] = talib.SMA(all_df.loc[mask, "Close"], timeperiod=20)
        all_df.loc[mask, "EMA_20"] = talib.EMA(all_df.loc[mask, "Close"], timeperiod=20)
        all_df.loc[mask, "RSI_14"] = talib.RSI(all_df.loc[mask, "Close"], timeperiod=14)
        all_df.loc[mask, "MACD"], all_df.loc[mask, "MACD_signal"], all_df.loc[mask, "MACD_hist"] = talib.MACD(all_df.loc[mask, "Close"])

    # Prepare common columns on full df
    all_df = prepare_df(all_df)

    # Compute all alphas on the full df (time-series with groupby, cross-sectional without)
    alpha_functions = [globals()[f'alpha{i}'] for i in range(1, 102)]

    for i, alpha_func in enumerate(alpha_functions, 1):
        # if(i<82):
        #     continue
        print(f"about to process alpha_{i}")
        all_df[f"alpha_{i}"] = alpha_func(all_df)
        print(f"processed alpha_{i}")
    
    print(f"Successfully calculated all 101 alpha factors!")
    print(f"DataFrame shape: {all_df.shape}")
    
    # Save the results
    all_df.to_csv('data/alpha_factors_results.csv', index=False)
    print("Alpha factors saved to 'data/alpha_factors_results.csv'")
    
    # Backtesting
    print("\nStarting backtesting...")
    backtest_results = backtest_alphas(all_df)
    
    # Plot results
    print("Plotting cumulative PnL...")
    plot_cumulative_pnl(backtest_results)
    
    # Alphalens analysis
    print("\nStarting Alphalens analysis...")
    try:
        alphalens_analysis(all_df, backtest_results)
    except ImportError:
        print("Alphalens not installed. Installing now...")
        import subprocess
        subprocess.run(["pip", "install", "alphalens-reloaded"], check=True)
        alphalens_analysis(all_df, backtest_results)
    except Exception as e:
        print(f"Alphalens analysis failed: {e}")
        print("Continuing without Alphalens analysis...")


if __name__ == "__main__":
    main()
