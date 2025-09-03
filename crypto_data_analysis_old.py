import pandas as pd
import talib

import pandas as pd
import numpy as np
from scipy.stats import rankdata

# Helper functions (same as before, with additions if needed)

def ts_min(df, col, d):
    return df.groupby('Ticker')[col].rolling(d, min_periods=1).min().reset_index(level=0, drop=True)

def ts_max(df, col, d):
    return df.groupby('Ticker')[col].rolling(d, min_periods=1).max().reset_index(level=0, drop=True)

def ts_argmin(df, col, d):
    def argmin_lags(window):
        if len(window) == 0:
            return np.nan
        return len(window) - 1 - np.argmin(window)
    return df.groupby('Ticker')[col].rolling(d, min_periods=1).apply(argmin_lags, raw=True).reset_index(level=0, drop=True)

def ts_argmax(series, d):
    def argmax_lags(window):
        if len(window) == 0:
            return np.nan
        return len(window) - 1 - np.argmax(window)
    if isinstance(series, np.ndarray):
        series = pd.Series(series)    
    return series.rolling(d, min_periods=1).apply(argmax_lags, raw=True)

def ts_rank(df, col, d):
    def rank_window(window):
        if len(window) == 0:
            return np.nan
        ranks = rankdata(window)
        return ranks[-1] / len(window)  # normalize to [0,1]
    return df.groupby('Ticker')[col].rolling(d, min_periods=1).apply(rank_window, raw=True).reset_index(level=0, drop=True)

def decay_linear(df, col, d):
    def wma(window):
        if len(window) == 0:
            return np.nan
        weights = np.arange(1, len(window) + 1) / np.sum(np.arange(1, len(window) + 1))
        return np.dot(window, weights)
    return df.groupby('Ticker')[col].rolling(d, min_periods=1).apply(wma, raw=True).reset_index(level=0, drop=True)

def stddev(df, col, d):
    return df.groupby('Ticker')[col].rolling(d, min_periods=1).std().reset_index(level=0, drop=True)

def ts_sum(df, col, d):
    return df.groupby('Ticker')[col].rolling(d, min_periods=1).sum().reset_index(level=0, drop=True)

def ts_product(df, col, d):
    return df.groupby('Ticker')[col].rolling(d, min_periods=1).prod().reset_index(level=0, drop=True)

# def ts_correlation(df, col1, col2, d):
#     return df.groupby('Ticker').apply(lambda g: g[col1].rolling(d, min_periods=1).corr(g[col2])).reset_index(level=0, drop=True)

def ts_correlation(df, col1, col2, d):
    # Only select the columns, not the grouping column
    result = (
        df.groupby('Ticker')[[col1, col2]]
        .apply(lambda g: g[col1].rolling(d, min_periods=1).corr(g[col2]))
    )
    # Flatten MultiIndex to align with df index
    result.index = result.index.droplevel(0)
    return result.sort_index()

def ts_covariance(df, col1, col2, d):
    return df.groupby('Ticker').apply(lambda g: g[col1].rolling(d, min_periods=1).cov(g[col2])).reset_index(level=0, drop=True)

def ts_delta(df, col, d):
    return df.groupby('Ticker')[col].diff(d)

def signed_power(df, col, a):
    return np.sign(df[col]) * np.abs(df[col]) ** a

def cs_rank(series):
    return series.rank(pct=True)

def cs_scale(df, col, a=1):
    abs_sum = df.groupby('Date')[col].transform(lambda g: np.sum(np.abs(g)))
    return a * df[col] / abs_sum.clip(lower=1e-12)  # avoid div by zero

def indneutralize(df, col, level=None):
    # Placeholder: subtract mean per date since no industry data
    means = df.groupby('Date')[col].transform('mean')
    return df[col] - means

def adv(df, d):
    return df.groupby('Ticker')['Volume USDT'].rolling(d, min_periods=1).mean().reset_index(level=0, drop=True)

# Prepare common columns
def prepare_df(df):
    df = df.copy()
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    df['vwap'] = df['Volume USDT'] / df['Volume'].clip(lower=1e-12)
    df['cap'] = 1  # Placeholder for market cap
    return df

def alpha1(df):
    df = prepare_df(df)
    inner = np.where(df['returns'] < 0, stddev(df, 'returns', 20), df['Close'])
    powered = inner ** 2
    argmax = ts_argmax(powered, 5)
    ranked = cs_rank(argmax)
    return ranked - 0.5

def alpha2(df):
    df = prepare_df(df)
    df["log_Volume"] = np.log(df['Volume'])
    delta_log_vol = ts_delta(df, "log_Volume", 2)
    df["ranked_delta"] = cs_rank(delta_log_vol)
    inner = (df['Close'] - df['Open']) / df['Open']
    df["ranked_inner"] = cs_rank(inner)
    corr = ts_correlation(df, 'ranked_delta', 'ranked_inner', 6)
    return -1 * corr

def alpha3(df):
    df = prepare_df(df)
    ranked_open = cs_rank(df['Open'])
    ranked_vol = cs_rank(df['Volume'])
    corr = ts_correlation(df, 'ranked_open', 'ranked_vol', 10)
    return -1 * corr

def alpha4(df):
    df = prepare_df(df)
    ranked_low = cs_rank(df['Low'])
    tsrank = ts_rank(ranked_low, 9)
    return -1 * tsrank

def alpha5(df):
    df = prepare_df(df)
    sum_vwap = ts_sum(df, 'vwap', 10)
    inner = df['Open'] - (sum_vwap / 10)
    ranked_close_vwap = cs_rank(df, df['Close'] - df['vwap'])
    abs_ranked = np.abs(ranked_close_vwap)
    return cs_rank(inner) * (-1 * abs_ranked)

def alpha6(df):
    df = prepare_df(df)
    corr = ts_correlation(df, 'Open', 'Volume', 10)
    return -1 * corr

def alpha7(df):
    df = prepare_df(df)
    df['adv20'] = adv(df, 20)
    condition = df['adv20'] < df['Volume']
    inner = -1 * ts_rank(np.abs(ts_delta(df, 'Close', 7)), 60) * np.sign(ts_delta(df, 'Close', 7))
    return np.where(condition, inner, -1)

def alpha8(df):
    df = prepare_df(df)
    sum_open = ts_sum(df, 'Open', 5)
    sum_returns = ts_sum(df, 'returns', 5)
    inner = sum_open * sum_returns
    delayed = ts_delta(df, 'inner', 10)  # wait, delay not delta
    # delay(x,d) = x shift d
    delayed = df.groupby('Ticker')['inner'].shift(10)
    diff = inner - delayed
    ranked = cs_rank(diff)
    return -1 * ranked

def alpha9(df):
    df = prepare_df(df)
    delta_close = ts_delta(df, 'Close', 1)
    condition1 = ts_min(df, 'delta_close', 5) > 0
    condition2 = ts_max(df, 'delta_close', 5) < 0
    return np.where(condition1, delta_close, np.where(condition2, delta_close, -1 * delta_close))

def alpha10(df):
    df = prepare_df(df)
    delta_close = ts_delta(df, 'Close', 1)
    condition1 = ts_min(df, 'delta_close', 4) > 0
    condition2 = ts_max(df, 'delta_close', 4) < 0
    inner = np.where(condition1, delta_close, np.where(condition2, delta_close, -1 * delta_close))
    return cs_rank(inner)


def alpha11(df):
    df = prepare_df(df)
    rank_max = cs_rank(ts_max(df['vwap'] - df['Close'], 3))
    rank_min = cs_rank(ts_min(df['vwap'] - df['Close'], 3))
    rank_delta_vol = cs_rank(ts_delta(df['Volume'], 3))
    return (rank_max + rank_min) * rank_delta_vol

def alpha12(df):
    df = prepare_df(df)
    sign_delta_vol = np.sign(ts_delta(df, 'Volume', 1))
    return sign_delta_vol * (-1 * ts_delta(df, 'Close', 1))

def alpha13(df):
    df = prepare_df(df)
    cov = ts_covariance(df, cs_rank(df['Close']), cs_rank(df['Volume']), 5)
    return -1 * cs_rank(cov)

def alpha14(df):
    df = prepare_df(df)
    rank_delta_ret = cs_rank(ts_delta(df['returns'], 3))
    corr = ts_correlation(df, 'Open', 'Volume', 10)
    return -1 * cs_rank(rank_delta_ret) * corr

def alpha15(df):
    df = prepare_df(df)
    corr_rank_high_vol = ts_correlation(df, cs_rank(df['High']), cs_rank(df['Volume']), 3)
    sum_corr = ts_sum(df, 'corr_rank_high_vol', 3)
    return -1 * sum_corr

def alpha16(df):
    df = prepare_df(df)
    cov = ts_covariance(df, cs_rank(df['High']), cs_rank(df['Volume']), 5)
    return -1 * cs_rank(cov)

def alpha17(df):
    df = prepare_df(df)
    rank_ts_close = cs_rank(ts_rank(df['Close'], 10))
    delta_delta_close = ts_delta(ts_delta(df['Close'], 1), 1)
    rank_delta_delta = cs_rank(delta_delta_close)
    rank_ts_vol_adv = cs_rank(ts_rank(df['Volume / adv20'], 5))
    return -1 * rank_ts_close * rank_delta_delta * rank_ts_vol_adv

def alpha18(df):
    df = prepare_df(df)
    std_abs_close_open = stddev(df, np.abs(df['Close'] - df['Open']), 5)
    corr_close_open = ts_correlation(df, 'Close', 'Open', 10)
    inner = std_abs_close_open + (df['Close'] - df['Open']) + corr_close_open
    return -1 * cs_rank(inner)

def alpha19(df):
    df = prepare_df(df)
    delta_close_7 = ts_delta(df, 'Close', 7)
    sign_delta7_plus_delta = np.sign(delta_close_7 + delta_close_7)
    sign_total = -1 * np.sign(delta_close_7 + sign_delta7_plus_delta)
    sum_ret_250 = ts_sum(df, 'returns', 250)
    rank_sum1 = cs_rank(1 + sum_ret_250)
    return sign_total * (1 + rank_sum1)

def alpha20(df):
    df = prepare_df(df)
    rank_open_delay_high = cs_rank(df, df['Open'] - df.groupby('Ticker')['High'].shift(1))
    rank_open_delay_close = cs_rank(df, df['Open'] - df.groupby('Ticker')['Close'].shift(1))
    rank_open_delay_low = cs_rank(df, df['Open'] - df.groupby('Ticker')['Low'].shift(1))
    return -1 * rank_open_delay_high * rank_open_delay_close * rank_open_delay_low

def alpha21(df):
    df = prepare_df(df)
    sum_close8_8 = ts_sum(df, 'Close', 8) / 8
    std_close8 = stddev(df, 'Close', 8)
    sum_close2_2 = ts_sum(df, 'Close', 2) / 2
    adv20 = adv(df, 20)
    condition1 = (sum_close8_8 + std_close8) < sum_close2_2
    condition2 = sum_close2_2 < (sum_close8_8 - std_close8)
    condition3 = 1 <= (df['Volume'] / adv20)
    return np.where(condition1, -1, np.where(condition2, 1, np.where(condition3, 1, -1)))

def alpha22(df):
    df = prepare_df(df)
    corr_high_vol = ts_correlation(df, 'High', 'Volume', 5)
    delta_corr = ts_delta(df, 'corr_high_vol', 5)
    rank_std_close20 = cs_rank(stddev(df['Close'], 20))
    return -1 * delta_corr * rank_std_close20

def alpha23(df):
    df = prepare_df(df)
    sum_high20_20 = ts_sum(df, 'High', 20) / 20
    condition = sum_high20_20 < df['High']
    return np.where(condition, -1 * ts_delta(df, 'High', 2), 0)

def alpha24(df):
    df = prepare_df(df)
    sum_close100_100 = ts_sum(df, 'Close', 100) / 100
    delta_sum = ts_delta(df, 'sum_close100_100', 100)
    delta_delay = delta_sum / df.groupby('Ticker')['Close'].shift(100)
    condition = delta_delay <= 0.05
    ts_min_close100 = ts_min(df, 'Close', 100)
    return np.where(condition, -1 * (df['Close'] - ts_min_close100), -1 * ts_delta(df, 'Close', 3))

def alpha25(df):
    df = prepare_df(df)
    adv20 = adv(df, 20)
    inner = -1 * df['returns'] * adv20 * df['vwap'] * (df['High'] - df['Close'])
    return cs_rank(inner)

def alpha26(df):
    df = prepare_df(df)
    tsrank_vol5 = ts_rank(df['Volume'], 5)
    tsrank_high5 = ts_rank(df['High'], 5)
    corr = ts_correlation(df, 'tsrank_vol5', 'tsrank_high5', 5)
    ts_max_corr = ts_max(df, 'corr', 3)
    return -1 * ts_max_corr

def alpha27(df):
    df = prepare_df(df)
    rank_vol = cs_rank(df['Volume'])
    rank_vwap = cs_rank(df['vwap'])
    corr = ts_correlation(df, 'rank_vol', 'rank_vwap', 6)
    sum_corr2 = ts_sum(df, 'corr', 2) / 2.0
    rank_sum = cs_rank(sum_corr2)
    condition = 0.5 < rank_sum
    return np.where(condition, -1, 1)

def alpha28(df):
    df = prepare_df(df)
    adv20 = adv(df, 20)
    corr_adv_low = ts_correlation(df, 'adv20', 'Low', 5)
    hl2 = (df['High'] + df['Low']) / 2
    inner = corr_adv_low + hl2 - df['Close']
    return cs_scale(inner)

def alpha29(df):
    df = prepare_df(df)
    delta_close5 = ts_delta(df, 'Close - 1', 5)  # assuming (close - 1)
    rank_delta = cs_rank(delta_close5)
    rank_rank = cs_rank(rank_delta)
    rank_scale_log = cs_rank(rank_rank)
    ts_min_rank = ts_min(rank_scale_log, 2)
    sum_ts_min = ts_sum(ts_min_rank, 1)
    log_sum = np.log(sum_ts_min)
    scale_log = cs_scale(log_sum)
    rank_scale = cs_rank(scale_log)
    rank_rank_scale = cs_rank(rank_scale)
    product_rank = ts_product(rank_rank_scale, 1)
    min_product = np.minimum(product_rank, 5)
    delay_ret6 = df.groupby('Ticker')['returns'].shift(6)
    ts_rank_delay = ts_rank(-1 * delay_ret6, 5)  # -1 * returns delayed
    return min_product + ts_rank_delay

def alpha30(df):
    df = prepare_df(df)
    sign_close_delay1 = np.sign(df['Close'] - df.groupby('Ticker')['Close'].shift(1))
    sign_delay1_delay2 = np.sign(df.groupby('Ticker')['Close'].shift(1) - df.groupby('Ticker')['Close'].shift(2))
    sign_delay2_delay3 = np.sign(df.groupby('Ticker')['Close'].shift(2) - df.groupby('Ticker')['Close'].shift(3))
    sum_sign = sign_close_delay1 + sign_delay1_delay2 + sign_delay2_delay3
    rank_sum_sign = cs_rank(df, 'sum_sign')
    sum_vol5 = ts_sum(df, 'Volume', 5)
    sum_vol20 = ts_sum(df, 'Volume', 20)
    inner = (1.0 - rank_sum_sign) * sum_vol5 / sum_vol20
    return inner

def alpha31(df):
    df = prepare_df(df)
    delta_close3 = ts_delta(df, 'Close', 3)
    delta_delta_close = ts_delta(df, 'delta_close3', 1)
    rank_delta_delta = cs_rank(df, 'delta_delta_close')
    rank_rank_delta = cs_rank(df, 'rank_delta_delta')
    decay_delta10 = decay_linear(df, '-1 * rank_rank_delta', 10)
    rank_decay = cs_rank(df, 'decay_delta10')
    rank_rank_rank = cs_rank(df, 'rank_decay')
    rank_delta_close3 = cs_rank(df, '-1 * delta_close3')
    corr_adv_low = ts_correlation(df, 'adv20', 'Low', 12)
    scale_corr = cs_scale(df, 'corr_adv_low')
    sign_scale = np.sign(scale_corr)
    return rank_rank_rank + rank_delta_close3 + sign_scale

def alpha32(df):
    df = prepare_df(df)
    sum_close7_7 = ts_sum(df, 'Close', 7) / 7
    inner1 = sum_close7_7 - df['Close']
    scale_inner1 = cs_scale(inner1)
    delay_close5 = df.groupby('Ticker')['Close'].shift(5)
    corr_vwap_delay = ts_correlation(df, 'vwap', 'delay_close5', 230)
    scale_corr = cs_scale(corr_vwap_delay)
    inner2 = 20 * scale_corr
    return scale_inner1 + inner2

def alpha33(df):
    df = prepare_df(df)
    inner = 1 - (df['Open'] / df['Close'])
    power = inner ** 1
    return cs_rank(-1 * power)

def alpha34(df):
    df = prepare_df(df)
    std_ret2 = stddev(df, 'returns', 2)
    std_ret5 = stddev(df, 'returns', 5)
    ratio_std = std_ret2 / std_ret5
    rank_ratio = cs_rank(ratio_std)
    rank_delta_close1 = cs_rank(ts_delta(df['Close'], 1))
    inner = (1 - rank_ratio) + (1 - rank_delta_close1)
    return cs_rank(inner)

def alpha35(df):
    df = prepare_df(df)
    ts_rank_vol32 = ts_rank(df['Volume'], 32)
    hl_low = (df['Close'] + df['High']) - df['Low']
    ts_rank_hl_low = ts_rank(hl_low, 16)
    ts_rank_ret32 = ts_rank(df['returns'], 32)
    inner1 = ts_rank_vol32 * (1 - ts_rank_hl_low)
    inner2 = 1 - ts_rank_ret32
    return inner1 * inner2

def alpha36(df):
    df = prepare_df(df)
    delay_vol1 = df.groupby('Ticker')['Volume'].shift(1)
    corr_close_open_delay_vol = ts_correlation(df, 'Close - Open', 'delay_vol1', 15)
    rank_corr = cs_rank(corr_close_open_delay_vol)
    part1 = 2.21 * rank_corr
    rank_open_close = cs_rank(df['Open'] - df['Close'])
    part2 = 0.7 * rank_open_close
    delay_neg_ret6 = df.groupby('Ticker')['returns'].shift(6)
    ts_rank_delay = ts_rank(-1 * delay_neg_ret6, 5)
    rank_ts = cs_rank(ts_rank_delay)
    part3 = 0.73 * rank_ts
    corr_vwap_adv = ts_correlation(df, 'vwap', 'adv20', 6)
    abs_corr = np.abs(corr_vwap_adv)
    rank_abs = cs_rank(abs_corr)
    part4 = rank_abs
    sum_close200_200 = ts_sum(df, 'Close', 200) / 200
    inner_sum_open = (sum_close200_200 - df['Open']) * (df['Close'] - df['Open'])
    rank_inner = cs_rank(inner_sum_open)
    part5 = 0.6 * rank_inner
    return part1 + part2 + part3 + part4 + part5

def alpha37(df):
    df = prepare_df(df)
    delay_open_close1 = df.groupby('Ticker')['Open - Close'].shift(1)
    corr_delay_close = ts_correlation(df, 'delay_open_close1', 'Close', 200)
    rank_corr = cs_rank(corr_delay_close)
    rank_open_close = cs_rank(df['Open'] - df['Close'])
    return rank_corr + rank_open_close

def alpha38(df):
    df = prepare_df(df)
    ts_rank_close10 = ts_rank(df['Close'], 10)
    rank_ts = cs_rank(ts_rank_close10)
    close_open = df['Close'] / df['Open']
    rank_close_open = cs_rank(close_open)
    return -1 * rank_ts * rank_close_open

def alpha39(df):
    df = prepare_df(df)
    decay_vol_adv9 = decay_linear(df, 'Volume / adv20', 9)
    rank_decay = cs_rank(decay_vol_adv9)
    delta_close7 = ts_delta(df, 'Close', 7)
    inner = delta_close7 * (1 - rank_decay)
    rank_inner = cs_rank(inner)
    sum_ret250 = ts_sum(df, 'returns', 250)
    rank_sum = cs_rank(sum_ret250)
    return -1 * rank_inner * (1 + rank_sum)

def alpha40(df):
    df = prepare_df(df)
    std_high10 = stddev(df, 'High', 10)
    rank_std = cs_rank(std_high10)
    corr_high_vol = ts_correlation(df, 'High', 'Volume', 10)
    return -1 * rank_std * corr_high_vol

def alpha41(df):
    df = prepare_df(df)
    hl_pow = (df['High'] * df['Low']) ** 0.5
    return hl_pow - df['vwap']

def alpha42(df):
    df = prepare_df(df)
    vwap_close = df['vwap'] - df['Close']
    rank_vwap_close = cs_rank(vwap_close)
    vwap_close_pos = df['vwap'] + df['Close']
    rank_vwap_close_pos = cs_rank(vwap_close_pos)
    return rank_vwap_close / rank_vwap_close_pos

def alpha43(df):
    df = prepare_df(df)
    adv20 = adv(df, 20)
    vol_adv = df['Volume'] / adv20
    ts_rank_vol_adv = ts_rank(vol_adv, 20)
    delta_close7 = ts_delta(df['Close'], 7)
    ts_rank_delta = ts_rank(-1 * delta_close7, 8)
    return ts_rank_vol_adv * ts_rank_delta

def alpha44(df):
    df = prepare_df(df)
    rank_vol = cs_rank(df['Volume'])
    corr_high_rank_vol = ts_correlation(df, 'High', rank_vol, 5)
    return -1 * corr_high_rank_vol

def alpha45(df):
    df = prepare_df(df)
    delay_close5 = df.groupby('Ticker')['Close'].shift(5)
    sum_delay20 = ts_sum(df, 'delay_close5', 20) / 20
    rank_sum = cs_rank(sum_delay20)
    corr_close_vol = ts_correlation(df, 'Close', 'Volume', 2)
    sum_close5 = ts_sum(df, 'Close', 5)
    sum_close20 = ts_sum(df, 'Close', 20)
    corr_sum = ts_correlation(df, 'sum_close5', 'sum_close20', 2)
    rank_corr_sum = cs_rank(corr_sum)
    inner = rank_sum * corr_close_vol * rank_corr_sum
    return -1 * inner

def alpha46(df):
    df = prepare_df(df)
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
    df = prepare_df(df)
    adv20 = adv(df, 20)
    vol_adv = (cs_rank(1 / df['Close']) * df['Volume']) / adv20
    rank_high_close = cs_rank(df['High'] - df['Close'])
    sum_high5_5 = ts_sum(df, 'High', 5) / 5
    inner = df['High'] * rank_high_close / sum_high5_5
    delay_vwap5 = df.groupby('Ticker')['vwap'].shift(5)
    rank_vwap_delay = cs_rank(df['vwap'] - delay_vwap5)
    return vol_adv * inner - rank_vwap_delay

def alpha48(df):
    df = prepare_df(df)
    delta_close1 = ts_delta(df['Close'], 1)
    delay_close1 = df.groupby('Ticker')['Close'].shift(1)
    delta_delay1 = ts_delta(delay_close1, 1)
    corr_delta = ts_correlation(df, 'delta_close1', 'delta_delay1', 250)
    inner = corr_delta * delta_close1 / df['Close']
    neutralized = indneutralize(df, inner)
    delay_close1_sq = (delta_close1 / delay_close1) ** 2
    sum_sq = ts_sum(delay_close1_sq, 250)
    return neutralized / sum_sq

def alpha49(df):
    df = prepare_df(df)
    delay_close20 = df.groupby('Ticker')['Close'].shift(20)
    delay_close10 = df.groupby('Ticker')['Close'].shift(10)
    part1 = (delay_close20 - delay_close10) / 10
    part2 = (delay_close10 - df['Close']) / 10
    diff = part1 - part2
    condition = diff < -0.1
    delta_close1 = ts_delta(df, 'Close', 1)
    return np.where(condition, 1, -1 * delta_close1)

def alpha50(df):
    df = prepare_df(df)
    rank_vol = cs_rank(df, 'Volume')
    rank_vwap = cs_rank(df, 'vwap')
    corr_rank = ts_correlation(df, 'rank_vol', 'rank_vwap', 5)
    rank_corr = cs_rank(df, 'corr_rank')
    ts_max_rank = ts_max(df, 'rank_corr', 5)
    return -1 * ts_max_rank

def alpha51(df):
    df = prepare_df(df)
    delay_close20 = df.groupby('Ticker')['Close'].shift(20)
    delay_close10 = df.groupby('Ticker')['Close'].shift(10)
    part1 = (delay_close20 - delay_close10) / 10
    part2 = (delay_close10 - df['Close']) / 10
    diff = part1 - part2
    condition = diff < -0.05
    delta_close1 = ts_delta(df, 'Close', 1)
    return np.where(condition, 1, -1 * delta_close1)

def alpha52(df):
    df = prepare_df(df)
    ts_min_low5 = ts_min(df, 'Low', 5)
    delay_ts_min5 = df.groupby('Ticker')['ts_min_low5'].shift(5)
    inner1 = -1 * ts_min_low5 + delay_ts_min5
    sum_ret240 = ts_sum(df, 'returns', 240)
    sum_ret20 = ts_sum(df, 'returns', 20)
    diff_sum = sum_ret240 - sum_ret20
    rank_diff = cs_rank(df, 'diff_sum / 220')
    ts_rank_vol5 = ts_rank(df, 'Volume', 5)
    return inner1 * rank_diff * ts_rank_vol5

def alpha53(df):
    df = prepare_df(df)
    hl_close = (df['Close'] - df['Low']) - (df['High'] - df['Close'])
    denom = df['Close'] - df['Low']
    inner = hl_close / denom
    return -1 * ts_delta(df, 'inner', 9)

def alpha54(df):
    df = prepare_df(df)
    low_close = df['Low'] - df['Close']
    open_pow5 = df['Open'] ** 5
    num = -1 * low_close * open_pow5
    low_high = df['Low'] - df['High']
    close_pow5 = df['Close'] ** 5
    denom = low_high * close_pow5
    return num / denom

def alpha55(df):
    df = prepare_df(df)
    ts_min_low12 = ts_min(df, 'Low', 12)
    ts_max_high12 = ts_max(df, 'High', 12)
    num = df['Close'] - ts_min_low12
    denom = ts_max_high12 - ts_min_low12
    ratio = num / denom
    rank_ratio = cs_rank(df, 'ratio')
    rank_vol = cs_rank(df, 'Volume')
    corr = ts_correlation(df, 'rank_ratio', 'rank_vol', 6)
    return -1 * corr

def alpha56(df):
    df = prepare_df(df)
    sum_ret10 = ts_sum(df, 'returns', 10)
    sum_sum_ret2_3 = ts_sum(df, ts_sum(df, 'returns', 2), 3) / 3
    inner = sum_ret10 / sum_sum_ret2_3
    rank_inner = cs_rank(df, 'inner')
    ret_cap = df['returns'] * df['cap']
    rank_ret_cap = cs_rank(df, 'ret_cap')
    return 0 - (1 * rank_inner * rank_ret_cap)

def alpha57(df):
    df = prepare_df(df)
    close_vwap = df['Close'] - df['vwap']
    rank_ts_argmax_close30 = ts_rank(df, ts_argmax(df, 'Close', 30), 2)
    decay_rank = decay_linear(df, 'rank_ts_argmax_close30', 2)
    inner = close_vwap / decay_rank
    return 0 - (1 * inner)

def alpha58(df):
    df = prepare_df(df)
    corr_vwap_vol = ts_correlation(df, 'vwap', 'Volume', 3.92795)
    decay_corr = decay_linear(df, 'corr_vwap_vol', 7.89291)
    rank_decay = cs_rank(df, 'decay_corr')
    neutralized_vwap = indneutralize(df, 'vwap')
    ts_rank_neut = ts_rank(df, 'rank_decay', 5.50322)
    return -1 * ts_rank_neut

def alpha59(df):
    df = prepare_df(df)
    open_weight = df['Open'] * (1 - 0.318108)
    vwap_weight = df['vwap'] * 0.318108
    weighted = vwap_weight + open_weight
    corr_weighted_adv = ts_correlation(df, 'weighted', 'adv180', 13.557)
    decay_corr = decay_linear(df, 'corr_weighted_adv', 12.2883)
    rank_decay = cs_rank(df, 'decay_corr')
    neutralized_close = indneutralize(df, 'Close')
    delta_neut = ts_delta(df, 'neutralized_close', 2.25164)
    decay_delta = decay_linear(df, 'delta_neut', 8.22237)
    rank_decay_delta = cs_rank(df, 'decay_delta')
    inner = rank_decay_delta - rank_decay
    return inner * -1

def alpha60(df):
    df = prepare_df(df)
    hl_vol = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])) * df['Volume']
    rank_hl_vol = cs_rank(df, 'hl_vol')
    scale_rank = cs_scale(df, 'rank_hl_vol')
    inner1 = 2 * scale_rank
    ts_argmax_close10 = ts_argmax(df, 'Close', 10)
    rank_ts = cs_rank(df, 'ts_argmax_close10')
    scale_rank_ts = cs_scale(df, 'rank_ts')
    inner2 = inner1 - scale_rank_ts
    return 0 - (1 * inner2)

def alpha61(df):
    df = prepare_df(df)
    ts_min_vwap = ts_min(df, 'vwap', 16.1219)
    vwap_min = df['vwap'] - ts_min_vwap
    rank_vwap_min = cs_rank(df, 'vwap_min')
    corr_vwap_adv = ts_correlation(df, 'vwap', 'adv180', 17.9282)
    rank_corr = cs_rank(df, 'corr_vwap_adv')
    return rank_vwap_min < rank_corr

def alpha62(df):
    df = prepare_df(df)
    sum_adv20 = ts_sum(df, 'adv20', 22.4101)
    corr_vwap_sum = ts_correlation(df, 'vwap', 'sum_adv20', 9.91009)
    rank_corr = cs_rank(df, 'corr_vwap_sum')
    rank_open1 = cs_rank(df, 'Open')
    rank_open2 = cs_rank(df, 'Open')
    sum_open = rank_open1 + rank_open2
    hl2 = (df['High'] + df['Low']) / 2
    rank_hl2 = cs_rank(df, 'hl2')
    rank_high = cs_rank(df, 'High')
    sum_right = rank_hl2 + rank_high
    condition = sum_open < sum_right
    rank_condition = cs_rank(df, 'condition')
    return rank_corr < rank_condition * -1

def alpha63(df):
    df = prepare_df(df)
    neutralized_close = indneutralize(df, 'Close')
    delta_neut = ts_delta(df, 'neutralized_close', 2.25164)
    decay_delta = decay_linear(df, 'delta_neut', 8.22237)
    rank_decay_delta = cs_rank(df, 'decay_delta')
    open_weight = df['Open'] * (1 - 0.318108)
    vwap_weight = df['vwap'] * 0.318108
    weighted = vwap_weight + open_weight
    sum_adv180 = ts_sum(df, 'adv180', 37.2467)
    corr_weighted_sum = ts_correlation(df, 'weighted', 'sum_adv180', 13.557)
    decay_corr = decay_linear(df, 'corr_weighted_sum', 12.2883)
    rank_decay_corr = cs_rank(df, 'decay_corr')
    inner = rank_decay_delta - rank_decay_corr
    return inner * -1

def alpha64(df):
    df = prepare_df(df)
    open_weight = df['Open'] * (1 - 0.178404)
    low_weight = df['Low'] * 0.178404
    weighted = low_weight + open_weight
    sum_weighted = ts_sum(df, 'weighted', 12.7054)
    sum_adv120 = ts_sum(df, 'adv120', 12.7054)
    corr_sum = ts_correlation(df, 'sum_weighted', 'sum_adv120', 16.6208)
    rank_corr = cs_rank(df, 'corr_sum')
    hl2 = (df['High'] + df['Low']) / 2
    hl2_weight = hl2 * 0.178404
    vwap_weight = df['vwap'] * (1 - 0.178404)
    weighted2 = hl2_weight + vwap_weight
    delta_weighted = ts_delta(df, 'weighted2', 3.69741)
    rank_delta = cs_rank(df, 'delta_weighted')
    inner = rank_corr < rank_delta
    return inner * -1

def alpha65(df):
    df = prepare_df(df)
    open_weight = df['Open'] * (1 - 0.00817205)
    vwap_weight = df['vwap'] * 0.00817205
    weighted = vwap_weight + open_weight
    sum_adv60 = ts_sum(df, 'adv60', 8.6911)
    corr_weighted_sum = ts_correlation(df, 'weighted', 'sum_adv60', 6.40374)
    rank_corr = cs_rank(df, 'corr_weighted_sum')
    ts_min_open = ts_min(df, 'Open', 13.635)
    open_min = df['Open'] - ts_min_open
    rank_open_min = cs_rank(df, 'open_min')
    inner = rank_corr < rank_open_min
    return inner * -1

def alpha66(df):
    df = prepare_df(df)
    delta_vwap = ts_delta(df, 'vwap', 3.51013)
    decay_delta = decay_linear(df, 'delta_vwap', 7.23052)
    rank_decay = cs_rank(df, 'decay_delta')
    low_weight = df['Low'] * (1 - 0.96633)
    low_weight2 = df['Low'] * 0.96633
    weighted_low = low_weight + low_weight2
    weighted_vwap = weighted_low - df['vwap']
    open_hl2 = df['Open'] - ((df['High'] + df['Low']) / 2)
    denom = open_hl2
    inner = weighted_vwap / denom
    corr_inner = ts_correlation(df, 'low', 'adv81', 19.569)
    decay_corr = decay_linear(df, 'corr_inner', 17.1543)
    ts_rank_decay = ts_rank(df, 'decay_corr', 6.72611)
    inner2 = rank_decay + ts_rank_decay
    return inner2 * -1

def alpha67(df):
    df = prepare_df(df)
    ts_min_high = ts_min(df, 'High', 2.14593)
    high_min = df['High'] - ts_min_high
    rank_high_min = cs_rank(df, 'high_min')
    neutralized_vwap = indneutralize(df, 'vwap')
    corr_vwap_adv = ts_correlation(df, 'neutralized_vwap', 'adv20', 6.02936)
    rank_corr = cs_rank(df, 'corr_vwap_adv')
    power = rank_high_min ** rank_corr
    return power * -1

def alpha68(df):
    df = prepare_df(df)
    rank_high = cs_rank(df, 'High')
    rank_adv15 = cs_rank(df, 'adv15')
    corr_rank = ts_correlation(df, 'rank_high', 'rank_adv15', 8.91644)
    ts_rank_corr = ts_rank(df, 'corr_rank', 13.9333)
    close_weight = df['Close'] * (1 - 0.518371)
    low_weight = df['Low'] * 0.518371
    weighted = low_weight + close_weight
    delta_weighted = ts_delta(df, 'weighted', 1.06157)
    rank_delta = cs_rank(df, 'delta_weighted')
    inner = ts_rank_corr < rank_delta
    return inner * -1

def alpha69(df):
    df = prepare_df(df)
    neutralized_vwap = indneutralize(df, 'vwap')
    delta_neut = ts_delta(df, 'neutralized_vwap', 2.72412)
    ts_max_delta = ts_max(df, 'delta_neut', 4.79344)
    rank_ts_max = cs_rank(df, 'ts_max_delta')
    close_weight = df['Close'] * (1 - 0.490655)
    vwap_weight = df['vwap'] * 0.490655
    weighted = vwap_weight + close_weight
    corr_weighted_adv = ts_correlation(df, 'weighted', 'adv20', 4.92416)
    ts_rank_corr = ts_rank(df, 'corr_weighted_adv', 9.0615)
    power = rank_ts_max ** ts_rank_corr
    return power * -1

def alpha70(df):
    df = prepare_df(df)
    delta_vwap = ts_delta(df, 'vwap', 1.29456)
    rank_delta = cs_rank(df, 'delta_vwap')
    neutralized_close = indneutralize(df, 'Close')
    corr_close_adv = ts_correlation(df, 'neutralized_close', 'adv50', 17.8256)
    ts_rank_corr = ts_rank(df, 'corr_close_adv', 17.9171)
    power = rank_delta ** ts_rank_corr
    return power * -1

def alpha71(df):
    df = prepare_df(df)
    ts_rank_close = ts_rank(df, 'Close', 3.43976)
    ts_rank_adv180 = ts_rank(df, 'adv180', 12.0647)
    corr_ts = ts_correlation(df, 'ts_rank_close', 'ts_rank_adv180', 18.0175)
    decay_corr = decay_linear(df, 'corr_ts', 4.20501)
    ts_rank_decay1 = ts_rank(df, 'decay_corr', 15.6948)
    rank_ts1 = np.maximum(ts_rank_decay1, ts_rank_decay1)  # max of same
    hl_close = (df['Low'] + df['Open']) - (df['vwap'] + df['vwap'])
    power_hl = hl_close ** 2
    rank_power = cs_rank(df, 'power_hl')
    decay_rank = decay_linear(df, 'rank_power', 16.4662)
    ts_rank_decay2 = ts_rank(df, 'decay_rank', 4.4388)
    return np.maximum(rank_ts1, ts_rank_decay2)

def alpha72(df):
    df = prepare_df(df)
    hl2 = (df['High'] + df['Low']) / 2
    corr_hl_adv = ts_correlation(df, 'hl2', 'adv40', 8.93345)
    decay_corr = decay_linear(df, 'corr_hl_adv', 10.1519)
    rank_decay = cs_rank(df, 'decay_corr')
    ts_rank_vwap = ts_rank(df, 'vwap', 3.72469)
    ts_rank_vol = ts_rank(df, 'Volume', 18.5188)
    corr_ts = ts_correlation(df, 'ts_rank_vwap', 'ts_rank_vol', 6.86671)
    decay_corr2 = decay_linear(df, 'corr_ts', 2.95011)
    rank_decay2 = cs_rank(df, 'decay_corr2')
    return rank_decay / rank_decay2

def alpha73(df):
    df = prepare_df(df)
    delta_vwap = ts_delta(df, 'vwap', 4.72775)
    decay_delta = decay_linear(df, 'delta_vwap', 2.91864)
    rank_decay = cs_rank(df, 'decay_decay')
    open_weight = df['Open'] * (1 - 0.147155)
    low_weight = df['Low'] * 0.147155
    weighted = low_weight + open_weight
    denom = weighted
    delta_weighted = ts_delta(df, 'weighted', 2.03608) / denom * -1
    decay_delta_weight = decay_linear(df, 'delta_weighted', 3.33829)
    ts_rank_decay = ts_rank(df, 'decay_delta_weight', 16.7411)
    inner = np.maximum(rank_decay, ts_rank_decay)
    return inner * -1

def alpha74(df):
    df = prepare_df(df)
    sum_adv30 = ts_sum(df, 'adv30', 37.4843)
    corr_close_sum = ts_correlation(df, 'Close', 'sum_adv30', 15.1365)
    rank_corr = cs_rank(df, 'corr_close_sum')
    high_weight = df['High'] * 0.0261661
    vwap_weight = df['vwap'] * (1 - 0.0261661)
    weighted = high_weight + vwap_weight
    rank_weighted = cs_rank(df, 'weighted')
    rank_vol = cs_rank(df, 'Volume')
    corr_rank = ts_correlation(df, 'rank_weighted', 'rank_vol', 11.4791)
    rank_corr2 = cs_rank(df, 'corr_rank')
    inner = rank_corr < rank_corr2
    return inner * -1

def alpha75(df):
    df = prepare_df(df)
    corr_vwap_vol = ts_correlation(df, 'vwap', 'Volume', 4.24304)
    rank_corr = cs_rank(df, 'corr_vwap_vol')
    rank_low = cs_rank(df, 'Low')
    rank_adv50 = cs_rank(df, 'adv50')
    corr_rank_low_adv = ts_correlation(df, 'rank_low', 'rank_adv50', 12.4413)
    rank_corr2 = cs_rank(df, 'corr_rank_low_adv')
    return rank_corr < rank_corr2

def alpha76(df):
    df = prepare_df(df)
    delta_vwap = ts_delta(df, 'vwap', 1.24383)
    decay_delta = decay_linear(df, 'delta_vwap', 11.8259)
    rank_decay = cs_rank(df, 'decay_delta')
    neutralized_low = indneutralize(df, 'Low')
    corr_low_adv = ts_correlation(df, 'neutralized_low', 'adv81', 8.14941)
    ts_rank_corr = ts_rank(df, 'corr_low_adv', 19.569)
    decay_ts_rank = decay_linear(df, 'ts_rank_corr', 17.1543)
    ts_rank_decay2 = ts_rank(df, 'decay_ts_rank', 19.383)
    inner = np.maximum(rank_decay, ts_rank_decay2)
    return inner * -1

def alpha77(df):
    df = prepare_df(df)
    hl2_high = ((df['High'] + df['Low']) / 2 + df['High']) - (df['vwap'] + df['High'])
    decay_hl = decay_linear(df, 'hl2_high', 20.0451)
    rank_decay = cs_rank(df, 'decay_hl')
    corr_hl_adv = ts_correlation(df, '(High + Low)/2', 'adv40', 3.1614)
    decay_corr = decay_linear(df, 'corr_hl_adv', 5.64125)
    rank_decay2 = cs_rank(df, 'decay_corr')
    return np.minimum(rank_decay, rank_decay2)

def alpha78(df):
    df = prepare_df(df)
    low_weight = df['Low'] * (1 - 0.352233)
    vwap_weight = df['vwap'] * 0.352233
    weighted = low_weight + vwap_weight
    sum_weighted = ts_sum(df, 'weighted', 19.7428)
    sum_adv40 = ts_sum(df, 'adv40', 19.7428)
    corr_sum = ts_correlation(df, 'sum_weighted', 'sum_adv40', 6.83313)
    rank_corr = cs_rank(df, 'corr_sum')
    power = rank_corr ** rank_corr  # ^rank(correlation(rank(vwap), rank(volume), 5.77492))
    # wait, from snippet: ^rank(correlation(rank(vwap), rank(volume), 5.77492))
    rank_vwap = cs_rank(df, 'vwap')
    rank_vol = cs_rank(df, 'Volume')
    corr_rank_vwap_vol = ts_correlation(df, 'rank_vwap', 'rank_vol', 5.77492)
    rank_corr2 = cs_rank(df, 'corr_rank_vwap_vol')
    power2 = rank_corr ** rank_corr2
    return power2

def alpha79(df):
    df = prepare_df(df)
    close_weight = df['Close'] * (1 - 0.60733)
    open_weight = df['Open'] * 0.60733
    weighted = open_weight + close_weight
    neutralized_weighted = indneutralize(df, 'weighted')
    delta_neut = ts_delta(df, 'neutralized_weighted', 1.23438)
    rank_delta = cs_rank(df, 'delta_neut')
    ts_rank_vwap = ts_rank(df, 'vwap', 3.60973)
    ts_rank_adv150 = ts_rank(df, 'adv150', 9.18637)
    corr_ts = ts_correlation(df, 'ts_rank_vwap', 'ts_rank_adv150', 14.6644)
    rank_corr = cs_rank(df, 'corr_ts')
    inner = rank_delta < rank_corr
    return inner

def alpha80(df):
    df = prepare_df(df)
    open_weight = df['Open'] * (1 - 0.868128)
    high_weight = df['High'] * 0.868128
    weighted = high_weight + open_weight
    neutralized_weighted = indneutralize(df, 'weighted')
    delta_neut = ts_delta(df, 'neutralized_weighted', 4.04545)
    sign_delta = np.sign(delta_neut)
    rank_sign = cs_rank(df, 'sign_delta')
    corr_high_adv = ts_correlation(df, 'High', 'adv10', 5.11456)
    ts_rank_corr = ts_rank(df, 'corr_high_adv', 5.53756)
    power = rank_sign ** ts_rank_corr
    return power * -1

def alpha81(df):
    df = prepare_df(df)
    sum_adv10 = ts_sum(df, 'adv10', 49.6054)
    corr_vwap_sum = ts_correlation(df, 'vwap', 'sum_adv10', 8.47743)
    rank_corr = cs_rank(df, 'corr_vwap_sum')
    power4 = rank_corr ** 4
    product_rank = ts_product(df, 'power4', 14.9655)
    log_product = np.log(product_rank)
    rank_log = cs_rank(df, 'log_product')
    rank_vwap = cs_rank(df, 'vwap')
    rank_vol = cs_rank(df, 'Volume')
    corr_rank_vwap_vol = ts_correlation(df, 'rank_vwap', 'rank_vol', 5.07914)
    rank_corr2 = cs_rank(df, 'corr_rank_vwap_vol')
    inner = rank_log < rank_corr2
    return inner * -1

def alpha82(df):
    df = prepare_df(df)
    open_delta = ts_delta(df, 'Open', 1.46063)
    decay_open = decay_linear(df, 'open_delta', 14.8717)
    rank_decay = cs_rank(df, 'decay_open')
    neutralized_vol = indneutralize(df, 'Volume')
    open_weight = df['Open'] * (1 - 0.634196)
    open_weight2 = df['Open'] * 0.634196
    weighted = open_weight + open_weight2
    corr_weighted_open = ts_correlation(df, 'neutralized_vol', 'weighted', 17.4842)
    decay_corr = decay_linear(df, 'corr_weighted_open', 6.92131)
    ts_rank_decay2 = ts_rank(df, 'decay_corr', 13.4283)
    inner = rank_decay, ts_rank_decay2  # min
    return np.minimum(rank_decay, ts_rank_decay2) * -1

def alpha83(df):
    df = prepare_df(df)
    sum_close5_5 = ts_sum(df, 'Close', 5) / 5
    hl_denom = (df['High'] - df['Low']) / sum_close5_5
    delay_hl_denom2 = df.groupby('Ticker')['hl_denom'].shift(2)
    rank_delay = cs_rank(df, 'delay_hl_denom2')
    rank_vol = cs_rank(df, 'Volume')
    num = rank_delay * rank_vol
    vwap_close = df['vwap'] - df['Close']
    denom = hl_denom / vwap_close
    return num / denom

def alpha84(df):
    df = prepare_df(df)
    ts_max_vwap = ts_max(df, 'vwap', 15.3217)
    vwap_ts_max = df['vwap'] - ts_max_vwap
    ts_rank_vwap = ts_rank(df, 'vwap_ts_max', 20.7127)
    delta_close = ts_delta(df, 'Close', 4.96796)
    return signed_power(df, 'ts_rank_vwap', 'delta_close')

def alpha85(df):
    df = prepare_df(df)
    high_weight = df['High'] * (1 - 0.876703)
    close_weight = df['Close'] * 0.876703
    weighted = close_weight + high_weight
    corr_weighted_adv = ts_correlation(df, 'weighted', 'adv30', 9.61331)
    rank_corr = cs_rank(df, 'corr_weighted_adv')
    hl2 = (df['High'] + df['Low']) / 2
    ts_rank_hl2 = ts_rank(df, 'hl2', 3.70596)
    ts_rank_vol = ts_rank(df, 'Volume', 10.1595)
    corr_ts = ts_correlation(df, 'ts_rank_hl2', 'ts_rank_vol', 7.11408)
    rank_corr2 = cs_rank(df, 'corr_ts')
    power = rank_corr ** rank_corr2
    return power

def alpha86(df):
    df = prepare_df(df)
    sum_adv20 = ts_sum(df, 'adv20', 14.7444)
    corr_close_sum = ts_correlation(df, 'Close', 'sum_adv20', 6.00049)
    ts_rank_corr = ts_rank(df, 'corr_close_sum', 20.4195)
    open_close = df['Open'] + df['Close']
    vwap_open = df['vwap'] + df['Open']
    diff = open_close - vwap_open
    rank_diff = cs_rank(df, 'diff')
    inner = ts_rank_corr < rank_diff
    return inner * -1

def alpha87(df):
    df = prepare_df(df)
    close_weight = df['Close'] * (1 - 0.369701)
    vwap_weight = df['vwap'] * 0.369701
    weighted = vwap_weight + close_weight
    delta_weighted = ts_delta(df, 'weighted', 1.91233)
    decay_delta = decay_linear(df, 'delta_weighted', 2.65461)
    rank_decay = cs_rank(df, 'decay_delta')
    neutralized_adv81 = indneutralize(df, 'adv81')
    corr_adv_close = ts_correlation(df, 'neutralized_adv81', 'Close', 13.4132)
    abs_corr = np.abs(corr_adv_close)
    decay_abs = decay_linear(df, 'abs_corr', 4.89768)
    ts_rank_decay = ts_rank(df, 'decay_abs', 14.4535)
    inner = np.maximum(rank_decay, ts_rank_decay)
    return inner * -1

def alpha88(df):
    df = prepare_df(df)
    rank_open = cs_rank(df, 'Open')
    rank_low = cs_rank(df, 'Low')
    sum_left = rank_open + rank_low
    rank_high = cs_rank(df, 'High')
    rank_close = cs_rank(df, 'Close')
    sum_right = rank_high + rank_close
    diff = sum_left - sum_right
    decay_diff = decay_linear(df, 'diff', 8.06882)
    rank_decay = cs_rank(df, 'decay_diff')
    ts_rank_close = ts_rank(df, 'Close', 8.44728)
    ts_rank_adv60 = ts_rank(df, 'adv60', 20.6966)
    corr_ts = ts_correlation(df, 'ts_rank_close', 'ts_rank_adv60', 8.01266)
    decay_corr = decay_linear(df, 'corr_ts', 6.65053)
    ts_rank_decay2 = ts_rank(df, 'decay_corr', 2.61957)
    return np.minimum(rank_decay, ts_rank_decay2)

def alpha89(df):
    df = prepare_df(df)
    low_weight = df['Low'] * (1 - 0.967285)
    low_weight2 = df['Low'] * 0.967285
    weighted = low_weight + low_weight2
    corr_weighted_adv = ts_correlation(df, 'weighted', 'adv10', 6.94279)
    decay_corr = decay_linear(df, 'corr_weighted_adv', 5.51607)
    ts_rank_decay = ts_rank(df, 'decay_corr', 3.79744)
    neutralized_vwap = indneutralize(df, 'vwap')
    delta_neut = ts_delta(df, 'neutralized_vwap', 3.48158)
    decay_delta = decay_linear(df, 'delta_neut', 10.1466)
    ts_rank_decay2 = ts_rank(df, 'decay_delta', 15.3012)
    return ts_rank_decay - ts_rank_decay2

def alpha90(df):
    df = prepare_df(df)
    ts_max_close = ts_max(df, 'Close', 4.66719)
    close_ts_max = df['Close'] - ts_max_close
    rank_close_ts = cs_rank(df, 'close_ts_max')
    neutralized_adv40 = indneutralize(df, 'adv40')
    corr_adv_low = ts_correlation(df, 'neutralized_adv40', 'Low', 5.38375)
    ts_rank_corr = ts_rank(df, 'corr_adv_low', 3.21856)
    power = rank_close_ts ** ts_rank_corr
    return power * -1

def alpha91(df):
    df = prepare_df(df)
    neutralized_close = indneutralize(df, 'Close')
    corr_neut_vol = ts_correlation(df, 'neutralized_close', 'Volume', 9.74928)
    decay_corr = decay_linear(df, 'corr_neut_vol', 16.398)
    decay_decay = decay_linear(df, 'decay_corr', 3.83219)
    ts_rank_decay = ts_rank(df, 'decay_decay', 4.8667)
    corr_vwap_adv = ts_correlation(df, 'vwap', 'adv30', 4.01303)
    decay_corr2 = decay_linear(df, 'corr_vwap_adv', 2.6809)
    rank_decay2 = cs_rank(df, 'decay_corr2')
    inner = ts_rank_decay - rank_decay2
    return inner * -1

def alpha92(df):
    df = prepare_df(df)
    hl2_close = ((df['High'] + df['Low']) / 2 + df['Close']) < (df['Low'] + df['Open'])
    decay_hl = decay_linear(df, 'hl2_close', 14.7221)
    ts_rank_decay = ts_rank(df, 'decay_hl', 18.8683)
    neutralized_vol = indneutralize(df, 'Volume')
    open_weight = df['Open'] * (1 - 0.634196)
    open_weight2 = df['Open'] * 0.634196
    weighted = open_weight + open_weight2
    corr_weighted = ts_correlation(df, 'neutralized_vol', 'weighted', 17.4842)
    decay_corr = decay_linear(df, 'corr_weighted', 6.92131)
    ts_rank_decay2 = ts_rank(df, 'decay_corr', 13.4283)
    inner = np.minimum(ts_rank_decay, ts_rank_decay2)
    return inner * -1

def alpha93(df):
    df = prepare_df(df)
    neutralized_vwap = indneutralize(df, 'vwap')
    corr_neut_adv = ts_correlation(df, 'neutralized_vwap', 'adv81', 17.4193)
    decay_corr = decay_linear(df, 'corr_neut_adv', 19.848)
    ts_rank_decay = ts_rank(df, 'decay_corr', 7.54455)
    close_weight = df['Close'] * (1 - 0.524434)
    vwap_weight = df['vwap'] * 0.524434
    weighted = vwap_weight + close_weight
    delta_weighted = ts_delta(df, 'weighted', 2.77377)
    decay_delta = decay_linear(df, 'delta_weighted', 16.2664)
    rank_decay = cs_rank(df, 'decay_delta')
    inner = ts_rank_decay / rank_decay
    return inner

def alpha94(df):
    df = prepare_df(df)
    ts_min_vwap = ts_min(df, 'vwap', 11.5783)
    vwap_ts_min = df['vwap'] - ts_min_vwap
    rank_vwap_ts = cs_rank(df, 'vwap_ts_min')
    ts_rank_vwap = ts_rank(df, 'vwap', 19.6462)
    ts_rank_adv60 = ts_rank(df, 'adv60', 4.02992)
    corr_ts = ts_correlation(df, 'ts_rank_vwap', 'ts_rank_adv60', 18.0926)
    ts_rank_corr = ts_rank(df, 'corr_ts', 2.70756)
    power = rank_vwap_ts ** ts_rank_corr
    return power * -1

def alpha95(df):
    df = prepare_df(df)
    ts_min_open = ts_min(df, 'Open', 12.4105)
    open_ts_min = df['Open'] - ts_min_open
    rank_open_ts = cs_rank(df, 'open_ts_min')
    hl2 = (df['High'] + df['Low']) / 2
    sum_hl2 = ts_sum(df, 'hl2', 19.1351)
    sum_adv40 = ts_sum(df, 'adv40', 19.1351)
    corr_sum = ts_correlation(df, 'sum_hl2', 'sum_adv40', 12.8742)
    power5 = corr_sum ** 5
    rank_power = cs_rank(df, 'power5')
    ts_rank_rank = ts_rank(df, 'rank_power', 11.7584)
    inner = rank_open_ts < ts_rank_rank
    return inner

def alpha96(df):
    df = prepare_df(df)
    corr_vwap_vol = ts_correlation(df, 'vwap', 'Volume', 3.83878)
    decay_corr = decay_linear(df, 'corr_vwap_vol', 4.16783)
    ts_rank_decay1 = ts_rank(df, 'decay_corr', 8.38151)
    ts_rank_close = ts_rank(df, 'Close', 7.45404)
    ts_rank_adv60 = ts_rank(df, 'adv60', 4.13242)
    corr_ts = ts_correlation(df, 'ts_rank_close', 'ts_rank_adv60', 3.65459)
    ts_argmax_corr = ts_argmax(df, 'corr_ts', 12.6556)
    decay_argmax = decay_linear(df, 'ts_argmax_corr', 14.0365)
    ts_rank_decay2 = ts_rank(df, 'decay_argmax', 13.4143)
    inner = np.maximum(ts_rank_decay1, ts_rank_decay2)
    return inner * -1

def alpha97(df):
    df = prepare_df(df)
    low_weight = df['Low'] * (1 - 0.721001)
    vwap_weight = df['vwap'] * 0.721001
    weighted = vwap_weight + low_weight
    neutralized_weighted = indneutralize(df, 'weighted')
    delta_neut = ts_delta(df, 'neutralized_weighted', 3.3705)
    decay_delta = decay_linear(df, 'delta_neut', 20.4523)
    rank_decay = cs_rank(df, 'decay_delta')
    ts_rank_low = ts_rank(df, 'Low', 7.87871)
    ts_rank_adv60 = ts_rank(df, 'adv60', 17.255)
    corr_ts = ts_correlation(df, 'ts_rank_low', 'ts_rank_adv60', 4.97547)
    ts_rank_corr = ts_rank(df, 'corr_ts', 18.5925)
    decay_ts_rank = decay_linear(df, 'ts_rank_corr', 15.7152)
    ts_rank_decay2 = ts_rank(df, 'decay_ts_rank', 6.71659)
    inner = rank_decay - ts_rank_decay2
    return inner * -1

def alpha98(df):
    df = prepare_df(df)
    sum_adv5 = ts_sum(df, 'adv5', 26.4719)
    corr_vwap_sum = ts_correlation(df, 'vwap', 'sum_adv5', 4.58418)
    decay_corr = decay_linear(df, 'corr_vwap_sum', 7.18088)
    rank_decay = cs_rank(df, 'decay_corr')
    rank_open = cs_rank(df, 'Open')
    rank_adv15 = cs_rank(df, 'adv15')
    corr_rank = ts_correlation(df, 'rank_open', 'rank_adv15', 20.8187)
    ts_argmin_corr = ts_argmin(df, 'corr_rank', 8.62571)
    ts_rank_argmin = ts_rank(df, 'ts_argmin_corr', 6.95668)
    decay_ts_rank = decay_linear(df, 'ts_rank_argmin', 8.07206)
    rank_decay2 = cs_rank(df, 'decay_ts_rank')
    inner = rank_decay - rank_decay2
    return inner

def alpha99(df):
    df = prepare_df(df)
    hl2 = (df['High'] + df['Low']) / 2
    sum_hl2 = ts_sum(df, 'hl2', 19.8975)
    sum_adv60 = ts_sum(df, 'adv60', 19.8975)
    corr_sum = ts_correlation(df, 'sum_hl2', 'sum_adv60', 8.8136)
    rank_corr = cs_rank(df, 'corr_sum')
    corr_low_vol = ts_correlation(df, 'Low', 'Volume', 6.28259)
    rank_corr2 = cs_rank(df, 'corr_low_vol')
    inner = rank_corr < rank_corr2
    return inner * -1

def alpha100(df):
    df = prepare_df(df)
    hl_vol = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])) * df['Volume']
    rank_hl_vol = cs_rank(df, 'hl_vol')
    neutralized_rank = indneutralize(df, 'rank_hl_vol')
    neutralized_neut = indneutralize(df, 'neutralized_rank')
    scale_neut = cs_scale(df, 'neutralized_neut', 1.5)
    corr_close_rank_adv = ts_correlation(df, 'Close', cs_rank(df, 'adv20'), 5)
    rank_ts_argmin_close = cs_rank(df, ts_argmin(df, 'Close', 30))
    inner_corr = corr_close_rank_adv - rank_ts_argmin_close
    neut_inner = indneutralize(df, 'inner_corr')
    scale_neut2 = cs_scale(df, 'neut_inner')
    inner = scale_neut - scale_neut2
    vol_adv = df['Volume'] / adv(df, 20)
    mult = inner * vol_adv
    return 0 - (1 * mult)

def alpha101(df):
    df = prepare_df(df)
    hl = df['High'] - df['Low'] + 0.001
    close_open = df['Close'] - df['Open']
    return close_open / hl

if __name__ == "__main__":
    # Load the combined dataframe
    all_df = pd.read_csv("data/Binance_AllCrypto_d.csv")
    all_df.index = range(all_df.shape[0])
    # all_df = pd.read_csv("data/ccxt_Binance_AllCrypto_d.csv")

    # Example: Get a list of all tickers
    tickers = all_df["Ticker"].unique()

    # Loop through each ticker and perform analysis
    for ticker in tickers:
        df = all_df[all_df["Ticker"] == ticker]
        df = df.sort_values("Date")

        # TA-Lib indicators
        mask = all_df["Ticker"] == ticker
        all_df.loc[mask, "SMA_20"] = talib.SMA(all_df.loc[mask, "Close"], timeperiod=20)
        all_df.loc[mask, "EMA_20"] = talib.EMA(all_df.loc[mask, "Close"], timeperiod=20)
        all_df.loc[mask, "RSI_14"] = talib.RSI(all_df.loc[mask, "Close"], timeperiod=14)
        all_df.loc[mask, "MACD"], all_df.loc[mask, "MACD_signal"], all_df.loc[mask, "MACD_hist"] = talib.MACD(all_df.loc[mask, "Close"])

        alpha_functions = [globals()[f'alpha{i}'] for i in range(1, 102)]

        for i, alpha_func in enumerate(alpha_functions, 1):
            df[f"alpha_{i}"] = alpha_func(df)
        # Optionally assign back to all_df or save per ticker
        all_df.loc[df.index, [f"alpha_{i}" for i in range(1, 102)]] = df[[f"alpha_{i}" for i in range(1, 102)]]
        
    print (all_df.shape)