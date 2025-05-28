import pandas as pd
import numpy as np
import requests
from itertools import product
import ta
from datetime import datetime, timedelta

# --- Fetch Binance mainnet historical klines for last 6 months ---
def get_binance_klines(symbol, interval, start_str, end_str=None, limit=1000):
    base_url = 'https://api.binance.com/api/v3/klines'
    df_list = []
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None

    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'startTime': start_ts,
        }
        if end_ts:
            params['endTime'] = end_ts
        response = requests.get(base_url, params=params)
        data = response.json()
        if not data:
            break

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df.astype({
            'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'
        })
        df_list.append(df[['open', 'high', 'low', 'close', 'volume']])

        # Increment start timestamp for next batch (last open_time + interval)
        last_open = df.index[-1]
        start_ts = int((last_open + pd.Timedelta(minutes=15)).timestamp() * 1000)

        if len(data) < limit or (end_ts and start_ts > end_ts):
            break

    full_df = pd.concat(df_list)
    return full_df[~full_df.index.duplicated(keep='first')]  # Remove duplicates if any

# --- Add technical indicators ---
def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close'])/3).cumsum() / df['volume'].cumsum()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df.dropna(inplace=True)
    return df

# --- Buy signal with weights ---
def buy_signal(row, weights):
    score = 0
    score += weights[0] * (row['rsi'] < 30)
    score += weights[1] * (row['macd'] > row['macd_signal'])
    score += weights[2] * (row['ema9'] > row['ema21'])
    score += weights[3] * (row['close'] > row['vwap'])
    score += weights[4] * (row['close'] < row['bb_lower'])
    score += weights[5] * (row['stoch_k'] < 20 and row['stoch_k'] > row['stoch_d'])
    return score >= 3

# --- Sell signal with weights ---
def sell_signal(row, weights):
    score = 0
    score += weights[0] * (row['rsi'] > 70)
    score += weights[1] * (row['macd'] < row['macd_signal'])
    score += weights[2] * (row['ema9'] < row['ema21'])
    score += weights[3] * (row['close'] < row['vwap'])
    score += weights[4] * (row['close'] > row['bb_upper'])
    score += weights[5] * (row['stoch_k'] > 80 and row['stoch_k'] < row['stoch_d'])
    return score >= 3

# --- Backtest with reinvestment and initial capital ---
def backtest_strategy(df, buy_weights, sell_weights, initial_capital=10000):
    capital = initial_capital
    position = 0  # units of XRP
    trades = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        if position == 0 and buy_signal(row, buy_weights):
            position = capital / row['close']
            trades += 1
        elif position > 0 and sell_signal(row, sell_weights):
            capital = position * row['close']
            position = 0
            trades += 1

    # Close position if still holding at the end
    if position > 0:
        capital = position * df.iloc[-1]['close']

    profit = capital - initial_capital
    return profit, trades

# --- Loop through all weight combinations ---
def optimize_weights(df):
    best_profit = float('-inf')
    best_buy_weights = None
    best_sell_weights = None
    weight_range = [0, 1, 2]

    total_combinations = len(weight_range)**6 * len(weight_range)**6
    print(f"Total combinations to test: {total_combinations}")

    count = 0
    results = []

    for buy_weights in product(weight_range, repeat=6):
        for sell_weights in product(weight_range, repeat=6):
            profit, trades = backtest_strategy(df, buy_weights, sell_weights)
            count += 1
            results.append((profit, trades, buy_weights, sell_weights))
            if profit > best_profit:
                best_profit = profit
                best_buy_weights = buy_weights
                best_sell_weights = sell_weights
                print(f"New Best -> Profit: {profit:.2f}, Trades: {trades}, Buy weights: {buy_weights}, Sell weights: {sell_weights}")
            if count % 1000 == 0:
                print(f"Tested {count}/{total_combinations} combinations...")

    print(f"\nBest Buy weights: {best_buy_weights}, Best Sell weights: {best_sell_weights}, Max Profit: {best_profit:.2f}")
    return best_buy_weights, best_sell_weights, results

def main():
    print("Fetching 6 months of historical data for XRPUSDT at 15m interval...")
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)  # ~6 months
    df = get_binance_klines('XRPUSDT', '15m', start_str=start_date.strftime('%Y-%m-%d %H:%M:%S'), end_str=end_date.strftime('%Y-%m-%d %H:%M:%S'))
    print(f"Data fetched: {len(df)} rows")

    print("Calculating indicators...")
    df = add_indicators(df)

    print("Starting optimization over all weight combinations (this will take time)...")
    best_buy_weights, best_sell_weights, results = optimize_weights(df)

    print("Backtesting best weights on full data...")
    final_profit, final_trades = backtest_strategy(df, best_buy_weights, best_sell_weights)
    print(f"Final profit with best weights: ${final_profit:.2f} after {final_trades} trades.")

if __name__ == "__main__":
    main()
