import pandas as pd
import requests
import time
import ta  # pip install ta

# --- Config ---
SYMBOL = 'XRPUSDT'
INTERVAL = '15m'
START_USDT = 10000
BUY_WEIGHTS = (0, 0, 1, 0, 2, 2)
SELL_WEIGHTS = (2, 0, 0, 1, 0, 1)

# --- Fetch historical data from Binance Mainnet ---
def get_binance_klines(symbol, interval, start_ts, end_ts):
    url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    while start_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000
        }
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        all_data += data
        start_ts = data[-1][0] + 1
        time.sleep(0.5)  # to respect rate limits
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df.astype({
        'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'
    })
    return df[['open', 'high', 'low', 'close', 'volume']]

# --- Add indicators ---
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

# --- Signal logic ---
def buy_signal(row, weights):
    score = 0
    score += weights[0] * (row['rsi'] < 30)
    score += weights[1] * (row['macd'] > row['macd_signal'])
    score += weights[2] * (row['ema9'] > row['ema21'])
    score += weights[3] * (row['close'] > row['vwap'])
    score += weights[4] * (row['close'] < row['bb_lower'])
    score += weights[5] * (row['stoch_k'] < 20 and row['stoch_k'] > row['stoch_d'])
    return score >= 3

def sell_signal(row, weights):
    score = 0
    score += weights[0] * (row['rsi'] > 70)
    score += weights[1] * (row['macd'] < row['macd_signal'])
    score += weights[2] * (row['ema9'] < row['ema21'])
    score += weights[3] * (row['close'] < row['vwap'])
    score += weights[4] * (row['close'] > row['bb_upper'])
    score += weights[5] * (row['stoch_k'] > 80 and row['stoch_k'] < row['stoch_d'])
    return score >= 3

# --- Backtesting logic ---
def backtest_strategy(df, buy_weights, sell_weights):
    usdt = START_USDT
    position = 0
    for i in range(1, len(df)):
        row = df.iloc[i]
        if position == 0 and buy_signal(row, buy_weights):
            position = usdt / row['close']
            entry_price = row['close']
        elif position > 0 and sell_signal(row, sell_weights):
            usdt = position * row['close']
            position = 0
    if position > 0:
        usdt = position * df.iloc[-1]['close']
    return usdt - START_USDT

# --- Main ---
def main():
    print("Fetching 6 months of historical data...")
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - 6 * 30 * 24 * 60 * 60 * 1000
    df = get_binance_klines(SYMBOL, INTERVAL, start_ts, end_ts)
    print("Calculating indicators...")
    df = add_indicators(df)
    print("Running backtest with best weights...")
    profit = backtest_strategy(df, BUY_WEIGHTS, SELL_WEIGHTS)
    print(f"Final profit over 6 months: ${profit:.2f} on ${START_USDT} initial investment")

if __name__ == '__main__':
    main()
