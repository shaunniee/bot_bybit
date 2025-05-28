import pandas as pd
import numpy as np
import requests
from itertools import product
import ta
import matplotlib.pyplot as plt
import datetime

class XRPBacktestBot:
    def __init__(self, initial_capital=10000, fee=0.00075, slippage=0.0005):
        self.symbol = 'XRPUSDT'
        self.interval = '15m'
        self.initial_capital = initial_capital
        self.fee = fee
        self.slippage = slippage
        self.df = None
        self.results = {}

    def fetch_historical_data(self, start_str, end_str=None):
        # Fetch historical klines from Binance API for XRPUSDT 15m
        # Using Binance API v3 klines endpoint; may require pagination for 6 months of data
        # This function assumes user passes start and end in 'YYYY-MM-DD' format
        
        base_url = 'https://api.binance.com/api/v3/klines'
        limit = 1000  # max per request
        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None
        
        all_klines = []
        while True:
            params = {
                'symbol': self.symbol,
                'interval': self.interval,
                'startTime': start_ts,
                'limit': limit
            }
            if end_ts:
                params['endTime'] = end_ts
            
            resp = requests.get(base_url, params=params)
            data = resp.json()
            if not data:
                break
            
            all_klines.extend(data)
            last_open_time = data[-1][0]
            start_ts = last_open_time + 1
            
            if len(data) < limit:
                break
        
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        self.df = df[['open', 'high', 'low', 'close', 'volume']]

    def add_indicators(self):
        df = self.df
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df.dropna(inplace=True)
        self.df = df

    def buy_signal(self, row, weights):
        # Adjust threshold to >=2.5 for XRP signals (tuned for volatility)
        score = 0
        score += weights[0] * (row['rsi'] < 35)
        score += weights[1] * (row['macd'] > row['macd_signal'])
        score += weights[2] * (row['ema9'] > row['ema21'])
        score += weights[3] * (row['close'] > row['vwap'])
        score += weights[4] * (row['close'] < row['bb_lower'])
        score += weights[5] * (row['stoch_k'] < 25 and row['stoch_k'] > row['stoch_d'])
        return score >= 2.5

    def sell_signal(self, row, weights):
        score = 0
        score += weights[0] * (row['rsi'] > 65)
        score += weights[1] * (row['macd'] < row['macd_signal'])
        score += weights[2] * (row['ema9'] < row['ema21'])
        score += weights[3] * (row['close'] < row['vwap'])
        score += weights[4] * (row['close'] > row['bb_upper'])
        score += weights[5] * (row['stoch_k'] > 75 and row['stoch_k'] < row['stoch_d'])
        return score >= 2.5

    def backtest(self, buy_weights, sell_weights, stop_loss_pct=0.03, take_profit_pct=0.05):
        df = self.df.copy()
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        portfolio_values = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            price = row['close']

            if position == 0:
                if self.buy_signal(row, buy_weights):
                    qty = capital / price
                    cost = qty * price * (1 + self.fee + self.slippage)
                    capital -= cost
                    position = qty
                    entry_price = price
                    trades.append({'type': 'buy', 'time': df.index[i], 'price': price, 'capital': capital})
            else:
                # Check stop-loss and take-profit first
                if price <= entry_price * (1 - stop_loss_pct):
                    proceeds = position * price * (1 - self.fee - self.slippage)
                    capital += proceeds
                    trades.append({'type': 'sell_stop_loss', 'time': df.index[i], 'price': price, 'capital': capital})
                    position = 0
                    entry_price = 0
                elif price >= entry_price * (1 + take_profit_pct):
                    proceeds = position * price * (1 - self.fee - self.slippage)
                    capital += proceeds
                    trades.append({'type': 'sell_take_profit', 'time': df.index[i], 'price': price, 'capital': capital})
                    position = 0
                    entry_price = 0
                elif self.sell_signal(row, sell_weights):
                    proceeds = position * price * (1 - self.fee - self.slippage)
                    capital += proceeds
                    trades.append({'type': 'sell_signal', 'time': df.index[i], 'price': price, 'capital': capital})
                    position = 0
                    entry_price = 0

            portfolio_value = capital + position * price
            portfolio_values.append({'time': df.index[i], 'value': portfolio_value})

        if position > 0:
            last_price = df['close'].iloc[-1]
            proceeds = position * last_price * (1 - self.fee - self.slippage)
            capital += proceeds
            trades.append({'type': 'sell_end', 'time': df.index[-1], 'price': last_price, 'capital': capital})
            position = 0

        total_profit = capital - self.initial_capital
        portfolio_df = pd.DataFrame(portfolio_values).set_index('time')
        self.results = {
            'profit': total_profit,
            'trades': trades,
            'portfolio': portfolio_df
        }
        return total_profit, trades, portfolio_df

    def plot_results(self):
        portfolio_df = self.results.get('portfolio')
        trades = self.results.get('trades', [])

        if portfolio_df is None or len(portfolio_df) == 0:
            print("No portfolio data to plot.")
            return

        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_df.index, portfolio_df['value'], label='Portfolio Value')

        for trade in trades:
            color = 'g' if trade['type'].startswith('buy') else 'r'
            marker = '^' if trade['type'].startswith('buy') else 'v'
            plt.scatter(trade['time'], trade['capital'], marker=marker, color=color, s=100)

        plt.title(f'XRP Backtest Portfolio Value')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value (USDT)')
        plt.legend()
        plt.grid()
        plt.show()


def main():
    bot = XRPBacktestBot(initial_capital=10000)

    print("Fetching 6 months XRP historical data...")
    # Example: 6 months ago from today
    today = datetime.datetime.utcnow()
    six_months_ago = today - datetime.timedelta(days=180)
    bot.fetch_historical_data(start_str=six_months_ago.strftime('%Y-%m-%d'))

    print("Calculating indicators...")
    bot.add_indicators()

    # Example weight combos you can optimize or set manually
    buy_weights = (1, 1, 1, 1, 1, 1)
    sell_weights = (1, 1, 1, 1, 1, 1)

    print("Starting backtest...")
    profit, trades, portfolio = bot.backtest(buy_weights, sell_weights, stop_loss_pct=0.03, take_profit_pct=0.05)
    print(f"Backtest completed. Profit: {profit:.2f} USDT, Trades executed: {len(trades)}")

    bot.plot_results()

if __name__ == '__main__':
    main()
