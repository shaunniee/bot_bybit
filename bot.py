import pandas as pd
import numpy as np
import requests
import datetime
import ta
import matplotlib.pyplot as plt

class AdvancedXRPBacktester:
    def __init__(self, initial_capital=10000, max_risk_per_trade=0.02, fee=0.00075, slippage=0.0005):
        self.symbol = 'XRPUSDT'
        self.interval = '15m'
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade  # max % capital risked per trade
        self.fee = fee
        self.slippage = slippage
        self.df = None
        self.trades = []
        self.position = 0
        self.entry_price = 0

    def fetch_data(self, start_str, end_str=None):
        print(f"Fetching historical data for {self.symbol}...")
        base_url = 'https://api.binance.com/api/v3/klines'
        limit = 1000
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
            response = requests.get(base_url, params=params)
            data = response.json()
            if not data:
                break
            all_klines.extend(data)
            last_open_time = data[-1][0]
            start_ts = last_open_time + 1
            if len(data) < limit:
                break

        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df.astype({
            'open': float, 'high': float, 'low': float,
            'close': float, 'volume': float
        })
        self.df = df[['open', 'high', 'low', 'close', 'volume']]

    def add_indicators(self):
        df = self.df
        # Multi-period RSI for robustness
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        # MACD with standard params and signal line
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        # EMA fast and slow
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        # ATR for volatility-based stop loss
        df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        # On-Balance Volume for volume confirmation
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df.dropna(inplace=True)
        self.df = df

    def calculate_signal_score(self, row, weights):
        # Normalize inputs to 0-1 scores for smooth scoring
        score = 0
        # RSI - normalized (low RSI bullish, high RSI bearish)
        rsi_norm = (70 - row['rsi_14']) / 70  # normalized to [0,1] roughly
        score += weights[0] * rsi_norm

        # MACD histogram (macd - signal)
        macd_hist = row['macd'] - row['macd_signal']
        macd_score = (macd_hist + 5) / 10  # assuming macd_hist ~[-5,5] range normalized
        score += weights[1] * macd_score

        # EMA crossover (fast - slow)
        ema_diff = (row['ema_9'] - row['ema_21']) / row['ema_21']
        ema_score = (ema_diff + 0.05) / 0.1  # normalize around zero
        score += weights[2] * ema_score

        # Price vs Bollinger Bands (buy near lower band)
        bb_range = row['bb_upper'] - row['bb_lower']
        bb_pos = (row['close'] - row['bb_lower']) / bb_range
        bb_score = 1 - bb_pos  # higher score if price closer to lower band
        score += weights[3] * bb_score

        # OBV momentum - normalize using recent pct change
        obv_mom = row['obv'] / max(1, row['obv'])  # simplistic norm
        score += weights[4] * obv_mom

        # ATR for volatility weight (lower ATR means stronger signals)
        atr_score = 1 - min(1, row['atr_14'] / row['close'])
        score += weights[5] * atr_score

        return score

    def dynamic_position_size(self, stop_loss_distance):
        # Calculate position size risking max_risk_per_trade of capital
        risk_amount = self.capital * self.max_risk_per_trade
        pos_size = risk_amount / stop_loss_distance
        return pos_size

    def backtest(self, buy_weights, sell_weights):
        df = self.df.copy()
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        portfolio_values = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            price = row['close']

            # Calculate stop-loss and take-profit dynamically using ATR
            stop_loss_dist = row['atr_14']
            take_profit_dist = stop_loss_dist * 1.5

            if self.position == 0:
                buy_score = self.calculate_signal_score(row, buy_weights)
                if buy_score > 2.5:  # Tuned threshold for strong buy
                    # Calculate position size based on ATR risk
                    position_size = self.dynamic_position_size(stop_loss_dist)
                    # Check if affordable, limit by capital
                    max_qty = self.capital / price
                    qty = min(position_size, max_qty)
                    cost = qty * price * (1 + self.fee + self.slippage)
                    if cost > self.capital:
                        continue
                    self.capital -= cost
                    self.position = qty
                    self.entry_price = price
                    self.trades.append({'type': 'buy', 'time': df.index[i], 'price': price, 'qty': qty, 'capital': self.capital})
            else:
                sell_score = self.calculate_signal_score(row, sell_weights)
                # Check stop loss and take profit
                if price <= self.entry_price - stop_loss_dist:
                    proceeds = self.position * price * (1 - self.fee - self.slippage)
                    self.capital += proceeds
                    self.trades.append({'type': 'stop_loss_sell', 'time': df.index[i], 'price': price, 'capital': self.capital})
                    self.position = 0
                    self.entry_price = 0
                elif price >= self.entry_price + take_profit_dist:
                    proceeds = self.position * price * (1 - self.fee - self.slippage)
                    self.capital += proceeds
                    self.trades.append({'type': 'take_profit_sell', 'time': df.index[i], 'price': price, 'capital': self.capital})
                    self.position = 0
                    self.entry_price = 0
                elif sell_score < 1.5:  # Threshold tuned for sell
                    proceeds = self.position * price * (1 - self.fee - self.slippage)
                    self.capital += proceeds
                    self.trades.append({'type': 'signal_sell', 'time': df.index[i], 'price': price, 'capital': self.capital})
                    self.position = 0
                    self.entry_price = 0
            portfolio_values.append(self.capital + self.position * price)

        df['portfolio_value'] = portfolio_values + [portfolio_values[-1]] * (len(df) - len(portfolio_values))
        return self.capital, df, self.trades

    def analyze_performance(self, df):
        df = df.dropna()
        returns = df['portfolio_value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)  # 15m intervals in 1 year approx
        drawdown = (df['portfolio_value'] / df['portfolio_value'].cummax()) - 1
        max_drawdown = drawdown.min()
        total_return = df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1
        win_trades = [t for t in self.trades if t['type'].startswith('sell') and t['capital'] > self.initial_capital]
        win_rate = len(win_trades) / max(1, len([t for t in self.trades if t['type'].startswith('sell')]))

        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")

        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['portfolio_value'], label='Portfolio Value')
        plt.title("Portfolio Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.show()

    def run_full_test(self, start_date, end_date, buy_weights, sell_weights):
        self.fetch_data(start_date, end_date)
        self.add_indicators()
        final_capital, df, trades = self.backtest(buy_weights, sell_weights)
        print(f"Initial Capital: ${self.initial_capital}")
        print(f"Final Capital after backtest: ${final_capital:.2f}")
        self.analyze_performance(df)
        return df, trades

# Example usage:
if __name__ == "__main__":
    bot = AdvancedXRPBacktester()
    buy_w = [1.0, 1.5, 1.0, 0.5, 0.5, 0.5]
    sell_w = [1.0, 1.0, 0.5, 1.0, 0.5, 0.5]
    start = '2024-01-01'
    end = '2024-06-30'
    df, trades = bot.run_full_test(start, end, buy_w, sell_w)
