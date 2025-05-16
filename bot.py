import time
import requests
import os
from pybit.unified_trading import HTTP
from telegram import Bot
from datetime import datetime

# === CONFIGURATION ===
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "XRPUSDT"
COOLDOWN_SECONDS = 86400  # 24h
TRADE_PERCENTAGE = 0.98
PROFIT_TARGET = 0.03
STOP_LOSS_PERCENTAGE = 0.03

# === INIT ===
session = HTTP(testnet=True, api_key=API_KEY, api_secret=API_SECRET)
bot = Bot(token=TELEGRAM_TOKEN)
in_cooldown = False

def send_telegram(msg):
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

def get_wallet_balance():
    balances = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["coin"]
    usdt = next(item for item in balances if item["coin"] == "USDT")
    return float(usdt["availableToTrade"])

def get_price():
    data = session.get_tickers(category="spot", symbol=SYMBOL)
    return float(data["result"]["list"][0]["lastPrice"])

def place_order(side, qty):
    return session.place_order(
        category="spot",
        symbol=SYMBOL,
        side=side,
        orderType="Market",
        qty=str(qty),
    )

def get_position():
    orders = session.get_open_orders(category="spot", symbol=SYMBOL)
    return any(order["side"] == "Buy" for order in orders["result"]["list"])

# === TRADING LOOP ===
buy_price = None
cooldown_start = None

while True:
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if in_cooldown:
            if time.time() - cooldown_start >= COOLDOWN_SECONDS:
                in_cooldown = False
                send_telegram("Cooldown period ended. Bot is resuming trades.")
            else:
                time.sleep(30)
                continue

        current_price = get_price()
        change_24h = float(session.get_tickers(category="spot", symbol=SYMBOL)["result"]["list"][0]["price24hPcnt"]) * 100
        print(f"{now} | 24h Change: {change_24h:.2f}% | Price: {current_price:.4f}")

        if buy_price:
            # Monitor sell condition
            price_change = (current_price - buy_price) / buy_price

            if price_change >= PROFIT_TARGET:
                balance = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["coin"]
                xrp_balance = next((coin for coin in balance if coin["coin"] == "XRP"), None)
                if xrp_balance and float(xrp_balance["availableToTrade"]) > 0:
                    place_order("Sell", float(xrp_balance["availableToTrade"]))
                    send_telegram(f"📈 Sold XRP at {current_price:.4f} (Profit Reached)")
                    buy_price = None

            elif price_change <= -STOP_LOSS_PERCENTAGE:
                balance = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["coin"]
                xrp_balance = next((coin for coin in balance if coin["coin"] == "XRP"), None)
                if xrp_balance and float(xrp_balance["availableToTrade"]) > 0:
                    place_order("Sell", float(xrp_balance["availableToTrade"]))
                    send_telegram(f"🔻 Stop-loss hit. Sold XRP at {current_price:.4f}")
                    buy_price = None
                    cooldown_start = time.time()
                    in_cooldown = True

        else:
            if change_24h <= -5:
                usdt = get_wallet_balance()
                trade_usdt = usdt * TRADE_PERCENTAGE
                buy_price = current_price
                qty = trade_usdt / current_price
                place_order("Buy", round(qty, 2))
                send_telegram(f"🟢 Bought XRP at {buy_price:.4f} | Qty: {round(qty, 2)}")

        time.sleep(30)

    except Exception as e:
        send_telegram(f"❌ Error: {e}")
        time.sleep(60)
