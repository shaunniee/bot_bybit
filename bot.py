import time
import os
from pybit.unified_trading import HTTP
from telegram import Bot
from datetime import datetime
import asyncio

# === CONFIGURATION ===
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "XRPUSDT"
COOLDOWN_SECONDS = 86400  # 24 hours
TRADE_PERCENTAGE = 0.98
PROFIT_TARGET = 0.03
STOP_LOSS_PERCENTAGE = 0.03

# === INIT ===
session = HTTP(testnet=True, api_key=API_KEY, api_secret=API_SECRET)
bot = Bot(token=TELEGRAM_TOKEN)
in_cooldown = False

async def send_telegram(msg):
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

def get_wallet_balance():
    balances = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")["result"]["list"][0]["coin"][0]["walletBalance"]
    return float(balances)

def get_price():
    data = session.get_tickers(category="spot", symbol=SYMBOL)
    return float(data["result"]["list"][0]["lastPrice"])

def place_order(side, qty):
    return session.place_order(
        category="spot",
        symbol=SYMBOL,
        side=side,
        orderType="Limit",
        qty=str(qty),
        price=get_price()
    )

def get_position():
    orders = session.get_open_orders(category="spot", symbol=SYMBOL)
    return any(order["side"] == "Buy" for order in orders["result"]["list"])

def has_xrp():
    balance = session.get_wallet_balance(accountType="UNIFIED", coin="XRP")["result"]["list"][0]["coin"][0]["walletBalance"]
    return float(balance) > 0

def get_last_buy_price():
    try:
        orders = session.get_order_history(category="spot", symbol=SYMBOL)
        for order in sorted(orders["result"]["list"], key=lambda x: x["createdTime"], reverse=True):
            if order["side"] == "Buy" and order["orderStatus"] == "Filled":
                return float(order["avgPrice"])
    except Exception as e:
        print(f"Error fetching last buy price: {e}")
    return None

# === INITIAL BUY PRICE CHECK ===
xrp_balance = session.get_wallet_balance(accountType="UNIFIED", coin="XRP")["result"]["list"][0]["coin"][0]["walletBalance"]
xrp_balance = float(xrp_balance)

if xrp_balance > 20:
    buy_price = get_last_buy_price()
    if buy_price:
        print(f"Detected XRP balance > 20. Setting buy_price to last filled buy price: {buy_price}")
    else:
        buy_price = None
        print("XRP balance > 20 but no filled buy order found. Setting buy_price to None.")
else:
    buy_price = None
    print("XRP balance <= 20. Setting buy_price to None.")

cooldown_start = None

# === TRADING LOOP ===
async def trading_loop():
    global buy_price, cooldown_start, in_cooldown

    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if in_cooldown:
                if time.time() - cooldown_start >= COOLDOWN_SECONDS:
                    in_cooldown = False
                    await send_telegram("Cooldown period ended. Bot is resuming trades.")
                else:
                    print(f"{now} | In cooldown. Waiting...")
                    await asyncio.sleep(600)
                    continue

            current_price = get_price()
            change_24h = float(session.get_tickers(category="spot", symbol=SYMBOL)["result"]["list"][0]["price24hPcnt"])
            print(f"{now} | 24h Change: {change_24h:.2f}% | Price: {current_price}")

            if buy_price:
                price_change = (current_price - buy_price) / buy_price
                if price_change >= PROFIT_TARGET:
                    xrp_balance = session.get_wallet_balance(accountType="UNIFIED", coin="XRP")["result"]["list"][0]["coin"][0]["walletBalance"]
                    if xrp_balance > 0:
                        place_order("Sell", float(xrp_balance))
                        await send_telegram(f"📈 Sold XRP at {current_price:.4f} (Profit Reached)")
                        buy_price = None

                elif price_change <= -STOP_LOSS_PERCENTAGE:
                    xrp_balance = session.get_wallet_balance(accountType="UNIFIED", coin="XRP")["result"]["list"][0]["coin"][0]["walletBalance"]
                    if xrp_balance > 0:
                        place_order("Sell", float(xrp_balance))
                        await send_telegram(f"🔻 Stop-loss hit. Sold XRP at {current_price:.4f}")
                        buy_price = None
                        cooldown_start = time.time()
                        in_cooldown = True

            else:
                usdt = get_wallet_balance()
                if change_24h <= -5 and not get_position() and usdt >= 10:
                    trade_usdt = usdt * TRADE_PERCENTAGE
                    buy_price = current_price
                    qty = trade_usdt / current_price
                    print(place_order("Buy", round(qty, 2)))
                    await send_telegram(f"🛒 Placed Buy Order for XRP at {current_price:.4f}")
                elif usdt < 10:
                    print(f"{now} | Skipping buy: USDT balance too low (${usdt:.2f})")

        except Exception as e:
            print(f"Error: {e}")

        await asyncio.sleep(600)  # Wait 10 minutes before next iteration

# Run the trading loop
asyncio.run(trading_loop())
