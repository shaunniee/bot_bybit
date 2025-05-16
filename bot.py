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
BUY_TOLERANCE_PERCENTAGE = 0.01  # 1% below market

# === INIT ===
session = HTTP(testnet=True, api_key=API_KEY, api_secret=API_SECRET)
bot = Bot(token=TELEGRAM_TOKEN)
in_cooldown = False
cooldown_start = None

async def send_telegram(msg):
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

def get_wallet_balance():
    balances = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")["result"]["list"][0]["coin"][0]["walletBalance"]
    return float(balances)

def get_price():
    data = session.get_tickers(category="spot", symbol=SYMBOL)
    return float(data["result"]["list"][0]["lastPrice"])

def place_conditional_buy(qty, trigger_price):
    response = session.place_order(
        category="spot",
        symbol=SYMBOL,
        side="Buy",
        orderType="Market",
        qty=str(qty),
        triggerPrice=str(trigger_price),
        triggerDirection=1,  # Trigger when price drops below
        timeInForce="GoodTillCancel"
    )
    return response["result"]["orderId"]

def check_order_filled(order_id):
    order_info = session.get_order(category="spot", symbol=SYMBOL, orderId=order_id)
    return order_info["result"]["orderStatus"] == "Filled"

def place_take_profit_and_stop_loss(qty, buy_price):
    take_profit_price = round(buy_price * (1 + PROFIT_TARGET), 4)
    stop_loss_price = round(buy_price * (1 - STOP_LOSS_PERCENTAGE), 4)

    # Take Profit
    session.place_order(
        category="spot",
        symbol=SYMBOL,
        side="Sell",
        orderType="Limit",
        qty=str(qty),
        price=str(take_profit_price)
    )

    # Stop Loss
    session.place_order(
        category="spot",
        symbol=SYMBOL,
        side="Sell",
        orderType="Market",
        qty=str(qty),
        stopLoss=str(stop_loss_price)
    )

    return take_profit_price, stop_loss_price

# === TRADING LOOP ===
async def trading_loop():
    global in_cooldown, cooldown_start

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

            usdt = get_wallet_balance()
            if change_24h <= -5 and usdt >= 10:
                trade_usdt = usdt * TRADE_PERCENTAGE
                trigger_price = current_price * (1 - BUY_TOLERANCE_PERCENTAGE)
                qty = round(trade_usdt / trigger_price, 2)

                order_id = place_conditional_buy(qty, round(trigger_price, 4))
                await send_telegram(f"🛒 Placed Conditional Market Buy Order for XRP\nTrigger: {trigger_price:.4f}")

                # Wait for the order to fill
                print("⏳ Waiting for buy order to fill...")
                for _ in range(60):  # Check for up to 10 minutes
                    if check_order_filled(order_id):
                        take_profit, stop_loss = place_take_profit_and_stop_loss(qty, trigger_price)
                        await send_telegram(
                            f"✅ Buy filled at {trigger_price:.4f}\n"
                            f"📈 Take Profit set at {take_profit:.4f}\n"
                            f"🔻 Stop Loss set at {stop_loss:.4f}"
                        )
                        break
                    await asyncio.sleep(10)
                else:
                    await send_telegram("⚠️ Buy order not filled within 10 minutes.")

            elif usdt < 10:
                print(f"{now} | Skipping buy: USDT balance too low (${usdt:.2f})")

        except Exception as e:
            print(f"Error: {e}")
            await send_telegram(f"⚠️ Error in trading loop: {e}")

        await asyncio.sleep(600)  # Wait 10 minutes before next iteration

# Run the trading loop
asyncio.run(trading_loop())
