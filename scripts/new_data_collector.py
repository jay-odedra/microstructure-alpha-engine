import pandas as pd
import numpy as np
from binance.client import Client
import time
from tqdm import tqdm
from pathlib import Path
import logging

pd.set_option("display.max_columns", None)

# --------------------------------------------------
# LOGGER
# --------------------------------------------------

def setup_logger():

    logger = logging.getLogger("data_collector")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler("./data_collection.log")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger()


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SYMBOL = "BTCUSDT"
LOOP_ITERATIONS = 100000
BUFFER_SIZE = 500

LOB_PATH = Path(
    "E:\\Quant_Projects\\microstructure-alpha-engine\\microstructure-alpha-engine\\data\\raw\\lob"
)
TRADE_PATH = Path(
    "E:\\Quant_Projects\\microstructure-alpha-engine\\microstructure-alpha-engine\\data\\raw\\trades"
)

client = Client(requests_params={"timeout": 5})


# --------------------------------------------------
# API HELPERS
# --------------------------------------------------

def fetch_order_book():
    """Fetch order book snapshot"""

    for _ in range(5):

        try:

            t0 = int(time.time() * 1000)

            depth = client.get_order_book(symbol=SYMBOL, limit=100)

            t1 = int(time.time() * 1000)

            latency = (t1 - t0) / 2

            bids = np.array(depth["bids"], dtype=float).tolist()
            asks = np.array(depth["asks"], dtype=float).tolist()

            return {
                "timestamp": int(time.time() * 1000),
                "latency": float(latency),
                "lastUpdateId": int(depth["lastUpdateId"]),
                "bids": bids,
                "asks": asks,
            }

        except Exception as e:

            logger.warning(f"Depth API error: {e}")
            time.sleep(2)

    return None


def fetch_trades(last_trade_id):
    """Fetch trades newer than last_trade_id"""

    for _ in range(5):

        try:
            trades = client.get_recent_trades(symbol=SYMBOL, limit=1000)
            break

        except Exception as e:

            logger.warning(f"Trade API error: {e}")
            time.sleep(2)

    else:
        return [], last_trade_id

    trades = [trade for trade in trades if trade["id"] > last_trade_id]

    if trades:
        last_trade_id = trades[-1]["id"]

    return trades, last_trade_id


# --------------------------------------------------
# SAVE FUNCTION
# --------------------------------------------------

def save_buffers(depth_buffer, trades_buffer, file_id):

    pd.DataFrame(depth_buffer).to_parquet(
        LOB_PATH / f"lob_data_raw_{file_id}.parquet",
        compression="snappy",
    )

    pd.DataFrame(trades_buffer).to_parquet(
        TRADE_PATH / f"trades_data_raw_{file_id}.parquet",
        compression="snappy",
    )

    #logger.info(f"Saved batch {file_id}")


# --------------------------------------------------
# MAIN COLLECTION LOOP
# --------------------------------------------------

def collect_data():

    logger.info("Starting data collection")

    depth_buffer = []
    trades_buffer = []

    file_id = 0
    last_trade_id = 0

    try:

        for _ in tqdm(range(LOOP_ITERATIONS)):

            loop_start = time.time()

            # -----------------------
            # ORDER BOOK
            # -----------------------

            depth_data = fetch_order_book()

            if depth_data:
                depth_buffer.append(depth_data)

            # -----------------------
            # TRADES
            # -----------------------

            trades, last_trade_id = fetch_trades(last_trade_id)

            trades_buffer.extend(trades)

            if len(trades_buffer) > 20000:
                trades_buffer = trades_buffer[-10000:]

            # -----------------------
            # SAVE BUFFERS
            # -----------------------

            if len(depth_buffer) >= BUFFER_SIZE:

                save_buffers(depth_buffer, trades_buffer, file_id)

                depth_buffer = []
                trades_buffer = []

                file_id += 1

            # -----------------------
            # MAINTAIN 1s SAMPLING
            # -----------------------

            elapsed = time.time() - loop_start
            time.sleep(max(0, 1 - elapsed))

    finally:

        if depth_buffer or trades_buffer:
            save_buffers(depth_buffer, trades_buffer, file_id)

        logger.info("Final buffers saved")


# --------------------------------------------------
# RUN SCRIPT
# --------------------------------------------------

if __name__ == "__main__":
    collect_data()