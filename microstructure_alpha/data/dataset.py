from pathlib import Path
import time

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm


def fetch_order_book(symbol: str, client, levels: int = 100):

    for attempt in range(5):

        try:
            t0 = int(time.time() * 1000)  # latency

            timestamp = int(time.time() * 1000)

            depth = client.get_order_book(symbol=symbol, limit=levels)  # get order book

            t1 = int(time.time() * 1000)

            latency = (t1 - t0) / 2

            bids = np.array(depth["bids"], dtype=float).tolist()
            asks = np.array(depth["asks"], dtype=float).tolist()

            return {
                "symbol": symbol,
                "timestamp": timestamp,
                "latency": float(latency),
                "lastUpdateId": int(depth["lastUpdateId"]),
                "bids": bids,
                "asks": asks,
            }

        except Exception as e:

            logger.warning(f"Depth API error (attempt {attempt+1}/5): {e}")
            time.sleep(2)

    return None


def fetch_trades(symbol: str, client, last_trade_id: int):

    for attempt in range(5):

        try:

            trades = client.get_recent_trades(symbol=symbol, limit=1000)  # get trades

            trades = [
                trade for trade in trades if trade["id"] > last_trade_id
            ]  # make sure we arent same trades multiple times

            if trades:
                last_trade_id = trades[-1]["id"]

            return trades, last_trade_id

        except Exception as e:

            logger.warning(f"Trade API error (attempt {attempt+1}/5): {e}")
            time.sleep(2)

    return [], last_trade_id


def save_buffers(
    depth_buffer: list,
    trades_buffer: list,
    file_id: int,
    lob_path: Path,
    trade_path: Path,
):

    # lob_path.mkdir(parents=True, exist_ok=True)
    # trade_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(depth_buffer).to_parquet(
        lob_path / f"lob_data_raw_{file_id}.parquet",
        compression="snappy",
    )

    pd.DataFrame(trades_buffer).to_parquet(
        trade_path / f"trades_data_raw_{file_id}.parquet",
        compression="snappy",
    )

    logger.info(f"Saved batch {file_id}")


def collect_data(
    symbol: str,
    client,
    lob_path: Path,
    trade_path: Path,
    loop_iterations: int,
    buffer_size: int,
    levels: int = 100,
):

    logger.info("Starting data collection")

    depth_buffer = []
    trades_buffer = []

    file_id = 0
    last_trade_id = 0

    try:

        for _ in tqdm(range(loop_iterations)):

            loop_start = time.time()

            # ORDER BOOK

            depth_data = fetch_order_book(symbol, client, levels)

            if depth_data:
                depth_buffer.append(depth_data)

            # TRADES

            trades, last_trade_id = fetch_trades(symbol, client, last_trade_id)

            trades_buffer.extend(trades)

            if len(trades_buffer) > 20000:
                trades_buffer = trades_buffer[-10000:]

            # SAVE BUFFERS

            if len(depth_buffer) >= buffer_size:

                save_buffers(depth_buffer, trades_buffer, file_id, lob_path, trade_path)

                depth_buffer = []
                trades_buffer = []

                file_id += 1

            # KEEP SAMPLING AT 1s for different latencies

            elapsed = time.time() - loop_start
            time.sleep(max(0, 1 - elapsed))

    finally:

        if depth_buffer or trades_buffer:
            save_buffers(depth_buffer, trades_buffer, file_id, lob_path, trade_path)

        logger.info("Final buffers saved")
