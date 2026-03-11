
import pandas as pd
import numpy as np
from binance.client import Client
import time
import black
from pympler import asizeof

from tqdm import tqdm

pd.set_option("display.max_columns", None)

client = Client()
depth = client.get_order_book(symbol="BTCUSDT")
trades = client.get_recent_trades(symbol="BTCUSDT", limit=1000)




depth_buffer = []
trades_buffer = []
last_trade_id = 0
buffer_size = 500
for i in tqdm(range(50000)):
    timestamp = int(time.time() * 1000)
    
    t0 = int(time.time() * 1000)

    depth = client.get_order_book(symbol="BTCUSDT", limit=100)
    
    t1 = int(time.time() * 1000)
    latency = (t1-t0) / 2
    
    bids = np.array(depth["bids"], dtype=float)
    asks = np.array(depth["asks"], dtype=float)
    depth_buffer.append(
        {
            "timestamp": int(timestamp),
            "latency": float(latency),
            "lastUpdateId": int(depth["lastUpdateId"]),
            "bids": bids,
            "asks": asks,
        }
    )
    trades = client.get_recent_trades(symbol="BTCUSDT", limit=1000)
    
    trades = [trade for trade in trades if trade["id"]>last_trade_id]
    if len(trades) > 0:
        last_trade_id = trades[-1]["id"]
    trades_buffer.extend(trades)
    
    if len(depth_buffer) >= buffer_size:    
        pd.DataFrame(depth_buffer).to_parquet(f"E:\\Quant_Projects\\microstructure-alpha-engine\\microstructure-alpha-engine\\data\\raw\\lob\\lob_data_raw_{i}.parquet",compression="snappy")
        pd.DataFrame(trades_buffer).to_parquet(f"E:\\Quant_Projects\\microstructure-alpha-engine\\microstructure-alpha-engine\\data\\raw\\trades\\trades_data_raw_{i}.parquet",compression="snappy")
        depth_buffer = []
        trades_buffer = []
    time.sleep(1)