from pathlib import Path

from binance.client import Client

import microstructure_alpha.data.dataset as m_alpha_data
from microstructure_alpha.utils.logger import setup_logger

logger = setup_logger(
    "C:\\Users\\jayod\\Documents\\Quant_Project\\microstructure-alpha-engine\\logs\\data_collection_raw_26_03_36_notebook.log"
)


SYMBOL = "BTCUSDT"
LOOP_ITERATIONS = 100000
BUFFER_SIZE = 500
ORDER_BOOK_DEPTH = 100

LOB_PATH = Path(
    "C:\\Users\\jayod\\Documents\\Quant_Project\\microstructure-alpha-engine\\data\\raw_26_03_36\\lob"
)
TRADE_PATH = Path(
    "C:\\Users\\jayod\\Documents\\Quant_Project\\microstructure-alpha-engine\\data\\raw_26_03_36\\trades"
)
client = Client(requests_params={"timeout": 5})


m_alpha_data.collect_data(
    SYMBOL, client, LOB_PATH, TRADE_PATH, LOOP_ITERATIONS, BUFFER_SIZE, ORDER_BOOK_DEPTH
)
