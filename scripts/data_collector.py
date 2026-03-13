from microstructure_alpha.utils.logger import setup_logger
import microstructure_alpha.data.dataset as m_alpha_data

import pandas as pd
import numpy as np
from binance.client import Client
import time
from tqdm import tqdm
from pathlib import Path


logger = setup_logger(
    "E:\\Quant_Projects\\microstructure-alpha-engine\\microstructure-alpha-engine\\logs\\data_collection_notebook.log"
)


SYMBOL = "BTCUSDT"
LOOP_ITERATIONS = 100000
BUFFER_SIZE = 500
ORDER_BOOK_DEPTH = 100

LOB_PATH = Path(
    "E:\\Quant_Projects\\microstructure-alpha-engine\\microstructure-alpha-engine\\data\\raw\\lob"
)
TRADE_PATH = Path(
    "E:\\Quant_Projects\\microstructure-alpha-engine\\microstructure-alpha-engine\\data\\raw\\trades"
)
client = Client(requests_params={"timeout": 5})


m_alpha_data.collect_data(
    SYMBOL, client, LOB_PATH, TRADE_PATH, LOOP_ITERATIONS, BUFFER_SIZE, ORDER_BOOK_DEPTH
)
