import gc
import glob

from loguru import logger
import pandas as pd
from tqdm import tqdm

from microstructure_alpha.data.alignment import process_lob, process_trade


def run_batch_build_dataset_pipeline(lob_path: str, trade_path: str, save_path: str):
    files_trade = sorted(glob.glob(trade_path + "trades_data_raw_*"))
    files_lob = sorted(glob.glob(lob_path + "lob_data_raw_*"))

    logger.info("Starting batch pipeline")
    logger.info(f"Found {len(files_trade)} trade files")
    logger.info(f"Found {len(files_lob)} lob files")

    total_batches = min(len(files_trade), len(files_lob))
    logger.info(f"Processing {total_batches} batches")

    for i, (trade_file, lob_file) in tqdm(
        enumerate(zip(files_trade, files_lob)),
        total=total_batches,
        desc="Processing batches",
    ):
        lob_df = pd.read_parquet(lob_file)
        trade_df = pd.read_parquet(trade_file)
        processed_lob = process_lob(lob_df)
        processed_trade = process_trade(trade_df, processed_lob)

        processed_lob.to_parquet(
            save_path + f"lob_interim_{i}.parquet", compression="snappy"
        )
        processed_trade.to_parquet(
            save_path + f"trade_interim_{i}.parquet", compression="snappy"
        )

        del lob_df
        del trade_df
        del processed_lob
        del processed_trade

        gc.collect()

    logger.info("Batch pipeline completed successfully")
    return total_batches
