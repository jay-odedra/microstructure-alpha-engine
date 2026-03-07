import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]



def setup_logger(log_name):
    LOG_DIR = ROOT / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    
    log_file = LOG_DIR / f"{log_name}.log"
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
