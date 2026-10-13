"""Utilities needed by the curated CLCRN training path."""
import logging
import sys
from pathlib import Path
import numpy as np


class StandardScaler:
    def __init__(self, mean, std):
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError('Training feature statistics must be finite with positive standard deviation.')
        self.mean = float(mean)
        self.std = float(std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def get_logger(log_dir, name='clcrn_revision'):
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f'{name}.{path.resolve()}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in [logging.FileHandler(path/'info.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
