from  pathlib import Path
import pandas as pd
import os


class TraceLoader:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        #class traceloader is used as a wrapper to keep path logic and I/) in one place
        #__init__ is used to stores the root directory as path object

    def load_metrics(self, provider: str, split: str) -> pd.DataFrame:
        """ Loads minute -level metrics for a given provider and split,
        Expected columns:
        [time, machine_id, cpu, mem , disk and net, throttle ,task running]"""
        f = self.base / "interim" / provider / f"metrics_{split}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}. Run scripts/make_windows.py first.")
        return pd.read_parquet(f)
    # the above function is used to read a parquet file containing machine performance metrics that is loaded into Pandas DataFrame
    #parquet file  is a open source, column oriented dat file format optimized for effiecient data storage and retrival, that is especially ised for big data  processing

    def load_events(selfself, provider: str, split: str) -> pd.DataFrame:
        """this loads per minute event counts
        """
        f = self.base / "interim" / provider / f"events_{split}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}. Run scripts/make_windows.py first.")
        return pd.read_parquet(f)
