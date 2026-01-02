import os
import glob
import pandas as pd

def load_latest_data(data_dir="data_processed"):
    """
    Loads the latest CSV file containing 'Backtest' in its name from the specified directory.
    """
    pattern = os.path.join(data_dir, "*Backtest*.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir} matching pattern '*Backtest*.csv'")
    
    data_file = sorted(files)[-1]
    print(f"[*] Loading data: {data_file}")
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    return df