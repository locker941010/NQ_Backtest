import argparse
import yfinance as yf
import pandas as pd
import os
import sys
import glob
from datetime import datetime, timedelta

# 預設儲存路徑
OUTPUT_DIR = "data_raw"

def get_optimal_period(interval: str) -> str:
    """無舊檔時，根據 yfinance 限制決定最大下載範圍"""
    interval = interval.lower()
    if any(x in interval for x in ['m']) and '60m' not in interval:
        return "59d" # 1m, 2m, 5m, 15m, 30m
    elif interval in ['60m', '1h', 'h1', '4h', 'h4']:
        return "730d" # Hourly
    else:
        return "max"  # Daily+

def get_recovery_period(last_date: datetime, interval: str) -> str:
    """
    有舊檔時，根據最後一筆資料的時間差，決定要下載的範圍以填補 Gap。
    """
    now = datetime.now()
    delta = now - last_date
    days_diff = delta.days
    
    interval = interval.lower()
    is_intraday = any(x in interval for x in ['m']) and '60m' not in interval

    print(f"[*] Data Gap Detected: {days_diff} days (Last: {last_date.strftime('%Y-%m-%d')})")

    # yfinance valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    if days_diff < 4:
        return "5d"   # 正常頻率更新，取5天確保重疊
    elif days_diff < 28:
        return "1mo"  # 差距在一個月內
    elif days_diff < 85:
        if is_intraday:
            print("[!] Warning: Gap > 1 month. Attempting max intraday (59d). Some data might be lost due to API limits.")
            return "59d"
        else:
            return "3mo"
    else:
        if is_intraday:
            print("[!] Critical: Gap is too large for 1m-30m data (limit ~60 days). Recent data will be fetched, but a gap is inevitable.")
            return "59d"
        return "1y"

def parse_interval(input_val: str) -> str:
    input_val = input_val.lower().strip()
    if input_val.isdigit():
        return f"{input_val}m"
    if input_val == "h1": return "1h"
    if input_val == "h4": return "4h"
    return input_val

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.index.tz is not None:
        df.index = df.index.tz_convert('America/New_York').tz_localize(None)
    
    df.sort_index(inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Download financial data (Smart Incremental).")
    parser.add_argument("-t", "--timeframe", type=str, required=True, help="Timeframe (e.g., 2, 5, 15).")
    parser.add_argument("--ticker", type=str, default="NQ=F", help="Ticker (default: NQ=F)")

    args = parser.parse_args()
    
    ticker = args.ticker
    interval = parse_interval(args.timeframe)
    safe_ticker = ticker.replace("=", "")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    search_pattern = os.path.join(OUTPUT_DIR, f"{safe_ticker}_{interval}_*.csv")
    existing_files = glob.glob(search_pattern)
    
    old_df = pd.DataFrame()
    old_filename = None
    period = ""
    
    # 1. 決定下載策略
    if existing_files:
        old_filename = max(existing_files, key=os.path.getmtime)
        print(f"[*] Found existing file: {old_filename}")
        
        try:
            old_df = pd.read_csv(old_filename, index_col=0, parse_dates=True)
            if not old_df.empty:
                last_dt = old_df.index[-1]
                # 確保 last_dt 沒有時區以便計算 (通常 read_csv 讀回來是 naive)
                if last_dt.tzinfo is not None:
                    last_dt = last_dt.tz_localize(None)
                
                # 動態決定下載範圍
                period = get_recovery_period(last_dt, interval)
                print(f"[*] Mode: Incremental Update (Fetch last {period})")
            else:
                period = get_optimal_period(interval)
        except Exception as e:
            print(f"[!] Error reading old file: {e}. Fallback to full download.")
            period = get_optimal_period(interval)
    else:
        period = get_optimal_period(interval)
        print(f"[*] Mode: Full Download (Period: {period})")

    # 2. 下載與處理
    try:
        print(f"[*] Downloading {ticker} [{interval}]...")
        new_df = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if new_df.empty:
            print(f"[!] Warning: No new data found for {ticker}.")
            if old_df.empty: sys.exit(1)
            final_df = old_df
        else:
            new_df = process_dataframe(new_df)
            
            if not old_df.empty:
                # 合併
                final_df = pd.concat([old_df, new_df])
                # 去重關鍵：保留最後出現的 (新數據覆蓋舊數據)
                final_df = final_df[~final_df.index.duplicated(keep='last')]
                final_df.sort_index(inplace=True)
                
                # 簡單驗證：檢查 Gap
                if len(final_df) > 1:
                    time_diffs = final_df.index.to_series().diff()
                    # 如果分鐘線出現超過 4 天的 Gap (考慮週末)，發出警告
                    max_gap = time_diffs.max()
                    if max_gap > timedelta(days=4):
                         print(f"[!] Warning: Significant data gap detected in final dataframe: {max_gap}")
                
                print(f"    Merged: {len(old_df)} (Old) + {len(new_df)} (New) -> {len(final_df)} (Total)")
            else:
                final_df = new_df

        # 3. 存檔
        if final_df.empty:
            print("[!] Final dataframe is empty. Nothing to save.")
            sys.exit(1)

        start_date = final_df.index.min().strftime("%Y%m%d")
        end_date = final_df.index.max().strftime("%Y%m%d")
        
        new_filename = f"{safe_ticker}_{interval}_{start_date}-{end_date}.csv"
        new_filepath = os.path.join(OUTPUT_DIR, new_filename)
        
        if old_filename and old_filename != new_filepath:
            os.remove(old_filename)
            print(f"[-] Removed old file: {old_filename}")
            
        final_df.to_csv(new_filepath)
        print(f"[V] Success: Saved to {new_filepath}")
        
    except Exception as e:
        print(f"[X] Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()