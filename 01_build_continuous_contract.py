import yfinance as yf
import pandas as pd
import os
import glob
import datetime
import calendar
import argparse
import sys

# --- 設定 ---
RAW_DIR = "data_raw"
TICKER_ROOT = "NQ"
VALID_INTERVALS = ['2m', '5m', '15m', '30m', '60m', '1h']

# TOS 邏輯：在結算週的「週一」就換月
ROLLOVER_OFFSET_DAYS = 4 

# 目標時區
TARGET_TIMEZONE = 'Asia/Taipei'

def get_third_friday(year, month):
    """計算指定年月的第三個星期五 (CME 到期日)"""
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    month_cal = c.monthdatescalendar(year, month)
    fridays = [day for week in month_cal for day in week if day.weekday() == 4 and day.month == month]
    return fridays[2]

def get_rollover_date(year, month):
    """計算換月日 (TOS 風格：結算週的週一)"""
    expiration_date = get_third_friday(year, month)
    return expiration_date - datetime.timedelta(days=ROLLOVER_OFFSET_DAYS)

def get_contract_code(date_obj):
    """輸入日期，返回該日期當下「應該」交易的主力合約"""
    year = date_obj.year
    month = date_obj.month
    
    futures_months = [3, 6, 9, 12]
    futures_codes = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}
    
    target_year = year
    target_month = 0
    
    for m in futures_months:
        rollover_date = get_rollover_date(year, m)
        if date_obj.date() < rollover_date:
            target_month = m
            break
    
    if target_month == 0:
        target_month = 3
        target_year = year + 1
        
    code = futures_codes[target_month]
    contract_year_short = target_year % 100
    
    current_rollover_limit = get_rollover_date(target_year, target_month)
    return f"{TICKER_ROOT}{code}{contract_year_short}.CME", current_rollover_limit

def get_latest_local_file(interval, pattern_prefix="NQF"):
    if not os.path.exists(RAW_DIR): os.makedirs(RAW_DIR)
    pattern = os.path.join(RAW_DIR, f"{pattern_prefix}_{interval}_*.csv")
    files = glob.glob(pattern)
    if not files: return None
    return sorted(files)[-1]

def download_segment(ticker, interval, start_date, end_date):
    """下載並轉換為當地時區，同時移除 Adj Close"""
    print(f"    -> Downloading {ticker} ({interval}) from {start_date.date()} to {end_date.date()}...")
    try:
        # 下載
        df = yf.download(ticker, interval=interval, start=start_date, end=end_date + datetime.timedelta(days=1), progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # [FIX] 這裡加入移除 Adj Close 的邏輯
        if 'Adj Close' in df.columns:
            df.drop(columns=['Adj Close'], inplace=True)
        
        df = df.dropna()
        
        if df.empty:
            return df

        # --- 時區轉換邏輯 ---
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # 轉為目標時區 (Asia/Taipei) 並移除時區資訊
        df.index = df.index.tz_convert(TARGET_TIMEZONE).tz_localize(None)
        
        return df

    except Exception as e:
        print(f"    [!] Error downloading {ticker}: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=str, required=True, help=f"Choices: {VALID_INTERVALS}")
    args = parser.parse_args()
    
    if args.interval not in VALID_INTERVALS:
        print(f"[!] Interval '{args.interval}' not supported directly.")
        sys.exit(1)

    interval = args.interval
    print(f"--- Smart Data Updater | Target: {interval} | TZ: {TARGET_TIMEZONE} ---")
    
    local_file = get_latest_local_file(interval)
    df_old = pd.DataFrame()
    last_timestamp = None
    
    if local_file:
        print(f"[*] Found local file: {os.path.basename(local_file)}")
        df_old = pd.read_csv(local_file, index_col=0, parse_dates=True)
        
        # [FIX] 讀取舊檔時也順便清洗一下，防呆
        if 'Adj Close' in df_old.columns:
            df_old.drop(columns=['Adj Close'], inplace=True)
            
        df_old = df_old[df_old.index.notna()]
        last_timestamp = df_old.index[-1]
        print(f"[*] Last Data Point (Local): {last_timestamp}")
    else:
        print(f"[*] No local file. Starting fresh (Max 60 days).")
        last_timestamp = datetime.datetime.now() - datetime.timedelta(days=59)

    # 下載起點
    start_download_time = last_timestamp - datetime.timedelta(hours=2)
    current_time = datetime.datetime.now()

    if start_download_time >= current_time:
        print("[*] Data is already up to date.")
        return

    # 判斷合約
    start_contract, start_rollover_date = get_contract_code(start_download_time)
    end_contract, end_rollover_date = get_contract_code(current_time)

    df_new_segment = pd.DataFrame()

    if start_contract == end_contract:
        print(f"[*] Single Contract Mode: {start_contract}")
        df_new_segment = download_segment(start_contract, interval, start_download_time, current_time)
    else:
        print(f"[*] ROLLOVER DETECTED!")
        print(f"    Old Contract: {start_contract} (Ends on {start_rollover_date})")
        print(f"    New Contract: {end_contract}")
        
        cutoff_dt = datetime.datetime.combine(start_rollover_date, datetime.time(23, 59))
        
        if start_download_time < cutoff_dt:
            df_part1 = download_segment(start_contract, interval, start_download_time, cutoff_dt)
        else:
            df_part1 = pd.DataFrame()

        # 新合約的開始時間往前推 1 天，以捕捉週日夜盤
        bridge_start = datetime.datetime.combine(start_rollover_date - datetime.timedelta(days=1), datetime.time(0, 0))
        
        df_part2 = download_segment(end_contract, interval, bridge_start, current_time)
        
        df_new_segment = pd.concat([df_part1, df_part2])

    if df_new_segment.empty:
        print("[!] Download returned empty data.")
        return

    print(f"[*] Merging {len(df_new_segment)} new rows...")
    
    if not df_old.empty:
        # 確保兩邊都沒有 Adj Close 再合併，避免欄位不對齊
        if 'Adj Close' in df_old.columns: df_old.drop(columns=['Adj Close'], inplace=True)
        if 'Adj Close' in df_new_segment.columns: df_new_segment.drop(columns=['Adj Close'], inplace=True)
        
        df_combined = pd.concat([df_old, df_new_segment])
    else:
        df_combined = df_new_segment

    # 去重 & 排序
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined.sort_index(inplace=True)

    # 存檔
    s_date = df_combined.index.min().strftime("%Y%m%d")
    e_date = df_combined.index.max().strftime("%Y%m%d")
    new_filename = f"NQF_{interval}_{s_date}-{e_date}.csv"
    new_filepath = os.path.join(RAW_DIR, new_filename)

    df_combined.to_csv(new_filepath)
    print(f"[V] Saved to: {new_filename}")
    
    if local_file and local_file != new_filepath:
        print(f"    (Removing old file: {os.path.basename(local_file)})")
        os.remove(local_file)

if __name__ == "__main__":
    main()