import pandas as pd
import pandas_ta as ta
import os
import glob
import sys
import argparse

# --- 設定 (Default Settings) ---
RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"
BASE_INT = "2m"  # 主結構
REF_INT = "5m"   # 參考資料

def get_latest_file(ticker_pattern: str, interval: str):
    """抓取資料夾中該時框最新的 csv"""
    pattern = os.path.join(RAW_DIR, f"{ticker_pattern}_{interval}_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return sorted(files)[-1]

# ==========================================
# Mode 1: Dual Trend (Legacy Logic)
# 邏輯：只計算 5m MA20，並將其 Merge 進 2m
# ==========================================
def process_dual_trend(df_base, df_ref, safe_ticker):
    print(f"[*] Mode: Dual Trend (Calculating 5m MA20 + Merge)")
    
    col_ma = f"MA20_{REF_INT}"
    df_ref[col_ma] = ta.sma(df_ref['Close'], length=20)
    
    # 只取 MA 欄位並 Shift 1
    df_to_merge = df_ref[[col_ma]].shift(1)
    
    df_final = pd.merge_asof(
        df_base.sort_index(),
        df_to_merge.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    return df_final, "w_5mMA"

# ==========================================
# Mode 2: Strategy C (Resample + Vis Columns)
# 邏輯：由 2m 重取樣 5m，並保留「策略用(Shift)」與「畫圖用(Original)」兩套數據
# ==========================================
def process_strategy_c(df_base, df_ref_ignored, safe_ticker):
    print(f"[*] Mode: Strategy C (Resampling 2m -> 5m with Vis Columns)")
    
    # 1. 執行重取樣 (2m -> 5m)
    # label='left', closed='left': 20:00 代表 20:00~20:05
    resample_rule = '5min'
    logic = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    df_5m_raw = df_base.resample(resample_rule, label='left', closed='left').agg(logic)
    df_5m_raw.dropna(inplace=True)
    
    # 2. 準備兩套數據
    
    # [Set A] 策略用 (Strategy Data) - 必須 Shift 1 防偷看
    # 命名：Open_5m, Close_5m...
    df_strat = df_5m_raw.copy().shift(1)
    df_strat.columns = [f"{c}_5m" for c in df_strat.columns]
    
    # [Set B] 視覺用 (Visual Data) - 保持原始時間，不 Shift
    # 命名：Open_5m_vis, Close_5m_vis...
    df_vis = df_5m_raw.copy() # No shift
    df_vis.columns = [f"{c}_5m_vis" for c in df_vis.columns]
    
    # 3. 合併這兩套數據 (先水平串接，因為它們索引相同)
    # 注意：df_strat 因為 shift 會有第一筆 NaN，concat 會自動處理
    df_merge_source = pd.concat([df_strat, df_vis], axis=1)
    
    # 4. Merge 進 2m 主數據 (Merge As-Of)
    # 20:00 (2m) 會對應到 20:00 (5m)
    # -> 取得 Set A (19:55 的數據) -> 安全
    # -> 取得 Set B (20:00 的數據) -> 畫圖用，剛好畫在 20:00 開始的位置 -> 直觀
    df_final = pd.merge_asof(
        df_base.sort_index(),
        df_merge_source.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )
    
    return df_final, "StrategyC_Resampled_Dual"

# ==========================================
# Main Controller
# ==========================================
# [請用這段覆蓋原本的 main 函數]
def main():
    parser = argparse.ArgumentParser(description="NQ Futures ETL Pipeline")
    parser.add_argument("--mode", type=str, choices=['dual_trend', 'strategy_c'], default='dual_trend', 
                        help="Processing Mode")
    parser.add_argument("--ticker", type=str, default="NQ=F", help="Ticker symbol")
    
    args = parser.parse_args()
    
    ticker_symbol = args.ticker
    safe_ticker = ticker_symbol.replace("=", "")
    
    print(f"--- Starting ETL Pipeline for {ticker_symbol} ---")

    # 1. 讀取 Base (2m)
    base_file = get_latest_file(safe_ticker, BASE_INT)
    if not base_file:
        print(f"[!] Error: Base data ({BASE_INT}) not found in {RAW_DIR}")
        sys.exit(1)
    
    print(f"    -> Loading Base: {os.path.basename(base_file)}")
    df_base = pd.read_csv(base_file, index_col=0, parse_dates=True)
    # [FIX] 清洗 Base 索引：移除任何解析失敗的日期 (NaT)
    df_base = df_base[df_base.index.notna()]
    
    # 讀取 Ref (僅 Dual Trend 需要)
    df_ref = None
    if args.mode == 'dual_trend':
        ref_file = get_latest_file(safe_ticker, REF_INT)
        if not ref_file:
            print(f"[!] Error: Ref data ({REF_INT}) not found.")
            sys.exit(1)
        print(f"    -> Loading Ref:  {os.path.basename(ref_file)}")
        df_ref = pd.read_csv(ref_file, index_col=0, parse_dates=True)
        # [FIX] 清洗 Ref 索引：這是解決報錯的關鍵
        df_ref = df_ref[df_ref.index.notna()]

    # 2. 處理
    df_final = None
    filename_tag = ""

    if args.mode == 'dual_trend':
        df_final, filename_tag = process_dual_trend(df_base, df_ref, safe_ticker)
    elif args.mode == 'strategy_c':
        df_final, filename_tag = process_strategy_c(df_base, None, safe_ticker)

    # 3. 存檔
    if df_final is not None:
        df_final.dropna(inplace=True)
        
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        s_date = df_final.index.min().strftime("%Y%m%d")
        e_date = df_final.index.max().strftime("%Y%m%d")
        
        out_name = f"{safe_ticker}_Backtest_{BASE_INT}_{filename_tag}_{s_date}-{e_date}.csv"
        out_path = os.path.join(PROCESSED_DIR, out_name)
        
        df_final.to_csv(out_path)
        print(f"[V] ETL Complete.")
        print(f"    Mode: {args.mode}")
        print(f"    Saved to: {out_path}")
        print(f"    Columns ({len(df_final.columns)}): {list(df_final.columns)}")
    else:
        print("[!] Error: ETL processing failed.")

if __name__ == "__main__":
    main()