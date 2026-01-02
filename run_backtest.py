import argparse
import sys
import os
import importlib
import inspect
import glob
import pandas as pd
from backtesting import Backtest

# 確保可以 import 當前目錄下的模組
sys.path.append(os.getcwd())

# 引入 Core 模組 (假設這些檔案將在下一步建立，這裡先做好接口)
try:
    from core.base_strategy import BaseStrategy
    # 暫時使用 Stub，待您完成 data_loader.py 與 analyzer.py 後即可完全運作
    from core import data_loader
    from core import analyzer
except ImportError as e:
    print(f"[Warning] Core modules not fully implemented yet: {e}")
    # 為了讓您能先測試 run_backtest 的邏輯，這裡定義簡單的 Mock
    if 'data_loader' not in sys.modules:
        class MockDataLoader:
            @staticmethod
            def load_latest_data(data_dir="data_processed"):
                pattern = os.path.join(data_dir, "*Backtest*.csv")
                files = glob.glob(pattern)
                if not files: raise FileNotFoundError("No data found.")
                data_file = sorted(files)[-1]
                print(f"[*] Loading data: {data_file}")
                return pd.read_csv(data_file, index_col=0, parse_dates=True)
        data_loader = MockDataLoader()
    
    if 'analyzer' not in sys.modules:
        class MockAnalyzer:
            @staticmethod
            def generate_report(strategy_instance, df_data, initial_cash=100000):
                print("[*] (Stub) Generating Report & Charts...")
                print(f"    Trades recorded: {len(strategy_instance.manual_trades)}")
        analyzer = MockAnalyzer()

def load_strategy_class(strategy_name: str):
    """
    Dynamically load a strategy class from the strategies/ folder.
    """
    try:
        # 嘗試 import strategies.{strategy_name}
        module_path = f"strategies.{strategy_name}"
        module = importlib.import_module(module_path)
        
        # 尋找該模組中繼承自 BaseStrategy 的類別
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                return obj
        
        raise ValueError(f"No class inheriting BaseStrategy found in {module_path}")
        
    except ModuleNotFoundError:
        print(f"[Error] Strategy file 'strategies/{strategy_name}.py' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to load strategy: {e}")
        sys.exit(1)

def scan_strategies():
    """
    Scan 'strategies/' directory for available strategy files.
    """
    strategy_files = glob.glob("strategies/*.py")
    strategies = []
    for f in strategy_files:
        basename = os.path.basename(f)
        if basename == "__init__.py": continue
        strategies.append(basename.replace(".py", ""))
    return strategies

def main():
    parser = argparse.ArgumentParser(description="NQ Futures Backtesting CLI")
    parser.add_argument("--strategy", type=str, help="Name of the strategy file (without .py), e.g., nq_dual_trend")
    parser.add_argument("--cash", type=float, default=100000, help="Initial cash (default: 100000)")
    
    args = parser.parse_args()
    
    # 1. 決定要執行的策略
    target_strategies = []
    if args.strategy:
        target_strategies = [args.strategy]
    else:
        print("[*] No strategy specified. Scanning 'strategies/' folder...")
        target_strategies = scan_strategies()
        if not target_strategies:
            print("[!] No strategies found in 'strategies/' folder.")
            sys.exit(0)
            
    # 2. 讀取數據 (只讀一次，假設所有策略用同一份數據)
    try:
        df = data_loader.load_latest_data()
    except Exception as e:
        print(f"[Error] Data loading failed: {e}")
        sys.exit(1)

    # 3. 依序執行策略
    for strat_name in target_strategies:
        print(f"\n{'='*60}")
        print(f"[*] Running Strategy: {strat_name}")
        print(f"{'='*60}")
        
        StrategyClass = load_strategy_class(strat_name)
        
        # 初始化 Backtest
        bt = Backtest(df, StrategyClass, cash=args.cash, commission=0, margin=0.05)
        
        # 執行回測
        stats = bt.run()
        
        # 獲取策略實例 (內含 manual_trades)
        strategy_instance = stats['_strategy']
        
        # 4. 呼叫 Analyzer 產出報表 (CSV + Plotly)
        # Analyzer 內部應包含: 數據清洗(Rounding/Direction), Console Print, CSV Save, Plotly Chart
        analyzer.generate_report(strategy_instance, df, initial_cash=args.cash)

if __name__ == "__main__":
    main()