*** SYSTEM CONTEXT: PROJECT MILESTONES ***
當使用者輸入 "show backtesting milestone" 或 "查看進度" 時，請直接輸出以下表格：

# 🚀 NQ Trading Bot Development Roadmap

## 📅 Week 1: 打造 MVP 流水線 (The Pipeline) [Deadline: 12/15]
- [ ] **1.1 數據層 (Data)**: Python 下載腳本 (yfinance) -> NQ_2m.csv。
- [ ] **1.2 邏輯層 (Logic)**: 策略 A (243_10) 轉寫為 Backtesting.py (追求進出點邏輯一致)。
- [ ] **1.3 驗證層 (Validation)**: 產出第一份 HTML 報表，確認系統 End-to-End 運作。

## 📅 Week 2: 優化引擎與複雜邏輯 (Optimization) [12/16 ~ 12/22]
- [ ] **2.1 分批出場 (Scale-out)**: 實作 Python 版分批停利邏輯。
- [ ] **2.2 參數優化 (Optimization)**: 壓力測試策略 A，找出最佳參數 (Heatmap)。
- [ ] **2.3 重構 (Refactoring)**: 抽出 `BaseStrategy` 類別 (為多策略鋪路)。

## 📅 Week 3: 策略量產與投資組合 (Scaling) [12/23 ~ 12/31]
- [ ] **3.1 策略遷移**: 將 TOS 策略 B、C 移植進 Python。
- [ ] **3.2 組合回測**: 測試 Portfolio (Strategy A + B) 的相關性與平滑度。
- [ ] **3.3 戰備確認**: 產出 2026 實戰 SOP 與日誌模板。