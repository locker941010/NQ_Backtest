import pandas as pd
import numpy as np
from backtesting import Strategy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class BaseStrategy(Strategy):
    """
    Base Strategy: Pure Manual Mode (Fixed Entry Time)
    """
    
    # --- Default Parameters ---
    p_sl_points = 16.0          
    p_tp_points = 20.0          
    p_runner_sl_offset = 7.0    
    p_point_value = 2.0 
    
    trade_size = 2              
    runner_size = 1             
    exit_calc_mode = 'FIXED_RUNNER' 

    def init(self):
        self.manual_trades: List[Dict[str, Any]] = []
        self.ideal_entry_price: Optional[float] = None
        self.entry_time_marker: Optional[pd.Timestamp] = None
        self.partial_exit_done: bool = False 
        self.log_file = None 

    def log(self, message: str):
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{message}\n")
            except: pass

    @staticmethod
    def round_to_tick(price: float, tick_size: float = 0.25) -> float:
        return round(round(price / tick_size) * tick_size, 2)

    def _calc_exit_qty(self) -> int:
        qty = 0
        if self.exit_calc_mode == 'HALF_PLUS_ONE':
            qty = (self.trade_size // 2) + 1
        else:
            qty = self.trade_size - self.runner_size
        return max(0, min(qty, self.trade_size))

    def check_intra_bar_outcome(self, open_p, high, low, close, entry_price, sl_price, tp_price, current_time, direction) -> str:
        is_entry_hit = False
        if direction == 1: 
            if low <= entry_price: is_entry_hit = True
        else: 
            if high >= entry_price: is_entry_hit = True
        if not is_entry_hit: return 'HOLD'

        if direction == -1: # Short
            is_sl_hit = high >= sl_price
            is_tp_hit = low <= tp_price
            is_close_win = close <= tp_price
        else: # Long
            is_sl_hit = low <= sl_price
            is_tp_hit = high >= tp_price
            is_close_win = close >= tp_price

        if is_sl_hit and not is_tp_hit: return 'SL'
        if not is_sl_hit and is_close_win: return 'TP'
        if not is_sl_hit and not is_tp_hit: return 'HOLD'

        dist_open_to_high = abs(open_p - high)
        dist_open_to_low = abs(open_p - low)
        dist_open_to_entry = abs(open_p - entry_price)

        if is_sl_hit and is_tp_hit:
            if direction == -1: return 'SL' if dist_open_to_high < dist_open_to_low else 'TP'
            else: return 'SL' if dist_open_to_low < dist_open_to_high else 'TP'

        if not is_sl_hit and is_tp_hit and not is_close_win:
            is_open_filled = (open_p >= entry_price) if direction == -1 else (open_p <= entry_price)
            if is_open_filled: return 'TP'
            dist_to_tp_side = dist_open_to_low if direction == -1 else dist_open_to_high
            if dist_open_to_entry < dist_to_tp_side: return 'TP'
            else: return 'HOLD'
        return 'HOLD'

    def record_trade(self, entry_time, exit_time, entry_price, exit_price, pnl, size, outcome_type):
        qty = abs(size)
        total_points = pnl * qty
        point_value = getattr(self, 'p_point_value', 1.0)
        total_usd = total_points * point_value
        
        self.manual_trades.append({
            'EntryTime': entry_time,
            'ExitTime': exit_time,
            'EntryPrice': self.round_to_tick(entry_price),
            'ExitPrice': self.round_to_tick(exit_price),
            'PnL': total_points,
            'PnL_Per_Unit': pnl,
            'PnL_USD': total_usd,
            'Size': size,
            'Type': outcome_type
        })

    # ============================================================
    # [CORE] Pure Manual Exit Logic (Fixed Entry Time)
    # ============================================================
    def check_and_execute_scale_out(self, ma_price: float):
        if not self.position: return

        # 取得當前持倉資訊
        current_trade = self.trades[-1]
        direction = 1 if self.position.is_long else -1
        
        # [CRITICAL FIX]
        # 強制使用引擎回報的 entry_time (成交 K 棒時間)，而不是掛單時的時間。
        # 這樣可以修正 Report 提早一根 K 棒的問題。
        self.ideal_entry_price = current_trade.entry_price
        self.entry_time_marker = current_trade.entry_time

        # --- Phase 1: 尚未分批出場 (檢查 TP1 和 初始 SL) ---
        if not self.partial_exit_done:
            if direction == -1: # Short
                tp1_price = self.ideal_entry_price - self.p_tp_points
                sl_price = self.ideal_entry_price + self.p_sl_points
            else: # Long
                tp1_price = self.ideal_entry_price + self.p_tp_points
                sl_price = self.ideal_entry_price - self.p_sl_points
            
            # 檢查是否觸發
            hit_sl = (self.data.High[-1] >= sl_price) if direction == -1 else (self.data.Low[-1] <= sl_price)
            hit_tp = (self.data.Low[-1] <= tp1_price) if direction == -1 else (self.data.High[-1] >= tp1_price)
            
            outcome = 'HOLD'
            if hit_sl and hit_tp: outcome = 'SL' # 保守判定
            elif hit_sl: outcome = 'SL'
            elif hit_tp: outcome = 'TP'

            if outcome == 'TP':
                exit_qty = self._calc_exit_qty()
                
                # [MANUAL ACTION] 平掉部分倉位
                if direction == -1: self.buy(size=exit_qty)
                else: self.sell(size=exit_qty)
                
                self.record_trade(
                    entry_time=self.entry_time_marker,
                    exit_time=self.data.index[-1],
                    entry_price=self.ideal_entry_price,
                    exit_price=tp1_price,
                    pnl=self.p_tp_points,
                    size=direction * exit_qty,
                    outcome_type='TP1 (Partial)'
                )
                self.partial_exit_done = True
                
                # 如果是全出 (例如 trade_size=1)，則重置
                if exit_qty == self.trade_size:
                    self.ideal_entry_price = None
                    return

            elif outcome == 'SL':
                # [MANUAL ACTION] 全倉停損
                self.position.close()
                
                self.record_trade(
                    entry_time=self.entry_time_marker,
                    exit_time=self.data.index[-1],
                    entry_price=self.ideal_entry_price,
                    exit_price=sl_price,
                    pnl=-self.p_sl_points,
                    size=direction * self.trade_size,
                    outcome_type='SL (Full)'
                )
                self.ideal_entry_price = None
                return

        # --- Phase 2: 已經分批出場 (檢查 Runner SL 和 MA Exit) ---
        else:
            if direction == -1: runner_sl = self.ideal_entry_price + self.p_runner_sl_offset
            else: runner_sl = self.ideal_entry_price - self.p_runner_sl_offset
            
            ma_exit = False
            if direction == -1: 
                if self.data.Close[-1] > ma_price: ma_exit = True
            else: 
                if self.data.Close[-1] < ma_price: ma_exit = True
            
            hit_runner_sl = (self.data.High[-1] >= runner_sl) if direction == -1 else (self.data.Low[-1] <= runner_sl)
            
            final_exit = False
            exit_type = ''
            exit_px = 0.0
            pnl_pts = 0.0

            if hit_runner_sl:
                final_exit = True
                exit_type = 'Runner SL'
                exit_px = runner_sl
                pnl_pts = -self.p_runner_sl_offset
            elif ma_exit:
                final_exit = True
                exit_type = 'MA Exit'
                exit_px = self.data.Close[-1]
                if direction == -1: pnl_pts = self.ideal_entry_price - exit_px
                else: pnl_pts = exit_px - self.ideal_entry_price

            if final_exit:
                # [MANUAL ACTION] 平掉剩餘倉位
                self.position.close()
                remaining_size = self.position.size 
                
                self.record_trade(
                    entry_time=self.entry_time_marker,
                    exit_time=self.data.index[-1],
                    entry_price=self.ideal_entry_price,
                    exit_price=exit_px,
                    pnl=pnl_pts,
                    size=remaining_size,
                    outcome_type=exit_type
                )
                self.ideal_entry_price = None
                self.partial_exit_done = False

    # ============================================================
    # [MODIFIED] Entry Logic (No Engine SL/TP)
    # ============================================================
    def execute_entry_with_intrabar_check(self, direction: int, raw_limit_price: float, sl_points: float, tp_points: float, trade_size: int):
        current_time = self.data.index[-1]
        limit_price = self.round_to_tick(raw_limit_price)

        if direction == 1:
            target_sl = limit_price - sl_points
            target_tp = limit_price + tp_points
        else:
            target_sl = limit_price + sl_points
            target_tp = limit_price - tp_points

        # 1. Intra-bar 模擬 (檢查是否在進場這根 K 棒就直接出場)
        outcome = self.check_intra_bar_outcome(
            self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1],
            limit_price, target_sl, target_tp, current_time, direction
        )

        self.partial_exit_done = False

        if outcome == 'SL' or outcome == 'TP':
            # 進場當下即出場，直接紀錄，不掛單
            pnl_points_per_unit = -sl_points if outcome == 'SL' else tp_points
            exit_px = target_sl if outcome == 'SL' else target_tp
            
            self.record_trade(
                entry_time=current_time,
                exit_time=current_time,
                entry_price=limit_price,
                exit_price=exit_px,
                pnl=pnl_points_per_unit,
                size=direction * trade_size,
                outcome_type=f"Intra-Bar {outcome}"
            )
            return True

        elif outcome == 'HOLD':
            # 這裡設定的 entry_time_marker 只是暫時的 (掛單時間)
            # 真正成交後，check_and_execute_scale_out 會用 current_trade.entry_time 覆蓋它
            self.ideal_entry_price = limit_price
            self.entry_time_marker = current_time
            
            if direction == 1:
                self.buy(limit=limit_price, size=trade_size) 
                self.log(f"[ORDER PLACED] LONG Limit @ {limit_price} (Pure Manual)")
            else:
                self.sell(limit=limit_price, size=trade_size)
                self.log(f"[ORDER PLACED] SHORT Limit @ {limit_price} (Pure Manual)")
            
            return False

        return False

    @abstractmethod
    def next(self):
        pass