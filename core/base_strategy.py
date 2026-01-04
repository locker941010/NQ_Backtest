import pandas as pd
import numpy as np
from backtesting import Strategy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class BaseStrategy(Strategy):
    """
    Base Strategy: Pure Manual Mode (No Cool-down)
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
        self.manually_closed_trades = set()
        # [REMOVED] Cool-down counter

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


        # Reverse Fill Protection
        #if direction == -1 and open_p < entry_price:
        #    if is_tp_hit and not is_close_win: is_tp_hit = False 
        #if direction == 1 and open_p > entry_price:
        #    if is_tp_hit and not is_close_win: is_tp_hit = False

        if is_sl_hit and not is_tp_hit: return 'SL'
        if not is_sl_hit and is_tp_hit: return 'TP'
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
    # [OVERRIDE] Scale-out Logic for NQ 5Eat 2Eat
    # Logic: 
    # 1. TP Fixed First (Must happen before MA exit).
    # 2. Runner Exit: Pullback to MA20 (Long: Low<=MA, Short: High>=MA).
    # ============================================================
    def check_and_execute_scale_out_5eat2eat(self, ma_price: float):
        if not self.position: return

        current_trade = self.trades[-1]
        direction = 1 if self.position.is_long else -1
        
        # 確保有紀錄理想進場價 (如果是市價單進場，這裡補救一下)
        if self.ideal_entry_price is None:
            self.ideal_entry_price = current_trade.entry_price
            self.entry_time_marker = current_trade.entry_time

        current_time = self.data.index[-1]
        is_entry_bar = (current_trade.entry_time == current_time)

        tp1_triggered = False
        runner_triggered = False
        remaining_qty = abs(self.position.size)
        
        # --- Phase 1: Fixed TP (Priority 1) ---
        # 必須先滿足這個，才會開啟 MA 離場的權限
        if not self.partial_exit_done:
            # 設定 TP1 和 SL 價格
            if direction == -1: # Short
                tp1_price = self.ideal_entry_price - self.p_tp_points
                sl_price = self.ideal_entry_price + self.p_sl_points
            else: # Long
                tp1_price = self.ideal_entry_price + self.p_tp_points
                sl_price = self.ideal_entry_price - self.p_sl_points
            
            # 檢查 Intra-bar 結果
            outcome = 'HOLD'
            if is_entry_bar:
                outcome = self.check_intra_bar_outcome(
                    self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1],
                    self.ideal_entry_price, sl_price, tp1_price, self.data.index[-1], direction
                )
            else:
                # 非進場 K 棒，直接看 High/Low
                hit_sl = (self.data.High[-1] >= sl_price) if direction == -1 else (self.data.Low[-1] <= sl_price)
                hit_tp = (self.data.Low[-1] <= tp1_price) if direction == -1 else (self.data.High[-1] >= tp1_price)
                if hit_sl: outcome = 'SL'
                elif hit_tp: outcome = 'TP'

            # 執行 Phase 1 離場
            if outcome == 'TP':
                exit_qty = self._calc_exit_qty()
                if direction == -1: self.buy(size=exit_qty)
                else: self.sell(size=exit_qty)
                
                self.record_trade(
                    entry_time=self.entry_time_marker,
                    exit_time=current_time,
                    entry_price=self.ideal_entry_price,
                    exit_price=tp1_price,
                    pnl=self.p_tp_points,
                    size=direction * exit_qty,
                    outcome_type='TP1 (Fixed)'
                )
                self.manually_closed_trades.add((self.entry_time_marker, current_time))
                self.partial_exit_done = True
                tp1_triggered = True
                remaining_qty -= exit_qty
                self.log(f"  [EXEC] TP1 Fixed Hit. Closed {exit_qty}. Remain {remaining_qty}.")

            elif outcome == 'SL':
                self.position.close()
                self.record_trade(
                    entry_time=self.entry_time_marker,
                    exit_time=current_time,
                    entry_price=self.ideal_entry_price,
                    exit_price=sl_price,
                    pnl=-self.p_sl_points,
                    size=direction * self.trade_size,
                    outcome_type='SL (Full)'
                )
                self.manually_closed_trades.add((self.entry_time_marker, current_time))
                self.ideal_entry_price = None
                return # 全部位出場，直接結束

        # --- Phase 2: Runner MA Touch Exit (Priority 2) ---
        # 只有在 TP1 已經達成後 (partial_exit_done is True) 才執行
        if self.partial_exit_done and remaining_qty > 0:
            
            # 1. 計算 Runner 的硬 SL (通常是保本或微幅虧損)
            if direction == -1: runner_sl = self.ideal_entry_price + self.p_runner_sl_offset
            else: runner_sl = self.ideal_entry_price - self.p_runner_sl_offset
            
            # 2. 檢查 MA Touch (拉回碰觸均線)
            # Long: 價格回落碰到 MA (Low <= MA)
            # Short: 價格反彈碰到 MA (High >= MA)
            ma_touch_exit = False
            if direction == -1: # Short
                if self.data.High[-1] >= ma_price: ma_touch_exit = True
            else: # Long
                if self.data.Low[-1] <= ma_price: ma_touch_exit = True
            
            # 3. 檢查 Runner SL
            hit_runner_sl = (self.data.High[-1] >= runner_sl) if direction == -1 else (self.data.Low[-1] <= runner_sl)
            
            exit_type = ''
            exit_px = 0.0
            pnl_pts = 0.0

            if hit_runner_sl:
                runner_triggered = True
                exit_type = 'Runner SL'
                exit_px = runner_sl
                pnl_pts = -self.p_runner_sl_offset
            elif ma_touch_exit:
                runner_triggered = True
                exit_type = 'MA Touch Exit'
                exit_px = ma_price # 假設在 MA 價格成交
                if direction == -1: pnl_pts = self.ideal_entry_price - exit_px
                else: pnl_pts = exit_px - self.ideal_entry_price

            if runner_triggered:
                runner_size = direction * remaining_qty
                # 執行平倉
                if direction == -1: self.buy(size=remaining_qty)
                else: self.sell(size=remaining_qty)

                self.record_trade(
                    entry_time=self.entry_time_marker,
                    exit_time=current_time,
                    entry_price=self.ideal_entry_price,
                    exit_price=exit_px,
                    pnl=pnl_pts,
                    size=runner_size,
                    outcome_type=exit_type
                )
                self.manually_closed_trades.add((self.entry_time_marker, current_time))
                self.log(f"  [EXEC] {exit_type} Triggered. Closed Runner.")
                
                # 重置狀態
                self.ideal_entry_price = None
                self.partial_exit_done = False

    # ============================================================
    # [CORE] Pure Manual Exit Logic
    # ============================================================
    def check_and_execute_scale_out(self, ma_price: float):
        if not self.position: return

        current_trade = self.trades[-1]
        direction = 1 if self.position.is_long else -1
        
        self.ideal_entry_price = current_trade.entry_price
        self.entry_time_marker = current_trade.entry_time

        current_time = self.data.index[-1]
        is_entry_bar = (current_trade.entry_time == current_time)

        tp1_triggered = False
        runner_triggered = False
        remaining_qty = abs(self.position.size)
        self.log(f"[Check and Exec Scale Out] {current_time}, Remain >> {remaining_qty} << Position")

        # --- Phase 1: TP1 & Initial SL ---
        if not self.partial_exit_done:
            if direction == -1:
                tp1_price = self.ideal_entry_price - self.p_tp_points
                sl_price = self.ideal_entry_price + self.p_sl_points
            else:
                tp1_price = self.ideal_entry_price + self.p_tp_points
                sl_price = self.ideal_entry_price - self.p_sl_points
            
            outcome = 'HOLD'
            if is_entry_bar:
                outcome = self.check_intra_bar_outcome(
                    self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1],
                    self.ideal_entry_price, sl_price, tp1_price, self.data.index[-1], direction
                )
            else:
                hit_sl = (self.data.High[-1] >= sl_price) if direction == -1 else (self.data.Low[-1] <= sl_price)
                hit_tp = (self.data.Low[-1] <= tp1_price) if direction == -1 else (self.data.High[-1] >= tp1_price)
                if hit_sl: outcome = 'SL'
                elif hit_tp: outcome = 'TP'

            if outcome == 'TP':
                exit_qty = self._calc_exit_qty()
                
                if direction == -1: self.buy(size=exit_qty)
                else: self.sell(size=exit_qty)
                
                self.record_trade(
                    entry_time=self.entry_time_marker,
                    exit_time=current_time,
                    entry_price=self.ideal_entry_price,
                    exit_price=tp1_price,
                    pnl=self.p_tp_points,
                    size=direction * exit_qty,
                    outcome_type='TP1 (Partial)'
                )
                self.manually_closed_trades.add((self.entry_time_marker, current_time))
                self.partial_exit_done = True
                tp1_triggered = True
                remaining_qty -= exit_qty
                
                self.log(f"  [EXEC] TP1 Triggered. Closing {exit_qty} units. Remain >> {remaining_qty} << Position")

            elif outcome == 'SL':
                self.log(f"  [EXIT EXEC] SL Triggered. Submitting Market Close. ++++ ")
                self.position.close()
                self.log(f"  [EXIT EXEC] SL Triggered. Submitting Market Close. Remain >> {remaining_qty} << Position")
                self.record_trade(
                    entry_time=self.entry_time_marker,
                    exit_time=current_time,
                    entry_price=self.ideal_entry_price,
                    exit_price=sl_price,
                    pnl=-self.p_sl_points,
                    size=direction * self.trade_size,
                    outcome_type='SL (Full)'
                )
                self.manually_closed_trades.add((self.entry_time_marker, current_time))
                self.ideal_entry_price = None
                return

        # --- Phase 2: Runner (MA Exit / Runner SL) ---
        if self.partial_exit_done and remaining_qty > 0:
            
            if direction == -1: runner_sl = self.ideal_entry_price + self.p_runner_sl_offset
            else: runner_sl = self.ideal_entry_price - self.p_runner_sl_offset
            
            ma_exit = False
            if direction == -1: 
                if self.data.Low[-1] <= ma_price: ma_exit = True 
            else: 
                if self.data.High[-1] >= ma_price: ma_exit = True 
            
            hit_runner_sl = (self.data.High[-1] >= runner_sl) if direction == -1 else (self.data.Low[-1] <= runner_sl)
            
            exit_type = ''
            exit_px = 0.0
            pnl_pts = 0.0

            if hit_runner_sl:
                runner_triggered = True
                exit_type = 'Runner SL'
                exit_px = runner_sl
                pnl_pts = -self.p_runner_sl_offset
            elif ma_exit:
                runner_triggered = True
                exit_type = 'MA Exit'
                exit_px = ma_price
                if direction == -1: pnl_pts = self.ideal_entry_price - exit_px
                else: pnl_pts = exit_px - self.ideal_entry_price

            if runner_triggered:
                self.log(f"  [EXIT EXEC] {exit_type} Triggered. Submitting Market Close. Remain >> {remaining_qty} << Position")
                runner_size = direction * remaining_qty
                
                self.record_trade(
                    entry_time=self.entry_time_marker,
                    exit_time=current_time,
                    entry_price=self.ideal_entry_price,
                    exit_price=exit_px,
                    pnl=pnl_pts,
                    size=runner_size,
                    outcome_type=exit_type
                )
                self.manually_closed_trades.add((self.entry_time_marker, current_time))
                self.ideal_entry_price = None
                self.partial_exit_done = False

        # --- Phase 3: Execution ---
        if runner_triggered:
            self.log(f"  [EXIT EXEC] {exit_type} Triggered. Position Close. ++++")
            if direction == -1: self.buy(size=1)
            else: self.sell(size=1)
            self.log(f"  [EXIT EXEC] {exit_type} Triggered. Position Close. ---- Remain >> {remaining_qty} << Position")
        elif tp1_triggered:
            pass

    # ============================================================
    # [MODIFIED] Entry Logic
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

        outcome = self.check_intra_bar_outcome(
            self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1],
            limit_price, target_sl, target_tp, current_time, direction
        )

        self.partial_exit_done = False

        if outcome == 'SL' or outcome == 'TP':
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
            self.ideal_entry_price = limit_price
            self.entry_time_marker = current_time
            
            if direction == 1:
                self.buy(limit=limit_price, size=trade_size) 
                self.log(f"[ORDER PLACED] LONG Limit @ {limit_price}")
            else:
                self.sell(limit=limit_price, size=trade_size)
                self.log(f"[ORDER PLACED] SHORT Limit @ {limit_price}")
            
            return False

        return False

    @abstractmethod
    def next(self):
        pass