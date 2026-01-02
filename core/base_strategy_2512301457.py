import pandas as pd
import numpy as np
from backtesting import Strategy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import os

class BaseStrategy(Strategy):
    """
    Base Strategy Class for NQ Futures.
    Encapsulates Intra-bar logic, Order Management, and Common Utilities.
    """
    
    # --- Default Parameters ---
    p_sl_points = 16.0          
    p_tp_points = 20.0          
    p_runner_sl_offset = 7.0    

    # [NEW] Point Value (1點多少錢)
    p_point_value = 2.0 
    
    trade_size = 2              
    runner_size = 1             
    exit_calc_mode = 'FIXED_RUNNER' 

    def init(self):
        self.manual_trades: List[Dict[str, Any]] = []
        self.ideal_entry_price: Optional[float] = None
        self.entry_time_marker: Optional[pd.Timestamp] = None
        self.partial_exit_done: bool = False 
        
        # Logging helper (Child strategies can override self.log_file)
        self.log_file = None 

    def log(self, message: str):
        """Generic logger: writes to file if set, otherwise prints."""
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{message}\n")
            except:
                pass
        # else: print(message) # Optional: print to console if no file

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

    def check_intra_bar_outcome(
        self, open_p: float, high: float, low: float, close: float, 
        entry_price: float, sl_price: float, tp_price: float, 
        current_time: Any, direction: int
    ) -> str:
        """
        Simulates price movement within a bar to determine if SL or TP is hit.
        Returns: 'SL', 'TP', or 'HOLD'
        """
        log_msg = (
                f"\n[INTRA-BAR] {current_time} | Direction: {direction}\n"
                f"  O: {open_p:.2f} | H: {high:.2f} | L: {low:.2f} | C: {close:.2f}\n"
                f"  Price - Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}\n"
            )
        self.log(log_msg)
        # 1. Check if Entry is triggered (Limit Order Logic)
        is_entry_hit = False
        if direction == 1: # Long
            if low <= entry_price: is_entry_hit = True
        else: # Short
            if high >= entry_price: is_entry_hit = True
        #self.log(f"  1. ENTRY trigger: {is_entry_hit}\n")
        if not is_entry_hit:
            return 'HOLD' # Entry didn't happen in this bar
        

        # 2. Check Outcome assuming Entry happened
        if direction == -1: # Short
            is_sl_hit = high >= sl_price
            is_tp_hit = low <= tp_price
            is_close_win = close <= tp_price
        else: # Long
            is_sl_hit = low <= sl_price
            is_tp_hit = high >= tp_price
            is_close_win = close >= tp_price
        log_msg = (
                f"  2. is SL: {is_sl_hit} | is TP: {is_tp_hit} | is close: {is_close_win}\n"
            )
        #self.log(log_msg)

        if is_sl_hit and not is_tp_hit: return 'SL'
        if not is_sl_hit and is_close_win: return 'TP'
        if not is_sl_hit and not is_tp_hit: return 'HOLD'

        # 3. Ambiguous Case (Both SL and TP prices touched)
        # Use Open price distance to estimate which happened first
        dist_open_to_high = abs(open_p - high)
        dist_open_to_low = abs(open_p - low)
        dist_open_to_entry = abs(open_p - entry_price)
        log_msg = (
                f"  3. O2H: {dist_open_to_high:.2f} | O2L: {dist_open_to_low:.2f} | O2E: {dist_open_to_entry:.2f}\n"
            )
        #self.log(log_msg)
        if is_sl_hit and is_tp_hit:
            if direction == -1:
                return 'SL' if dist_open_to_high < dist_open_to_low else 'TP'
            else:
                return 'SL' if dist_open_to_low < dist_open_to_high else 'TP'

        # 4. Entry vs TP conflict (Did we enter before hitting TP?)
        if not is_sl_hit and is_tp_hit and not is_close_win:
            is_open_filled = (open_p >= entry_price) if direction == -1 else (open_p <= entry_price)
            #self.log(f"  4.1 is open filled: {is_open_filled}")
            if is_open_filled: return 'TP'
            
            dist_to_tp_side = dist_open_to_low if direction == -1 else dist_open_to_high
            #self.log(f"  4.2 O2E: {dist_open_to_entry} | 2TP: {dist_to_tp_side}")
            if dist_open_to_entry < dist_to_tp_side:
                return 'TP'
            else:
                return 'HOLD' # Hit TP price first (so no entry), then pulled back to entry? Conservative: HOLD
        #self.log(f"  5. return HOLD")
        return 'HOLD'

    def record_trade(self, entry_time, exit_time, entry_price, exit_price, pnl, size, outcome_type):
        qty = abs(size)
        
        # pnl 參數現在統一代表「單口點數」
        total_points = pnl * qty
        
        # 取得點值 (如果策略有設定 p_point_value，否則預設 1.0)
        point_value = getattr(self, 'p_point_value', 1.0)
        
        # 計算總金額
        total_usd = total_points * point_value
        
        self.manual_trades.append({
            'EntryTime': entry_time,
            'ExitTime': exit_time,
            'EntryPrice': self.round_to_tick(entry_price),
            'ExitPrice': self.round_to_tick(exit_price),
            'PnL': total_points,       # 保持紀錄總點數
            'PnL_Per_Unit': pnl,       # 單口點數
            'PnL_USD': total_usd,      # 新增：總金額
            'Size': size,
            'Type': outcome_type
        })

    def execute_entry_with_intrabar_check(self, direction: int, raw_limit_price: float, sl_points: float, tp_points: float, trade_size: int):
        """
        [CORE LOGIC] Handles Limit Entry with Intra-bar simulation.
        - If trade opens and closes in the same bar: Records it manually.
        - If trade opens but holds: Places a Limit Order.
        """
        current_time = self.data.index[-1]
        
        # 1. Round Price
        limit_price = self.round_to_tick(raw_limit_price)

        # 2. Calculate SL/TP
        if direction == 1: # Long
            target_sl = limit_price - sl_points
            target_tp = limit_price + tp_points
        else: # Short
            target_sl = limit_price + sl_points
            target_tp = limit_price - tp_points

        # 3. Check Intra-bar Outcome
        outcome = self.check_intra_bar_outcome(
            self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1],
            limit_price, target_sl, target_tp, current_time, direction
        )
        self.log(f"[INTRA-BAR OUTCOME] {outcome}")
        # Reset partial exit flag for new trade
        self.partial_exit_done = False

        if outcome == 'SL' or outcome == 'TP':
            # --- Case A: Intra-bar Full Cycle ---
            # Manually record trade, DO NOT place order to engine
            pnl_points = -sl_points if outcome == 'SL' else tp_points
            exit_px = target_sl if outcome == 'SL' else target_tp
            
            self.record_trade(
                entry_time=current_time,
                exit_time=current_time,
                entry_price=limit_price,
                exit_price=exit_px,
                pnl=pnl_points,
                size=direction * trade_size,
                outcome_type=f"Intra-Bar {outcome}"
            )
            self.log(f"[INTRA-BAR EXEC] {outcome} | Entry: {limit_price} | Exit: {exit_px}")
            
            # Return True indicating a trade was completed immediately
            return True

        elif outcome == 'HOLD':
            # --- Case B: Pending Order ---
            # Place standard Limit Order
            self.ideal_entry_price = limit_price
            self.entry_time_marker = current_time
            
            if direction == 1:
                self.buy(limit=limit_price, sl=target_sl, tp=target_tp, size=trade_size)
                self.log(f"[ORDER PLACED] LONG Limit @ {limit_price}")
            else:
                self.sell(limit=limit_price, sl=target_sl, tp=target_tp, size=trade_size)
                self.log(f"[ORDER PLACED] SHORT Limit @ {limit_price}")
            
            return False # Trade not completed yet

        # Case C: No Entry (Price didn't touch limit)
        # self.log(f"[ORDER PENDING] Price didn't touch {limit_price}")
        return False

    def check_and_execute_scale_out(self, ma_price: float):
        """
        [Core Logic] Handles the entire exit lifecycle:
        1. Phase 1: Check TP1 (Partial Exit) & Initial SL.
        2. Phase 2: Check Runner SL & MA Exit.
        """
        if not self.position:
            return

        # 1. Setup Context
        current_trade = self.trades[-1]
        is_entry_bar = (current_trade.entry_bar == len(self.data) - 1)
        direction = 1 if self.position.is_long else -1
        
        # Ensure entry price is recorded
        if self.ideal_entry_price is None:
            self.ideal_entry_price = current_trade.entry_price
            self.entry_time_marker = current_trade.entry_time

        # ============================================================
        # Phase 1: Pre-Scale Out (Check TP1 & Initial SL)
        # ============================================================
        if not self.partial_exit_done:
            if direction == -1: # Short
                tp1_price = self.ideal_entry_price - self.p_tp_points
                sl_price = self.ideal_entry_price + self.p_sl_points
            else: # Long
                tp1_price = self.ideal_entry_price + self.p_tp_points
                sl_price = self.ideal_entry_price - self.p_sl_points
            
            outcome = 'HOLD'
            
            # A. Intra-bar Check (Entry Bar)
            if is_entry_bar:
                outcome = self.check_intra_bar_outcome(
                    self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1],
                    self.ideal_entry_price, sl_price, tp1_price, self.data.index[-1], direction
                )
            # B. Standard Bar Check
            else:
                hit_sl = (self.data.High[-1] >= sl_price) if direction == -1 else (self.data.Low[-1] <= sl_price)
                hit_tp = (self.data.Low[-1] <= tp1_price) if direction == -1 else (self.data.High[-1] >= tp1_price)
                
                if hit_sl and hit_tp:
                    outcome = 'SL' # Conservative assumption
                elif hit_sl: outcome = 'SL'
                elif hit_tp: outcome = 'TP'

            # C. Execution
            if outcome == 'TP':
                exit_qty = self._calc_exit_qty()
                
                # Execute Partial Close
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
                
                # Handle Early Full Exit (if exit_qty == trade_size)
                if exit_qty == self.trade_size:
                    self.ideal_entry_price = None
                    return

            elif outcome == 'SL':
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

        # ============================================================
        # Phase 2: Post-Scale Out (Runner Management)
        # ============================================================
        else:
            # 1. Calculate Runner SL
            if direction == -1: # Short
                runner_sl = self.ideal_entry_price + self.p_runner_sl_offset
            else: # Long
                runner_sl = self.ideal_entry_price - self.p_runner_sl_offset
            
            # 2. Check MA Exit
            ma_exit = False
            if direction == -1: # Short
                if self.data.Close[-1] > ma_price: ma_exit = True
            else: # Long
                if self.data.Close[-1] < ma_price: ma_exit = True
            
            # 3. Check Trigger
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
                self.position.close()
                remaining_size = self.position.size # Backtesting.py handles size sign
                
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

    @abstractmethod
    def next(self):
        pass