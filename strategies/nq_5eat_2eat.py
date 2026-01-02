import pandas as pd
import numpy as np
from backtesting import Strategy
from core.base_strategy import BaseStrategy
import os

# --- Helper Functions ---
def SMA(array, n):
    """Simple Moving Average using Pandas Rolling."""
    return pd.Series(array).rolling(n).mean()

class NQ5Eat2Eat(BaseStrategy):
    """
    NQ_5E2E Strategy (NQ 5-Eat 2-Eat) - Relaxed 5m Filter
    
    Updates:
    1. Added 'p_5m_eat_lookback': Allows the 5m Eat pattern to have occurred 
       within the last N 5m-bars (default 2).
    """

    # --- Strategy Parameters ---
    p_ma_fast = 20          # MA20
    p_ma_slow = 43          # MA43
    p_lookback = 20         # Lookback for Boss Candle
    p_entry_window = 3      # Window to check for Signal
    p_turn_threshold = 5    # Bars since MA20 turn
    p_order_validity = 5    # Cancel order if not filled after N bars
    
    # --- MA Touch Parameters ---
    p_ma_touch_lookback = 3     
    p_ma_touch_tolerance = 0.5  
    
    # --- 5m Filter Parameters (NEW) ---
    p_5m_eat_lookback = 2   # Check current and previous 1 bar (Total 2)

    # --- Money Management ---
    trade_size = 1          
    
    # --- Exit Parameters ---
    p_sl_points = 16.0
    p_tp_points = 20.0

    def init(self):
        super().init()

        # --- Indicators ---
        self.ma20 = self.I(SMA, self.data.Close, self.p_ma_fast)
        self.ma43 = self.I(SMA, self.data.Close, self.p_ma_slow)

        # Debugging Visualization
        self.debug_2m_eat = self.I(lambda x: np.full(len(x), np.nan), self.data.Close, name='Debug_2m_Eat', overlay=True)
        self.debug_5m_eat = self.I(lambda x: np.full(len(x), np.nan), self.data.Close, name='Debug_5m_Eat', overlay=True)
        
        # --- Trade Tracking ---
        self.last_trade_count = 0 

        # --- Logging Setup ---
        self.log_file = 'log.txt'
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"--- Backtest Log Started: {pd.Timestamp.now()} ---\n")
        except Exception as e:
            print(f"[Error] Could not create log file: {e}")

    def _log(self, message):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
        except:
            pass

    def next(self):
        current_time = self.data.index[-1]

        # --- 0. TRADE RECORDER ---
        if len(self.closed_trades) > self.last_trade_count:
            for i in range(self.last_trade_count, len(self.closed_trades)):
                trade = self.closed_trades[i]
                outcome = 'TP' if trade.pl > 0 else 'SL'
                self.record_trade(
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    pnl=trade.pl,
                    size=trade.size,
                    outcome_type=outcome
                )
                self._log(f"[TRADE CLOSED] {outcome} | PnL: {trade.pl:.2f}")
            self.last_trade_count = len(self.closed_trades)

        # --- 0.5 ORDER MONITOR ---
        if len(self.orders) > 0:
            for order in list(self.orders):
                limit_price = order.limit if order.limit else 0
                side = "LONG" if order.size > 0 else "SHORT"
                
                pos_size = self.position.size
                is_exit = (pos_size > 0 and side == "SHORT") or (pos_size < 0 and side == "LONG")
                order_type = "EXIT" if is_exit else "ENTRY"

                bar_high = self.data.High[-1]
                bar_low = self.data.Low[-1]
                
                # Pre-emptive TP Cancel Logic
                if order_type == "ENTRY":
                    should_cancel = False
                    if side == "LONG":
                        theoretical_tp = limit_price + self.p_tp_points
                        if bar_high >= theoretical_tp: should_cancel = True
                    elif side == "SHORT":
                        theoretical_tp = limit_price - self.p_tp_points
                        if bar_low <= theoretical_tp: should_cancel = True
                    
                    if should_cancel:
                        order.cancel()
                        self._log(f"[CANCEL] Pre-emptive TP hit.")
                        continue

                status_msg = (
                    f"[ORDER MONITOR] {current_time} | {order_type} {side} @ {limit_price}\n"
                    f"  Current Bar: High {bar_high} | Low {bar_low}\n"
                )
                
                if side == "LONG":
                    if bar_low <= limit_price: status_msg += "  -> EXECUTING..."
                    else: status_msg += f"  -> PENDING (Missed by {bar_low - limit_price:.2f})"
                else: 
                    if bar_high >= limit_price: status_msg += "  -> EXECUTING..."
                    else: status_msg += f"  -> PENDING (Missed by {limit_price - bar_high:.2f})"
                
                self._log(status_msg)

        # Skip if not enough data
        if len(self.data) < max(self.p_ma_slow, self.p_lookback) + self.p_turn_threshold + 5:
            return

        # Only enter if flat
        if self.position:
            return

        # --- 1. MA Trend & Turn Analysis ---
        ma20_slope = self.ma20[-1] - self.ma20[-2]
        ma43_slope = self.ma43[-1] - self.ma43[-2]
        
        trend_dir = 0
        if ma20_slope > 0 and ma43_slope > 0:
            trend_dir = 1 
        elif ma20_slope < 0 and ma43_slope < 0:
            trend_dir = -1 
        
        # Check MA20 Slope Stability
        recent_slopes = np.diff(self.ma20[-self.p_turn_threshold-1:])
        is_stable = True
        if trend_dir == 1:
            if np.any(recent_slopes <= 0): is_stable = False
        elif trend_dir == -1:
            if np.any(recent_slopes >= 0): is_stable = False

        # --- 2. MA Touch Filter ---
        has_ma_touch = False
        for i in range(self.p_ma_touch_lookback):
            idx = -1 - i
            ma_val = self.ma20[idx]
            upper_bound = ma_val + self.p_ma_touch_tolerance
            lower_bound = ma_val - self.p_ma_touch_tolerance
            
            if self.data.High[idx] >= lower_bound and self.data.Low[idx] <= upper_bound:
                has_ma_touch = True
                break

        # --- 3. 5m Filter Check (Relaxed) ---
        is_5m_valid = False
        boss_5m_open = 0.0
        
        if trend_dir != 0:
            direction_str = 'LONG' if trend_dir == 1 else 'SHORT'
            is_5m_valid, boss_5m_open = self._check_5m_eat(direction_str)

        # --- 4. Signal Logic (2m) ---
        is_2m_valid = False
        boss_2m_open = 0.0
        
        if trend_dir != 0:
            opens = self.data.Open
            closes = self.data.Close
            vols = self.data.Volume
            curr_idx = len(self.data) - 1
            
            is_2m_valid, boss_2m_open = self._find_boss_and_check_eat(
                opens, closes, vols, curr_idx, trend_dir
            )
            
            if is_2m_valid:
                if trend_dir == 1: self.debug_2m_eat[-1] = self.data.Low[-1] * 0.999
                else: self.debug_2m_eat[-1] = self.data.High[-1] * 1.001

        # --- 5. Execution ---
        if is_2m_valid:
            if len(self.orders) > 0:
                self._log(f"[INFO] {current_time} | New Signal -> Cancelling {len(self.orders)} old orders.")
                for o in self.orders: o.cancel()

            log_msg = (
                f"\n[POTENTIAL] {current_time}\n"
                f"  Trend: {trend_dir} (MA20: {ma20_slope:.2f})\n"
                f"  MA Stable: {is_stable}\n"
                f"  MA Touch ({self.p_ma_touch_lookback} bars): {has_ma_touch}\n"
                f"  5m Filter (Lookback {self.p_5m_eat_lookback}): {is_5m_valid} (Boss 5m Open: {boss_5m_open})\n"
                f"  2m Trigger: {is_2m_valid} (Boss 2m Open: {boss_2m_open})"
            )
            
            if trend_dir != 0 and is_stable and has_ma_touch and is_5m_valid and is_2m_valid:
                if trend_dir == 1:
                    limit_price = min(boss_2m_open, boss_5m_open)
                    log_msg += f"\n  >>> PLACING LONG LIMIT @ {limit_price}"
                    self._log(log_msg)
                    self.execute_entry_with_intrabar_check(1, limit_price, self.p_sl_points, self.p_tp_points, self.trade_size)
                else:
                    limit_price = max(boss_2m_open, boss_5m_open)
                    log_msg += f"\n  >>> PLACING SHORT LIMIT @ {limit_price}"
                    self._log(log_msg)
                    self.execute_entry_with_intrabar_check(-1, limit_price, self.p_sl_points, self.p_tp_points, self.trade_size)


    # --- Logic Helpers ---
    def _find_boss_and_check_eat(self, opens, closes, vols, curr_idx, direction) -> tuple[bool, float]:
        curr_vol = vols[curr_idx]
        curr_open = opens[curr_idx]
        curr_close = closes[curr_idx]
        
        if direction == 1: 
            if curr_close <= curr_open: return False, 0.0
        else: 
            if curr_close >= curr_open: return False, 0.0

        ref_idx = -1
        start_search = curr_idx - 1
        end_search = max(0, curr_idx - self.p_lookback)
        
        for i in range(start_search, end_search, -1):
            c_open = opens[i]
            c_close = closes[i]
            c_vol = vols[i]
            
            is_same_color = False
            if direction == 1: 
                if c_close > c_open: is_same_color = True
            else: 
                if c_close < c_open: is_same_color = True
            
            if is_same_color and c_vol > curr_vol:
                max_opp_vol = 0
                for k in range(i + 1, curr_idx):
                    k_open = opens[k]
                    k_close = closes[k]
                    k_vol = vols[k]
                    
                    is_opp = False
                    if direction == 1: 
                        if k_close < k_open: is_opp = True
                    else: 
                        if k_close > k_open: is_opp = True
                    
                    if is_opp and k_vol > max_opp_vol:
                        max_opp_vol = k_vol
                
                if c_vol >= max_opp_vol:
                    ref_idx = i
                    break 
        
        if ref_idx == -1: return False, 0.0
            
        boss_vol = -1.0
        boss_open = 0.0
        found_boss = False
        
        for k in range(ref_idx + 1, curr_idx):
            k_open = opens[k]
            k_close = closes[k]
            k_vol = vols[k]
            
            is_opp = False
            if direction == 1: 
                if k_close < k_open: is_opp = True
            else: 
                if k_close > k_open: is_opp = True
            
            if is_opp:
                if k_vol > boss_vol:
                    boss_vol = k_vol
                    boss_open = k_open
                    found_boss = True
        
        if not found_boss: return False, 0.0
            
        if curr_vol >= boss_vol: return False, 0.0
            
        # Tolerance Check (0.25)
        if direction == 1: # Long
            if curr_close >= (boss_open - 0.25): return True, boss_open
        else: # Short
            if curr_close <= (boss_open + 0.25): return True, boss_open
                
        return False, 0.0

    def _check_5m_eat(self, direction: str) -> tuple[bool, float]:
        try:
            _ = self.data.Open_5m
        except AttributeError:
            return False, 0.0

        lookback_limit = 60 
        if len(self.data) < lookback_limit: return False, 0.0
        
        raw_o5 = self.data.Open_5m[-lookback_limit:]
        raw_c5 = self.data.Close_5m[-lookback_limit:]
        raw_v5 = self.data.Volume_5m[-lookback_limit:]
        
        unique_5m_opens = []
        unique_5m_closes = []
        unique_5m_vols = []
        
        change_indices = []
        for i in range(len(raw_o5) - 1):
            if raw_o5[i] != raw_o5[i+1]:
                change_indices.append(i)
        change_indices.append(len(raw_o5) - 1)
        
        for idx in change_indices:
            unique_5m_opens.append(raw_o5[idx])
            unique_5m_closes.append(raw_c5[idx])
            unique_5m_vols.append(raw_v5[idx])
            
        u_opens = np.array(unique_5m_opens)
        u_closes = np.array(unique_5m_closes)
        u_vols = np.array(unique_5m_vols)
        
        curr_5m_idx = len(u_opens) - 1
        dir_int = 1 if direction == 'LONG' else -1
        
        # --- MODIFIED: Lookback Loop ---
        # Check current and previous N bars
        for i in range(self.p_5m_eat_lookback):
            check_idx = curr_5m_idx - i
            if check_idx < 0: break
            
            is_valid, boss_open = self._find_boss_and_check_eat(
                u_opens, u_closes, u_vols, check_idx, dir_int
            )
            
            if is_valid:
                if direction == 'LONG': self.debug_5m_eat[-1] = self.data.Low[-1]
                else: self.debug_5m_eat[-1] = self.data.High[-1]
                return True, boss_open # Return immediately if found
            
        return False, 0.0