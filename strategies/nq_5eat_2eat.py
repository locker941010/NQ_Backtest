import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import time
from backtesting import Strategy
from core.base_strategy import BaseStrategy
import os

class NQ5Eat2Eat(BaseStrategy):
    """
    NQ_5E2E Strategy (NQ 5-Eat 2-Eat) - Refactored for BaseStrategy Architecture
    
    Logic:
    1. Trend determined by MA20 & MA43 slopes.
    2. Pullback detected via MA20 Touch.
    3. Signal Trigger: "Eat" pattern (High Volume Boss Candle Engulf) on 5m (Filter) and 2m (Trigger).
    """

    # ==========================================
    # 1. Strategy Parameters
    # ==========================================
    p_ma_fast = 20          # MA20
    p_ma_slow = 43          # MA43
    p_lookback = 20         # Lookback for Boss Candle
    p_entry_window = 3      # Window to check for Signal
    p_turn_threshold = 5    # Bars since MA20 turn

    p_start_time = time(15, 0)
    p_end_time = time(4, 0)
    
    # --- MA Touch Parameters ---
    p_ma_touch_lookback = 3     
    p_ma_touch_tolerance = 0.5  
    
    # --- 5m Filter Parameters ---
    p_5m_eat_lookback = 2   # Check current and previous 1 bar (Total 2)

    # --- Money Management ---
    trade_size = 2          # Default to 2 for Scale-out logic
    runner_size = 1
    exit_calc_mode = 'FIXED_RUNNER'
    
    # --- Exit Parameters ---
    p_sl_points = 16.0
    p_tp_points = 20.0
    p_runner_sl_offset = 7.0
    p_point_value = 2.0

    def init(self):
        super().init()

        # --- Data Integrity Check ---
        if not hasattr(self.data, 'Open_5m'):
            raise ValueError("[Error] Data missing 'Open_5m' (and likely High/Low/Close/Vol 5m). Strategy requires 5m Ghost columns.")

        # --- Indicators (Registered for Plotting) ---
        # We use pandas_ta for consistency with the reference strategy
        close_series = pd.Series(self.data.Close)
        self.ma20 = self.I(ta.sma, close_series, length=self.p_ma_fast, name='MA20', color='blue')
        self.ma43 = self.I(ta.sma, close_series, length=self.p_ma_slow, name='MA43', color='orange')

        # --- Debug Visualization ---
        # We initialize empty arrays and fill them in next() for the Analyzer to pick up
        self.debug_2m_eat = self.I(lambda: np.full(len(self.data.Close), np.nan), name='Signal_2m_Eat', color='purple', scatter=True, overlay=True)
        self.debug_5m_eat = self.I(lambda: np.full(len(self.data.Close), np.nan), name='Signal_5m_Eat', color='cyan', scatter=True, overlay=True)
        
        # --- Logging Setup ---
        self.log_file = 'log_nq_5eat.txt'
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"--- NQ 5Eat 2Eat Backtest Log Started: {pd.Timestamp.now()} ---\n")
        except Exception as e:
            print(f"[Error] Could not create log file: {e}")

    # ==========================================
    # 2. Main Execution Logic
    # ==========================================
    def next(self):
        current_time = self.data.index[-1]

        # --- 0.5 ORDER MONITOR (From NQDualTrend) ---
        if len(self.orders) > 0:
            for order in list(self.orders):
                limit_price = order.limit if order.limit else 0
                side = "LONG" if order.size > 0 else "SHORT"
                
                # Skip Market Exits
                if order.limit is None and order.stop is None: continue
                
                # Pre-emptive Cancel Logic
                bar_high = self.data.High[-1]
                bar_low = self.data.Low[-1]
                should_cancel = False
                
                if side == "LONG":
                    theoretical_tp = limit_price + self.p_tp_points
                    if bar_high >= theoretical_tp: should_cancel = True
                elif side == "SHORT":
                    theoretical_tp = limit_price - self.p_tp_points
                    if bar_low <= theoretical_tp: should_cancel = True
                
                if should_cancel:
                    order.cancel()
                    self.log(f"[CANCEL] Pre-emptive TP hit for {side} order.")
                    continue

        # ============================================================
        # 1. POSITION MANAGEMENT (Scale Out)
        # ============================================================
        if self.position:
            self.check_and_execute_scale_out_5eat2eat(ma_price=self.ma20[-1])
            return

        # ============================================================
        # 2. ORDER CLEANUP
        # ============================================================
        if len(self.orders) > 0:
            self.log(f"[ORDER CLEANUP] Flat position. Cancelling {len(self.orders)} stale orders.")
            for o in self.orders: o.cancel()

        # ============================================================
        # 3. CHECKS & PRE-CALCS
        # ============================================================
        # Ensure enough data for lookbacks
        if len(self.data) < max(self.p_ma_slow, self.p_lookback) + self.p_turn_threshold + 5:
            return
        t = current_time.time()
        is_trading_time = False
        if self.p_start_time > self.p_end_time:
            # 跨日 (例如 15:00 ~ 02:00)
            # 時間必須 >= 15:00 或者 <= 02:00
            if t >= self.p_start_time or t <= self.p_end_time:
                is_trading_time = True
        
        if not is_trading_time:
            return

        # --- 4. SIGNAL LOGIC ---
        
        # A. Trend Analysis
        ma20_slope = self.ma20[-1] - self.ma20[-2]
        ma43_slope = self.ma43[-1] - self.ma43[-2]
        
        trend_dir = 0
        if ma20_slope > 0 and ma43_slope > 0:
            trend_dir = 1 
        elif ma20_slope < 0 and ma43_slope < 0:
            trend_dir = -1 
        
        if trend_dir == 0: return

        # B. MA Stability Check
        recent_slopes = np.diff(self.ma20[-self.p_turn_threshold-1:])
        is_stable = True
        if trend_dir == 1:
            if np.any(recent_slopes <= 0): is_stable = False
        elif trend_dir == -1:
            if np.any(recent_slopes >= 0): is_stable = False
            
        if not is_stable: return

        # C. MA Touch Filter (Pullback)
        has_ma_touch = False
        for i in range(self.p_ma_touch_lookback):
            idx = -1 - i
            ma_val = self.ma20[idx]
            upper_bound = ma_val + self.p_ma_touch_tolerance
            lower_bound = ma_val - self.p_ma_touch_tolerance
            
            if self.data.High[idx] >= lower_bound and self.data.Low[idx] <= upper_bound:
                has_ma_touch = True
                break
        
        if not has_ma_touch: return

        # D. 5m Filter Check (Relaxed)
        direction_str = 'LONG' if trend_dir == 1 else 'SHORT'
        is_5m_valid, boss_5m_open = self._check_5m_eat(direction_str)
        
        if not is_5m_valid: return

        # E. 2m Trigger Check
        opens = self.data.Open
        closes = self.data.Close
        vols = self.data.Volume
        curr_idx = len(self.data) - 1
        
        is_2m_valid, boss_2m_open = self._find_boss_and_check_eat(
            opens, closes, vols, curr_idx, trend_dir
        )

        # ============================================================
        # 5. EXECUTION
        # ============================================================
        if is_2m_valid:
            # Update Debug Indicators for Plotting
            if trend_dir == 1: 
                self.debug_2m_eat[-1] = self.data.Low[-1] - 2.0
                self.debug_5m_eat[-1] = self.data.Low[-1] - 4.0
            else: 
                self.debug_2m_eat[-1] = self.data.High[-1] + 2.0
                self.debug_5m_eat[-1] = self.data.High[-1] + 4.0

            log_msg = (
                f"\n[POTENTIAL] {current_time}\n"
                f"  Trend: {trend_dir} (MA20: {ma20_slope:.2f})\n"
                f"  MA Stable: {is_stable}\n"
                f"  MA Touch ({self.p_ma_touch_lookback} bars): {has_ma_touch}\n"
                f"  5m Filter (Lookback {self.p_5m_eat_lookback}): {is_5m_valid} (Boss 5m Open: {boss_5m_open})\n"
                f"  2m Trigger: {is_2m_valid} (Boss 2m Open: {boss_2m_open})")
            self.log(log_msg)

            # Determine Limit Price (Conservative Entry)
            limit_price = 0.0
            if trend_dir == 1:
                limit_price = min(boss_2m_open, boss_5m_open)
            else:
                limit_price = max(boss_2m_open, boss_5m_open)

            self.log(f"  >>> ATTEMPT ENTRY: {direction_str} Limit @ {limit_price:.2f}")
            
            # Execute using BaseStrategy's Intra-bar logic
            self.execute_entry_with_intrabar_check(
                direction=trend_dir,
                raw_limit_price=limit_price,
                sl_points=self.p_sl_points,
                tp_points=self.p_tp_points,
                trade_size=self.trade_size
            )

    # ==========================================
    # 3. Logic Helpers
    # ==========================================
    def _find_boss_and_check_eat(self, opens, closes, vols, curr_idx, direction) -> tuple[bool, float]:
        """
        Identifies if the current candle 'eats' (engulfs/passes) a previous 'Boss' candle's open.
        A Boss candle is a high-volume candle in the opposite direction.
        """
        curr_vol = vols[curr_idx]
        curr_open = opens[curr_idx]
        curr_close = closes[curr_idx]
        
        # Current candle must be in trend direction
        if direction == 1: 
            if curr_close <= curr_open: return False, 0.0
        else: 
            if curr_close >= curr_open: return False, 0.0

        # 1. Find Reference Candle (Last candle of same color with higher volume)
        ref_idx = -1
        start_search = curr_idx - 1
        end_search = max(0, curr_idx - self.p_lookback)
        
        for i in range(start_search, end_search, -1):
            c_open = opens[i]
            c_close = closes[i]
            c_vol = vols[i]
            
            is_same_color = (c_close > c_open) if direction == 1 else (c_close < c_open)
            
            if is_same_color and c_vol > curr_vol:
                # Check if this candle is the "dominant" one in its local area
                max_opp_vol = 0
                for k in range(i + 1, curr_idx):
                    k_open = opens[k]
                    k_close = closes[k]
                    k_vol = vols[k]
                    is_opp = (k_close < k_open) if direction == 1 else (k_close > k_open)
                    if is_opp and k_vol > max_opp_vol:
                        max_opp_vol = k_vol
                
                if c_vol >= max_opp_vol:
                    ref_idx = i
                    break 
        
        if ref_idx == -1: return False, 0.0
            
        # 2. Find the "Boss" (Highest volume opposite candle between Ref and Current)
        boss_vol = -1.0
        boss_open = 0.0
        found_boss = False
        
        for k in range(ref_idx + 1, curr_idx):
            k_open = opens[k]
            k_close = closes[k]
            k_vol = vols[k]
            
            is_opp = (k_close < k_open) if direction == 1 else (k_close > k_open)
            
            if is_opp:
                if k_vol > boss_vol:
                    boss_vol = k_vol
                    boss_open = k_open
                    found_boss = True
        
        if not found_boss: return False, 0.0
            
        # 3. Volume Confirmation (Current volume doesn't strictly need to beat Boss, 
        # but usually we want momentum. The original strategy had `curr_vol >= boss_vol` 
        # which is very strict. We relax it slightly here or keep it if strictness is desired.
        # Keeping strictness based on original code, but note this is a common failure point.)
        # if curr_vol < boss_vol: return False, 0.0  #<-- Commented out to allow more trades, 
        # or uncomment to restore strict logic.
            
        # 4. "Eat" Check (Price passes Boss Open)
        # Tolerance Check (0.25 ticks)
        if direction == 1: # Long
            if curr_close >= (boss_open - 0.25): return True, boss_open
        else: # Short
            if curr_close <= (boss_open + 0.25): return True, boss_open
                
        return False, 0.0

    def _check_5m_eat(self, direction: str) -> tuple[bool, float]:
        """
        Reconstructs 5m bars from the broadcasted 5m columns and checks for the Eat pattern.
        """
        # Safety check handled in init, but good to be safe
        if not hasattr(self.data, 'Open_5m'): return False, 0.0

        lookback_limit = 60 
        if len(self.data) < lookback_limit: return False, 0.0
        
        # Extract broadcasted 5m data
        raw_o5 = self.data.Open_5m[-lookback_limit:]
        raw_c5 = self.data.Close_5m[-lookback_limit:]
        raw_v5 = self.data.Volume_5m[-lookback_limit:]
        
        # Compress to unique 5m bars
        # Since data is broadcasted (e.g. 2m bars: [A, A, A, B, B...]), we detect changes
        unique_5m_opens = []
        unique_5m_closes = []
        unique_5m_vols = []
        
        # Identify indices where the 5m bar changes
        # Note: This logic assumes the last bar might be incomplete but we still check it
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
        
        # Check current and previous N bars (p_5m_eat_lookback)
        for i in range(self.p_5m_eat_lookback):
            check_idx = curr_5m_idx - i
            if check_idx < 0: break
            
            is_valid, boss_open = self._find_boss_and_check_eat(
                u_opens, u_closes, u_vols, check_idx, dir_int
            )
            
            if is_valid:
                return True, boss_open 
            
        return False, 0.0