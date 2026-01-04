import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import time
from backtesting import Strategy
from core.base_strategy import BaseStrategy
import os

class NQDualTrend(BaseStrategy):
    """
    NQ Dual Trend Strategy (No Cool-down, Enhanced Logging)
    """
    
    # ==========================================
    # 1. Strategy Parameters
    # ==========================================
    p_ma20_len = 20
    p_ma43_len = 43
    p_spread_threshold = 20.0
    p_deduction_ago = 34  
    p_entry_offset = 2.0  
    p_tolerance = 1.0     
    p_start_time = time(15, 0)
    p_end_time = time(4, 0)
    p_sl_points = 16.0      
    p_tp_points = 20.0      
    p_runner_sl_offset = 7.0 
    trade_size = 2          
    runner_size = 1         
    exit_calc_mode = 'FIXED_RUNNER'
    p_point_value = 2.0 

    def init(self):
        super().init()
        if not hasattr(self.data, 'MA20_5m'):
            raise ValueError("[Error] Data missing 'MA20_5m'.")

        close = pd.Series(self.data.Close)
        self.ma20_2m = self.I(ta.sma, close, length=self.p_ma20_len, name='MA20', color='blue')
        self.ma43_2m = self.I(ta.sma, close, length=self.p_ma43_len, name='MA43', color='orange')
        self.ma20_5m_ref = self.I(lambda x: x, self.data.MA20_5m, name='MA20_5m', color='purple', overlay=True)
        
        self.debug_signal_short = self.I(self.calc_signal_short_array, name='Signal_Short', color='red', scatter=True)
        self.debug_signal_long = self.I(self.calc_signal_long_array, name='Signal_Long', color='green', scatter=True)
        self.debug_deduction_dots = self.I(self.calc_deduction_marker_array, name='Deduction', color='orange', scatter=True)

        self.last_trade_count = 0 
        self.handled_trade_entries = set()

        self.log_file = 'log_dual_trend.txt'
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"--- DualTrend Backtest Log Started: {pd.Timestamp.now()} ---\n")
        except Exception as e:
            print(f"[Error] Could not create log file: {e}")

    # ... (Signal Calculation Helpers 保持不變) ...
    def calc_signal_short_array(self):
        close = self.data.Close
        ma20 = ta.sma(pd.Series(close), length=self.p_ma20_len).to_numpy()
        ma43 = ta.sma(pd.Series(close), length=self.p_ma43_len).to_numpy()
        ma20_5m = self.data.MA20_5m
        signals = np.full(len(close), np.nan)
        for i in range(60, len(close)):
            if (abs(ma20[i] - ma43[i]) >= self.p_spread_threshold and
                ma20[i] < ma43[i] and
                (ma20[i] < ma20[i-1] and ma20[i-1] < ma20[i-2] and ma20[i-2] < ma20[i-3]) and
                min(close[i-43 : i-33]) > close[i] and
                close[i] < ma20_5m[i]):
                signals[i] = 1 
        return signals

    def calc_signal_long_array(self):
        close = self.data.Close
        ma20 = ta.sma(pd.Series(close), length=self.p_ma20_len).to_numpy()
        ma43 = ta.sma(pd.Series(close), length=self.p_ma43_len).to_numpy()
        ma20_5m = self.data.MA20_5m
        signals = np.full(len(close), np.nan)
        for i in range(60, len(close)):
            if (abs(ma20[i] - ma43[i]) >= self.p_spread_threshold and
                ma20[i] > ma43[i] and
                (ma20[i] > ma20[i-1] and ma20[i-1] > ma20[i-2] and ma20[i-2] > ma20[i-3]) and
                max(close[i-43 : i-33]) < close[i] and
                close[i] > ma20_5m[i]):
                signals[i] = 1 
        return signals

    def calc_deduction_marker_array(self):
        sig_s = self.calc_signal_short_array()
        sig_l = self.calc_signal_long_array()
        high = self.data.High
        low = self.data.Low
        markers = np.full(len(high), np.nan)
        for i in range(60, len(high)):
            deduction_idx = i - 20 
            if deduction_idx >= 0:
                if sig_s[i] == 1: markers[deduction_idx] = high[deduction_idx] + 5.0
                elif sig_l[i] == 1: markers[deduction_idx] = low[deduction_idx] - 5.0
        return markers

    # ==========================================
    # 3. Main Execution Logic
    # ==========================================
    def next(self):
        current_time = self.data.index[-1]

        # --- 0.5 ORDER MONITOR (Enhanced) ---
        if len(self.orders) > 0:
            for order in list(self.orders):
                limit_price = order.limit if order.limit else 0
                side = "LONG" if order.size > 0 else "SHORT"
                
                # [DEBUG] Distinguish Market Exit vs Limit Entry
                if order.limit is None and order.stop is None:
                    self.log(f"[ORDER MONITOR] PENDING MARKET EXIT (Size: {order.size}) - Closing Position")
                    continue
                
                # Entry Order Logic
                order_type = "ENTRY"
                bar_high = self.data.High[-1]
                bar_low = self.data.Low[-1]
                
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
                        self.log(f"[CANCEL] Pre-emptive TP hit.")
                        continue
                
                self.log(f"[ORDER MONITOR] PENDING LIMIT ENTRY {side} @ {limit_price} | Bar: {bar_low}-{bar_high}")

        # ============================================================
        # 1. POSITION MANAGEMENT
        # ============================================================
        if self.position:
            self.check_and_execute_scale_out(ma_price=self.ma20_2m[-1])
            return

        # ============================================================
        # 2. ORDER CLEANUP
        # ============================================================
        if len(self.orders) > 0:
            self.log(f"[ORDER CLEANUP] Flat position. Cancelling {len(self.orders)} stale orders.")
            for o in self.orders: o.cancel()

        # ============================================================
        # 3. CHECKS
        # ============================================================
        if len(self.data) < 60: return
        t = current_time.time()
        is_trading_time = False
        
        if self.p_start_time > self.p_end_time:
            # 跨日 (例如 15:00 ~ 02:00)
            # 時間必須 >= 15:00 或者 <= 02:00
            if t >= self.p_start_time or t <= self.p_end_time:
                is_trading_time = True
        
        if not is_trading_time:
            return
        
        # [REMOVED] Cool-down Check

        # ============================================================
        # 4. SIGNAL LOGIC
        # ============================================================
        is_short = (self.debug_signal_short[-1] == 1)
        is_long = (self.debug_signal_long[-1] == 1)
        
        ma20 = self.ma20_2m[-1]
        ma43 = self.ma43_2m[-1]
        spread = abs(ma20 - ma43)
        ma20_5m_val = self.ma20_5m_ref[-1]
        close = self.data.Close[-1]

        # ============================================================
        # 5. EXECUTION
        # ============================================================
        if is_short or is_long:
            dir_str = "SHORT" if is_short else "LONG"
            self.log(f"\n[SIGNAL FOUND] {current_time} | {dir_str}")
            self.log(f"  Spread: {spread:.2f} (>={self.p_spread_threshold})")
            self.log(f"  MA20: {ma20:.2f} | MA43: {ma43:.2f}")
            self.log(f"  Price: {close:.2f} | 5m MA: {ma20_5m_val:.2f}")

            target_direction = 0
            raw_limit_price = 0.0
            
            if is_short:
                target_direction = -1
                raw_limit_price = ma20_5m_val + self.p_entry_offset
            elif is_long:
                target_direction = 1
                raw_limit_price = ma20_5m_val - self.p_entry_offset

            if target_direction != 0:
                self.log(f"  >>> ATTEMPT ENTRY: {dir_str} Limit @ {raw_limit_price:.2f}")
                self.execute_entry_with_intrabar_check(
                    direction=target_direction,
                    raw_limit_price=raw_limit_price,
                    sl_points=self.p_sl_points,
                    tp_points=self.p_tp_points,
                    trade_size=self.trade_size
                )