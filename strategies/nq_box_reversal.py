import pandas as pd
import numpy as np
from core.base_strategy import BaseStrategy

def SMA(array, n):
    """Simple Moving Average"""
    return pd.Series(array).rolling(n).mean()

class NQBoxReversal(BaseStrategy):
    """
    Strategy: Smart Box Breakout & Pullback (Close-Based).
    
    Logic:
    1. Identify a consolidation 'Box' based on Closing Prices (ignoring wicks).
       - Dynamic reverse scan to determine box length.
    2. Trend Filter: The Box Start must occur while MA20 is continuously descending.
    3. Signal: Price breaks and holds above MA43 (2 bars).
    4. Deduction Logic (OR Condition):
       - Valid if MA20 Deduction (Close[t-20]) OR MA43 Deduction (Close[t-43])
         is inside the Box Range.
    5. Entry: Limit Buy at the Lowest Close of the Box.
    """

    # --- Strategy Parameters ---
    p_min_box_len = 10          # Minimum bars to form a valid box
    p_max_box_len = 60          # Maximum lookback for box scanning
    p_box_tol = 5.0             # Tolerance points to extend box range during scan
    p_trend_lookback = 3        # Number of bars MA20 must be falling BEFORE box start
    
    p_ma20_period = 20
    p_ma43_period = 43
    p_deduction_buffer = 3      # Min bars required between Box End and Deduction Index

    def init(self):
        # Initialize BaseStrategy
        super().init()

        # 1. Standard Indicators
        self.ma20_2m = self.I(SMA, self.data.Close, self.p_ma20_period, name='MA20')
        self.ma43_2m = self.I(SMA, self.data.Close, self.p_ma43_period, name='MA43')

        # 2. Debug/Visualization Arrays
        full_length = len(self.data)
        
        self.arr_box_high = np.full(full_length, np.nan)
        self.arr_box_low = np.full(full_length, np.nan)
        self.arr_deduction_20 = np.full(full_length, np.nan)
        self.arr_deduction_43 = np.full(full_length, np.nan)
        
        # 3. Register Visuals
        self.debug_box_high = self.I(lambda: self.arr_box_high, name='Box High', overlay=True, color='cyan')
        self.debug_box_low = self.I(lambda: self.arr_box_low, name='Box Low', overlay=True, color='cyan')
        
        self.debug_deduction_20 = self.I(lambda: self.arr_deduction_20, name='Ded20 Dot', overlay=True, color='blue', scatter=True)
        self.debug_deduction_43 = self.I(lambda: self.arr_deduction_43, name='Ded43 Dot', overlay=True, color='magenta', scatter=True)

        # 4. State variables
        self.box_end_index = None       
        self.entry_limit_price = None
        self.active_deduction_mode = None # '20', '43', or 'BOTH'

    def next(self):
        # Current absolute index
        current_idx = len(self.data) - 1

        # ---------------------------------------------------------------------
        # 1. Position Management
        # ---------------------------------------------------------------------
        if self.position:
            self.box_end_index = None
            self.entry_limit_price = None
            self.active_deduction_mode = None
            self.check_and_execute_scale_out(ma_price=self.ma20_2m[-1])
            return

        # ---------------------------------------------------------------------
        # 2. Pending Order Management (Dynamic Cancellation)
        # ---------------------------------------------------------------------
        if len(self.orders) > 0 and self.box_end_index is not None:
            deduction_idx_20 = current_idx - self.p_ma20_period
            deduction_idx_43 = current_idx - self.p_ma43_period
            
            buffer_20 = self.box_end_index - deduction_idx_20
            buffer_43 = self.box_end_index - deduction_idx_43
            
            should_cancel = False

            # Logic: Keep order alive if the SPECIFIC mode that triggered it is still valid.
            if self.active_deduction_mode == '20':
                if buffer_20 < self.p_deduction_buffer: should_cancel = True
            elif self.active_deduction_mode == '43':
                if buffer_43 < self.p_deduction_buffer: should_cancel = True
            elif self.active_deduction_mode == 'BOTH':
                if (buffer_20 < self.p_deduction_buffer) and (buffer_43 < self.p_deduction_buffer):
                    should_cancel = True
            
            if should_cancel:
                for order in self.orders:
                    order.cancel()
                self.box_end_index = None
                self.entry_limit_price = None
                self.active_deduction_mode = None
            return

        # ---------------------------------------------------------------------
        # 3. Signal Generation
        # ---------------------------------------------------------------------
        if len(self.data) < self.p_max_box_len + 5:
            return

        # A. Breakout Confirmation (Close > MA43 for 2 bars)
        is_breakout = (self.data.Close[-1] > self.ma43_2m[-1]) and \
                      (self.data.Close[-2] > self.ma43_2m[-2])
        
        if not is_breakout:
            return

        # B. Smart Box Definition (Dynamic Reverse Scan)
        box_end_idx_rel = -3
        box_end_abs_idx = current_idx + box_end_idx_rel
        
        current_box_top = self.data.Close[box_end_idx_rel]
        current_box_bottom = self.data.Close[box_end_idx_rel]
        
        valid_box_len = 1
        
        # Scan Backwards
        for i in range(1, self.p_max_box_len):
            check_idx_rel = box_end_idx_rel - i
            check_close = self.data.Close[check_idx_rel]
            
            upper_limit = current_box_top + self.p_box_tol
            lower_limit = current_box_bottom - self.p_box_tol
            
            if lower_limit <= check_close <= upper_limit:
                current_box_top = max(current_box_top, check_close)
                current_box_bottom = min(current_box_bottom, check_close)
                valid_box_len += 1
            else:
                break
        
        if valid_box_len < self.p_min_box_len:
            return

        # C. Trend Filter (MA20 Continuous Descent at Box Start)
        # Calculate absolute start index of the box
        box_start_abs_idx = box_end_abs_idx - valid_box_len + 1
        
        # Safety check for array bounds
        if box_start_abs_idx < self.p_trend_lookback:
            return

        # Check: MA20 must be strictly falling for 'p_trend_lookback' bars 
        # leading up to and including the box start.
        # Logic: MA[start] < MA[start-1] < MA[start-2] ...
        is_downtrend = True
        for k in range(self.p_trend_lookback):
            # Check index: box_start - k
            idx_curr = box_start_abs_idx - k
            idx_prev = idx_curr - 1
            
            if not (self.ma20_2m[idx_curr] < self.ma20_2m[idx_prev]):
                is_downtrend = False
                break
        
        if not is_downtrend:
            return

        # D. Flexible Deduction Logic (OR Condition)
        deduction_idx_20_abs = current_idx - self.p_ma20_period
        deduction_idx_43_abs = current_idx - self.p_ma43_period
        
        val_20 = self.data.Close[-self.p_ma20_period]
        val_43 = self.data.Close[-self.p_ma43_period]
        
        # Check 1: Value inside Box Range (Inclusive)
        in_range_20 = (current_box_bottom <= val_20 <= current_box_top)
        in_range_43 = (current_box_bottom <= val_43 <= current_box_top)
        
        # Check 2: Time Buffer
        buffer_20 = box_end_abs_idx - deduction_idx_20_abs
        buffer_43 = box_end_abs_idx - deduction_idx_43_abs
        
        valid_20 = in_range_20 and (buffer_20 >= self.p_deduction_buffer)
        valid_43 = in_range_43 and (buffer_43 >= self.p_deduction_buffer)
        
        if not (valid_20 or valid_43):
            return

        # Determine Mode
        if valid_20 and valid_43:
            mode = 'BOTH'
        elif valid_20:
            mode = '20'
        else:
            mode = '43'

        # ---------------------------------------------------------------------
        # 4. Visualization Updates
        # ---------------------------------------------------------------------
        
        if valid_20 and deduction_idx_20_abs >= 0:
            self.arr_deduction_20[deduction_idx_20_abs] = val_20
            
        if valid_43 and deduction_idx_43_abs >= 0:
            self.arr_deduction_43[deduction_idx_43_abs] = val_43

        # Draw Box (Back-fill only)
        start_draw = max(0, box_start_abs_idx)
        end_draw = current_idx + 1 
        
        self.arr_box_high[start_draw : end_draw] = current_box_top
        self.arr_box_low[start_draw : end_draw] = current_box_bottom

        # ---------------------------------------------------------------------
        # 5. Execution
        # ---------------------------------------------------------------------
        limit_price = float(current_box_bottom)
        limit_price = self.round_to_tick(limit_price)

        sl_price = limit_price - self.p_sl_points
        tp_price = limit_price + self.p_tp_points

        self.buy(
            limit=limit_price,
            sl=sl_price,
            tp=tp_price,
            size=self.trade_size
        )

        self.box_end_index = box_end_abs_idx
        self.entry_limit_price = limit_price
        self.active_deduction_mode = mode