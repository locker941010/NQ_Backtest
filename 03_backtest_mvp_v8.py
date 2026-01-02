import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest
import numpy as np
import os
import glob
import time
import csv # [新增] 用於自定義 CSV 寫入

# [引入 Plotly]
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 工具函數
# ==========================================
def round_to_tick(price, tick_size=0.25):
    return round(round(price / tick_size) * tick_size, 2)

# ==========================================
# 2. 策略邏輯 (保持 v7 邏輯)
# ==========================================
class Strategy_NQ_Dual_v7(Strategy):
    p_ma20_len = 20
    p_ma43_len = 43
    p_sl_points = 16.0
    p_tp_points = 20.0
    p_spread_threshold = 20.0
    p_entry_offset = 0.0 
    p_tolerance = 1.0 
    
    p_deduction_ago = 34
    p_deduction_window = 10

    def init(self):
        if not hasattr(self.data, 'MA20_5m'):
            raise ValueError("[Error] Data missing 'MA20_5m'.")

        close = pd.Series(self.data.Close)
        self.ma20_2m = self.I(ta.sma, close, length=self.p_ma20_len, name='MA20', color='blue')
        self.ma43_2m = self.I(ta.sma, close, length=self.p_ma43_len, name='MA43', color='orange')
        self.ma20_5m_ref = self.I(lambda x: x, self.data.MA20_5m, name='MA20_5m', color='purple', overlay=True)
        
        self.debug_signal_short = self.I(self.calc_signal_short_array, name='Signal_Short', color='red', scatter=True)
        self.debug_signal_long = self.I(self.calc_signal_long_array, name='Signal_Long', color='green', scatter=True)
        self.debug_deduction_dots = self.I(self.calc_deduction_marker_array, name='Deduction', color='orange', scatter=True)

        self.manual_trades = [] 
        self.ideal_entry_price = None 

    # --- 核心邏輯 ---
    def check_intra_bar_outcome(self, open_p, high, low, close, entry_price, sl_price, tp_price, current_time, direction):
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
            if direction == -1:
                return 'SL' if dist_open_to_high < dist_open_to_low else 'TP'
            else:
                return 'SL' if dist_open_to_low < dist_open_to_high else 'TP'

        if not is_sl_hit and is_tp_hit and not is_close_win:
            is_open_filled = (open_p >= entry_price) if direction == -1 else (open_p <= entry_price)
            if is_open_filled: return 'TP'
            
            dist_to_tp_side = dist_open_to_low if direction == -1 else dist_open_to_high
            if dist_open_to_entry < dist_to_tp_side:
                return 'TP'
            else:
                return 'HOLD'

        return 'HOLD'

    # --- 輔助計算 ---
    def calc_signal_short_array(self):
        close = self.data.Close
        s_close = pd.Series(close)
        ma20 = ta.sma(s_close, length=self.p_ma20_len).to_numpy()
        ma43 = ta.sma(s_close, length=self.p_ma43_len).to_numpy()
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
        s_close = pd.Series(close)
        ma20 = ta.sma(s_close, length=self.p_ma20_len).to_numpy()
        ma43 = ta.sma(s_close, length=self.p_ma43_len).to_numpy()
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
        for i in range(20, len(high)):
            deduction_idx = i - 20
            if sig_s[i] == 1:
                markers[deduction_idx] = high[deduction_idx] + 5.0
            elif sig_l[i] == 1:
                markers[deduction_idx] = low[deduction_idx] - 5.0
        return markers

    def next(self):
        for order in self.orders:
            order.cancel()

        if self.position:
            current_trade = self.trades[-1]
            is_entry_bar = (current_trade.entry_bar == len(self.data) - 1)
            direction = 1 if self.position.is_long else -1
            
            if self.ideal_entry_price is None:
                self.ideal_entry_price = current_trade.entry_price
                self.entry_time_marker = current_trade.entry_time

            if direction == -1:
                target_sl = self.ideal_entry_price + self.p_sl_points
                target_tp = self.ideal_entry_price - self.p_tp_points
            else:
                target_sl = self.ideal_entry_price - self.p_sl_points
                target_tp = self.ideal_entry_price + self.p_tp_points
            
            outcome = 'HOLD'
            
            if is_entry_bar:
                outcome = self.check_intra_bar_outcome(
                    self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1],
                    self.ideal_entry_price, target_sl, target_tp, self.data.index[-1], direction
                )
            else:
                if direction == -1:
                    hit_sl = self.data.High[-1] >= target_sl
                    hit_tp = self.data.Low[-1] <= target_tp
                else:
                    hit_sl = self.data.Low[-1] <= target_sl
                    hit_tp = self.data.High[-1] >= target_tp
                
                if hit_sl and hit_tp:
                    dist_high = abs(self.data.Open[-1] - self.data.High[-1])
                    dist_low = abs(self.data.Open[-1] - self.data.Low[-1])
                    if direction == -1:
                        outcome = 'SL' if dist_high < dist_low else 'TP'
                    else:
                        outcome = 'SL' if dist_low < dist_high else 'TP'
                elif hit_sl:
                    outcome = 'SL'
                elif hit_tp:
                    outcome = 'TP'
            
            if outcome != 'HOLD':
                exit_price = target_sl if outcome == 'SL' else target_tp
                pnl = -self.p_sl_points if outcome == 'SL' else self.p_tp_points
                entry_time = getattr(self, 'entry_time_marker', self.data.index[-1])

                self.manual_trades.append({
                    'EntryTime': entry_time,
                    'ExitTime': self.data.index[-1],
                    'EntryPrice': self.ideal_entry_price,
                    'ExitPrice': exit_price,
                    'PnL': pnl,
                    'Size': direction,
                    'Type': outcome
                })
                self.position.close()
                self.ideal_entry_price = None
            return

        if len(self.data) < 60: return

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        open_p = self.data.Open
        ma20 = self.ma20_2m
        ma43 = self.ma43_2m
        ma20_5m = self.ma20_5m_ref
        
        current_spread = abs(ma20[-1] - ma43[-1])
        cond_spread = current_spread >= self.p_spread_threshold
        
        short_trend = (ma20[-1] < ma43[-1]) and (ma20[-1] < ma20[-2] and ma20[-2] < ma20[-3] and ma20[-3] < ma20[-4])
        short_deduction = min(close[-44:-34]) > close[-1]
        short_location = close[-1] < ma20_5m[-1]
        short_signal = short_trend and cond_spread and short_deduction and short_location

        long_trend = (ma20[-1] > ma43[-1]) and (ma20[-1] > ma20[-2] and ma20[-2] > ma20[-3] and ma20[-3] > ma20[-4])
        long_deduction = max(close[-44:-34]) < close[-1]
        long_location = close[-1] > ma20_5m[-1]
        long_signal = long_trend and cond_spread and long_deduction and long_location

        target_direction = 0
        if short_signal:
            target_direction = -1
            raw_limit_price = ma20_5m[-1] + self.p_entry_offset
        elif long_signal:
            target_direction = 1
            raw_limit_price = ma20_5m[-1] - self.p_entry_offset
            
        if target_direction != 0:
            is_hit = False
            if target_direction == -1:
                is_hit = (raw_limit_price - high[-1]) <= self.p_tolerance
            else:
                is_hit = (low[-1] - raw_limit_price) <= self.p_tolerance
            
            if is_hit:
                limit_price = round_to_tick(raw_limit_price, 0.25)
                if target_direction == -1:
                    target_sl = limit_price + self.p_sl_points
                    target_tp = limit_price - self.p_tp_points
                else:
                    target_sl = limit_price - self.p_sl_points
                    target_tp = limit_price + self.p_tp_points
                
                outcome = self.check_intra_bar_outcome(
                    open_p[-1], high[-1], low[-1], close[-1],
                    limit_price, target_sl, target_tp, self.data.index[-1], target_direction
                )
                
                if outcome == 'SL' or outcome == 'TP':
                    pnl = -self.p_sl_points if outcome == 'SL' else self.p_tp_points
                    exit_px = target_sl if outcome == 'SL' else target_tp
                    self.manual_trades.append({
                        'EntryTime': self.data.index[-1],
                        'ExitTime': self.data.index[-1],
                        'EntryPrice': limit_price,
                        'ExitPrice': exit_px,
                        'PnL': pnl,
                        'Size': target_direction,
                        'Type': outcome
                    })
                elif outcome == 'HOLD':
                    self.ideal_entry_price = limit_price
                    self.entry_time_marker = self.data.index[-1]
                    if target_direction == -1: self.sell(size=1)
                    else: self.buy(size=1)
            else:
                if target_direction == -1: self.sell(limit=raw_limit_price, size=1)
                else: self.buy(limit=raw_limit_price, size=1)

# ==========================================
# 3. 自定義繪圖函數 (Plotly - 儀表板版 v17)
# ==========================================
def plot_custom_chart(df, strategy, trades_df, filename, initial_cash=100000):
    print(f"[*] Generating Custom Plotly Chart: {filename} ...")
    
    x_axis = np.arange(len(df))
    pnl_array = np.zeros(len(df))
    if not trades_df.empty:
        trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime'])
        time_to_idx = {t: i for i, t in enumerate(df.index)}
        
        for _, row in trades_df.iterrows():
            exit_time = row['ExitTime']
            pnl_val = row['PnL'] * 5.0 - 4.0 
            
            if exit_time in time_to_idx:
                idx = time_to_idx[exit_time]
                pnl_array[idx] += pnl_val
            else:
                try:
                    idx = df.index.searchsorted(exit_time)
                    if idx < len(df):
                        pnl_array[idx] += pnl_val
                except:
                    pass

    equity_curve = np.cumsum(pnl_array) + initial_cash
    
    fig = make_subplots(
        rows=5, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        subplot_titles=('Price & Trades', 'Volume', 'Signals & Deduction', 'Profit/Loss (Per Trade)', 'Equity Curve'),
        row_heights=[0.5, 0.1, 0.1, 0.15, 0.15],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]]
    )

    # Row 1: Price
    fig.add_trace(go.Candlestick(
        x=x_axis, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC', increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=strategy.ma20_2m, name='MA20 (2m)', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=strategy.ma43_2m, name='MA43 (2m)', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=strategy.ma20_5m_ref, name='MA20 (5m)', line=dict(color='purple', width=1.5)), row=1, col=1)

    long_trades = trades_df[trades_df['Size'] > 0]
    short_trades = trades_df[trades_df['Size'] < 0]

    if not long_trades.empty:
        fig.add_trace(go.Scatter(x=long_trades['EntryBar'], y=long_trades['EntryPrice'], mode='markers', marker=dict(symbol='arrow-up', size=12, color='red', line=dict(width=2, color='black')), name='Long Entry'), row=1, col=1)
        fig.add_trace(go.Scatter(x=long_trades['ExitBar'], y=long_trades['ExitPrice'], mode='markers', marker=dict(symbol='x', size=8, color='black'), name='Long Exit'), row=1, col=1)

    if not short_trades.empty:
        fig.add_trace(go.Scatter(x=short_trades['EntryBar'], y=short_trades['EntryPrice'], mode='markers', marker=dict(symbol='arrow-down', size=12, color='green', line=dict(width=2, color='black')), name='Short Entry'), row=1, col=1)
        fig.add_trace(go.Scatter(x=short_trades['ExitBar'], y=short_trades['ExitPrice'], mode='markers', marker=dict(symbol='x', size=8, color='black'), name='Short Exit'), row=1, col=1)

    # Row 2: Volume
    colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=x_axis, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # Row 3: Signals
    sig_short = strategy.debug_signal_short
    sig_long = strategy.debug_signal_long
    deduction = strategy.debug_deduction_dots
    idx_short = np.where(~np.isnan(sig_short))[0]
    idx_long = np.where(~np.isnan(sig_long))[0]
    idx_deduction = np.where(~np.isnan(deduction))[0]

    if len(idx_short) > 0: fig.add_trace(go.Scatter(x=idx_short, y=[-1]*len(idx_short), mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Short Signal'), row=3, col=1)
    if len(idx_long) > 0: fig.add_trace(go.Scatter(x=idx_long, y=[1]*len(idx_long), mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Long Signal'), row=3, col=1)
    if len(idx_deduction) > 0: fig.add_trace(go.Scatter(x=idx_deduction, y=[0]*len(idx_deduction), mode='markers', marker=dict(symbol='circle', size=6, color='orange'), name='Deduction Time'), row=3, col=1)
    fig.update_yaxes(range=[-1.5, 1.5], tickvals=[-1, 0, 1], ticktext=['Short', 'Deduction', 'Long'], row=3, col=1)

    # Row 4: PnL
    nonzero_idx = np.nonzero(pnl_array)[0]
    if len(nonzero_idx) > 0:
        pnl_vals = pnl_array[nonzero_idx]
        pnl_colors = ['red' if p > 0 else 'green' for p in pnl_vals]
        fig.add_trace(go.Bar(x=nonzero_idx, y=pnl_vals, marker_color=pnl_colors, width=30, name='Trade PnL ($)'), row=4, col=1)

    # Row 5: Equity
    fig.add_trace(go.Scatter(x=x_axis, y=equity_curve, name='Equity', line=dict(color='dodgerblue', width=2), fill='tozeroy', fillcolor='rgba(30, 144, 255, 0.1)'), row=5, col=1)
    eq_min, eq_max = np.min(equity_curve), np.max(equity_curve)
    padding = max((eq_max - eq_min) * 0.1, 1000)
    fig.update_yaxes(range=[max(0, eq_min - padding), eq_max + padding], row=5, col=1)

    # Layout
    num_ticks = 20
    tick_indices = np.linspace(0, len(df)-1, num_ticks, dtype=int)
    tick_texts = df.index[tick_indices].strftime('%m-%d %H:%M')
    fig.update_layout(title='NQ Strategy Backtest Dashboard (Gapless)', height=1200, template='plotly_white', xaxis_rangeslider_visible=False, showlegend=True)
    fig.update_xaxes(tickmode='array', tickvals=tick_indices, ticktext=tick_texts, showgrid=True)
    fig.update_yaxes(autorange=True, fixedrange=False, row=1, col=1)
    fig.update_yaxes(autorange=True, fixedrange=False, row=2, col=1)
    fig.update_yaxes(autorange=True, fixedrange=False, row=4, col=1)

    fig.write_html(filename)
    print(f"[V] Custom Chart Saved: {filename}")

# ==========================================
# 4. 統計計算輔助函數
# ==========================================
def get_stats_list(df):
    """回傳 [Trades, Win, Loss, WinRate, PnL] 的列表"""
    if df.empty:
        return [0, 0, 0, "0.00%", "0.00"]
    
    total = len(df)
    wins = len(df[df['PnL'] > 0])
    losses = len(df[df['PnL'] <= 0])
    rate = (wins / total * 100)
    pnl = df['PnL'].sum()
    
    return [total, wins, losses, f"{rate:.2f}%", f"{pnl:.2f}"]

def print_stats(df, label):
    stats = get_stats_list(df)
    print(f" {label:10} | Trades: {stats[0]:3} | W/L: {stats[1]:3}/{stats[2]:3} | Rate: {stats[3]:>7} | PnL: {stats[4]:>8}")

# ==========================================
# 5. 主程式
# ==========================================
def run_backtest():
    data_dir = "data_processed"
    pattern = os.path.join(data_dir, "*Backtest*.csv")
    files = glob.glob(pattern)
    if not files: return
    data_file = sorted(files)[-1]
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    print("[*] Running Final Backtest (v18 - CSV Report Fix)...")
    initial_cash = 100000
    bt = Backtest(df, Strategy_NQ_Dual_v7, cash=initial_cash, commission=0, margin=0.05)
    stats = bt.run()
    
    strategy_instance = stats['_strategy']
    manual_trades = strategy_instance.manual_trades
    
    if len(manual_trades) > 0:
        df_manual = pd.DataFrame(manual_trades)
        
        # 1. 價格修正
        price_cols = ['EntryPrice', 'ExitPrice']
        for col in price_cols:
            df_manual[col] = df_manual[col].apply(lambda x: round_to_tick(x, 0.25))

        # 2. Intra-bar 標記
        for idx, row in df_manual.iterrows():
            if row['EntryTime'] == row['ExitTime']:
                if 'Intra-Bar' not in str(row['Type']):
                    df_manual.at[idx, 'Type'] = f"Intra-Bar ({row['Type']})"
        
        # 3. [新增] Direction 欄位
        df_manual['Direction'] = df_manual['Size'].apply(lambda x: 'Long' if x > 0 else 'Short')
        
        # 4. 補回索引計算 (為了畫圖)
        time_to_idx = {t: i for i, t in enumerate(df.index)}
        df_manual['EntryBar'] = df_manual['EntryTime'].map(time_to_idx).fillna(0).astype(int)
        df_manual['ExitBar'] = df_manual['ExitTime'].map(time_to_idx).fillna(0).astype(int)
        
        # 5. Console 輸出
        print("\n" + "="*80)
        print(f"   Strategy Performance Report (Detailed)")
        print("="*80)
        
        df_long = df_manual[df_manual['Size'] > 0]
        df_short = df_manual[df_manual['Size'] < 0]
        
        print_stats(df_long, "LONG")
        print_stats(df_short, "SHORT")
        print("-" * 80)
        print_stats(df_manual, "TOTAL")
        print("="*80)

        # 6. [新增] 自定義 CSV 輸出 (含表頭統計)
        output_file = "nq_trade_report_final_v18.csv"
        
        # 準備統計數據
        stats_long = get_stats_list(df_long)
        stats_short = get_stats_list(df_short)
        stats_total = get_stats_list(df_manual)
        
        # 調整 DataFrame 欄位順序
        cols_order = [
            'EntryTime', 'ExitTime', 'Direction', 'Type', 'Size', 
            'EntryPrice', 'ExitPrice', 'PnL', 
            'EntryBar', 'ExitBar'
        ]
        # 確保只輸出存在的欄位
        cols_to_write = [c for c in cols_order if c in df_manual.columns]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 寫入統計區塊
            writer.writerow(['Statistics', 'Trades', 'Win', 'Loss', 'Win Rate', 'Total PnL'])
            writer.writerow(['Long'] + stats_long)
            writer.writerow(['Short'] + stats_short)
            writer.writerow(['Total'] + stats_total)
            writer.writerow([]) # 空行
            
            # 寫入交易明細標題
            writer.writerow(cols_to_write)
            
        # 附加交易明細數據
        df_manual[cols_to_write].to_csv(output_file, mode='a', header=False, index=False)
        
        print(f"\n[V] Report Saved: {output_file}")
        
        timestamp = int(time.time())
        html_filename = f'Chart_Dashboard_v18_{timestamp}.html'
        
        plot_custom_chart(df, strategy_instance, df_manual, html_filename, initial_cash)
        
    else:
        print("[!] No trades found.")

if __name__ == "__main__":
    run_backtest()