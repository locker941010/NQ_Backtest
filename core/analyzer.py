import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import time
import os

class AdaptiveAnalyzer:
    """
    Adaptive Analyzer: Dynamically visualizes strategy results.
    
    Updates for DualTrend:
    1. Adds MA20_5m trace specifically for DualTrend.
    2. Injects MA values into the Main Candlestick Hover tooltip for DualTrend.
    """
    
    @staticmethod
    def round_to_tick(price, tick_size=0.25):
        return round(round(price / tick_size) * tick_size, 2)

    @staticmethod
    def get_aggregated_stats(df):
        if df.empty:
            return [0, 0, 0, "0.00%", "0.00", "0.00"]
        grouped = df.groupby('EntryTime')['PnL'].sum()
        total_signals = len(grouped)
        wins = (grouped > 0).sum()
        losses = (grouped <= 0).sum()
        total_pnl = grouped.sum()
        win_rate = (wins / total_signals * 100) if total_signals > 0 else 0.0
        avg_pnl = (total_pnl / total_signals) if total_signals > 0 else 0.0
        return [total_signals, wins, losses, f"{win_rate:.2f}%", f"{total_pnl:.2f}", f"{avg_pnl:.2f}"]

    @staticmethod
    def print_stats(df, label):
        stats = AdaptiveAnalyzer.get_aggregated_stats(df)
        print(f" {label:10} | Signals: {stats[0]:3} | W/L: {stats[1]:3}/{stats[2]:3} | Rate: {stats[3]:>7} | Net PnL: {stats[4]:>8} | Avg: {stats[5]:>6}")

    @staticmethod
    def _plot_strategy_specifics(fig, strategy, x_axis, df):
        """
        Plots strategy-specific indicators using WebGL.
        """
        strat_name = strategy.__class__.__name__
        is_dual_trend = 'DualTrend' in strat_name
        
        # A. Calculate MAs dynamically
        ma20 = df['Close'].rolling(window=20).mean()
        ma43 = df['Close'].rolling(window=43).mean()

        # B. Plot MA20 (Red)
        fig.add_trace(go.Scattergl(
            x=x_axis, y=ma20, 
            name='MA20 (2m)', 
            line=dict(color='red', width=1.5),
            mode='lines',
            hoverinfo='skip' 
        ), row=1, col=1)

        # C. Plot MA43 (Purple)
        fig.add_trace(go.Scattergl(
            x=x_axis, y=ma43, 
            name='MA43 (2m)', 
            line=dict(color='purple', width=1.5),
            mode='lines',
            hoverinfo='skip' 
        ), row=1, col=1)

        # --- DUAL TREND SPECIFIC: MA20_5m ---
        if is_dual_trend:
            # Try to get MA20_5m from strategy or data
            ma20_5m = None
            if hasattr(strategy, 'ma20_5m_ref') and strategy.ma20_5m_ref is not None:
                ma20_5m = strategy.ma20_5m_ref
            elif 'MA20_5m' in df.columns:
                ma20_5m = df['MA20_5m']
            
            if ma20_5m is not None:
                fig.add_trace(go.Scattergl(
                    x=x_axis, y=ma20_5m,
                    name='MA20 (5m)',
                    line=dict(color='orange', width=2, dash='dot'), # Orange Dotted for visibility
                    mode='lines',
                    hoverinfo='skip'
                ), row=1, col=1)

        # D. Original Strategy MA (Generic fallback)
        if not is_dual_trend and hasattr(strategy, 'ma') and strategy.ma is not None:
             fig.add_trace(go.Scattergl(
                 x=x_axis, y=strategy.ma, 
                 name='MA (Base)', 
                 line=dict(color='blue', width=1, dash='dot'),
                 hoverinfo='skip'
             ), row=1, col=1)

        # Signals
        for attr_name in dir(strategy):
            if attr_name.startswith('debug_'):
                attr_val = getattr(strategy, attr_name)
                if isinstance(attr_val, np.ndarray) and len(attr_val) == len(df):
                    if not np.all(np.isnan(attr_val)):
                        color = 'gray'
                        symbol = 'circle'
                        if 'long' in attr_name or 'bull' in attr_name: 
                            color = 'green'; symbol = 'triangle-up'
                        elif 'short' in attr_name or 'bear' in attr_name: 
                            color = 'red'; symbol = 'triangle-down'
                        elif 'eat' in attr_name:
                            color = 'blue' if '2m' in attr_name else 'purple'
                            symbol = 'diamond'

                        indices = np.where(~np.isnan(attr_val))[0]
                        vals = attr_val[indices]
                        
                        if len(indices) > 0:
                            fig.add_trace(go.Scattergl(
                                x=indices, y=vals, 
                                mode='markers', 
                                marker=dict(symbol=symbol, size=7, color=color, line=dict(width=1, color='black')),
                                name=f"Signal: {attr_name}"
                            ), row=3, col=1)

    @staticmethod
    def generate_report(strategy_instance, df_data, initial_cash=100000):
        manual_trades = strategy_instance.manual_trades
        strat_name = strategy_instance.__class__.__name__
        is_dual_trend = 'DualTrend' in strat_name
        
        print("\n" + "="*90)
        print(f"   Strategy Performance Report (Aggregated by Signal)")
        print("="*90)
        
        if len(manual_trades) > 0:
            df_manual = pd.DataFrame(manual_trades)
            df_long = df_manual[df_manual['Size'] > 0]
            df_short = df_manual[df_manual['Size'] < 0]
            AdaptiveAnalyzer.print_stats(df_long, "LONG")
            AdaptiveAnalyzer.print_stats(df_short, "SHORT")
            print("-" * 90)
            AdaptiveAnalyzer.print_stats(df_manual, "TOTAL")
            
            output_file = f"nq_trade_report_{strat_name}.csv"
            cols_to_write = [c for c in ['EntryTime', 'ExitTime', 'Type', 'Size', 'EntryPrice', 'ExitPrice', 'PnL_Per_Unit', 'PnL', 'PnL_USD'] if c in df_manual.columns]
            df_manual[cols_to_write].to_csv(output_file, index=False)
            print(f"[V] Report CSV Saved: {output_file}")
        else:
            print("[!] No trades found.")

        print("="*90)

        # --- Plot Logic ---
        timestamp = int(time.time())
        html_filename = f'Report_{strat_name}_{timestamp}.html'
        print(f"[*] Generating Adaptive Chart: {html_filename} ...")
        
        x_axis = np.arange(len(df_data))
        date_strings = df_data.index.strftime('%Y-%m-%d %H:%M')
        
        # PnL Calculation
        pnl_array = np.zeros(len(df_data))
        time_to_idx = {t: i for i, t in enumerate(df_data.index)}
        if len(manual_trades) > 0:
            df_trades = pd.DataFrame(manual_trades)
            for _, row in df_trades.iterrows():
                exit_time = row['ExitTime']
                qty = abs(row['Size'])
                pnl_val = (row['PnL'] * 5.0) - (4.0 * qty) 
                idx = time_to_idx.get(exit_time)
                if idx is None:
                    idx = df_data.index.searchsorted(exit_time)
                    if idx >= len(df_data): idx = len(df_data) - 1
                pnl_array[idx] += pnl_val
        
        equity_curve = np.cumsum(pnl_array) + initial_cash

        # --- Subplots ---
        fig = make_subplots(
            rows=5, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.02, 
            subplot_titles=('Price Action (2m + 5m Ghost)', 'Volume (2m + 5m Ghost)', 'Debug Signals', 'Trade PnL', 'Equity Curve'),
            row_heights=[0.5, 0.15, 0.1, 0.1, 0.15]
        )

        # ---------------------------------------------------------
        # PRE-CALCULATION: Time-Based Grouping
        # ---------------------------------------------------------
        use_vis_cols = 'Open_5m_vis' in df_data.columns
        has_5m = use_vis_cols or 'Open_5m' in df_data.columns
        
        col_open = 'Open_5m_vis' if use_vis_cols else 'Open_5m'
        col_high = 'High_5m_vis' if use_vis_cols else 'High_5m'
        col_low  = 'Low_5m_vis'  if use_vis_cols else 'Low_5m'
        col_close= 'Close_5m_vis' if use_vis_cols else 'Close_5m'
        col_vol = 'Volume_5m_vis' if 'Volume_5m_vis' in df_data.columns else 'Volume_5m'
        
        # Arrays
        poly_red_x, poly_red_y = [], []
        poly_green_x, poly_green_y = [], []
        wick_red_x, wick_red_y = [], []
        wick_green_x, wick_green_y = [], []
        vol_red_x, vol_red_y = [], []
        vol_green_x, vol_green_y = [], []
        hover_x, hover_y, hover_text = [], [], []
        vol_hover_x, vol_hover_y, vol_hover_text = [], [], []
        
        if has_5m:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df_data.index):
                    times = pd.to_datetime(df_data.index)
                else:
                    times = df_data.index
                
                group_keys = times.floor('5min')
                codes, uniques = pd.factorize(group_keys)
                change_indices = np.concatenate(([0], np.where(codes[1:] != codes[:-1])[0] + 1, [len(df_data)]))
                
                for i in range(len(change_indices) - 1):
                    start_idx = change_indices[i]
                    end_idx = change_indices[i+1] 
                    
                    x_left = start_idx - 0.45
                    x_right = (end_idx - 1) + 0.45
                    x_center = (x_left + x_right) / 2
                    
                    op = df_data[col_open].iloc[start_idx]
                    cl = df_data[col_close].iloc[start_idx]
                    hi = df_data[col_high].iloc[start_idx]
                    lo = df_data[col_low].iloc[start_idx]
                    vol = df_data[col_vol].iloc[start_idx] if col_vol in df_data.columns else 0
                    
                    t_str = times[start_idx].strftime('%H:%M')
                    h_txt = (f"<b>5m Ghost ({t_str})</b><br>"
                             f"O: {op:.2f}<br>H: {hi:.2f}<br>L: {lo:.2f}<br>C: {cl:.2f}")
                    hover_x.append(x_center)
                    hover_y.append(hi)
                    hover_text.append(h_txt)
                    
                    v_txt = f"<b>5m Vol ({t_str})</b><br>Vol: {vol}"
                    vol_hover_x.append(x_center)
                    vol_hover_y.append(vol)
                    vol_hover_text.append(v_txt)

                    if cl >= op: # Red
                        poly_red_x.extend([x_left, x_left, x_right, x_right, x_left, None])
                        poly_red_y.extend([op, cl, cl, op, op, None])
                        wick_red_x.extend([x_center, x_center, None])
                        wick_red_y.extend([lo, hi, None])
                        vol_red_x.extend([x_left, x_left, x_right, x_right, x_left, None])
                        vol_red_y.extend([0, vol, vol, 0, 0, None])
                    else: # Green
                        poly_green_x.extend([x_left, x_left, x_right, x_right, x_left, None])
                        poly_green_y.extend([op, cl, cl, op, op, None])
                        wick_green_x.extend([x_center, x_center, None])
                        wick_green_y.extend([lo, hi, None])
                        vol_green_x.extend([x_left, x_left, x_right, x_right, x_left, None])
                        vol_green_y.extend([0, vol, vol, 0, 0, None])
                    
            except Exception as e:
                print(f"[!] Error in Ghost Candle Calculation: {e}")
        
        # ---------------------------------------------------------
        # Row 1: Price Action
        # ---------------------------------------------------------
        
        # A. Ghost Wicks
        if wick_red_x:
            fig.add_trace(go.Scattergl(x=wick_red_x, y=wick_red_y, mode='lines', line=dict(color='rgba(255, 0, 0, 0.4)', width=1), hoverinfo='skip', name='Ghost Wick'), row=1, col=1)
        if wick_green_x:
            fig.add_trace(go.Scattergl(x=wick_green_x, y=wick_green_y, mode='lines', line=dict(color='rgba(0, 128, 0, 0.4)', width=1), hoverinfo='skip', name='Ghost Wick'), row=1, col=1)
            
        # B. Ghost Bodies
        if poly_red_x:
            fig.add_trace(go.Scattergl(x=poly_red_x, y=poly_red_y, mode='lines', fill='toself', fillcolor='rgba(255, 0, 0, 0.25)', line=dict(width=0), hoverinfo='skip', name='Ghost Body'), row=1, col=1)
        if poly_green_x:
            fig.add_trace(go.Scattergl(x=poly_green_x, y=poly_green_y, mode='lines', fill='toself', fillcolor='rgba(0, 128, 0, 0.25)', line=dict(width=0), hoverinfo='skip', name='Ghost Body'), row=1, col=1)

        # C. Ghost Hover
        if hover_x:
            fig.add_trace(go.Scattergl(x=hover_x, y=hover_y, mode='markers', marker=dict(size=0, opacity=0), text=hover_text, hoverinfo='text', name='Ghost Info'), row=1, col=1)

        # D. Primary Candles (2m) - CUSTOM HOVER FOR DUAL TREND
        # Prepare Custom Data for Hover
        custom_data = None
        hover_template = '<b>%{text}</b><br>O: %{open:.2f}<br>H: %{high:.2f}<br>L: %{low:.2f}<br>C: %{close:.2f}<extra></extra>'
        
        if is_dual_trend:
            # Calculate MAs for customdata
            ma20 = df_data['Close'].rolling(window=20).mean().fillna(0)
            ma43 = df_data['Close'].rolling(window=43).mean().fillna(0)
            ma20_5m = df_data['MA20_5m'] if 'MA20_5m' in df_data.columns else ma20 # Fallback
            
            # Stack them: [MA20, MA43, MA20_5m]
            custom_data = np.stack((ma20, ma43, ma20_5m), axis=-1)
            
            # Update Template to show MAs
            hover_template = (
                '<b>%{text}</b><br>'
                'O: %{open:.2f}<br>H: %{high:.2f}<br>L: %{low:.2f}<br>C: %{close:.2f}<br>'
                'MA20: %{customdata[0]:.2f}<br>'
                'MA43: %{customdata[1]:.2f}<br>'
                'MA20_5m: %{customdata[2]:.2f}'
                '<extra></extra>'
            )

        fig.add_trace(go.Candlestick(
            x=x_axis, 
            open=df_data['Open'], high=df_data['High'], 
            low=df_data['Low'], close=df_data['Close'],
            name='OHLC (2m)', 
            increasing_line_color='red', 
            decreasing_line_color='green',
            text=date_strings,
            customdata=custom_data, # Inject Data
            hovertemplate=hover_template # Apply Template
        ), row=1, col=1)
        
        # --- Trade Visualization ---
        if len(manual_trades) > 0:
            entry_long_x, entry_long_y, entry_long_text = [], [], []
            entry_short_x, entry_short_y, entry_short_text = [], [], []
            exit_win_x, exit_win_y, exit_win_text = [], [], []
            exit_loss_x, exit_loss_y, exit_loss_text = [], [], []
            trade_lines_x, trade_lines_y = [], []

            for t in manual_trades:
                e_idx = time_to_idx.get(t['EntryTime'])
                x_idx = time_to_idx.get(t['ExitTime'])
                if e_idx is not None and x_idx is not None:
                    if t['Size'] > 0:
                        entry_long_x.append(e_idx)
                        entry_long_y.append(t['EntryPrice'])
                        entry_long_text.append(f"Long Entry<br>{t['EntryPrice']}")
                    else:
                        entry_short_x.append(e_idx)
                        entry_short_y.append(t['EntryPrice'])
                        entry_short_text.append(f"Short Entry<br>{t['EntryPrice']}")
                    
                    pnl = t['PnL']
                    if pnl > 0:
                        exit_win_x.append(x_idx)
                        exit_win_y.append(t['ExitPrice'])
                        exit_win_text.append(f"Win Exit<br>{t['ExitPrice']}<br>PnL: {pnl:.2f}")
                    else:
                        exit_loss_x.append(x_idx)
                        exit_loss_y.append(t['ExitPrice'])
                        exit_loss_text.append(f"Loss Exit<br>{t['ExitPrice']}<br>PnL: {pnl:.2f}")
                    
                    trade_lines_x.extend([e_idx, x_idx, None])
                    trade_lines_y.extend([t['EntryPrice'], t['ExitPrice'], None])

            if trade_lines_x:
                fig.add_trace(go.Scattergl(x=trade_lines_x, y=trade_lines_y, mode='lines', line=dict(color='gray', width=1, dash='dash'), hoverinfo='skip', name='Trade Path'), row=1, col=1)
            if entry_long_x:
                fig.add_trace(go.Scattergl(x=entry_long_x, y=entry_long_y, mode='markers', marker=dict(symbol='triangle-up', size=10, color='blue'), text=entry_long_text, hoverinfo='text', name='Long Entry'), row=1, col=1)
            if entry_short_x:
                fig.add_trace(go.Scattergl(x=entry_short_x, y=entry_short_y, mode='markers', marker=dict(symbol='triangle-down', size=10, color='purple'), text=entry_short_text, hoverinfo='text', name='Short Entry'), row=1, col=1)
            if exit_win_x:
                fig.add_trace(go.Scattergl(x=exit_win_x, y=exit_win_y, mode='markers', marker=dict(symbol='circle', size=8, color='green'), text=exit_win_text, hoverinfo='text', name='Win Exit'), row=1, col=1)
            if exit_loss_x:
                fig.add_trace(go.Scattergl(x=exit_loss_x, y=exit_loss_y, mode='markers', marker=dict(symbol='x', size=8, color='red'), text=exit_loss_text, hoverinfo='text', name='Loss Exit'), row=1, col=1)
        
        AdaptiveAnalyzer._plot_strategy_specifics(fig, strategy_instance, x_axis, df_data)

        # ---------------------------------------------------------
        # Row 2: Volume (Multi-TF)
        # ---------------------------------------------------------
        if vol_red_x:
            fig.add_trace(go.Scattergl(x=vol_red_x, y=vol_red_y, mode='lines', fill='toself', fillcolor='rgba(255, 0, 0, 0.25)', line=dict(width=0), hoverinfo='skip', name='Vol 5m'), row=2, col=1)
        if vol_green_x:
            fig.add_trace(go.Scattergl(x=vol_green_x, y=vol_green_y, mode='lines', fill='toself', fillcolor='rgba(0, 128, 0, 0.25)', line=dict(width=0), hoverinfo='skip', name='Vol 5m'), row=2, col=1)
        if vol_hover_x:
            fig.add_trace(go.Scattergl(x=vol_hover_x, y=vol_hover_y, mode='markers', marker=dict(size=0, opacity=0), text=vol_hover_text, hoverinfo='text', name='Vol 5m Info'), row=2, col=1)

        v2_red_x, v2_red_y = [], []
        v2_green_x, v2_green_y = [], []
        opens = df_data['Open'].values
        closes = df_data['Close'].values
        vols = df_data['Volume'].values
        
        for i in range(len(df_data)):
            x_l, x_r, v = i - 0.4, i + 0.4, vols[i]
            pts_x = [x_l, x_l, x_r, x_r, x_l, None]
            pts_y = [0, v, v, 0, 0, None]
            if closes[i] >= opens[i]:
                v2_red_x.extend(pts_x)
                v2_red_y.extend(pts_y)
            else:
                v2_green_x.extend(pts_x)
                v2_green_y.extend(pts_y)

        if v2_red_x:
            fig.add_trace(go.Scattergl(x=v2_red_x, y=v2_red_y, mode='lines', fill='toself', fillcolor='red', line=dict(width=0), hoverinfo='skip', name='Vol 2m'), row=2, col=1)
        if v2_green_x:
            fig.add_trace(go.Scattergl(x=v2_green_x, y=v2_green_y, mode='lines', fill='toself', fillcolor='green', line=dict(width=0), hoverinfo='skip', name='Vol 2m'), row=2, col=1)
            
        fig.add_trace(go.Scattergl(x=x_axis, y=df_data['Volume'], mode='markers', marker=dict(size=0, opacity=0), name='Volume (2m)', text=date_strings, hovertemplate='<b>%{text}</b><br>Vol: %{y}<extra></extra>'), row=2, col=1)

        # ---------------------------------------------------------
        # Row 3, 4, 5
        # ---------------------------------------------------------
        fig.add_trace(go.Scattergl(x=[0], y=[np.nan], mode='markers', name='_hidden'), row=3, col=1)
        
        nonzero_idx = np.nonzero(pnl_array)[0]
        if len(nonzero_idx) > 0:
            pnl_vals = pnl_array[nonzero_idx]
            pnl_colors = ['red' if p > 0 else 'green' for p in pnl_vals]
            fig.add_trace(go.Bar(x=nonzero_idx, y=pnl_vals, marker_color=pnl_colors, name='Trade PnL'), row=4, col=1)

        fig.add_trace(go.Scattergl(x=x_axis, y=equity_curve, name='Equity', line=dict(color='dodgerblue', width=2), fill='tozeroy'), row=5, col=1)

        # Layout
        num_ticks = 20
        tick_indices = np.linspace(0, len(df_data)-1, num_ticks, dtype=int)
        tick_texts = df_data.index[tick_indices].strftime('%m-%d %H:%M')
        
        fig.update_layout(
            title=f'Backtest Report: {strat_name}', 
            height=1400, 
            template='plotly_white', 
            xaxis_rangeslider_visible=False,
            showlegend=True,
            barmode='overlay',
            hovermode='x' 
        )
        
        fig.update_yaxes(tickformat='.2f', row=1, col=1)
        fig.update_yaxes(fixedrange=False)
        
        for r in range(1, 6):
            fig.update_xaxes(
                tickmode='array', 
                tickvals=tick_indices, 
                ticktext=tick_texts, 
                showgrid=True,
                showticklabels=(r==5), 
                row=r, col=1
            )

        fig.write_html(html_filename)
        
        # JS Injection (Same as before)
        try:
            with open(html_filename, 'a', encoding='utf-8') as f:
                f.write("""
<style>
    #custom-crosshair { position: absolute; top: 0; bottom: 0; width: 0px; border-left: 1px dashed rgba(80, 80, 80, 0.8); background-color: transparent; pointer-events: none; display: none; z-index: 1000; }
    #deduction-line { position: absolute; top: 0; bottom: 0; width: 0px; border-left: 2px dashed rgba(255, 165, 0, 0.9); background-color: transparent; pointer-events: none; display: none; z-index: 999; }
    #deduction-label { position: absolute; top: 50px; background: rgba(255, 165, 0, 0.9); color: white; padding: 2px 5px; font-size: 10px; border-radius: 3px; pointer-events: none; display: none; z-index: 1001; }
    #vol-label { position: absolute; top: 55%; background: rgba(0, 0, 0, 0.7); color: #fff; padding: 2px 5px; font-size: 11px; border-radius: 3px; pointer-events: none; display: none; z-index: 1002; }
</style>
<script>
    (function() {
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (!gd) return;
        var crosshair = document.createElement('div'); crosshair.id = 'custom-crosshair'; gd.appendChild(crosshair);
        var dedLine = document.createElement('div'); dedLine.id = 'deduction-line'; gd.appendChild(dedLine);
        var dedLabel = document.createElement('div'); dedLabel.id = 'deduction-label'; dedLabel.innerText = 'MA20 Ded. (21)'; gd.appendChild(dedLabel);
        var volLabel = document.createElement('div'); volLabel.id = 'vol-label'; gd.appendChild(volLabel);
        var ticking = false; var lastEvt = null; var volTrace = null;

        function update() {
            if (!lastEvt) { ticking = false; return; }
            if (!volTrace && gd.data) { volTrace = gd.data.find(t => t.name === 'Volume (2m)' && t.mode === 'markers'); }
            var rect = gd.getBoundingClientRect();
            var xMouse = lastEvt.clientX - rect.left;
            var xaxis = gd._fullLayout.xaxis;
            var marginLeft = gd._fullLayout.margin.l;
            var graphHeight = rect.height;
            var xRel = xMouse - marginLeft;
            var dataIndex = xaxis.p2c(xRel);
            var snappedIndex = Math.round(dataIndex);
            var xPixelSnapped = xaxis.c2p(snappedIndex);
            var finalPos = marginLeft + xPixelSnapped + 8;
            
            crosshair.style.display = 'block'; crosshair.style.left = finalPos + 'px'; crosshair.style.height = graphHeight + 'px';
            
            var dedIndex = snappedIndex - 20;
            if (dedIndex >= 0) {
                var xPixelDed = xaxis.c2p(dedIndex);
                var dedPos = marginLeft + xPixelDed;
                dedLine.style.display = 'block'; dedLine.style.left = dedPos + 'px'; dedLine.style.height = graphHeight + 'px';
                dedLabel.style.display = 'block'; dedLabel.style.left = (dedPos + 5) + 'px';
            } else { dedLine.style.display = 'none'; dedLabel.style.display = 'none'; }

            if (volTrace && volTrace.y && volTrace.y[snappedIndex] !== undefined) {
                var volVal = volTrace.y[snappedIndex];
                volLabel.innerText = 'Vol: ' + volVal;
                volLabel.style.display = 'block'; volLabel.style.left = (finalPos + 5) + 'px';
            } else { volLabel.style.display = 'none'; }
            ticking = false;
        }
        gd.addEventListener('mousemove', function(evt) { lastEvt = evt; if (!ticking) { requestAnimationFrame(update); ticking = true; } });
        gd.addEventListener('mouseleave', function() { crosshair.style.display = 'none'; dedLine.style.display = 'none'; dedLabel.style.display = 'none'; volLabel.style.display = 'none'; lastEvt = null; });
    })();
</script>
""")
        except: pass

# ==========================================
# Compatibility Layer
# ==========================================
_analyzer_instance = AdaptiveAnalyzer()

def generate_report(strategy_instance, df_data, initial_cash=100000):
    return AdaptiveAnalyzer.generate_report(strategy_instance, df_data, initial_cash)

instance = _analyzer_instance