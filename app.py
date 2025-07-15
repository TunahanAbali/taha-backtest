import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta, date
import backtrader as bt
import importlib.util
import sys
from backtrader.feeds import PandasData
import matplotlib.pyplot as plt
import io

# --- Import custom strategies and indicators from run_adausdt.py ---
spec = importlib.util.spec_from_file_location("run_adausdt", "run_adausdt.py")
if spec is not None and spec.loader is not None:
    run_adausdt = importlib.util.module_from_spec(spec)
    sys.modules["run_adausdt"] = run_adausdt
    spec.loader.exec_module(run_adausdt)
    CombinedStrategy = run_adausdt.CombinedStrategy
else:
    st.error("Could not import run_adausdt.py")

# --- WaveTrend Strategy ---
class WaveTrendStrategy(bt.Strategy):
    params = (
        ('obLevel1', 80),
        ('obLevel2', -20),
        ('osLevel1', -80),
        ('osLevel2', 20),
        ('wt_n1', 10),
        ('wt_n2', 21),
    )

    def __init__(self):
        class WaveTrend(bt.Indicator):
            lines = ('wt1', 'wt2')
            params = (('n1', 10), ('n2', 21))
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.addminperiod(self.p.n1 + self.p.n2 + 4)
            def next(self):
                hlc3 = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
                if not hasattr(self, 'esa_list'):
                    self.esa_list = []
                    self.d_list = []
                    self.ci_list = []
                n1 = self.p.n1
                n2 = self.p.n2
                # ESA
                if len(self.esa_list) == 0:
                    esa = hlc3
                else:
                    esa = (hlc3 * (2/(n1+1))) + (self.esa_list[-1] * (1-(2/(n1+1))))
                self.esa_list.append(esa)
                # D
                diff = abs(hlc3 - esa)
                if len(self.d_list) == 0:
                    d = diff
                else:
                    d = (diff * (2/(n1+1))) + (self.d_list[-1] * (1-(2/(n1+1))))
                self.d_list.append(d)
                # CI
                ci = (hlc3 - esa) / (0.015 * d) if d != 0 else 0
                self.ci_list.append(ci)
                # TCI
                if len(self.ci_list) < n2:
                    tci = 0
                else:
                    tci = sum(self.ci_list[-n2:]) / n2
                self.l.wt1[0] = tci
                # WT2
                if len(self.l.wt1.get(size=4)) < 4:
                    self.l.wt2[0] = 0
                else:
                    self.l.wt2[0] = sum(self.l.wt1.get(size=4)) / 4
        self.wavetrend = WaveTrend(self.data, n1=self.p.wt_n1, n2=self.p.wt_n2)
        self.trades = []
        self._last_entry_datetime = None
        self._last_entry_price = None
        self._long = False
        self._short = False
        self._last_entry_bar = None

    def notify_order(self, order):
        if order.status == order.Completed and order.isbuy():
            self._last_entry_datetime = self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0])
            self._last_entry_price = order.executed.price
            self._last_entry_bar = len(self)
        elif order.status == order.Completed and order.issell():
            self._last_entry_datetime = self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0])
            self._last_entry_price = order.executed.price
            self._last_entry_bar = len(self)

    def notify_trade(self, trade):
        if trade.isclosed:
            entry_datetime = self._last_entry_datetime if self._last_entry_datetime is not None else 'N/A'
            entry_price = self._last_entry_price if self._last_entry_price is not None else 'N/A'
            self.trades.append({
                'entry_datetime': entry_datetime,
                'entry_price': entry_price,
                'exit_datetime': self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0]),
                'exit_price': self.data.close[0],
                'size': trade.size,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm
            })
            self._last_entry_datetime = None
            self._last_entry_price = None
            self._last_entry_bar = None

    def next(self):
        wt1 = self.wavetrend.wt1[0]
        wt1_prev = self.wavetrend.wt1[-1]
        # LONG ENTRY: wt1 crosses above osLevel1
        if not self.position and wt1_prev < self.p.osLevel1 and wt1 >= self.p.osLevel1:
            self.buy()
            self._long = True
            self._short = False
            self._last_entry_bar = len(self)
        # SHORT ENTRY: wt1 crosses below obLevel1
        elif not self.position and wt1_prev > self.p.obLevel1 and wt1 <= self.p.obLevel1:
            self.sell()
            self._short = True
            self._long = False
            self._last_entry_bar = len(self)
        # LONG EXIT: first negative slope of wt1 above obLevel2
        elif self.position and self._long and self._last_entry_bar is not None:
            if (len(self) > self._last_entry_bar) and wt1 > self.p.obLevel2 and wt1 < wt1_prev:
                self.close()
                self._long = False
        # SHORT EXIT: first positive slope of wt1 below osLevel2
        elif self.position and self._short and self._last_entry_bar is not None:
            if (len(self) > self._last_entry_bar) and wt1 < self.p.osLevel2 and wt1 > wt1_prev:
                self.close()
                self._short = False

# --- ADX/DI Strategy ---
class ADXStrategy(bt.Strategy):
    params = (
        ('adx_period', 14),
        ('adx_threshold', 20),
    )
    def __init__(self):
        class ADX_DI(bt.Indicator):
            lines = ('adx', 'diplus', 'diminus')
            params = (('period', 14),)
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.addminperiod(self.p.period + 1)
            def next(self):
                n = self.p.period
                high = self.data.high[0]
                low = self.data.low[0]
                close = self.data.close[0]
                prev_high = self.data.high[-1]
                prev_low = self.data.low[-1]
                prev_close = self.data.close[-1]
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                dmplus = high - prev_high if (high - prev_high) > (prev_low - low) and (high - prev_high) > 0 else 0
                dmminus = prev_low - low if (prev_low - low) > (high - prev_high) and (prev_low - low) > 0 else 0
                if not hasattr(self, 'tr_list'):
                    self.tr_list = []
                    self.dmplus_list = []
                    self.dmminus_list = []
                self.tr_list.append(tr)
                self.dmplus_list.append(dmplus)
                self.dmminus_list.append(dmminus)
                if len(self.tr_list) < n:
                    smtr = sum(self.tr_list) / len(self.tr_list)
                    smdmplus = sum(self.dmplus_list) / len(self.dmplus_list)
                    smdmminus = sum(self.dmminus_list) / len(self.dmminus_list)
                else:
                    smtr = sum(self.tr_list[-n:]) / n
                    smdmplus = sum(self.dmplus_list[-n:]) / n
                    smdmminus = sum(self.dmminus_list[-n:]) / n
                diplus = 100 * smdmplus / smtr if smtr != 0 else 0
                diminus = 100 * smdmminus / smtr if smtr != 0 else 0
                dx = 100 * abs(diplus - diminus) / (diplus + diminus) if (diplus + diminus) != 0 else 0
                if not hasattr(self, 'dx_list'):
                    self.dx_list = []
                self.dx_list.append(dx)
                if len(self.dx_list) < n:
                    adx = sum(self.dx_list) / len(self.dx_list)
                else:
                    adx = sum(self.dx_list[-n:]) / n
                self.l.diplus[0] = diplus
                self.l.diminus[0] = diminus
                self.l.adx[0] = adx
        self.adxdi = ADX_DI(self.data, period=self.p.adx_period)
        self.trades = []
        self._last_entry_datetime = None
        self._last_entry_price = None
        self._long = False
        self._short = False
        self._last_entry_bar = None

    def notify_order(self, order):
        if order.status == order.Completed and order.isbuy():
            self._last_entry_datetime = self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0])
            self._last_entry_price = order.executed.price
            self._last_entry_bar = len(self)
        elif order.status == order.Completed and order.issell():
            self._last_entry_datetime = self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0])
            self._last_entry_price = order.executed.price
            self._last_entry_bar = len(self)

    def notify_trade(self, trade):
        if trade.isclosed:
            entry_datetime = self._last_entry_datetime if self._last_entry_datetime is not None else 'N/A'
            entry_price = self._last_entry_price if self._last_entry_price is not None else 'N/A'
            self.trades.append({
                'entry_datetime': entry_datetime,
                'entry_price': entry_price,
                'exit_datetime': self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0]),
                'exit_price': self.data.close[0],
                'size': trade.size,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm
            })
            self._last_entry_datetime = None
            self._last_entry_price = None
            self._last_entry_bar = None

    def next(self):
        adx = self.adxdi.adx[0]
        adx_prev = self.adxdi.adx[-1]
        # LONG ENTRY: ADX < threshold
        if not self.position and adx < self.p.adx_threshold:
            self.buy()
            self._long = True
            self._short = False
            self._last_entry_bar = len(self)
        # SHORT ENTRY: ADX < threshold
        elif not self.position and adx < self.p.adx_threshold:
            self.sell()
            self._short = True
            self._long = False
            self._last_entry_bar = len(self)
        # LONG EXIT: ADX > threshold
        elif self.position and self._long and adx > self.p.adx_threshold:
            self.close()
            self._long = False
        # SHORT EXIT: ADX > threshold
        elif self.position and self._short and adx > self.p.adx_threshold:
            self.close()
            self._short = False

# --- Combined Strategy ---
class CombinedStrategy(bt.Strategy):
    params = (
        ('obLevel1', 60),
        ('obLevel2', 53),
        ('osLevel1', -60),
        ('osLevel2', -53),
        ('wt_n1', 10),
        ('wt_n2', 21),
        ('adx_period', 14),
        ('adx_threshold', 20),
    )
    def __init__(self):
        # WaveTrend
        class WaveTrend(bt.Indicator):
            lines = ('wt1', 'wt2')
            params = (('n1', 10), ('n2', 21))
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.addminperiod(self.p.n1 + self.p.n2 + 4)
            def next(self):
                hlc3 = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3
                if not hasattr(self, 'esa_list'):
                    self.esa_list = []
                    self.d_list = []
                    self.ci_list = []
                n1 = self.p.n1
                n2 = self.p.n2
                # ESA
                if len(self.esa_list) == 0:
                    esa = hlc3
                else:
                    esa = (hlc3 * (2/(n1+1))) + (self.esa_list[-1] * (1-(2/(n1+1))))
                self.esa_list.append(esa)
                # D
                diff = abs(hlc3 - esa)
                if len(self.d_list) == 0:
                    d = diff
                else:
                    d = (diff * (2/(n1+1))) + (self.d_list[-1] * (1-(2/(n1+1))))
                self.d_list.append(d)
                # CI
                ci = (hlc3 - esa) / (0.015 * d) if d != 0 else 0
                self.ci_list.append(ci)
                # TCI
                if len(self.ci_list) < n2:
                    tci = 0
                else:
                    tci = sum(self.ci_list[-n2:]) / n2
                self.l.wt1[0] = tci
                # WT2
                if len(self.l.wt1.get(size=4)) < 4:
                    self.l.wt2[0] = 0
                else:
                    self.l.wt2[0] = sum(self.l.wt1.get(size=4)) / 4
        self.wavetrend = WaveTrend(self.data, n1=self.p.wt_n1, n2=self.p.wt_n2)
        # ADX/DI
        class ADX_DI(bt.Indicator):
            lines = ('adx', 'diplus', 'diminus')
            params = (('period', 14),)
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.addminperiod(self.p.period + 1)
            def next(self):
                n = self.p.period
                high = self.data.high[0]
                low = self.data.low[0]
                close = self.data.close[0]
                prev_high = self.data.high[-1]
                prev_low = self.data.low[-1]
                prev_close = self.data.close[-1]
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                dmplus = high - prev_high if (high - prev_high) > (prev_low - low) and (high - prev_high) > 0 else 0
                dmminus = prev_low - low if (prev_low - low) > (high - prev_high) and (prev_low - low) > 0 else 0
                if not hasattr(self, 'tr_list'):
                    self.tr_list = []
                    self.dmplus_list = []
                    self.dmminus_list = []
                self.tr_list.append(tr)
                self.dmplus_list.append(dmplus)
                self.dmminus_list.append(dmminus)
                if len(self.tr_list) < n:
                    smtr = sum(self.tr_list) / len(self.tr_list)
                    smdmplus = sum(self.dmplus_list) / len(self.dmplus_list)
                    smdmminus = sum(self.dmminus_list) / len(self.dmminus_list)
                else:
                    smtr = sum(self.tr_list[-n:]) / n
                    smdmplus = sum(self.dmplus_list[-n:]) / n
                    smdmminus = sum(self.dmminus_list[-n:]) / n
                diplus = 100 * smdmplus / smtr if smtr != 0 else 0
                diminus = 100 * smdmminus / smtr if smtr != 0 else 0
                dx = 100 * abs(diplus - diminus) / (diplus + diminus) if (diplus + diminus) != 0 else 0
                if not hasattr(self, 'dx_list'):
                    self.dx_list = []
                self.dx_list.append(dx)
                if len(self.dx_list) < n:
                    adx = sum(self.dx_list) / len(self.dx_list)
                else:
                    adx = sum(self.dx_list[-n:]) / n
                self.l.diplus[0] = diplus
                self.l.diminus[0] = diminus
                self.l.adx[0] = adx
        self.adxdi = ADX_DI(self.data, period=self.p.adx_period)
        self.trades = []
        self._last_entry_datetime = None
        self._last_entry_price = None
        self._long = False
        self._short = False
        self._last_entry_bar = None

    def notify_order(self, order):
        if order.status == order.Completed and order.isbuy():
            self._last_entry_datetime = self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0])
            self._last_entry_price = order.executed.price
            self._last_entry_bar = len(self)
        elif order.status == order.Completed and order.issell():
            self._last_entry_datetime = self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0])
            self._last_entry_price = order.executed.price
            self._last_entry_bar = len(self)

    def notify_trade(self, trade):
        if trade.isclosed:
            entry_datetime = self._last_entry_datetime if self._last_entry_datetime is not None else 'N/A'
            entry_price = self._last_entry_price if self._last_entry_price is not None else 'N/A'
            self.trades.append({
                'entry_datetime': entry_datetime,
                'entry_price': entry_price,
                'exit_datetime': self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0]),
                'exit_price': self.data.close[0],
                'size': trade.size,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm
            })
            self._last_entry_datetime = None
            self._last_entry_price = None
            self._last_entry_bar = None

    def next(self):
        wt1 = self.wavetrend.wt1[0]
        wt1_prev = self.wavetrend.wt1[-1]
        adx = self.adxdi.adx[0]
        adx_prev = self.adxdi.adx[-1]
        # LONG ENTRY: both strategies' entry
        if not self.position and wt1_prev < self.p.osLevel1 and wt1 >= self.p.osLevel1 and adx < self.p.adx_threshold:
            self.buy()
            self._long = True
            self._short = False
            self._last_entry_bar = len(self)
        # SHORT ENTRY: both strategies' entry
        elif not self.position and wt1_prev > self.p.obLevel1 and wt1 <= self.p.obLevel1 and adx < self.p.adx_threshold:
            self.sell()
            self._short = True
            self._long = False
            self._last_entry_bar = len(self)
        # LONG EXIT: both strategies' exit
        elif self.position and self._long and self._last_entry_bar is not None:
            if (len(self) > self._last_entry_bar) and wt1 > self.p.obLevel2 and wt1 < wt1_prev and adx > self.p.adx_threshold:
                self.close()
                self._long = False
        # SHORT EXIT: both strategies' exit
        elif self.position and self._short and self._last_entry_bar is not None:
            if (len(self) > self._last_entry_bar) and wt1 < self.p.osLevel2 and wt1 > wt1_prev and adx > self.p.adx_threshold:
                self.close()
                self._short = False

st.set_page_config(page_title="Crypto Multi-Backtester", layout="wide")

# --- 3. Backtest Logic ---
def run_backtest(df, obLevel1, obLevel2, osLevel1, osLevel2, adx_threshold, use_wavetrend, use_adx):
    try:
        min_version = '1.9.74.123'
        if tuple(map(int, bt.__version__.split('.'))) < tuple(map(int, min_version.split('.'))):
            st.error(f"Backtrader version {bt.__version__} is too old. Please upgrade to at least {min_version} for PandasData support.")
            return None, []
        cerebro = bt.Cerebro()
        # Strategy selection logic
        if use_wavetrend and use_adx:
            cerebro.addstrategy(CombinedStrategy, obLevel1=obLevel1, obLevel2=obLevel2, osLevel1=osLevel1, osLevel2=osLevel2, adx_threshold=adx_threshold)
        elif use_wavetrend:
            cerebro.addstrategy(WaveTrendStrategy, obLevel1=obLevel1, obLevel2=obLevel2, osLevel1=osLevel1, osLevel2=osLevel2)
        elif use_adx:
            cerebro.addstrategy(ADXStrategy, adx_threshold=adx_threshold)
        else:
            st.error("Please select at least one strategy.")
            return None, []
        df_bt = df.copy()
        df_bt.set_index('datetime', inplace=True)
        data = PandasData(dataname=df_bt)
        cerebro.adddata(data)
        cerebro.broker.setcash(1000)
        cerebro.addsizer(bt.sizers.AllInSizer)
        cerebro.addobserver(bt.observers.BuySell)
        cerebro.addobserver(bt.observers.Trades)
        result = cerebro.run()
        final_value = cerebro.broker.getvalue()
        strategy = result[0]
        trades = getattr(strategy, 'trades', [])
        return final_value, trades
    except Exception as e:
        st.error(f"Backtest error for DataFrame: {e}")
        st.write(df.head())
        raise

# --- 4. Streamlit UI ---
st.title("Crypto Multi-Backtester (Top 200 by Market Cap, Local Data)")

# Diagnostic info for debugging PandasData linter error
st.write(f'Backtrader version: {bt.__version__}')
import inspect
st.write(f'PandasData class: {PandasData}')
st.write(f'PandasData __init__ signature: {inspect.signature(PandasData.__init__)}')

st.sidebar.header("Backtest Settings")

default_end = datetime.utcnow().date()
default_start = default_end - timedelta(days=4*365)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)

# --- Strategy selection ---
use_wavetrend = st.sidebar.checkbox("Use WaveTrend Strategy", value=True)
use_adx = st.sidebar.checkbox("Use ADX/DI Strategy", value=False)

# --- WaveTrend parameters ---
obLevel1 = st.sidebar.number_input("WaveTrend Overbought Level 1", value=80)
obLevel2 = st.sidebar.number_input("WaveTrend Overbought Level 2", value=-20)
osLevel1 = st.sidebar.number_input("WaveTrend Oversold Level 1", value=-80)
osLevel2 = st.sidebar.number_input("WaveTrend Oversold Level 2", value=20)

# --- ADX/DI parameters ---
adx_threshold = st.sidebar.number_input("ADX Threshold", value=20)

# --- List available coins from local data folder ---
data_dir = "data"
if not os.path.exists(data_dir):
    st.error(f"Data folder '{data_dir}' not found. Please upload OHLCV CSVs.")
    st.stop()
csv_files = [f for f in os.listdir(data_dir) if f.endswith("_4h.csv")]
coin_options = [f.replace("_4h.csv", "") for f in csv_files]

selected_symbols = st.sidebar.multiselect(
    "Select coins (USDT pairs)", options=coin_options, default=coin_options[:10]
)

def load_ohlcv(symbol):
    csv_path = os.path.join(data_dir, f"{symbol}_4h.csv")
    df = pd.read_csv(csv_path)
    # Filter by date range
    df["datetime"] = pd.to_datetime(df["datetime"])
    mask = (df["datetime"] >= pd.Timestamp(start_date)) & (df["datetime"] <= pd.Timestamp(end_date))
    return df.loc[mask].copy()

if st.sidebar.button("Run Backtest"):
    st.info("Running backtest. This may take a few minutes...")
    results = []
    trade_details = {}
    all_trades = []
    total_final_capital = 0
    for symbol in selected_symbols:
        try:
            df = load_ohlcv(symbol)
            if not isinstance(df, pd.DataFrame) or df.empty:
                result = {
                    'symbol': symbol,
                    'bars': 0,
                    'start': None,
                    'end': None,
                    'final_value': None,
                    'num_trades': 0,
                    'note': 'No data in CSV'
                }
                trade_details[symbol] = []
            else:
                final_value, trades = run_backtest(df, obLevel1, obLevel2, osLevel1, osLevel2, adx_threshold, use_wavetrend, use_adx)
                result = {
                    'symbol': symbol,
                    'bars': len(df),
                    'start': df['datetime'].min(),
                    'end': df['datetime'].max(),
                    'final_value': final_value,
                    'num_trades': len(trades),
                    'note': 'Backtest OK'
                }
                trade_details[symbol] = trades
                all_trades.extend(trades)
                if final_value is not None:
                    total_final_capital += final_value
            results.append(result)
        except Exception as e:
            st.error(f"Exception for {symbol}: {e}")
            results.append({'symbol': symbol, 'bars': 0, 'start': None, 'end': None, 'final_value': None, 'num_trades': 0, 'note': f'Error: {e}'} )
            trade_details[symbol] = []
    results_df = pd.DataFrame(results)
    # --- SUMMARY TABLE FOR ALL COINS ---
    if all_trades:
        all_trades_df = pd.DataFrame(all_trades)
        # Group trades by coin
        summary_rows = []
        for symbol in selected_symbols:
            trades = trade_details.get(symbol, [])
            if not trades:
                continue
            trades_df = pd.DataFrame(trades)
            num_trades = len(trades_df)
            num_wins = (trades_df['pnl'] > 0).sum()
            success_pct = (num_wins / num_trades * 100) if num_trades > 0 else 0
            starting_capital = 1000
            final_capital = None
            for r in results:
                if r['symbol'] == symbol:
                    final_capital = r['final_value']
                    break
            # Calculate profit % for each trade
            trades_df['profit_$'] = trades_df['pnl']
            trades_df['profit_%'] = 100 * trades_df['pnl'] / starting_capital
            # Prepare trade summary for display
            trade_display = trades_df[['entry_datetime','exit_datetime','profit_$','profit_%']].copy()
            trade_display = trade_display.rename(columns={'entry_datetime':'Entry Date','exit_datetime':'Exit Date','profit_$':'Profit ($)','profit_%':'Profit (%)'})
            summary_rows.append({
                'Coin': symbol,
                'Total Trades': num_trades,
                'Success Rate (%)': success_pct,
                'Starting Capital': starting_capital,
                'Final Capital': final_capital,
                'Trades': trade_display
            })
        # Sort by success rate descending
        summary_rows = sorted(summary_rows, key=lambda x: x['Success Rate (%)'], reverse=True)
        # Build summary DataFrame for display
        summary_df = pd.DataFrame([{k: v for k, v in row.items() if k != 'Trades'} for row in summary_rows])
        st.markdown("**Summary Table (All Coins, Sorted by Success Rate)**")
        st.dataframe(summary_df)
        # Show trades per coin in expandable sections
        for row in summary_rows:
            with st.expander(f"Trades for {row['Coin']}"):
                st.dataframe(row['Trades'])
    # --- TRADE DETAILS PER COIN ---
    for symbol in selected_symbols:
        with st.expander(f"Trade Details: {symbol}"):
            trades = trade_details.get(symbol, [])
            if trades:
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df)
                st.download_button(f"Download {symbol} Trades CSV", trades_df.to_csv(index=False), file_name=f"{symbol}_trades.csv")
                # Show summary stats
                num_trades = len(trades_df)
                num_wins = (trades_df['pnl'] > 0).sum()
                success_pct = (num_wins / num_trades * 100) if num_trades > 0 else 0
                total_pnl = trades_df['pnl'].sum()
                st.markdown(f"**Number of trades:** {num_trades}")
                st.markdown(f"**Success percentage:** {success_pct:.2f}%")
                st.markdown(f"**Total P&L:** {total_pnl:.2f}")
                # Plot price with entry/exit markers
                df = load_ohlcv(symbol)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df['datetime'], df['close'], label='Close Price', color='blue')
                # Entry/exit markers
                for _, trade in trades_df.iterrows():
                    try:
                        entry_dt = pd.to_datetime(trade['entry_datetime'])
                        exit_dt = pd.to_datetime(trade['exit_datetime'])
                        entry_px = float(trade['entry_price'])
                        exit_px = float(trade['exit_price'])
                        ax.scatter(entry_dt, entry_px, color='green', marker='^', label='Entry', zorder=5)
                        ax.scatter(exit_dt, exit_px, color='red', marker='v', label='Exit', zorder=5)
                    except Exception:
                        continue
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                ax.set_title(f"{symbol} Price with Trade Entries/Exits")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                st.pyplot(fig)
            else:
                st.write("No trades.")
else:
    st.write("Select your settings and click 'Run Backtest' to begin.") 