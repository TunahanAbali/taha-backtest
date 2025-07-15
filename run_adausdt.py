import backtrader as bt
import argparse
import pandas as pd
import datetime

# --- Custom ADX/DI Indicator (from adx_di.txt logic) ---
class ADX_DI(bt.Indicator):
    lines = ('adx', 'diplus', 'diminus')
    params = (('period', 14),)

    def __init__(self):
        len_ = self.p.period
        high = self.data.high
        low = self.data.low
        close = self.data.close

        tr = bt.indicators.Max(
            bt.indicators.Max(high - low, abs(high - close(-1))),
            abs(low - close(-1))
        )
        self.tr = tr

        dmplus = bt.If((high - high(-1)) > (low(-1) - low),
                       bt.Max(high - high(-1), 0), 0)
        dmminus = bt.If((low(-1) - low) > (high - high(-1)),
                        bt.Max(low(-1) - low, 0), 0)

        self.smtr = smtr = bt.indicators.ExponentialMovingAverage(tr, period=len_)
        self.smdmplus = smdmplus = bt.indicators.ExponentialMovingAverage(dmplus, period=len_)
        self.smdmminus = smdmminus = bt.indicators.ExponentialMovingAverage(dmminus, period=len_)

        self.l.diplus = diplus = 100 * smdmplus / smtr
        self.l.diminus = diminus = 100 * smdmminus / smtr
        dx = 100 * abs(diplus - diminus) / (diplus + diminus)
        self.l.adx = bt.indicators.ExponentialMovingAverage(dx, period=len_)

# --- Custom LazyBear WaveTrend Indicator (from lazybear_pinescript.txt logic) ---
class WaveTrend(bt.Indicator):
    # Output lines: wt1 (main), wt2 (signal)
    lines = ('wt1', 'wt2')
    params = (
        ('n1', 10),   # Channel Length (fast EMA period)
        ('n2', 21),   # Average Length (slow EMA period)
    )

    def __init__(self):
        # Calculate HLC3 (average of high, low, close)
        hlc3 = (self.data.high + self.data.low + self.data.close) / 3
        # ESA: Exponential Moving Average of HLC3 (fast period)
        esa = bt.indicators.ExponentialMovingAverage(hlc3, period=self.p.n1)
        # d: EMA of absolute difference between HLC3 and ESA (fast period)
        d = bt.indicators.ExponentialMovingAverage(abs(hlc3 - esa), period=self.p.n1)
        # CI: Channel Index (normalized deviation from ESA)
        ci = (hlc3 - esa) / (0.015 * d)
        # TCI: EMA of CI (slow period)
        tci = bt.indicators.ExponentialMovingAverage(ci, period=self.p.n2)
        # wt1: Main WaveTrend line
        self.l.wt1 = tci
        # wt2: Signal line (SMA of wt1, period=4)
        self.l.wt2 = bt.indicators.SimpleMovingAverage(tci, period=4)

# --- Strategy combining both indicators ---
class CombinedStrategy(bt.Strategy):
    # Strategy parameters: overbought/oversold levels, ADX threshold, indicator periods
    params = (
        ('obLevel1', 80),  # Overbought level for WaveTrend
        ('osLevel1', -80), # Oversold level for WaveTrend
        ('adx_threshold', 25), # Minimum ADX value to allow trades
        ('wt_n1', 10),     # WaveTrend channel length
        ('wt_n2', 21),     # WaveTrend average length
        ('adx_period', 14),# ADX/DI period
    )

    def __init__(self):
        # Instantiate indicators with strategy parameters
        self.wavetrend = WaveTrend(self.data, n1=self.p.wt_n1, n2=self.p.wt_n2)
        self.adxdi = ADX_DI(self.data, period=self.p.adx_period)
        # Detect when WaveTrend crosses above oversold (buy) or below overbought (sell)
        self.crossup = bt.indicators.CrossUp(self.wavetrend.wt1, self.p.osLevel1)
        self.crossdown = bt.indicators.CrossDown(self.wavetrend.wt1, self.p.obLevel1)
        self.trades = []  # Store trade info for later export
        self._last_entry_datetime = None  # Track last entry datetime for trade logging
        self._last_entry_price = None     # Track last entry price for trade logging

    def notify_order(self, order):
        # Called on order status changes
        if order.status == order.Completed and order.isbuy():
            # Store entry info for the trade when a buy order is completed
            self._last_entry_datetime = self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0])
            self._last_entry_price = order.executed.price

    def notify_trade(self, trade):
        # Called when a trade is closed
        if trade.isclosed:
            entry_datetime = self._last_entry_datetime if self._last_entry_datetime is not None else 'N/A'
            entry_price = self._last_entry_price if self._last_entry_price is not None else 'N/A'
            # Record trade details for later export
            self.trades.append({
                'entry_datetime': entry_datetime,
                'entry_price': entry_price,
                'exit_datetime': self.data.datetime.datetime(0).isoformat() if hasattr(self.data.datetime, 'datetime') else str(self.data.datetime[0]),
                'exit_price': self.data.close[0],
                'size': trade.size,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm
            })
            # Reset entry info for next trade
            self._last_entry_datetime = None
            self._last_entry_price = None

    def next(self):
        # Called on every new bar/candle
        if not self.position:
            # If not in a position, check for buy signal
            if self.crossup[0] and self.adxdi.adx[0] > self.p.adx_threshold:
                self.buy()  # Enter long position
        else:
            # If in a position, check for sell signal
            if self.crossdown[0] and self.adxdi.adx[0] > self.p.adx_threshold:
                self.sell()  # Exit position

# --- WaveTrend-Only Strategy for AVAXUSDT (2024) ---
class WaveTrendOnlyStrategy(bt.Strategy):
    params = (
        ('obLevel1', 60),  # Overbought level for WaveTrend
        ('osLevel1', -60), # Oversold level for WaveTrend
        ('obLevel2', 20),  # Overbought level 2 for exit
        ('osLevel2', -20), # Oversold level 2 for cover
        ('wt_n1', 10),     # WaveTrend channel length
        ('wt_n2', 21),     # WaveTrend average length
    )

    def __init__(self):
        self.wavetrend = WaveTrend(self.data, n1=self.p.wt_n1, n2=self.p.wt_n2)
        self.crossup = bt.indicators.CrossUp(self.wavetrend.wt1, self.p.osLevel1)
        self.crossdown = bt.indicators.CrossDown(self.wavetrend.wt1, self.p.obLevel1)
        self.trades = []
        self._last_entry_datetime = None
        self._last_entry_price = None
        self._long = False
        self._short = False
        self._last_entry_bar = None  # Track bar index of last entry

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
        dt = self.data.datetime.datetime(0)
        if not (datetime.datetime(2024, 1, 1) <= dt < datetime.datetime(2025, 1, 1)):
            return
        wt1 = self.wavetrend.wt1[0]
        wt1_prev = self.wavetrend.wt1[-1]
        # LONG ENTRY: wt1 crosses -60 upwards
        if not self.position and self.crossup[0]:
            self.buy()
            self._long = True
            self._short = False
            self._last_entry_bar = len(self)
        # SHORT ENTRY: wt1 crosses +60 downwards
        elif not self.position and self.crossdown[0]:
            self.sell()
            self._short = True
            self._long = False
            self._last_entry_bar = len(self)
        # LONG EXIT: first negative slope above +20, at least one bar after entry
        elif self.position and self._long and self._last_entry_bar is not None:
            if (len(self) > self._last_entry_bar) and wt1 > self.p.obLevel2 and wt1 < wt1_prev:
                self.close()
                self._long = False
        # SHORT EXIT: first positive slope below -20, at least one bar after entry
        elif self.position and self._short and self._last_entry_bar is not None:
            if (len(self) > self._last_entry_bar) and wt1 < self.p.osLevel2 and wt1 > wt1_prev:
                self.close()
                self._short = False

# --- Main script ---
def run():
    parser = argparse.ArgumentParser(description='Run Backtrader with ADAUSDT.csv and custom indicators.')
    parser.add_argument('--data', type=str, default='ADAUSDT.csv', help='CSV data file')
    parser.add_argument('--obLevel1', type=float, default=80, help='WaveTrend Overbought Level 1')
    parser.add_argument('--osLevel1', type=float, default=-80, help='WaveTrend Oversold Level 1')
    parser.add_argument('--adx_threshold', type=float, default=20, help='ADX threshold')
    parser.add_argument('--wt_n1', type=int, default=10, help='WaveTrend n1 (Channel Length)')
    parser.add_argument('--wt_n2', type=int, default=21, help='WaveTrend n2 (Average Length)')
    parser.add_argument('--adx_period', type=int, default=14, help='ADX/DI period')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    args = parser.parse_args()

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        CombinedStrategy,
        obLevel1=args.obLevel1,
        osLevel1=args.osLevel1,
        adx_threshold=args.adx_threshold,
        wt_n1=args.wt_n1,
        wt_n2=args.wt_n2,
        adx_period=args.adx_period
    )

    data = bt.feeds.GenericCSVData(
        dataname=args.data,
        dtformat=('%Y-%m-%dT%H:%M:%S'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=60,
        headers=True
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(10000)
    cerebro.addsizer(bt.sizers.AllInSizer)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Trades)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    result = cerebro.run()
    print('Final Portfolio Value: %.4f' % cerebro.broker.getvalue())

    # Write trades to txt file
    strategy = result[0]  # Only one strategy instance
    with open('trades.txt', 'w') as f:
        f.write('Entry Date\tEntry Price\tExit Date\tExit Price\tSize\tPnL\tPnL (Comm)\n')
        for trade in getattr(strategy, 'trades', []):
            f.write(f"{trade['entry_datetime']}\t{trade['entry_price']}\t{trade['exit_datetime']}\t{trade['exit_price']}\t{trade['size']}\t{trade['pnl']}\t{trade['pnlcomm']}\n")

    if args.plot:
        cerebro.plot()

# --- Main script for AVAXUSDT WaveTrend Only ---
def run_wavetrend_only():
    parser = argparse.ArgumentParser(description='Run Backtrader with AVAXUSDT.csv and WaveTrend indicator.')
    parser.add_argument('--data', type=str, default='AVAXUSDT.csv', help='CSV data file')
    parser.add_argument('--obLevel1', type=float, default=60, help='WaveTrend Overbought Level 1')
    parser.add_argument('--osLevel1', type=float, default=-60, help='WaveTrend Oversold Level 1')
    parser.add_argument('--wt_n1', type=int, default=10, help='WaveTrend n1 (Channel Length)')
    parser.add_argument('--wt_n2', type=int, default=21, help='WaveTrend n2 (Average Length)')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    args = parser.parse_args()

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        WaveTrendOnlyStrategy,
        obLevel1=args.obLevel1,
        osLevel1=args.osLevel1,
        wt_n1=args.wt_n1,
        wt_n2=args.wt_n2
    )

    data = bt.feeds.GenericCSVData(
        dataname=args.data,
        dtformat=('%Y-%m-%dT%H:%M:%S'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=60,
        headers=True
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(10000)
    cerebro.addsizer(bt.sizers.AllInSizer)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Trades)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    result = cerebro.run()
    print('Final Portfolio Value: %.4f' % cerebro.broker.getvalue())

    # Write trades to txt file
    strategy = result[0]
    with open('trades_wavetrend_only.txt', 'w') as f:
        f.write('Entry Date\tEntry Price\tExit Date\tExit Price\tSize\tPnL\tPnL (Comm)\n')
        for trade in getattr(strategy, 'trades', []):
            f.write(f"{trade['entry_datetime']}\t{trade['entry_price']}\t{trade['exit_datetime']}\t{trade['exit_price']}\t{trade['size']}\t{trade['pnl']}\t{trade['pnlcomm']}\n")

    if args.plot:
        cerebro.plot()

if __name__ == '__main__':
    run_wavetrend_only() 