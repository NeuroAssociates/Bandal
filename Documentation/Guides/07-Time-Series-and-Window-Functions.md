# Guide 07: Time Series Analysis & Window Functions

> **Language**: **English** | [한국어](07-Time-Series-and-Window-Functions.ko.md)  
> **Navigation**: [← Guide 06: Grouping, Joining & Reshaping](06-Grouping-Joining-and-Reshaping.md) | [API Reference](../API.md) | **Next**: [Guide 08: MLX Acceleration & End-to-End Workflows →](08-MLX-Acceleration-and-Workflows.md)

---

## Overview

Financial tick streams, IoT telemetry, and user activity logs are ordered in time. Analyzing them requires specialized window and temporal alignment primitives that go beyond static group aggregations:
- **Row-Based Rolling Windows** (`df.rolling(_:window:minPeriods:using:)`): Moving averages, sums, volatility (`standardDeviation`), and extrema over a fixed number of preceding rows.
- **Time-Interval Rolling Windows** (`df.rolling(_:on:interval:using:)`): Trailing windows defined by a physical duration (`TimeInterval` in seconds) over irregular timestamps.
- **Expanding Windows** (`df.expanding(_:using:)`): Cumulative prefix aggregations from row `0` up to the current row `i`.
- **Exponentially Weighted Moving Average (`EWMA`)** (`df.exponentiallyWeighted` / `Series.exponentiallyWeighted`): Recency-weighted smoothing specified by `alpha` or `span`.
- **As-Of Joins (`pd.merge_asof`)** (`df.joinAsOf`): Matching each left timestamp (e.g., a trade) to the closest preceding, succeeding, or nearest right timestamp (e.g., a quote), optionally partitioned `by` ticker and bounded by a time `tolerance`.
- **Time-Bucket Resampling (`df.resample`)**: Downsampling irregular `.date` rows into fixed-width UTC time buckets (`1m`, `5m`, `1h`, `1d`) with multiple aggregations.

---

## 1. Quick Comparison: `pandas` vs. `Bandal`

| Operation | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| Row-based rolling window | `df["price"].rolling(3, min_periods=1).mean()` | `try df.rolling("price", window: 3, minPeriods: 1, using: .mean)` |
| Time-based rolling window | `df.rolling("60s", on="ts")["price"].sum()` | `try df.rolling("price", on: "ts", interval: 60, using: .sum)` |
| Expanding window | `df["price"].expanding().max()` | `try df.expanding("price", using: .max)` |
| EWMA by `alpha` | `df["price"].ewm(alpha=0.5, adjust=False).mean()` | `try df.exponentiallyWeighted("price", alpha: 0.5, as: "ewma")` |
| EWMA by `span` | `df["price"].ewm(span=3, adjust=False).mean()` | `try df.exponentiallyWeighted("price", span: 3.0, as: "ewma_span3")` |
| As-Of Join (`merge_asof`) | `pd.merge_asof(trades, quotes, on="ts", by="sym", direction="backward", tolerance=pd.Timedelta("5s"))` | `try trades.joinAsOf(quotes, on: "ts", by: "sym", direction: .backward, tolerance: 5.0)` |
| Time-bucket resampling | `df.resample("3600s", on="ts").agg({"price": ["first", "last"], "vol": "sum"})` | `try df.resample(on: "ts", interval: 3600, aggregations: [...])` |

---

## 2. Row-Based Rolling Windows (`rolling(_:window:minPeriods:using:)`)

`df.rolling(_:window:minPeriods:using:)` computes a trailing window aggregation over up to `window` rows ending at the current row and appends a new column named **`<column>_<operation>_rolling_<window>`**.

- **`minPeriods`**: Minimum number of non-missing observations required inside the window to produce a value (defaults to `window`, meaning the first `window - 1` rows are `.missing`). Set `minPeriods: 1` to compute partial windows from the very first row.
- **Hardware Acceleration**:
  - `.sum`, `.mean`, and `.count` use a single-pass $O(N)$ sliding accumulator.
  - Large non-null `.min`, `.max`, `.variance`, and `.standardDeviation` windows use zero-copy 2D strided MLX views (`MLX.asStrided`) to reduce all windows in parallel.

```swift
import Bandal
import Foundation

let stock = try DataFrame(columns: [
    (name: "day", values: [1, 2, 3, 4, 5]),
    (name: "close", values: [100.0, 106.0, 103.0, 109.0, 112.0]),
])
.rolling("close", window: 3, using: .mean)               // First 2 rows are .missing
.rolling("close", window: 3, minPeriods: 1, using: .max) // Partial windows allowed

print(stock)
```

**Console Output**:
```text
   day  close  close_mean_rolling_3  close_max_rolling_3
0    1    100                   NaN                  100
1    2    106                   NaN                  106
2    3    103                   103                  106
3    4    109                   106                  109
4    5    112                   108                  112

[5 rows x 4 columns]
```

---

## 3. Time-Interval Rolling Windows (`rolling(_:on:interval:using:)`)

Real-world event logs rarely arrive at uniform intervals. When your frame has a `.date` column, `df.rolling(_:on:interval:using:)` defines the window for row $i$ (at timestamp $t_i$) as all rows $j \le i$ whose timestamp $t_j$ falls inside the half-open interval **$(t_i - \text{interval}, t_i]$**.

The appended column is named **`<valueColumn>_<operation>_rolling_<interval>s`**.

```swift
let baseTime = Date(timeIntervalSince1970: 1_700_000_000)

// Events at t = 0s, 10s, 25s, 70s
let requests = try DataFrame(columns: [
    (name: "ts", values: [
        .date(baseTime),
        .date(baseTime.addingTimeInterval(10)),
        .date(baseTime.addingTimeInterval(25)),
        .date(baseTime.addingTimeInterval(70)),
    ]),
    (name: "latency_ms", values: [12.0, 18.0, 30.0, 15.0]),
])
// Trailing 30-second window:
// - At t=0s:  ( -30s,  0s] -> [12]         -> sum = 12
// - At t=10s: ( -20s, 10s] -> [12, 18]     -> sum = 30
// - At t=25s: (  -5s, 25s] -> [12, 18, 30] -> sum = 60
// - At t=70s: (  40s, 70s] -> [15]         -> sum = 15
.rolling("latency_ms", on: "ts", interval: 30, using: .sum)

print(requests)
```

**Console Output**:
```text
                     ts  latency_ms  latency_ms_sum_rolling_30s
0  2023-11-14T22:13:20Z          12                          12
1  2023-11-14T22:13:30Z          18                          30
2  2023-11-14T22:13:45Z          30                          60
3  2023-11-14T22:14:30Z          15                          15

[4 rows x 3 columns]
```

---

## 4. Expanding Windows & Exponentially Weighted Moving Average (`EWMA`)

### 4.1 Expanding Windows (`expanding(_:using:)`)

`expanding(_:using:)` aggregates all observations from index `0` through the current row `i`, appending `<column>_<operation>_expanding`:

```swift
let expandingDemo = try DataFrame(columns: [
    (name: "step", values: [1, 2, 3, 4]),
    (name: "pnl", values: [10.0, -4.0, 18.0, 8.0]),
])
.expanding("pnl", using: .mean)
.expanding("pnl", using: .max)

print(expandingDemo)
```

**Console Output**:
```text
   step  pnl  pnl_mean_expanding  pnl_max_expanding
0     1   10                  10                 10
1     2   -4                   3                 10
2     3   18                   8                 18
3     4    8                   8                 18

[4 rows x 4 columns]
```

### 4.2 Exponentially Weighted Moving Average (`exponentiallyWeighted` / `ewm`)

`exponentiallyWeighted` computes the unadjusted recursive EWMA (`pandas` `ewm(..., adjust=False).mean()`):

$$y_0 = x_0, \qquad y_t = \alpha x_t + (1 - \alpha) y_{t-1}$$

Missing values carry the previous state forward unchanged until the next valid observation arrives. You can specify either:
- **`alpha`**: Smoothing factor $\alpha \in (0, 1]$ directly, or
- **`span`**: Span $\ge 1$, where $\alpha = \frac{2}{\text{span} + 1}$.

```swift
let signalDF = try DataFrame(columns: [
    (name: "t", values: [1, 2, 3, 4]),
    (name: "price", values: [100.0, 110.0, 105.0, 115.0]),
])
.exponentiallyWeighted("price", alpha: 0.5, as: "ewma_alpha05")
.exponentiallyWeighted("price", span: 3.0, as: "ewma_span3") // span=3 -> alpha = 2/(3+1) = 0.5

print(signalDF)
```

**Console Output**:
```text
   t  price  ewma_alpha05  ewma_span3
0  1    100           100         100
1  2    110           105         105
2  3    105         105.0       105.0
3  4    115           110         110

[4 rows x 4 columns]
```

---

## 5. Time-Series As-Of Joins (`joinAsOf` / `pd.merge_asof`)

When combining two asynchronous streams—such as trades and order-book quotes, or user clicks and A/B test configuration changes—exact timestamp joins (`join(on: "ts")`) almost never match because timestamps differ by milliseconds.

`df.joinAsOf(_:on:by:direction:tolerance:suffixes:)` solves this in $O(N_{\text{left}} + N_{\text{right}} \log N_{\text{right}})$ time via binary search:
- **`on: String`**: The `.date` or `.number` timestamp/sequence column present in both frames.
- **`by: String? = nil`**: Optional secondary key column (e.g. `"ticker"` or `"sensor_id"`) that must match **exactly** before searching for the closest timestamp.
- **`direction: AsOfDirection = .backward`**:
  - `.backward`: Matches the latest right row with $t_{\text{right}} \le t_{\text{left}}$ (no lookahead bias!).
  - `.forward`: Matches the earliest right row with $t_{\text{right}} \ge t_{\text{left}}$.
  - `.nearest`: Matches the right row with the smallest absolute gap $|t_{\text{right}} - t_{\text{left}}|$.
- **`tolerance: TimeInterval? = nil`**: Maximum allowed distance $|t_{\text{right}} - t_{\text{left}}|$ (in seconds for `.date` keys, or numeric units for `.number` keys). Matches exceeding `tolerance` become `.missing`.

```swift
let t0 = Date(timeIntervalSince1970: 1_700_000_000)

// Trades occurring at t = +2s, +8s, +20s
let trades = try DataFrame(columns: [
    (name: "ts", values: [
        .date(t0.addingTimeInterval(2)),
        .date(t0.addingTimeInterval(8)),
        .date(t0.addingTimeInterval(20)),
    ]),
    (name: "ticker", values: ["AAPL", "AAPL", "AAPL"]),
    (name: "trade_qty", values: [100.0, 50.0, 200.0]),
])

// Quotes published at t = 0s and t = +6s
let quotes = try DataFrame(columns: [
    (name: "ts", values: [
        .date(t0),
        .date(t0.addingTimeInterval(6)),
    ]),
    (name: "ticker", values: ["AAPL", "AAPL"]),
    (name: "bid", values: [185.0, 185.5]),
])

// Match each trade to the most recent quote within 5 seconds
// - Trade at +2s matches quote at 0s   (gap = 2s <= 5s) -> bid = 185.0
// - Trade at +8s matches quote at +6s  (gap = 2s <= 5s) -> bid = 185.5
// - Trade at +20s has gap = 14s > 5s tolerance          -> bid = NaN (.missing)
let matchedTrades = try trades.joinAsOf(
    quotes,
    on: "ts",
    by: "ticker",
    direction: .backward,
    tolerance: 5.0
)

print(matchedTrades)
```

**Console Output**:
```text
                     ts  ticker  trade_qty    bid
0  2023-11-14T22:13:22Z    AAPL        100    185
1  2023-11-14T22:13:28Z    AAPL         50  185.5
2  2023-11-14T22:13:40Z    AAPL        200    NaN

[3 rows x 4 columns]
```

---

## 6. Fixed-Interval Time Resampling (`resample`)

`df.resample(on:interval:aggregations:)` groups a `.date` column into non-overlapping, UTC-epoch-aligned time buckets of width `interval` seconds ($\lfloor t / \text{interval} \rfloor \times \text{interval}$) and applies your `[Aggregation]` list to each populated bucket in chronological order.

### Example: Constructing OHLCV Candlesticks from Tick Data

In quantitative finance, raw trade ticks can be converted into 1-minute (`interval: 60`) Open-High-Low-Close-Volume (**OHLCV**) bars in a single `resample` call:

```swift
let minute0 = Date(timeIntervalSince1970: 1_700_000_040) // 22:14:00 UTC

let ticks = try DataFrame(columns: [
    (name: "ts", values: [
        .date(minute0.addingTimeInterval(5)),   // 22:14:05 (Bucket 1)
        .date(minute0.addingTimeInterval(25)),  // 22:14:25 (Bucket 1)
        .date(minute0.addingTimeInterval(50)),  // 22:14:50 (Bucket 1)
        .date(minute0.addingTimeInterval(70)),  // 22:15:10 (Bucket 2)
        .date(minute0.addingTimeInterval(110)), // 22:15:50 (Bucket 2)
    ]),
    (name: "price", values: [100.0, 104.0, 99.0, 99.5, 102.0]),
    (name: "volume", values: [10.0, 25.0, 15.0, 30.0, 20.0]),
])

let ohlcv = try ticks.resample(
    on: "ts",
    interval: 60, // 60-second (1-minute) bars
    aggregations: [
        Aggregation("price", using: .first, as: "open"),
        Aggregation("price", using: .max,   as: "high"),
        Aggregation("price", using: .min,   as: "low"),
        Aggregation("price", using: .last,  as: "close"),
        Aggregation("volume", using: .sum,  as: "volume"),
    ]
)

print(ohlcv)
```

**Console Output**:
```text
                     ts  open  high   low  close  volume
0  2023-11-14T22:14:00Z   100   104    99     99      50
1  2023-11-14T22:15:00Z  99.5   102  99.5    102      50

[2 rows x 6 columns]
```

---

> **Next Step**: Proceed to **[Guide 08: MLX Acceleration & End-to-End Workflows →](08-MLX-Acceleration-and-Workflows.md)** to understand Bandal's hybrid execution engine (`Accelerate` + `mlx-swift`), `ExecutionPolicy`, direct `MLXArray` interop, and a complete end-to-end data analysis case study.
