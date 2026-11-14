# Guide 07: Time Series Analysis & Window Functions

Analyze ordered temporal data using row-based and time-interval rolling windows, expanding windows, exponentially weighted moving averages (`EWMA`), `joinAsOf` (`merge_asof`), and fixed-interval `resample`.

## Overview

Financial tick streams, IoT telemetry, and user activity logs are ordered in time. Analyzing them requires specialized window and temporal alignment primitives that go beyond static group aggregations:
- **Row-Based Rolling Windows** (``DataFrame/rolling(_:window:minPeriods:using:)``): Moving averages, sums, volatility (`standardDeviation`), and extrema over a fixed number of preceding rows.
- **Time-Interval Rolling Windows** (``DataFrame/rolling(_:on:interval:using:)``): Trailing windows defined by a physical duration (`TimeInterval` in seconds) over irregular timestamps.
- **Expanding Windows** (``DataFrame/expanding(_:using:)``): Cumulative prefix aggregations from row `0` up to the current row `i`.
- **Exponentially Weighted Moving Average (`EWMA`)** (``DataFrame/exponentiallyWeighted(_:alpha:adjust:as:)``): Recency-weighted smoothing specified by `alpha` or `span`.
- **As-Of Joins (`pd.merge_asof`)** (``DataFrame/joinAsOf(with:on:by:direction:tolerance:)``): Matching each left timestamp (e.g., a trade) to the closest preceding, succeeding, or nearest right timestamp (e.g., a quote), optionally partitioned `by` ticker and bounded by a time `tolerance`.
- **Time-Bucket Resampling** (``DataFrame/resample(on:interval:aggregations:)``): Downsampling irregular `.date` rows into fixed-width UTC time buckets (`1m`, `5m`, `1h`, `1d`) with multiple aggregations.

---

## 1. Row-Based & Time-Interval Rolling Windows

```swift
import Bandal
import Foundation

let stock = try DataFrame(columns: [
    (name: "day", values: [1, 2, 3, 4, 5]),
    (name: "close", values: [100.0, 106.0, 103.0, 109.0, 112.0]),
])
.rolling("close", window: 3, using: .mean)
.rolling("close", window: 3, minPeriods: 1, using: .max)

let baseTime = Date(timeIntervalSince1970: 1_700_000_000)
let requests = try DataFrame(columns: [
    (name: "ts", values: [
        .date(baseTime),
        .date(baseTime.addingTimeInterval(10)),
        .date(baseTime.addingTimeInterval(25)),
        .date(baseTime.addingTimeInterval(70)),
    ]),
    (name: "latency_ms", values: [12.0, 18.0, 30.0, 15.0]),
])
.rolling("latency_ms", on: "ts", interval: 30, using: .sum)
```

---

## 2. Expanding Windows & Exponentially Weighted Moving Average (`EWMA`)

```swift
let signalDF = try DataFrame(columns: [
    (name: "t", values: [1, 2, 3, 4]),
    (name: "price", values: [100.0, 110.0, 105.0, 115.0]),
])
.expanding("price", using: .max)
.exponentiallyWeighted("price", alpha: 0.5, as: "ewma_alpha05")
.exponentiallyWeighted("price", span: 3.0, as: "ewma_span3")
```

---

## 3. Time-Series As-Of Joins (`joinAsOf`)

``DataFrame/joinAsOf(with:on:by:direction:tolerance:)`` matches asynchronous streams in $O(N_{\text{left}} + N_{\text{right}} \log N_{\text{right}})$ time via binary search using ``AsOfDirection`` (`.backward`, `.forward`, or `.nearest`):

```swift
let t0 = Date(timeIntervalSince1970: 1_700_000_000)

let trades = try DataFrame(columns: [
    (name: "ts", values: [
        .date(t0.addingTimeInterval(2)),
        .date(t0.addingTimeInterval(8)),
        .date(t0.addingTimeInterval(20)),
    ]),
    (name: "ticker", values: ["AAPL", "AAPL", "AAPL"]),
    (name: "trade_qty", values: [100.0, 50.0, 200.0]),
])

let quotes = try DataFrame(columns: [
    (name: "ts", values: [
        .date(t0),
        .date(t0.addingTimeInterval(6)),
    ]),
    (name: "ticker", values: ["AAPL", "AAPL"]),
    (name: "bid", values: [185.0, 185.5]),
])

let matchedTrades = try trades.joinAsOf(
    quotes,
    on: "ts",
    by: "ticker",
    direction: .backward,
    tolerance: 5.0
)
```

---

## 4. Fixed-Interval Time Resampling (`resample`)

``DataFrame/resample(on:interval:aggregations:)`` groups a `.date` column into non-overlapping, UTC-epoch-aligned time buckets of width `interval` seconds:

```swift
let minute0 = Date(timeIntervalSince1970: 1_700_000_040)

let ticks = try DataFrame(columns: [
    (name: "ts", values: [
        .date(minute0.addingTimeInterval(5)),
        .date(minute0.addingTimeInterval(25)),
        .date(minute0.addingTimeInterval(50)),
        .date(minute0.addingTimeInterval(70)),
        .date(minute0.addingTimeInterval(110)),
    ]),
    (name: "price", values: [100.0, 104.0, 99.0, 99.5, 102.0]),
    (name: "volume", values: [10.0, 25.0, 15.0, 30.0, 20.0]),
])

let ohlcv = try ticks.resample(
    on: "ts",
    interval: 60,
    aggregations: [
        Aggregation("price", using: .first, as: "open"),
        Aggregation("price", using: .max,   as: "high"),
        Aggregation("price", using: .min,   as: "low"),
        Aggregation("price", using: .last,  as: "close"),
        Aggregation("volume", using: .sum,  as: "volume"),
    ]
)
```

> Next Step: Proceed to <doc:08-MLX-Acceleration-and-Workflows> to understand Bandal's hybrid execution engine (`Accelerate` + `mlx-swift`), ``ExecutionPolicy``, direct ``MLXVector`` interop, and a complete end-to-end data analysis case study.
