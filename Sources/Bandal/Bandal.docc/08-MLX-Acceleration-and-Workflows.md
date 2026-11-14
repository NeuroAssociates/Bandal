# Guide 08: MLX Acceleration, Runtime Policy & End-to-End Workflows

Configure Bandal's hybrid `Accelerate` + `mlx-swift` execution engine with `ExecutionPolicy` and `BandalRuntime`, interoperate with `MLXVector`, and build end-to-end analytical pipelines.

## Overview

Just as `pandas` relies on `NumPy` C/Fortran arrays under the hood for numerical speed, Bandal combines **typed columnar storage**, **Apple `Accelerate` (`vDSP`)**, and **`mlx-swift` (`MLXArray`)** into a unified hybrid execution engine designed specifically for Apple Silicon.

In this guide you will learn:
- **How Bandal's Hybrid Engine Works**: Typed columnar buffers (`Float64Column`, `CategoricalColumn`, `BitSet`) and automatic dispatch between `vDSP` and `mlx-swift`.
- **Controlling Execution Policy**: ``ExecutionPolicy`` (`.automatic`, `.cpu`, `.gpu`, `.gpuFloat32`) and ``BandalRuntime/withExecutionPolicy(_:operation:)-17sk0``.
- **Zero-Boilerplate `MLXVector` Interop**: Passing ``DataFrame`` columns or ``Series`` into ``MLXVector`` via ``DataFrame/toMLXVector(_:)`` and ``Series/toMLXVector()`` and converting back via ``Series/init(name:mlxVector:)`` or ``DataFrame/init(mlxColumns:)``.
- **Complete End-to-End Case Study**: A full data-science pipeline from CSV ingestion and cleaning to feature engineering, relational joins, group broadcasts, and Markdown report generation.

---

## 1. Architecture: Columnar Storage & Hybrid Execution

Unlike naive Swift implementations that store a table as `[[Any]]` or `[[DataValue]]`, Bandal stores each column in a specialized contiguous buffer paired with a 64-bit word-packed null bitmap (`BitSet`):

| Internal Storage Type | Memory Layout | Best For |
| --- | --- | --- |
| `Float64Column` | `ContiguousArray<Double>` + `BitSet` | Vectorized `vDSP` & `MLX` numerical computing |
| `Int64Column` | `ContiguousArray<Int64>` + `BitSet` | Exact 64-bit integer IDs & counters |
| `BoolColumn` | Word-packed `BitSet` values + `BitSet` nulls | Compact boolean flags & masks |
| `DateColumn` | `ContiguousArray<Double>` (UTC epoch seconds) + `BitSet` | $O(N)$ radix sorting, `joinAsOf`, `resample` |
| `CategoricalColumn` | `ContiguousArray<UInt32>` dictionary codes + `[String]` dictionary | Low-cardinality strings (fast $O(N)$ `groupBy` & `leftJoin`) |
| `StringColumn` | `ContiguousArray<String>` + `BitSet` | High-cardinality free-form text |

---

## 2. Controlling Execution Policy (`ExecutionPolicy` & `BandalRuntime`)

You can inspect MLX availability or override the execution policy within a task-scoped closure using ``BandalRuntime`` and ``ExecutionPolicy``:

```swift
import Bandal

print("MLX Available: \(BandalRuntime.isMLXAvailable)")
print("Active Backend: \(BandalRuntime.executionBackend)")
print("Current Policy: \(BandalRuntime.executionPolicy)")

let df = try DataFrame(columns: [
    (name: "x", values: [10.0, 20.0, 30.0, 40.0, 50.0]),
    (name: "y", values: [2.0,  4.0,  6.0,  8.0,  10.0]),
])

// Scoped override: force MLX GPU/CPU execution regardless of element count
let mlxResult = try BandalRuntime.withExecutionPolicy(.gpu) {
    try df
        .combining("x", "y", using: .multiply, as: "xy")
        .applying("xy", function: .sqrt, as: "sqrt_xy")
}

// Scoped override: force native Swift/Accelerate CPU execution
let cpuResult = try BandalRuntime.withExecutionPolicy(.cpu) {
    try df.correlation()
}

// Scoped override: custom threshold for automatic MLX dispatch
let customAutoResult = try BandalRuntime.withExecutionPolicy(.automatic(minimumElementCount: 1024)) {
    try df.statistics()
}
```

---

## 3. Interoperating with `MLXVector`

``MLXVector`` keeps chaining operations in contiguous `Double` buffers with a parallel `[Bool]` missing-value mask without per-step `[DataValue]` boxing:

```swift
import Bandal

let features = try DataFrame(columns: [
    (name: "id", values: ["S1", "S2", "S3", "S4"]),
    (name: "signal", values: [1.0, nil, 3.0, 4.0]),
])

// 1. Extract an MLXVector from a DataFrame column or Series
let vec: MLXVector = try features.toMLXVector("signal")

// 2. Chain vectorized arithmetic and reductions directly on MLXVector
let transformedVec = (vec * 10.0) + 5.0
print("Mean of transformed signal: \(transformedVec.mean() ?? .nan)")

// 3. Convert back to Series or construct a new DataFrame from MLXVector columns
let signalSeries = transformedVec.toSeries(name: "signal_transformed")
let mlxFrame = try DataFrame(mlxColumns: [
    (name: "raw", vector: vec),
    (name: "transformed", vector: transformedVec),
])
print(mlxFrame)
```

---

## 4. End-to-End Case Study: E-Commerce Analytics Pipeline

```swift
import Bandal
import Foundation

let rawOrders = try DataFrame(columns: [
    (name: "order_id", values: ["ORD-01", "ORD-01", "ORD-02", "ORD-03", "ORD-04", "ORD-05"]),
    (name: "store_id", values: [" s_01 ", " s_01 ", " s_01 ", " s_01 ", " s_02 ", " s_02 "]),
    (name: "date", values: ["2026-03-01", "2026-03-01", "2026-03-02", "2026-03-03", "2026-03-01", "2026-03-02"]),
    (name: "revenue", values: [1200.0, 1200.0, nil, 1800.0, 900.0, 1100.0]),
    (name: "items", values: [10.0, 10.0, 12.0, 15.0, 9.0, 11.0]),
])

let cleanedOrders = try rawOrders
    .dropDuplicates(in: ["order_id"], keep: .first)
    .trimmed("store_id")
    .uppercased("store_id")
    .castColumn("date", to: .date, dateFormat: "yyyy-MM-dd")
    .sorted(by: [
        SortDescriptor("store_id", order: .ascending),
        SortDescriptor("date", order: .ascending),
    ])
    .interpolated("revenue")
    .combining("revenue", "items", using: .divide, as: "avg_item_price")
    .rounded("avg_item_price", decimals: 2)

let storeDirectory = try DataFrame(columns: [
    (name: "store_id", values: ["S_01", "S_02"]),
    (name: "city", values: ["Seoul", "Busan"]),
])

let enrichedOrders = try cleanedOrders
    .leftJoin(storeDirectory, on: "store_id")
    .groupBy("store_id")
    .transform("revenue", using: .sum)
    .combining("revenue", "revenue_sum", using: .divide, as: "store_rev_share")
    .rounded("store_rev_share", decimals: 3)

let citySummary = try enrichedOrders
    .groupBy("city")
    .aggregate([
        Aggregation("order_id", using: .count, as: "days_active"),
        Aggregation("revenue",  using: .sum,   as: "total_revenue"),
        Aggregation("revenue",  using: .mean,  as: "daily_avg_rev"),
        Aggregation("items",    using: .sum,   as: "total_items"),
    ])
    .sorted(by: [SortDescriptor("total_revenue", order: .descending)])

print(citySummary.toMarkdown())
```
