# Guide 06: Grouping, Relational Joins & Reshaping

Aggregate, broadcast (`transform`), and filter groups with `groupBy`, join or concatenate tables, and reshape between wide and long layouts (`melt`, `pivotTable`, `pivoted`, `crossTabulated`, `exploded`, `transposed`).

## Overview

In relational and tabular data analysis, three operations form the backbone of business intelligence and feature engineering:
1. **Split-Apply-Combine (`groupBy`)**: Grouping rows by one or more categorical keys to compute summary metrics (``GroupedDataFrame/aggregate(_:)``), broadcast group-level statistics back to individual rows (``GroupedDataFrame/transform(_:using:)``), or filter entire groups (``GroupedDataFrame/filter(_:)``).
2. **Relational Joins & Concatenation**: Combining multiple tables horizontally by matching key columns (``DataFrame/join(_:on:kind:)``, ``DataFrame/leftJoin(_:on:)``) or stacking tables vertically (``DataFrame/concatenatingRows(_:)``).
3. **Reshaping**: Converting between wide and long table layouts (``DataFrame/melt(idColumns:variableName:valueName:)``, ``DataFrame/pivotTable(index:columns:values:using:)``, ``DataFrame/pivoted(index:columns:values:)``, ``DataFrame/crossTabulated(index:columns:normalize:)``, ``DataFrame/exploded(_:separator:)``, ``DataFrame/transposed(indexColumnName:rowPrefix:)``).

---

## 1. Split-Apply-Combine (`groupBy` & `GroupedDataFrame`)

Bandal supports 10 built-in aggregation functions via ``AggregateFunction`` (`.count`, `.sum`, `.mean`, `.min`, `.max`, `.median`, `.variance`, `.standardDeviation`, `.first`, `.last`).

```swift
import Bandal

let sales = try DataFrame(columns: [
    (name: "region", values: ["Seoul", "Seoul", "Busan", "Seoul", "Busan"]),
    (name: "channel", values: ["Online", "Retail", "Online", "Online", "Retail"]),
    (name: "amount", values: [120.0, 80.0, 150.0, 200.0, 90.0]),
    (name: "rep", values: ["Kim", "Lee", "Park", "Choi", "Jung"]),
])

// Compute multiple aggregations in a single grouping pass with custom column names
let byRegion = try sales
    .groupBy("region")
    .aggregate([
        Aggregation("amount", using: .count, as: "orders"),
        Aggregation("amount", using: .sum, as: "total_sales"),
        Aggregation("amount", using: .mean, as: "avg_order"),
        Aggregation("rep", using: .first, as: "lead_rep"),
    ])

// Group broadcast (transform)
let withRegionalShare = try sales
    .groupBy("region")
    .transform("amount", using: .sum)
    .combining("amount", "amount_sum", using: .divide, as: "regional_share")

// Group filter
let highVolumeRegions = try sales
    .groupBy("region")
    .filter { $0.rowCount >= 3 }
```

---

## 2. Relational Joins & Vertical Concatenation

Use ``DataFrame/join(_:on:kind:)`` (with ``JoinKind``: `.inner`, `.left`, `.right`, `.full`), ``DataFrame/leftJoin(_:on:)``, or ``DataFrame/join(_:leftOn:rightOn:kind:suffixes:)`` for multi-key joins with overlapping column suffixes:

```swift
let customers = try DataFrame(columns: [
    (name: "cust_id", values: ["C1", "C2", "C3"]),
    (name: "name", values: ["Alice", "Bob", "Charlie"]),
])

let purchases = try DataFrame(columns: [
    (name: "cust_id", values: ["C1", "C1", "C4"]),
    (name: "total", values: [50.0, 120.0, 300.0]),
])

let leftJoined = try customers.leftJoin(purchases, on: "cust_id")
let fullJoined = try customers.join(purchases, on: "cust_id", kind: .full)
```

---

## 3. Reshaping & Pivot Tables

```swift
let wideMetrics = try DataFrame(columns: [
    (name: "city", values: ["Seoul", "Busan"]),
    (name: "q1", values: [100.0, 60.0]),
    (name: "q2", values: [120.0, 75.0]),
])

// 1. Wide-to-long (melt)
let longMetrics = try wideMetrics.melt(
    idColumns: ["city"],
    variableName: "quarter",
    valueName: "revenue"
)

// 2. Non-aggregating pivot (pivoted)
let unmelted = try longMetrics.pivoted(
    index: "city",
    columns: "quarter",
    values: "revenue"
)

// 3. Aggregating pivot table (pivotTable)
let summaryPivot = try sales.pivotTable(
    index: ["region"],
    columns: "channel",
    values: "amount",
    using: .sum
)

// 4. Contingency table (crossTabulated)
let ctProps = try sales.crossTabulated(
    index: "region",
    columns: "channel",
    normalize: true
)
```

> Next Step: Proceed to <doc:07-Time-Series-and-Window-Functions> to learn about rolling windows (row-based and time-interval-based), expanding windows, exponentially weighted moving averages (`ewm`), `joinAsOf`, and time-bucket resampling.
