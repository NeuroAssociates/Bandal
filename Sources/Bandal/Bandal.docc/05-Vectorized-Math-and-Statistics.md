# Guide 05: Vectorized Math, Cumulative Ops & Structured Statistics

Perform hardware-accelerated column and scalar arithmetic, elementary math functions, cumulative/lag reductions, and structured descriptive statistics, quantiles, covariance, and correlation.

## Overview

In Bandal, numerical columns are backed by contiguous `Float64Column` buffers (`NumericBuffer`) accelerated by Apple's `Accelerate` (`vDSP`) framework for small-to-medium columns and `mlx-swift` (`MLXArray`) for large columns. Furthermore, Bandal's statistical APIs follow the **Structured Result Types** design: every summary returns a strongly typed Swift struct (``DataFrameDescription``, ``DataFrameStatistics``, ``ColumnStatistics``, ``QuantileTable``, ``UniqueCounts``) that renders a `pandas`-style table when printed while exposing typed properties for programmatic logic.

In this guide you will learn:
- **Vectorized Column & Scalar Arithmetic** (`+`, `-`, `*`, `/`, ``DataFrame/combining(_:_:using:as:)``, ``DataFrame/scaling(_:by:using:as:)``)
- **Elementary Mathematical Functions** (``UnaryMathFunction``, ``DataFrame/applying(_:function:as:)``, ``DataFrame/power(_:exponent:as:)``, ``DataFrame/abs(_:)``, ``DataFrame/clipped(_:lower:upper:)``, ``DataFrame/rounded(_:decimals:)``)
- **Cumulative, Lag, Difference, Percent Change & Ranking** (``DataFrame/cumulative(_:using:)``, ``DataFrame/shifted(_:by:)``, ``DataFrame/difference(_:periods:)``, ``DataFrame/percentChange(_:periods:)``, ``DataFrame/ranked(_:ascending:as:)``)
- **Structured Descriptive Statistics** (``DataFrame/describeStructured(percentiles:include:)``, ``DataFrameDescription``, ``DataFrame/statistics()``, ``ColumnStatistics``, ``DataFrame/summaryTable()``, ``DataFrame/quantiles(_:)``)
- **Frequency Tables, Covariance & Correlation Matrices** (``DataFrame/uniqueSummary()``, ``DataFrame/valueCounts(of:)``, ``DataFrame/covariance()``, ``DataFrame/correlation()``)

---

## 1. Vectorized Arithmetic (`Series` & `DataFrame`)

Two numeric ``Series`` of equal length (or a ``Series`` and a `Double` scalar) can be combined directly with `+`, `-`, `*`, `/`:

```swift
import Bandal

let price = Series(name: "price", values: [100.0, 200.0, nil, 50.0])
let qty   = Series(name: "qty",   values: [2.0,   3.0,   5.0, 4.0])

let revenue = try price * qty
let discounted = revenue * 0.9
```

Inside a ``DataFrame`` pipeline, use ``DataFrame/combining(_:_:using:as:)`` and ``DataFrame/scaling(_:by:using:as:)``:

```swift
let orders = try DataFrame(columns: [
    (name: "item", values: ["A", "B", "C"]),
    (name: "price", values: [100.0, 250.0, 80.0]),
    (name: "cost", values: [60.0, 150.0, 50.0]),
    (name: "qty", values: [10.0, 4.0, 20.0]),
])
.combining("price", "cost", using: .subtract, as: "unit_margin")
.combining("unit_margin", "qty", using: .multiply, as: "total_profit")
.scaling("price", by: 1.1, using: .multiply, as: "price_with_vat")
```

---

## 2. Elementary Mathematical Functions

Bandal provides vectorized mathematical transformations via ``UnaryMathFunction`` (`.sqrt`, `.exp`, `.log`, `.log10`, `.log1p`, `.sin`, `.cos`, `.tan`, `.sign`) as well as ``DataFrame/power(_:exponent:as:)``, ``DataFrame/abs(_:)``, ``DataFrame/clipped(_:lower:upper:)``, and ``DataFrame/rounded(_:decimals:)``:

```swift
let mathDF = try DataFrame(columns: [
    (name: "x", values: [-4.0, 1.0, 4.0, 9.0, 100.0]),
])
.applying("x", function: .sqrt, as: "sqrt_x")     // -4.0 -> .missing (NaN)
.applying("x", function: .log10, as: "log10_x")   // -4.0 -> .missing (NaN)
.power("x", exponent: 2.0, as: "x_squared")
.clipped("x", lower: 0.0, upper: 50.0)
```

---

## 3. Structured Descriptive Statistics

Bandal provides two complementary descriptive statistics APIs:
1. ``DataFrame/describe(percentiles:include:)``: Renders the classic `pandas`-style text table directly.
2. ``DataFrame/describeStructured(percentiles:include:)``: Returns a strongly-typed ``DataFrameDescription`` struct that both prints as a table and exposes typed dictionaries (`desc.numeric[col]` and `desc.nonNumeric[col]`).

```swift
let store = try DataFrame(columns: [
    (name: "branch", values: ["Seoul", "Seoul", "Busan", "Seoul", "Busan"]),
    (name: "sales", values: [120.0, 150.0, 90.0, 200.0, 140.0]),
    (name: "returns", values: [2.0, 5.0, 1.0, 4.0, 3.0]),
])

let desc: DataFrameDescription = store.describeStructured(percentiles: [0.25, 0.5, 0.75])
print(desc)

if let salesStats: ColumnStatistics = desc.numeric["sales"] {
    print("Sales Mean: \(salesStats.mean ?? .nan)")
}

let qTable: QuantileTable = try store.quantiles([0.1, 0.5, 0.9])
let p90Sales: Double? = qTable["sales", 0.9]
```

---

## 4. Covariance & Correlation Matrices

``DataFrame/covariance()`` and ``DataFrame/correlation()`` compute pairwise matrices across all numeric columns, ignoring rows where either column in a pair is `.missing`:

```swift
let cov = try store.covariance()
let corr = try store.correlation()
print(corr)
```

> Next Step: Continue to <doc:06-Grouping-Joining-and-Reshaping> to master `groupBy` aggregations, window broadcasting (`transform`), SQL-style joins, and pivot tables.
