# ``Bandal``

Pandas-style tabular data analysis for Swift, accelerated by MLX and Apple Accelerate.

## Overview

Bandal provides immutable, `Sendable` value types — ``DataFrame``, ``Series``, and ``DataRow`` — backed by specialized columnar buffers and 64-bit null bitmaps, with streaming CSV and JSON I/O, vectorized boolean masks, composable queries, group-by aggregations, relational and as-of joins, pivot/reshape tools, window functions, and structured descriptive statistics. Large numeric workloads run on `mlx-swift` (`MLXArray`) and fall back to native Swift and Apple `Accelerate` (`vDSP`) automatically.

Every result type conforms to `CustomStringConvertible` and prints a `pandas`-style aligned table.

```swift
import Bandal

let df = try DataFrame(columns: [
    (name: "city", values: ["Seoul", "Busan", "Seoul", "Incheon"]),
    (name: "revenue", values: [120.0, 95.0, 180.0, nil]),
    (name: "orders", values: [12, 9, 18, 5]),
])

let summary = try df
    .fillMissing(in: ["revenue"], with: 100.0)
    .combining("revenue", "orders", using: .divide, as: "avg_order")
    .groupBy("city")
    .aggregate([
        Aggregation("revenue", using: .sum, as: "total_revenue"),
        Aggregation("avg_order", using: .mean, as: "mean_order"),
    ])

print(summary)
```

## Topics

### Guides & Tutorials

- <doc:01-Core-Data-Structures>
- <doc:02-Data-IO-and-Inspection>
- <doc:03-Indexing-Filtering-and-Querying>
- <doc:04-Cleaning-and-Feature-Engineering>
- <doc:05-Vectorized-Math-and-Statistics>
- <doc:06-Grouping-Joining-and-Reshaping>
- <doc:07-Time-Series-and-Window-Functions>
- <doc:08-MLX-Acceleration-and-Workflows>

### Core Data Structures

- ``DataFrame``
- ``Series``
- ``DataRow``
- ``DataValue``
- ``DataType``
- ``DataFrameError``

### Data I/O & Rendering

- ``CSVReadOptions``
- ``RenderOptions``
- ``DataFrameInfo``

### Filtering, Sorting & Sampling

- ``BooleanMask``
- ``RowPredicate``
- ``SortOrder``
- ``SortDescriptor``
- ``DuplicateKeep``
- ``DateComponent``

### Grouping, Joining & Reshaping

- ``GroupedDataFrame``
- ``AggregateFunction``
- ``Aggregation``
- ``JoinKind``
- ``AsOfDirection``

### Vectorized Math & Structured Statistics

- ``ArithmeticOperator``
- ``UnaryMathFunction``
- ``CumulativeFunction``
- ``ColumnSummary``
- ``ColumnStatistics``
- ``NonNumericSummary``
- ``DataFrameStatistics``
- ``DataFrameSummary``
- ``DataFrameDescription``
- ``SeriesDescription``
- ``QuantileTable``
- ``MissingCounts``
- ``ColumnTypes``
- ``UniqueCounts``

### MLX Acceleration & Runtime Policy

- ``BandalRuntime``
- ``ExecutionPolicy``
- ``MLXVector``
