# Bandal

[한국어](README.ko.md) · [API Reference](Documentation/API.md) · [API 레퍼런스](Documentation/API.ko.md)

**Bandal** is a Swift package for tabular data analysis, modeled after Python's [pandas](https://pandas.pydata.org/) and designed according to the [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/).

`DataFrame`, `Series`, `DataRow`, and all summary/statistics result types are immutable value types (`Sendable`, `Equatable`, `Hashable`, `Codable`) that render pandas-style text when passed to `print(...)`. For large numeric columns, Bandal accelerates reductions, grouped aggregations, window functions, element-wise arithmetic, and covariance/correlation matrices through [MLX Swift](https://github.com/ml-explore/mlx-swift) (`float64` on the MLX CPU stream by default, or `float32` on the GPU stream via `ExecutionPolicy.gpu`) while automatically falling back to native Swift on small inputs or when the Metal library is unavailable.

---

## Highlights

- **Value-type `DataFrame`, `Series` & `DataRow`**: Immutable, `Sendable`, `Hashable`, and `Codable`, with literal-expressible `DataValue` cells (`.number(Double)`, `.string(String)`, `.boolean(Bool)`, `.date(Date)`, `.missing`).
- **Printable structured results**: Every inspection and statistical operation (`DataFrame`, `Series`, `DataRow`, `DataFrameDescription`, `DataFrameStatistics`, `DataFrameSummary`, `DataFrameInfo`, `MissingCounts`, `UniqueCounts`, `ColumnTypes`) is a Swift struct with typed properties and a pandas-like `description` / `describe()`.
- **Streaming CSV & JSON I/O**: RFC 4180 streaming CSV reader/writer in 64 KB chunks plus JSON row-object reader/writer (`init(jsonData:)`, `init(contentsOfJSON:)`, `jsonData(prettyPrinted:)`, `writeJSON(to:)`).
- **Rich querying & transformations**: Composable `RowPredicate` and expression-based `query("temp >= 20 && active == true")`, row-computed columns (`withColumn(name:computing:)`), UTC `DateComponent` extraction, and vectorized string operations (`lowercased`, `uppercased`, `trimmed`, `stringLength`, `replacingSubstring`, `containsSubstring`).
- **Grouping, joins, reshaping & windows**: Multi-key `groupBy` (`aggregate`, `transform`), single- and multi-key joins (`.left`, `.inner`, `.right`, `.full` with custom `suffixes`), `melt`, `pivotTable`, `exploded`, `transposed`, row-count and time-interval `rolling`, and `expanding` windows.
- **MLX-Swift numeric engine**: Vectorized moments, reductions, grouped aggregations, prefix-sum windows, element-wise math (`combining`, `scaling`, `cumulative`, `abs`, `clipped`, `rounded`, `difference`, `percentChange`), `MLX.matmul` covariance/correlation matrices, `ExecutionPolicy` scoping (`.automatic`, `.native`, `.cpu`, `.gpu`), and `MLXVector` tensor interop.

---

## Requirements

- macOS 14 or later (Apple silicon recommended for MLX acceleration)
- Swift 6.4 or later
- Xcode with the Metal toolchain installed when running MLX-accelerated kernels

---

## Installation

Add Bandal to your `Package.swift` dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/NeuroAssociates/bandal.git", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "Bandal", package: "Bandal"),
        ]
    ),
]
```

---

## Quick Start

```swift
import Bandal

let sales = try DataFrame(columns: [
    (name: "region", values: ["East", "East", "West", "West"]),
    (name: "revenue", values: [120.0, 180.0, 90.0, nil]),
    (name: "cost", values: [70.0, 100.0, 50.0, 40.0]),
])

print(sales)
//    region  revenue  cost
// 0  East    120.0    70.0
// 1  East    180.0    100.0
// 2  West    90.0     50.0
// 3  West    NaN      40.0
//
// [4 rows x 3 columns]

let analyzed = try sales
    .fillMissing(in: ["revenue"], with: 110.0)
    .combining("revenue", "cost", using: .subtract, as: "profit")
    .query("profit >= 50")

print(analyzed)

let byRegion = try analyzed
    .groupBy("region")
    .aggregate([
        Aggregation("revenue", using: .sum),
        Aggregation("profit", using: .mean),
    ])
print(byRegion)

// Structured descriptive statistics — inspect properties or print directly
let desc = analyzed.describeStructured()
print(desc.numeric["profit"]?.mean ?? .nan)
print(desc)
```

---

## pandas to Bandal Cheat Sheet

| pandas | Bandal |
| --- | --- |
| `pd.DataFrame({...})` | `try DataFrame(columns: [...])` |
| `pd.read_csv(path, ...)` / `df.to_csv(path)` | `try DataFrame(contentsOfCSV: url, options: ...)` / `try df.writeCSV(to: url)` |
| `pd.read_json(path)` / `df.to_json(path)` | `try DataFrame(contentsOfJSON: url)` / `try df.writeJSON(to: url)` |
| `df["col"]` / `df[["a", "b"]]` | `df["col"]` (`Series?`) or `try df.series("col")` / `try df.select("a", "b")` |
| `df.iloc[i]` / `df.iloc[a:b]` / `df.at[i, "col"]` | `df[row: i]` (`DataRow?`) / `df[rows: a..<b]` / `df["col", i]` |
| `df.head(n)` / `df.tail(n)` | `try df.head(n)` / `try df.tail(n)` |
| `df.nlargest(n, "col")` / `df.nsmallest(n, "col")` | `try df.largest(n, by: "col")` / `try df.smallest(n, by: "col")` |
| `df.drop(columns=[...])` / `df.rename(columns={...})` | `try df.drop(columns: [...])` / `try df.renamedColumns([...])` |
| `df.astype(...)` | `try df.castColumn("col", to: .number)` |
| `df[df["x"] >= 10]` / `df.query("x >= 10")` | `try df.filterRows { ... }`, `try df.query(.column("x", greaterThanOrEqual: 10))`, or `try df.query("x >= 10")` |
| `df.assign(y=lambda r: ...)` | `try df.withColumn(name: "y") { row in ... }` |
| `df["col"].str.lower()` / `.upper()` / `.strip()` / `.len()` / `.contains()` | `try df.lowercased("col")` / `uppercased` / `trimmed` / `stringLength` / `containsSubstring` |
| `df["date"].dt.year` / `.month` / `.day` | `try df.dateComponent(.year, from: "date")` |
| `df.isna()` / `df.isna().sum()` | `try df.missingMask()` / `df.missingSummary()` (`MissingCounts`) |
| `df.fillna(v)` / `df.ffill()` / `df.bfill()` / `df.dropna()` | `try df.fillMissing(with: v)` / `try df.forwardFilled()` / `try df.backwardFilled()` / `try df.dropMissing()` |
| `df.duplicated()` / `df.drop_duplicates()` | `try df.duplicatedRows()` / `try df.dropDuplicates()` |
| `df.sort_values(by=[...])` | `try df.sorted(by: [SortDescriptor(...)])` |
| `df.sample(n, random_state=42)` | `try df.sample(n, seed: 42)` |
| `pd.concat([df1, df2])` | `try DataFrame.concatenatingRows([df1, df2])` |
| `df.describe()` / `df.info()` / `df.dtypes` / `df.nunique()` | `df.describeStructured()` / `df.infoSummary()` / `df.dtypes` (`ColumnTypes`) / `df.uniqueSummary()` (`UniqueCounts`) |
| `s.value_counts()` / `s.quantile(q)` / `s.rank()` | `try df.valueCounts(of: "col")` / `try df.quantile(q)` / `try df.ranked("col")` |
| `df.cumsum()` / `df.shift()` / `df.diff()` / `df.pct_change()` | `try df.cumulative("col", using: .sum)` / `shifted` / `difference` / `percentChange` |
| `df.corr()` / `df.cov()` | `try df.correlation()` / `try df.covariance()` |
| `df.groupby(...).agg(...)` / `.transform(...)` | `try df.groupBy(...).aggregate(...)` / `try df.groupBy(...).transform(...)` |
| `df.merge(other, left_on=..., right_on=..., how=...)` | `try df.join(other, leftOn: [...], rightOn: [...], kind: ...)` |
| `df.melt(...)` / `df.pivot_table(...)` | `try df.melt(...)` / `try df.pivotTable(...)` |
| `df.explode("col")` / `df.T` | `try df.exploded("col")` / `try df.transposed()` |
| `df.rolling(w).mean()` / `df.rolling("60s", on=...).mean()` / `df.expanding().sum()` | `try df.rolling("col", window: w, using: .mean)` / `try df.rolling("col", on: "ts", interval: 60, using: .mean)` / `try df.expanding("col", using: .sum)` |
| `df.to_markdown()` | `df.toMarkdown()` |

---

## MLX Backend & Execution Policy

Bandal uses [MLX Swift](https://github.com/ml-explore/mlx-swift) as its vectorized numeric engine. By default (`ExecutionPolicy.automatic`), numeric columns above the internal size threshold are evaluated in `float64` on MLX's CPU stream to preserve full `Double` precision, while smaller inputs run in native Swift without tensor-allocation overhead.

You can inspect or override the execution policy at runtime:

```swift
print(BandalRuntime.isMLXAvailable)
print(BandalRuntime.executionBackend)

// Explicitly run a workload on the MLX GPU stream (float32 when lossless) or CPU stream
let corr = try BandalRuntime.withExecutionPolicy(.gpu) {
    try frame.correlation()
}

// Convert a column directly to an MLXVector / MLXArray and back
let vector = try frame.toMLXVector("revenue")
let mlxArray = vector.makeMLXArray()
let restored = try DataFrame(mlxColumns: [(name: "revenue", vector: vector)])
```

Set `BANDAL_DISABLE_MLX=1` in the environment to force the native Swift implementation everywhere.

---

## Testing & Benchmarks

- **Standard test suite (native path)**:
  ```sh
  xcrun swift test
  ```
- **Full test suite with MLX Metal library colocated**:
  ```sh
  Scripts/test-with-mlx.sh
  ```
- **Benchmarks**:
  ```sh
  xcrun swift run -c release BandalBenchmarks 1000 10000 100000
  ```

---

## Documentation

- [English API Reference (`Documentation/API.md`)](Documentation/API.md)
- [Korean API Reference (`Documentation/API.ko.md`)](Documentation/API.ko.md)
- [Contributing Guide (`CONTRIBUTING.md`)](CONTRIBUTING.md)
- [Changelog (`CHANGELOG.md`)](CHANGELOG.md)

---

## License
Created by [Ned Park](https://github.com/nedpark) of Neuro Associates. All rights reserved.
Released under the [MIT License](LICENSE).
