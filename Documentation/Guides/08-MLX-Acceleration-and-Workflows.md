# Guide 08: MLX Acceleration, Runtime Policy & End-to-End Workflows

> **Language**: **English** | [한국어](08-MLX-Acceleration-and-Workflows.ko.md)  
> **Navigation**: [← Guide 07: Time Series & Window Functions](07-Time-Series-and-Window-Functions.md) | [API Reference](../API.md) | [🏠 Back to Guide 01](01-Core-Data-Structures.md)

---

## Overview

Just as `pandas` relies on `NumPy` C/Fortran arrays under the hood for numerical speed, Bandal combines **typed columnar storage**, **Apple `Accelerate` (`vDSP`)**, and **`mlx-swift` (`MLXArray`)** into a unified hybrid execution engine designed specifically for Apple Silicon.

In this final guide you will learn:
- **How Bandal's Hybrid Engine Works**: Typed columnar buffers (`Float64Column`, `CategoricalColumn`, `BitSet`) and automatic dispatch between `vDSP` and `mlx-swift`.
- **Controlling Execution Policy**: `ExecutionPolicy` (`.automatic`, `.nativeOnly`, `.mlxPreferred`), `BandalRuntime.withExecutionPolicy`, and `MLXBackend`.
- **Zero-Boilerplate `MLXVector` & `MLXArray` Interop**: Passing `DataFrame` columns directly into `mlx-swift` neural networks or custom tensor pipelines and bringing results back into a `DataFrame`.
- **Complete End-to-End Case Study**: A full data-science pipeline from raw CSV ingestion and cleaning to feature engineering, relational joins, rolling windows, group aggregations, and Markdown report generation.

---

## 1. Architecture: Columnar Storage & Hybrid Execution

### 1.1 Typed Columnar Storage + 64-Bit Null Bitmaps

Unlike naive Swift implementations that store a table as `[[Any]]` or `[[DataValue]]`, Bandal stores each column in a specialized contiguous buffer paired with a 64-bit word-packed null bitmap (`BitSet`):

| Internal Storage Type | Memory Layout | Best For |
| --- | --- | --- |
| `Float64Column` | `ContiguousArray<Double>` + `BitSet` | Vectorized `vDSP` & `MLX` numerical computing |
| `Int64Column` | `ContiguousArray<Int64>` + `BitSet` | Exact 64-bit integer IDs & counters |
| `BoolColumn` | Word-packed `BitSet` values + `BitSet` nulls | Compact boolean flags & masks |
| `DateColumn` | `ContiguousArray<Double>` (UTC epoch seconds) + `BitSet` | $O(N)$ radix sorting, `joinAsOf`, `resample` |
| `CategoricalColumn` | `ContiguousArray<UInt32>` dictionary codes + `[String]` dictionary | Low-cardinality strings (fast $O(N)$ `groupBy` & `leftJoin`) |
| `StringColumn` | `ContiguousArray<String>` + `BitSet` | High-cardinality free-form text |

### 1.2 Automatic Dispatch: Native `vDSP` vs. `mlx-swift`

Launching a tensor graph on MLX carries a small fixed dispatch overhead. For a 50-row column, an in-place CPU SIMD loop (`vDSP`) finishes in nanoseconds, whereas for a 100,000-row column or a 2D strided rolling window, MLX's unified-memory tensor engine provides massive throughput.

By default (`.automatic(minimumElementCount: 4096)`), Bandal automatically selects the fastest path:
- **Below `4,096` elements**: Executes via native Swift and Apple `Accelerate` (`vDSP_vaddD`, `vDSP_vmulD`, `vDSP_dotprD`, etc.).
- **At or above `4,096` elements** (when the Metal/MLX runtime is available): Dispatches to `mlx-swift` (`MLXArray`) on the CPU `float64` stream (preserving full 64-bit `Double` precision) for reductions, element-wise arithmetic, scatter group reductions (`at.add`, `at.maximum`, `at.minimum`), 2D strided rolling windows (`MLX.asStrided`), and pairwise correlation/covariance matrix multiplications (`MLX.matmul`).
- **Automatic Fallback**: Whenever an operation involves non-numeric types or runs in an environment where Metal/MLX shaders are unavailable (e.g., command-line `swift test` without a Metal bundle), Bandal transparently falls back to its native `Accelerate` implementation with identical numerical results.

---

## 2. Controlling Execution Policy (`ExecutionPolicy` & `BandalRuntime`)

You can inspect MLX availability or override the execution policy globally or within a scoped closure using `BandalRuntime` and `MLXBackend`:

```swift
import Bandal

// 1. Check MLX backend status
print("MLX Available: \(MLXBackend.isAvailable)")
print("Default threshold: \(MLXBackend.defaultMinimumElements) elements")

let df = try DataFrame(columns: [
    (name: "x", values: [10.0, 20.0, 30.0, 40.0, 50.0]),
    (name: "y", values: [2.0,  4.0,  6.0,  8.0,  10.0]),
])

// 2. Scoped override: force MLX execution even for small columns (when MLX is available)
let mlxResult = try BandalRuntime.withExecutionPolicy(.mlxPreferred) {
    try df
        .combining("x", "y", using: .multiply, as: "xy")
        .applying("xy", function: .sqrt, as: "sqrt_xy")
}

// 3. Scoped override: force native Swift/Accelerate execution
let nativeResult = try BandalRuntime.withExecutionPolicy(.nativeOnly) {
    try df.correlation()
}

// 4. Custom threshold for automatic dispatch (e.g., use MLX for >= 1,024 elements)
BandalRuntime.executionPolicy = .automatic(minimumElementCount: 1024)
// Reset back to default
BandalRuntime.executionPolicy = .automatic()
```

---

## 3. Interoperating with `MLXVector` and `MLXArray`

When building machine learning models in Swift with `mlx-swift`, you often want to preprocess tabular data in Bandal and hand the resulting numerical features directly to `MLXArray` tensors—or convert model predictions back into a `DataFrame`.

`MLXVector` bridges Bandal's null-aware columnar buffers with `MLXArray`:

```swift
public struct MLXVector: Equatable, Sendable {
    public let values: [Double]    // Contiguous IEEE-754 doubles (0.0 at missing positions)
    public let isMissing: [Bool]   // True where the original cell was .missing
    public var count: Int { values.count }
    public var hasMissing: Bool { ... }

    // Converts to a 1-D float32 MLXArray, replacing missing positions with `fillValue`
    public func makeMLXArray(replacingMissingWith fillValue: Float = 0) -> MLXArray

    // Converts the missingness mask to a 1-D Bool MLXArray
    public func missingMaskArray() -> MLXArray
}
```

### Example: Extracting `MLXVector` / `MLXArray` and Writing Back to `DataFrame`

```swift
import Bandal
import MLX

let features = try DataFrame(columns: [
    (name: "id", values: ["S1", "S2", "S3", "S4"]),
    (name: "signal", values: [1.0, nil, 3.0, 4.0]),
])

// 1. Extract an MLXVector from a DataFrame column (or via `try features.series("signal").mlxVector()`)
let vec: MLXVector = try features.mlxVector(for: "signal")
print("Has missing: \(vec.hasMissing), Values buffer: \(vec.values)")

// 2. Convert to an MLXArray tensor (replacing .missing with 0.0) when MLX is available
if MLXBackend.isAvailable {
    let tensor: MLXArray = vec.makeMLXArray(replacingMissingWith: 0.0)
    let maskTensor: MLXArray = vec.missingMaskArray()
    print("Tensor shape: \(tensor.shape), dtype: \(tensor.dtype)")
    print("Mask shape: \(maskTensor.shape)")
}

// 3. Construct a new MLXVector (preserving the original missing mask) and attach to the DataFrame
let scaledValues = vec.values.map { $0 * 10.0 }
let outputVec = MLXVector(values: scaledValues, isMissing: vec.isMissing)

let updatedDF = try features.withColumn(name: "signal_x10", vector: outputVec)
print(updatedDF)
```

**Console Output**:
```text
Has missing: true, Values buffer: [1.0, 0.0, 3.0, 4.0]
   id  signal  signal_x10
0  S1       1          10
1  S2     NaN         NaN
2  S3       3          30
3  S4       4          40

[4 rows x 3 columns]
```

---

## 4. End-to-End Case Study: E-Commerce Analytics Pipeline

Let's bring together everything from Guides 01 through 08 in a single, self-contained analytical workflow. Suppose we receive:
1. A dirty CSV export of daily store orders (`rawOrdersCSV`) containing whitespace, currency symbols, missing values, and duplicate entries.
2. A store metadata table (`storeDirectory`) with regional tiers.

Our goal is to clean the data, enrich it via a relational join, compute per-store rolling and group-broadcast metrics, and output a Markdown executive summary table.

```swift
import Bandal
import Foundation

// --- Step 1: Ingest Raw CSV with Custom Options ---
let rawOrdersCSV = """
order_id,store_id,date,revenue,items
ORD-01, s_01 ,2026-03-01,"$1,200.00",10
ORD-01, s_01 ,2026-03-01,"$1,200.00",10
ORD-02, s_01 ,2026-03-02,N/A,12
ORD-03, s_01 ,2026-03-03,"$1,800.00",15
ORD-04, s_02 ,2026-03-01,"$900.00",9
ORD-05, s_02 ,2026-03-02,"$1,100.00",11
"""

let csvOptions = CSVReadOptions(
    additionalNullLiterals: ["N/A"],
    dateFormat: "yyyy-MM-dd"
)
let rawOrders = try DataFrame(csv: rawOrdersCSV, options: csvOptions)

// --- Step 2: Clean, Deduplicate, Interpolate & Engineer Features ---
let cleanedOrders = try rawOrders
    // Remove duplicate order_id rows
    .dropDuplicates(in: ["order_id"], keep: .first)
    // Normalize store_id strings (" s_01 " -> "S_01")
    .trimmed("store_id")
    .uppercased("store_id")
    // Sort chronologically within each store before interpolating
    .sorted(by: [
        SortDescriptor("store_id", order: .ascending),
        SortDescriptor("date", order: .ascending),
    ])
    // Linearly interpolate missing revenue (1200 -> [1500] -> 1800)
    .interpolated("revenue")
    // Compute average item price = revenue / items
    .combining("revenue", "items", using: .divide, as: "avg_item_price")
    .rounded("avg_item_price", decimals: 2)

print("=== 1. Cleaned & Interpolated Orders ===")
print(cleanedOrders)

// --- Step 3: Enrich with Store Metadata via Left Join ---
let storeDirectory = try DataFrame(columns: [
    (name: "store_id", values: ["S_01", "S_02"]),
    (name: "city", values: ["Seoul", "Busan"]),
])

let enrichedOrders = try cleanedOrders
    .leftJoin(storeDirectory, on: "store_id")
    // Broadcast each store's total revenue to every row
    .groupBy("store_id")
    .transform("revenue", using: .sum)
    // Calculate each day's share of the store's total revenue
    .combining("revenue", "revenue_sum", using: .divide, as: "store_rev_share")
    .rounded("store_rev_share", decimals: 3)

print("\n=== 2. Enriched Orders with Store Share ===")
print(enrichedOrders)

// --- Step 4: Executive Summary by City & Markdown Export ---
let citySummary = try enrichedOrders
    .groupBy("city")
    .aggregate([
        Aggregation("order_id", using: .count, as: "days_active"),
        Aggregation("revenue",  using: .sum,   as: "total_revenue"),
        Aggregation("revenue",  using: .mean,  as: "daily_avg_rev"),
        Aggregation("items",    using: .sum,   as: "total_items"),
    ])
    .sorted(by: [SortDescriptor("total_revenue", order: .descending)])

print("\n=== 3. Executive Summary (Markdown) ===")
print(citySummary.toMarkdown())
```

**Console Output**:
```text
=== 1. Cleaned & Interpolated Orders ===
   order_id  store_id                  date  revenue  items  avg_item_price
0    ORD-01      S_01  2026-03-01T00:00:00Z     1200     10             120
1    ORD-02      S_01  2026-03-02T00:00:00Z     1500     12             125
2    ORD-03      S_01  2026-03-03T00:00:00Z     1800     15             120
3    ORD-04      S_02  2026-03-01T00:00:00Z      900      9             100
4    ORD-05      S_02  2026-03-02T00:00:00Z     1100     11             100

[5 rows x 6 columns]

=== 2. Enriched Orders with Store Share ===
   order_id  store_id                  date  revenue  items  avg_item_price   city  revenue_sum  store_rev_share
0    ORD-01      S_01  2026-03-01T00:00:00Z     1200     10             120  Seoul         4500            0.267
1    ORD-02      S_01  2026-03-02T00:00:00Z     1500     12             125  Seoul         4500            0.333
2    ORD-03      S_01  2026-03-03T00:00:00Z     1800     15             120  Seoul         4500              0.4
3    ORD-04      S_02  2026-03-01T00:00:00Z      900      9             100  Busan         2000             0.45
4    ORD-05      S_02  2026-03-02T00:00:00Z     1100     11             100  Busan         2000             0.55

[5 rows x 9 columns]

=== 3. Executive Summary (Markdown) ===
| city | days_active | total_revenue | daily_avg_rev | total_items |
| :--- | ---: | ---: | ---: | ---: |
| Seoul | 3 | 4500 | 1500 | 37 |
| Busan | 2 | 2000 | 1000 | 20 |
```

---

## Summary of All Self-Study Guides

1. **[Guide 01: Core Data Structures (`DataValue`, `Series`, `DataFrame`, `DataRow`)](01-Core-Data-Structures.md)**
2. **[Guide 02: Data I/O (CSV & JSON) & Schema Inspection (`infoSummary`, `RenderOptions`)](02-Data-IO-and-Inspection.md)**
3. **[Guide 03: Indexing, Slicing, Boolean Masks & Querying (`BooleanMask`, `RowPredicate`)](03-Indexing-Filtering-and-Querying.md)**
4. **[Guide 04: Data Cleaning, Missing Values & Feature Engineering](04-Cleaning-and-Feature-Engineering.md)**
5. **[Guide 05: Vectorized Math, Cumulative Ops & Structured Statistics](05-Vectorized-Math-and-Statistics.md)**
6. **[Guide 06: Grouping (`groupBy`), Relational Joins & Reshaping](06-Grouping-Joining-and-Reshaping.md)**
7. **[Guide 07: Time Series Analysis & Window Functions (`rolling`, `joinAsOf`, `resample`)](07-Time-Series-and-Window-Functions.md)**
8. **[Guide 08: MLX Acceleration, Runtime Policy & End-to-End Workflows](08-MLX-Acceleration-and-Workflows.md)**
- **[Complete API Reference (`Documentation/API.md`)](../API.md)**
