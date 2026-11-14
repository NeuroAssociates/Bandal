# Guide 05: Vectorized Math, Cumulative Ops & Structured Statistics

> **Language**: **English** | [한국어](05-Vectorized-Math-and-Statistics.ko.md)  
> **Navigation**: [← Guide 04: Cleaning & Feature Engineering](04-Cleaning-and-Feature-Engineering.md) | [API Reference](../API.md) | **Next**: [Guide 06: Grouping, Joining & Reshaping →](06-Grouping-Joining-and-Reshaping.md)

---

## Overview

In `pandas` and `NumPy`, fast numerical analysis relies on vectorized operations (`df["a"] + df["b"]`, `np.log(s)`, `df.cumsum()`, `df.describe()`, `df.corr()`).

In Bandal, numerical columns are backed by contiguous `Float64Column` buffers (`NumericBuffer`) accelerated by Apple's `Accelerate` (`vDSP`) framework for small-to-medium columns and `mlx-swift` (`MLXArray`) for large columns. Furthermore, Bandal's statistical APIs follow the **Structured Result Types** design: every summary returns a strongly typed Swift struct (`DataFrameDescription`, `DataFrameStatistics`, `ColumnStatistics`, `QuantileTable`, `UniqueCounts`) that renders a `pandas`-style table when printed while exposing typed properties for programmatic logic.

In this guide you will learn:
- **Vectorized Column & Scalar Arithmetic** (`+`, `-`, `*`, `/`, `combining`, `scaling`)
- **Elementary Mathematical Functions** (`UnaryMathFunction`, `applying`, `power`, `abs`, `clipped`, `rounded`)
- **Cumulative, Lag, Difference, Percent Change & Ranking** (`cumulative`, `shifted`, `difference`, `percentChange`, `ranked`)
- **Structured Descriptive Statistics** (`describeStructured`, `DataFrameDescription`, `statistics`, `ColumnStatistics`, `summaryTable`, `quantiles`)
- **Frequency Tables, Covariance & Correlation Matrices** (`uniqueSummary`, `valueCounts`, `covariance`, `correlation`)

---

## 1. Quick Comparison: `pandas` / `NumPy` vs. `Bandal`

| Operation | Python `pandas` / `NumPy` | Swift `Bandal` |
| --- | --- | --- |
| Series + Series / Scalar | `s1 + s2`, `s1 * 1.1` | `try s1 + s2`, `try s1 * 1.1` |
| Combine 2 DataFrame columns | `df["total"] = df["p"] * df["q"]` | `try df.combining("p", "q", using: .multiply, as: "total")` |
| Scale DataFrame column | `df["p_tax"] = df["p"] * 1.1` | `try df.scaling("p", by: 1.1, using: .multiply, as: "p_tax")` |
| Element-wise math (`log`, `sqrt`, `exp`…) | `np.log(df["x"])` | `try df.applying("x", function: .log, as: "log_x")` |
| Power ($x^p$) | `df["x"] ** 2` | `try df.power("x", exponent: 2.0, as: "x_sq")` |
| Clip / Round / Abs | `df["x"].clip(0, 100).round(2).abs()` | `try df.clipped("x", lower: 0, upper: 100).rounded("x", decimals: 2).abs("x")` |
| Cumulative sum / max / min / prod | `df["x"].cumsum()` | `try df.cumulative("x", using: .sum)` |
| Lag shift / Diff / Pct change | `s.shift(1)` / `s.diff(1)` / `s.pct_change(1)` | `try df.shifted("x", by: 1)` / `try df.difference("x", periods: 1)` / `try df.percentChange("x", periods: 1)` |
| Average-tie ranking | `df["x"].rank(ascending=False)` | `try df.ranked("x", ascending: false, as: "rank")` |
| Descriptive statistics | `df.describe(percentiles=[0.1, 0.5, 0.9])` | `df.describeStructured(percentiles: [0.1, 0.5, 0.9])` |
| Multi-quantile table | `df.quantile([0.25, 0.5, 0.75])` | `try df.quantiles([0.25, 0.5, 0.75])` (`QuantileTable`) |
| Distinct counts / Value counts | `df.nunique()` / `df["cat"].value_counts()` | `df.uniqueSummary()` / `try df.valueCounts(of: "cat")` |
| Covariance / Pearson correlation | `df.cov()` / `df.corr()` | `try df.covariance()` / `try df.correlation()` |

---

## 2. Vectorized Arithmetic (`Series` & `DataFrame`)

### 2.1 `Series` Operators (`+`, `-`, `*`, `/`)

Two numeric `Series` of equal length (or a `Series` and a `Double` scalar) can be combined directly with `+`, `-`, `*`, `/`. Missing values (`.missing`) propagate automatically as `.missing`, and non-finite results (such as division by `0.0`) safely become `.missing` instead of crashing or polluting buffers with `NaN`/`Inf`.

```swift
import Bandal

let price = Series(name: "price", values: [100.0, 200.0, nil, 50.0])
let qty   = Series(name: "qty",   values: [2.0,   3.0,   5.0, 4.0])

let revenue = try price * qty
let discounted = try revenue * 0.9
print(discounted)
```

**Console Output**:
```text
   price
0    180
1    540
2    NaN
3    180
Name: price, Length: 4
```

### 2.2 `DataFrame.combining` and `DataFrame.scaling`

When working inside a `DataFrame` pipeline, `combining(_:_:using:as:)` and `scaling(_:by:using:as:)` let you perform vectorized column arithmetic without breaking your method chain:

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

print(orders)
```

**Console Output**:
```text
   item  price  cost  qty  unit_margin  total_profit  price_with_vat
0     A    100    60   10           40           400             110
1     B    250   150    4          100           400             275
2     C     80    50   20           30           600              88

[3 rows x 7 columns]
```

---

## 3. Elementary Mathematical Functions

Bandal provides vectorized mathematical transformations on both `Series` and `DataFrame`:

- **`UnaryMathFunction`**:
  - `.sqrt` ($\sqrt{x}$), `.exp` ($e^x$), `.log` ($\ln x$), `.log10` ($\log_{10} x$), `.log1p` ($\ln(1 + x)$)
  - `.sin`, `.cos`, `.tan`, `.sign` ($-1, 0, +1$)
- **`power(_:exponent:as:)`**: Raises each element to `exponent` ($x^p$).
- **`abs(_:)`**, **`clipped(_:lower:upper:)`**, **`rounded(_:decimals:)`**

> **Domain Safety**: Any out-of-domain input (such as `.sqrt` or `.log` of a negative number) automatically yields `.missing` for that cell rather_than throwing or storing `NaN`.

```swift
let mathDF = try DataFrame(columns: [
    (name: "x", values: [-4.0, 1.0, 4.0, 9.0, 100.0]),
])
.applying("x", function: .sqrt, as: "sqrt_x")     // -4.0 -> .missing (NaN)
.applying("x", function: .log10, as: "log10_x")   // -4.0 -> .missing (NaN)
.power("x", exponent: 2.0, as: "x_squared")
.clipped("x", lower: 0.0, upper: 50.0)            // Clamps "x" in place to 0...50

print(mathDF)
```

**Console Output**:
```text
     x  sqrt_x  log10_x  x_squared
0    0     NaN      NaN         16
1    1       1        0          1
2    4       2  0.60206         16
3    9       3  0.95424         81
4   50      10        2      10000

[5 rows x 4 columns]
```

---

## 4. Cumulative Reductions, Lag/Diff, Percent Change & Ranking

For sequential or financial data analysis, Bandal supports `cumulative`, `shifted`, `difference`, `percentChange`, and `ranked` on both `Series` and `DataFrame`:

```swift
let dailyRevenue = try DataFrame(columns: [
    (name: "day", values: [1, 2, 3, 4]),
    (name: "rev", values: [100.0, 120.0, 90.0, 120.0]),
])

// Create derived columns using Series methods and attach them to the DataFrame
let rev = try dailyRevenue.series("rev")

let enriched = try dailyRevenue
    .withColumn(try rev.cumulative(using: .sum).renamed("cum_sum"))
    .withColumn(try rev.cumulative(using: .maximum).renamed("running_peak"))
    .withColumn(rev.shifted(by: 1).renamed("prev_rev"))
    .withColumn(try rev.difference(periods: 1).renamed("diff"))
    .withColumn(try rev.percentChange(periods: 1).rounded(decimals: 3).renamed("pct_change"))
    .ranked("rev", ascending: false, as: "rev_rank") // Ties receive average rank (1.5)

print(enriched)
```

*(Note: On `DataFrame`, calling `df.cumulative("rev", using: .sum)`, `df.shifted("rev", by: 1)`, `df.difference("rev", periods: 1)`, or `df.percentChange("rev", periods: 1)` replaces `"rev"` in place, or you can use `Series` methods with `withColumn(_:)` to keep both columns.)*

```swift
let inPlaceExample = try dailyRevenue
    .ranked("rev", ascending: false, as: "rev_rank")
    .percentChange("rev", periods: 1)
    .rounded("rev", decimals: 3)

print(inPlaceExample)
```

**Console Output**:
```text
   day     rev  rev_rank
0    1     NaN         3
1    2     0.2       1.5
2    3   -0.25         4
3    4   0.333       1.5

[4 rows x 3 columns]
```

---

## 5. Structured Descriptive Statistics

In `pandas`, `df.describe()` returns another `DataFrame` where numeric and categorical summaries are mixed or omitted unless `include="all"` is passed.

Bandal provides two complementary APIs:
1. **`df.describe(percentiles:include:) -> String`**: Renders the classic `pandas`-style text table directly.
2. **`df.describeStructured(percentiles:include:) -> DataFrameDescription`**: Returns a strongly-typed Swift struct that **both** prints as the exact same table (`print(desc)` or `desc.describe()`) **and** gives you typed dictionaries (`desc.numeric[col]` and `desc.nonNumeric[col]`).

### 5.1 `describeStructured()` and `DataFrameDescription`

```swift
let store = try DataFrame(columns: [
    (name: "branch", values: ["Seoul", "Seoul", "Busan", "Seoul", "Busan"]),
    (name: "sales", values: [120.0, 150.0, 90.0, 200.0, 140.0]),
    (name: "returns", values: [2.0, 5.0, 1.0, 4.0, 3.0]),
])

// 1. Get the structured description (with custom percentiles if desired)
let desc: DataFrameDescription = store.describeStructured(percentiles: [0.25, 0.5, 0.75])

// Printing `desc` renders the full pandas-style table
print(desc)

// 2. Access individual numeric & non-numeric statistics in Swift code!
if let salesStats: ColumnStatistics = desc.numeric["sales"] {
    print("\nSales Mean: \(salesStats.mean ?? .nan), IQR: \((salesStats.q3 ?? 0) - (salesStats.q1 ?? 0))")
}
if let branchSummary: NonNumericSummary = desc.nonNumeric["branch"] {
    print("Top Branch: \(branchSummary.top ?? .missing) (appeared \(branchSummary.freq) times out of \(branchSummary.count))")
}
```

**Console Output**:
```text
        branch    sales  returns
count   5         5      5      
mean    NaN       140    3      
std     NaN       40.6202  1.58114
min     NaN       90     1      
25%     NaN       120    2      
50%     NaN       140    3      
75%     NaN       150    4      
max     NaN       200    5      
unique  2         NaN    NaN    
top     Seoul     NaN    NaN    
freq    3         NaN    NaN    

Sales Mean: 140.0, IQR: 30.0
Top Branch: Seoul (appeared 3 times out of 5)
```

### 5.2 `statistics()`, `summaryTable()`, and `quantiles(_:)`

- **`df.statistics() -> DataFrameStatistics`**: Computes `ColumnStatistics` (`count`, `sum`, `mean`, `variance`, `standardDeviation`, `min`, `q1`, `median`, `q3`, `max`) for all numeric columns.
- **`df.summaryTable() -> DataFrameSummary`**: Lightweight 6-number summary (`count`, `mean`, `std`, `min`, `50%`, `max`).
- **`df.quantiles(_ probabilities: [Double]) -> QuantileTable`**: Computes arbitrary quantiles across all numeric columns and exposes `table[column, probability]` subscripting as well as `table.toDataFrame()`.

```swift
let qTable: QuantileTable = try store.quantiles([0.1, 0.5, 0.9])
print(qTable)

// Direct programmatic lookup by (column, probability)
let p90Sales: Double? = qTable["sales", 0.9]
print("\n90th percentile of sales: \(p90Sales ?? .nan)")
```

**Console Output**:
```text
      sales  returns
0.1     102      1.4
0.5     140        3
0.9     180      4.6

90th percentile of sales: 180.0
```

---

## 6. Distinct Counts, Frequency Tables, Covariance & Correlation

### 6.1 `uniqueSummary()` and `valueCounts(of:)`

```swift
// 1. Number of distinct non-missing values per column (like df.nunique())
let uniques: UniqueCounts = store.uniqueSummary()
print(uniques)

// 2. Frequency table for a categorical column (like df["branch"].value_counts())
print("\nBranch Value Counts:")
print(try store.valueCounts(of: "branch"))
```

**Console Output**:
```text
branch     2
sales      5
returns    5
Columns: 3 (rows: 5)

Branch Value Counts:
   value  count
0  Seoul      3
1  Busan      2

[2 rows x 2 columns]
```

### 6.2 Pairwise Sample Covariance (`covariance()`) & Pearson Correlation (`correlation()`)

`covariance()` and `correlation()` compute pairwise matrices across all numeric columns, ignoring rows where either column in a pair is `.missing`. For large frames, the matrix multiplication is automatically accelerated on MLX (`float64` CPU stream) or Apple `Accelerate` (`vDSP_dotprD`).

```swift
print("--- Covariance Matrix ---")
print(try store.covariance())

print("\n--- Pearson Correlation Matrix ---")
print(try store.correlation())
```

**Console Output**:
```text
--- Covariance Matrix ---
   sales  returns
0   1650       55
1     55      2.5

[2 rows x 2 columns]

--- Pearson Correlation Matrix ---
      sales   returns
0         1  0.856349
1  0.856349         1

[2 rows x 2 columns]
```

---

> **Next Step**: Continue to **[Guide 06: Grouping, Joining & Reshaping →](06-Grouping-Joining-and-Reshaping.md)** to master `groupBy` aggregations, window broadcasting (`transform`), SQL-style joins, and pivot tables.
