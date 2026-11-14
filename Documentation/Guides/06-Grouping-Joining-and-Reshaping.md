# Guide 06: Grouping, Relational Joins & Reshaping

> **Language**: **English** | [한국어](06-Grouping-Joining-and-Reshaping.ko.md)  
> **Navigation**: [← Guide 05: Vectorized Math & Statistics](05-Vectorized-Math-and-Statistics.md) | [API Reference](../API.md) | **Next**: [Guide 07: Time Series & Window Functions →](07-Time-Series-and-Window-Functions.md)

---

## Overview

In relational and tabular data analysis, three operations form the backbone of business intelligence and feature engineering:
1. **Split-Apply-Combine (`groupBy`)**: Grouping rows by one or more categorical keys to compute summary metrics (`aggregate`), broadcast group-level statistics back to individual rows (`transform`), or filter entire groups (`filter`).
2. **Relational Joins & Concatenation (`join`, `leftJoin`, `concatenatingRows`)**: Combining multiple tables horizontally by matching key columns or stacking tables vertically.
3. **Reshaping (`melt`, `pivotTable`, `pivoted`, `crossTabulated`, `exploded`, `transposed`)**: Converting between wide and long table layouts.

This guide walks through each of these capabilities with side-by-side `pandas` comparisons and runnable Swift examples.

---

## 1. Quick Comparison: `pandas` vs. `Bandal`

| Operation | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| Single aggregation | `df.groupby("dept")["salary"].mean()` | `try df.groupBy("dept").aggregate("salary", using: .mean)` |
| Multi-column / named aggregations | `df.groupby(["dept", "tier"]).agg(total=("sales", "sum"), avg=("sales", "mean"))` | `try df.groupBy("dept", "tier").aggregate([Aggregation("sales", using: .sum, as: "total"), Aggregation("sales", using: .mean, as: "avg")])` |
| Group broadcast (`transform`) | `df["dept_mean"] = df.groupby("dept")["salary"].transform("mean")` | `try df.groupBy("dept").transform("salary", using: .mean)` |
| Group filter (`filter`) | `df.groupby("dept").filter(lambda g: len(g) >= 2)` | `try df.groupBy("dept").filter { $0.rowCount >= 2 }` |
| Inner / Left / Right / Full Join | `df1.merge(df2, on="id", how="left")` | `try df1.join(df2, on: "id", kind: .left)` (or `try df1.leftJoin(df2, on: "id")`) |
| Multi-key join with suffixes | `df1.merge(df2, left_on=["a"], right_on=["b"], suffixes=("_l", "_r"))` | `try df1.join(df2, leftOn: ["a"], rightOn: ["b"], kind: .inner, suffixes: ("_l", "_r"))` |
| Vertical row concatenation | `pd.concat([df1, df2], ignore_index=True)` | `try DataFrame.concatenatingRows([df1, df2])` |
| Wide-to-long (`melt`) | `df.melt(id_vars=["id"], var_name="metric", value_name="val")` | `try df.melt(idColumns: ["id"], variableName: "metric", valueName: "val")` |
| Aggregating `pivot_table` | `df.pivot_table(index=["region"], columns="qtr", values="rev", aggfunc="sum")` | `try df.pivotTable(index: ["region"], columns: "qtr", values: "rev", using: .sum)` |
| Non-aggregating `pivot` | `df.pivot(index="date", columns="ticker", values="close")` | `try df.pivoted(index: "date", columns: "ticker", values: "close")` |
| Contingency table (`crosstab`) | `pd.crosstab(df["dept"], df["remote"], normalize=True)` | `try df.crossTabulated(index: "dept", columns: "remote", normalize: true)` |
| Explode delimited list column | `df.assign(tag=df["tag"].str.split(",")).explode("tag")` | `try df.exploded("tag", separator: ",")` |
| Transpose (`df.T`) | `df.T` | `try df.transposed()` |

---

## 2. Split-Apply-Combine (`groupBy` & `GroupedDataFrame`)

### 2.1 Available `AggregateFunction` Cases

Bandal supports 10 built-in aggregation functions via `AggregateFunction`:

| Case | Default Output Suffix | Description |
| --- | --- | --- |
| `.count` | `<col>_count` | Number of non-missing values in the group |
| `.sum` | `<col>_sum` | Sum of numeric values |
| `.mean` | `<col>_mean` | Arithmetic mean |
| `.min` / `.max` | `<col>_min` / `<col>_max` | Minimum / Maximum value |
| `.median` | `<col>_median` | 50th percentile (median) |
| `.variance` | `<col>_variance` | Sample variance ($N - 1$ denominator) |
| `.standardDeviation` | `<col>_std` | Sample standard deviation ($N - 1$ denominator) |
| `.first` / `.last` | `<col>_first` / `<col>_last` | First / Last non-missing value (works on any column type!) |

> **Important Note on Grouping Keys**: Just like `df.groupby(..., dropna=True)` in `pandas`, rows where any grouping key is `.missing` are omitted from the grouped result, and output group keys are sorted in ascending order.

### 2.2 Single & Multi-Aggregation with Custom Aliases (`Aggregation`)

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

print(byRegion)
```

**Console Output**:
```text
   region  orders  total_sales  avg_order  lead_rep
0   Busan       2          240        120      Park
1   Seoul       3          400    133.333       Kim

[2 rows x 5 columns]
```

---

## 3. Group Broadcast (`transform`) & Group Filtering (`filter`)

### 3.1 `GroupedDataFrame.transform(_:using:)`

In `pandas`, `df.groupby("region")["amount"].transform("mean")` computes the group mean and broadcasts it back to every source row so you can compare each row against its group benchmark.

In Bandal, `groupBy(...).transform(_:using:)` appends a new column named `<column>_<operation>` (e.g., `"amount_mean"`) aligned with the original rows:

```swift
// Compute each order's share of its regional total sales
let withRegionalShare = try sales
    .groupBy("region")
    .transform("amount", using: .sum) // Appends "amount_sum"
    .combining("amount", "amount_sum", using: .divide, as: "regional_share")
    .rounded("regional_share", decimals: 3)

print(withRegionalShare)
```

**Console Output**:
```text
   region  channel  amount   rep  amount_sum  regional_share
0   Seoul   Online     120   Kim         400             0.3
1   Seoul   Retail      80   Lee         400             0.2
2   Busan   Online     150  Park         240           0.625
3   Seoul   Online     200  Choi         400             0.5
4   Busan   Retail      90  Jung         240           0.375

[5 rows x 6 columns]
```

### 3.2 `GroupedDataFrame.filter(_:)`

`filter(_:)` passes each group's sub-`DataFrame` to your closure and retains only the rows belonging to groups that return `true`, preserving original row order:

```swift
// Keep only regions that have at least 3 orders
let highVolumeRegions = try sales
    .groupBy("region")
    .filter { groupDF in
        groupDF.rowCount >= 3
    }

print(highVolumeRegions)
```

**Console Output**:
```text
   region  channel  amount   rep
0   Seoul   Online     120   Kim
1   Seoul   Retail      80   Lee
2   Seoul   Online     200  Choi

[3 rows x 4 columns]
```

---

## 4. Relational Joins (`join`, `leftJoin`) & Vertical Concatenation

### 4.1 Single-Key Joins (`JoinKind`: `.inner`, `.left`, `.right`, `.full`)

When both frames share the same key column name and have no overlapping non-key column names, use `join(_:on:kind:)` or the shorthand `leftJoin(_:on:)`:

```swift
let customers = try DataFrame(columns: [
    (name: "cust_id", values: ["C1", "C2", "C3"]),
    (name: "name", values: ["Alice", "Bob", "Charlie"]),
])

let purchases = try DataFrame(columns: [
    (name: "cust_id", values: ["C1", "C1", "C4"]),
    (name: "total", values: [50.0, 120.0, 300.0]),
])

// Left Join: keeps all customers (C1, C2, C3); C2 & C3 get .missing for "total"
let leftJoined = try customers.leftJoin(purchases, on: "cust_id")
print("--- Left Join ---")
print(leftJoined)

// Full Outer Join: includes unmatched right rows (C4) as well
let fullJoined = try customers.join(purchases, on: "cust_id", kind: .full)
print("\n--- Full Outer Join ---")
print(fullJoined)
```

**Console Output**:
```text
--- Left Join ---
   cust_id     name  total
0       C1    Alice     50
1       C1    Alice    120
2       C2      Bob    NaN
3       C3  Charlie    NaN

[4 rows x 3 columns]

--- Full Outer Join ---
   cust_id     name  total
0       C1    Alice     50
1       C1    Alice    120
2       C2      Bob    NaN
3       C3  Charlie    NaN
4       C4      nil    300

[5 rows x 3 columns]
```

### 4.2 Multi-Key Joins & Overlapping Column Suffixes (`join(_:leftOn:rightOn:kind:suffixes:)`)

When key columns have different names on the left and right tables, or when both tables share non-key column names (such as `"updated_at"` or `"score"`), use `join(_:leftOn:rightOn:kind:suffixes:)`:

```swift
let targetQ1 = try DataFrame(columns: [
    (name: "dept_id", values: ["D1", "D2"]),
    (name: "revenue", values: [1000.0, 800.0]),
])

let actualQ1 = try DataFrame(columns: [
    (name: "id", values: ["D1", "D2"]),
    (name: "revenue", values: [1150.0, 750.0]),
])

let comparison = try targetQ1.join(
    actualQ1,
    leftOn: ["dept_id"],
    rightOn: ["id"],
    kind: .inner,
    suffixes: (left: "_target", right: "_actual")
)
print(comparison)
```

**Console Output**:
```text
   dept_id  revenue_target   id  revenue_actual
0       D1            1000   D1            1150
1       D2             800   D2             750

[2 rows x 4 columns]
```

### 4.3 Vertical Row Concatenation (`DataFrame.concatenatingRows`)

To stack multiple `DataFrame`s with identical schemas vertically (like `pd.concat([df1, df2], axis=0)`):

```swift
let jan = try DataFrame(columns: [
    (name: "month", values: ["Jan"]),
    (name: "rev", values: [100.0]),
])
let feb = try DataFrame(columns: [
    (name: "month", values: ["Feb"]),
    (name: "rev", values: [130.0]),
])

let combined = try DataFrame.concatenatingRows([jan, feb])
print(combined)
```

```text
   month  rev
0    Jan  100
1    Feb  130

[2 rows x 2 columns]
```

---

## 5. Reshaping & Pivot Tables

### 5.1 Unpivoting Wide-to-Long (`melt`)

`melt(idColumns:variableName:valueName:)` converts wide metric columns into `(variable, value)` rows:

```swift
let wideMetrics = try DataFrame(columns: [
    (name: "city", values: ["Seoul", "Busan"]),
    (name: "q1", values: [100.0, 60.0]),
    (name: "q2", values: [120.0, 75.0]),
])

let longMetrics = try wideMetrics.melt(
    idColumns: ["city"],
    variableName: "quarter",
    valueName: "revenue"
)
print(longMetrics)
```

**Console Output**:
```text
    city  quarter  revenue
0  Seoul       q1      100
1  Busan       q1       60
2  Seoul       q2      120
3  Busan       q2       75

[4 rows x 3 columns]
```

### 5.2 Reshape without Aggregation (`pivoted`) vs. Aggregating `pivotTable`

Bandal provides two distinct pivot methods:
1. **`pivoted(index:columns:values:)` (`df.pivot`)**: Reshapes unique `(index, columns)` pairs into wide columns named directly after the pivot values (`"q1"`, `"q2"`). Throws `.duplicateColumnName` if duplicate `(index, column)` pairs exist.
2. **`pivotTable(index:columns:values:using:)` (`df.pivot_table`)**: Groups and aggregates multiple rows per `(index, columns)` pair using an `AggregateFunction`.

```swift
// 1. Exact inverse of `melt` when (city, quarter) pairs are unique:
let unmelted = try longMetrics.pivoted(
    index: "city",
    columns: "quarter",
    values: "revenue"
)
print("--- pivoted (Unique Pairs) ---")
print(unmelted)

// 2. Aggregating pivotTable when multiple rows share the same (region, channel):
let summaryPivot = try sales.pivotTable(
    index: ["region"],
    columns: "channel",
    values: "amount",
    using: .sum
)
print("\n--- pivotTable (Aggregated Sum) ---")
print(summaryPivot)
```

**Console Output**:
```text
--- pivoted (Unique Pairs) ---
    city   q1   q2
0  Busan   60   75
1  Seoul  100  120

[2 rows x 3 columns]

--- pivotTable (Aggregated Sum) ---
   region  amount_string_Online_sum  amount_string_Retail_sum
0   Busan                       150                        90
1   Seoul                       320                        80

[2 rows x 3 columns]
```

### 5.3 Contingency Tables (`crossTabulated` / `pd.crosstab`)

`crossTabulated(index:columns:normalize:)` computes a frequency table (or joint probability table when `normalize: true`) between two categorical columns:

```swift
let ctCounts = try sales.crossTabulated(index: "region", columns: "channel", normalize: false)
print("--- Cross-Tabulation (Counts) ---")
print(ctCounts)

let ctProps = try sales.crossTabulated(index: "region", columns: "channel", normalize: true)
print("\n--- Cross-Tabulation (Normalized Proportions) ---")
print(ctProps)
```

**Console Output**:
```text
--- Cross-Tabulation (Counts) ---
   region  Online  Retail
0   Busan       1       1
1   Seoul       2       1

[2 rows x 3 columns]

--- Cross-Tabulation (Normalized Proportions) ---
   region  Online  Retail
0   Busan     0.2     0.2
1   Seoul     0.4     0.2

[2 rows x 3 columns]
```

### 5.4 Exploding Delimited Cells (`exploded`) & Transposing (`transposed`)

```swift
let articles = try DataFrame(columns: [
    (name: "id", values: [1, 2]),
    (name: "tags", values: ["swift, mlx, data", "pandas, python"]),
])

// Expand comma-separated tags into individual rows
let explodedTags = try articles.exploded("tags", separator: ",")
print("--- Exploded ---")
print(explodedTags)

// Transpose rows and columns
let transposed = try wideMetrics.transposed(indexColumnName: "metric", rowPrefix: "city_")
print("\n--- Transposed ---")
print(transposed)
```

**Console Output**:
```text
--- Exploded ---
   id    tags
0   1   swift
1   1     mlx
2   1    data
3   2  pandas
4   2  python

[5 rows x 2 columns]

--- Transposed ---
   metric  city_0  city_1
0    city   Seoul   Busan
1      q1     100      60
2      q2     120      75

[3 rows x 3 columns]
```

---

> **Next Step**: Proceed to **[Guide 07: Time Series & Window Functions →](07-Time-Series-and-Window-Functions.md)** to learn about rolling windows (row-based and time-interval-based), expanding windows, exponentially weighted moving averages (`ewm`), `joinAsOf`, and time-bucket resampling.
