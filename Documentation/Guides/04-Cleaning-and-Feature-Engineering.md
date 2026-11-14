# Guide 04: Data Cleaning, Missing Values & Feature Engineering

> **Language**: **English** | [한국어](04-Cleaning-and-Feature-Engineering.ko.md)  
> **Navigation**: [← Guide 03: Indexing, Filtering & Querying](03-Indexing-Filtering-and-Querying.md) | [API Reference](../API.md) | **Next**: [Guide 05: Vectorized Math & Statistics →](05-Vectorized-Math-and-Statistics.md)

---

## Overview

Before statistical modeling or machine learning, raw datasets require cleaning and feature engineering: diagnosing and imputing missing values, removing duplicate records, standardizing strings, extracting calendar features from timestamps, scaling numerical variables, discretizing continuous columns into bins, and one-hot encoding categorical columns.

This guide covers all of Bandal's data-wrangling and feature-engineering APIs:
- **Missing Value Diagnostics & Imputation** (`missingSummary()`, `MissingCounts`, `fillMissing`, `forwardFilled`, `backwardFilled`, `interpolated`, `dropMissing`, `replacing`, `Series.where`)
- **Deduplication, Sorting & Reproducible Sampling** (`duplicatedRows`, `dropDuplicates`, `sorted(by:)`, `sample`)
- **Adding, Mapping, Renaming & Casting Columns** (`withColumn`, `mapped`, `renamedColumns`, `castColumn`)
- **Vectorized String & Date Feature Extraction** (`lowercased`, `uppercased`, `trimmed`, `stringLength`, `replacingSubstring`, `containsSubstring`, `dateComponent`)
- **ML Feature Engineering** (`standardized`, `minMaxScaled`, `binned`, `quantileBinned`, `oneHotEncoded`)

---

## 1. Quick Comparison: `pandas` vs. `Bandal`

| Task | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| Count missing values per column | `df.isna().sum()` | `df.missingSummary()` (`MissingCounts`) or `df.missingCounts()` |
| Boolean mask DataFrame of nulls | `df.isna()` | `try df.missingMask()` |
| Fill missing values with constant | `df.fillna(0)` / `df[["a"]].fillna(0)` | `try df.fillMissing(in: ["a"], with: 0.0)` |
| Forward / backward fill | `df.ffill()` / `df.bfill()` | `try df.forwardFilled()` / `try df.backwardFilled()` |
| Linear interpolation | `df["temp"].interpolate()` | `try df.interpolated("temp", as: "temp_interp")` |
| Drop rows with missing values | `df.dropna(subset=["a"])` | `try df.dropMissing(in: ["a"])` |
| Conditional element replacement | `s.where(mask, other=0)` | `try s.where(mask, other: 0.0)` |
| Drop duplicate rows | `df.drop_duplicates(subset=["id"], keep="first")` | `try df.dropDuplicates(in: ["id"], keep: .first)` |
| Multi-column sort | `df.sort_values(["dept", "salary"], ascending=[True, False])` | `try df.sorted(by: [SortDescriptor("dept"), SortDescriptor("salary", order: .descending)])` |
| Random sample (count / fraction) | `df.sample(n=10, random_state=42)` / `df.sample(frac=0.2)` | `try df.sample(10, seed: 42)` / `try df.sample(fraction: 0.2, seed: 42)` |
| Type casting | `df["a"].astype(float)` | `try df.castColumn("a", to: .number)` |
| Z-score standardization | `(s - s.mean()) / s.std()` | `try df.standardized("price", as: "price_z")` |
| Min-Max scaling | `(s - s.min()) / (s.max() - s.min())` | `try df.minMaxScaled("price", to: 0.0...1.0, as: "price_norm")` |
| Fixed-edge / quantile binning | `pd.cut(...)` / `pd.qcut(...)` | `try df.binned(...)` / `try df.quantileBinned(...)` |
| One-hot encoding | `pd.get_dummies(df, columns=["city"], drop_first=True)` | `try df.oneHotEncoded("city", dropFirst: true)` |

---

## 2. Diagnosing & Handling Missing Values

### 2.1 Inspecting Missingness (`missingSummary`, `missingCounts`, `missingMask`, `isMissing`)

`df.missingSummary()` returns a structured `MissingCounts` instance that prints an aligned summary table (with a total-missing footer) while providing subscript access (`missing["price"]`) and `totalMissing`:

```swift
import Bandal

let sensorData = try DataFrame(columns: [
    (name: "hour", values: [1, 2, 3, 4, 5]),
    (name: "temp", values: [20.0, nil, nil, 26.0, 28.0]),
    (name: "status", values: ["OK", "WARN", nil, "OK", "ERR"]),
])

let missing: MissingCounts = sensorData.missingSummary()
print(missing)
print("Missing in 'temp': \(missing["temp"] ?? 0), Total: \(missing.totalMissing)")
```

**Console Output**:
```text
hour      0
temp      2
status    1
Total missing: 3 (rows: 5)
Missing in 'temp': 2, Total: 3
```

### 2.2 Imputing Missing Values (`fillMissing`, `forwardFilled`, `backwardFilled`, `interpolated`)

Bandal provides four imputation strategies:
1. **Constant replacement**: `df.fillMissing(in: selectedColumns, with: replacement)`
2. **Forward fill (`ffill`)**: `df.forwardFilled()` carries the last known observation forward.
3. **Backward fill (`bfill`)**: `df.backwardFilled()` carries the next valid observation backward.
4. **Linear interpolation**: `df.interpolated(_:as:)` (or `Series.interpolated()`) linearly interpolates across interior gaps of any length while leaving leading/trailing `.missing` values untouched.

```swift
// 1. Constant fill on specific column(s)
let filledStatus = try sensorData.fillMissing(in: ["status"], with: "UNKNOWN")

// 2. Forward fill across the frame
let ffilled = try sensorData.forwardFilled()
print("--- Forward Filled ---")
print(ffilled)

// 3. Linear interpolation on 'temp' (20.0 -> [22.0, 24.0] -> 26.0 -> 28.0)
let interpolated = try sensorData.interpolated("temp", as: "temp_linear")
print("\n--- Linearly Interpolated ---")
print(interpolated)
```

**Console Output**:
```text
--- Forward Filled ---
   hour  temp  status
0     1    20      OK
1     2    20    WARN
2     3    20    WARN
3     4    26      OK
4     5    28     ERR

[5 rows x 3 columns]

--- Linearly Interpolated ---
   hour  temp  status  temp_linear
0     1    20      OK           20
1     2   NaN    WARN           22
2     3   NaN     NaN           24
3     4    26      OK           26
4     5    28     ERR           28

[5 rows x 4 columns]
```

### 2.3 Dropping Missing Rows & Replacing Sentinel Values (`dropMissing`, `replacing`, `where`)

```swift
// Remove rows that have missing values in "status" (omit `in:` to check all columns)
let completeStatus = try sensorData.dropMissing(in: ["status"])

// Replace sentinel values (e.g., "ERR" -> .missing) across specified columns
let cleanedStatus = try sensorData.replacing("ERR", with: .missing, in: ["status"])

// Conditional Series replacement (`pd.Series.where`):
// Keep values where mask is true; replace false positions with `other`
let tempSeries = try interpolated.series("temp_linear")
let normalRangeMask = try (tempSeries >= 21.0) && (tempSeries <= 27.0)
let cappedSeries = try tempSeries.where(normalRangeMask, other: .missing)
print(cappedSeries)
```

**Console Output**:
```text
   temp_linear
0          NaN
1           22
2           24
3           26
4          NaN
Name: temp_linear, Length: 5
```

---

## 3. Deduplication, Sorting & Sampling

### 3.1 Detecting & Dropping Duplicates (`duplicatedRows`, `dropDuplicates`)

`DuplicateKeep` controls which occurrence of a duplicate key is preserved:
- `.first` (default): Keeps the first occurrence, drops subsequent duplicates.
- `.last`: Keeps the last occurrence, drops earlier duplicates.
- `.none`: Drops **every** row whose key appears more than once.

```swift
let logs = try DataFrame(columns: [
    (name: "user", values: ["u1", "u2", "u1", "u3", "u2"]),
    (name: "event", values: ["login", "view", "login", "buy", "buy"]),
    (name: "seq", values: [1, 2, 3, 4, 5]),
])

// Deduplicate by "user", keeping only each user's latest event (.last)
let latestPerUser = try logs.dropDuplicates(in: ["user"], keep: .last)
print(latestPerUser)
```

**Console Output**:
```text
   user  event  seq
0    u1  login    3
1    u3    buy    4
2    u2    buy    5

[3 rows x 3 columns]
```

### 3.2 Multi-Column Sorting (`sorted(by:)`)

`sorted(by: [SortDescriptor])` performs a stable sort (using an $O(N)$ LSD radix sort for numeric/date columns $\ge 512$ rows). Missing values (`.missing`) are always placed last in both `.ascending` and `.descending` order.

```swift
let sortedLogs = try logs.sorted(by: [
    SortDescriptor("user", order: .ascending),
    SortDescriptor("seq", order: .descending),
])
print(sortedLogs)
```

```text
   user  event  seq
0    u1  login    3
1    u1  login    1
2    u2    buy    5
3    u2   view    2
4    u3    buy    4

[5 rows x 3 columns]
```

### 3.3 Reproducible Random Sampling (`sample`)

Pass an optional `seed: UInt64` (using a deterministic `SplitMix64` PRNG) for reproducible train/test or exploratory sub-sampling:

```swift
// Sample exact row count
let threeRows = try logs.sample(3, seed: 2026)

// Sample 40% of rows
let fortyPercent = try logs.sample(fraction: 0.4, seed: 2026)
print(fortyPercent)
```

```text
   user  event  seq
0    u1  login    3
1    u3    buy    4

[2 rows x 3 columns]
```

---

## 4. Adding, Mapping, Renaming & Casting Columns

```swift
let rawProducts = try DataFrame(columns: [
    (name: "sku_code", values: [" ab-10 ", " cd-20 ", " ef-30 "]),
    (name: "raw_price", values: ["$120.00", "$250.50", "$80.00"]),
])

let transformed = try rawProducts
    // 1. Rename columns
    .renamedColumns(["sku_code": "sku", "raw_price": "price"])
    // 2. Cast string currency column to .number
    .castColumn("price", to: .number)
    // 3. Transform a single column element-by-element
    .mapped("price") { val in
        guard let p = val.numberValue else { return .missing }
        return .number(p * 1.1) // Add 10% tax
    }
    // 4. Add a new computed column from each DataRow (like pandas df.assign)
    .withColumn(name: "tier") { row in
        guard let p = row["price"]?.numberValue else { return .missing }
        return p >= 200.0 ? "Premium" : "Standard"
    }

print(transformed)
```

**Console Output**:
```text
       sku   price      tier
0   ab-10      132  Standard
1   cd-20   275.55   Premium
2   ef-30       88  Standard

[3 rows x 3 columns]
```

---

## 5. String & Date Column Operations

### 5.1 String Cleaning & Extraction (`trimmed`, `uppercased`, `lowercased`, `replacingSubstring`, `stringLength`, `containsSubstring`)

 Every string method accepts an optional `as outputColumn: String? = nil` parameter. When `as` is `nil`, the source column is updated in place (or defaults to `"<col>_length"` / `"<col>_contains"`).

```swift
let cleanedStrings = try transformed
    .trimmed("sku")
    .uppercased("sku")
    .replacingSubstring(in: "sku", target: "-", with: "_")
    .stringLength("sku", as: "sku_len")
    .containsSubstring("sku", "CD", caseSensitive: true, as: "is_cd")

print(cleanedStrings)
```

**Console Output**:
```text
     sku   price      tier  sku_len  is_cd
0  AB_10     132  Standard        5  false
1  CD_20  275.55   Premium        5   true
2  EF_30      88  Standard        5  false

[3 rows x 5 columns]
```

### 5.2 Extracting Calendar Components from `.date` Columns (`dateComponent`)

`dateComponent(_:from:as:)` extracts UTC calendar components (`.year`, `.month`, `.day`, `.weekday`, `.hour`, `.minute`, `.second`) into a numeric column:

```swift
let events = try DataFrame(columns: [
    (name: "ts", values: ["2026-03-15", "2026-07-04", "2026-12-25"]),
])
.castColumn("ts", to: .date, dateFormat: "yyyy-MM-dd")
.dateComponent(.year, from: "ts", as: "year")
.dateComponent(.month, from: "ts", as: "month")
.dateComponent(.weekday, from: "ts", as: "weekday")

print(events)
```

**Console Output**:
```text
                     ts  year  month  weekday
0  2026-03-15T00:00:00Z  2026      3        1
1  2026-07-04T00:00:00Z  2026      7        7
2  2026-12-25T00:00:00Z  2026     12        6

[3 rows x 4 columns]
```

---

## 6. Feature Scaling, Discretization (Binning) & One-Hot Encoding

Bandal includes built-in preprocessing transformers commonly found in `scikit-learn` and `pandas`:

### 6.1 Z-Score Standardization & Min-Max Scaling (`standardized`, `minMaxScaled`)

- **`standardized(_:as:)`**: Centers the column to mean `0.0` and sample standard deviation `1.0` ($\frac{x - \mu}{s}$).
- **`minMaxScaled(_:to:as:)`**: Rescales values linearly into `range` (default `0.0...1.0`).

```swift
let scores = try DataFrame(columns: [
    (name: "city", values: ["Seoul", "Busan", "Incheon", "Seoul"]),
    (name: "score", values: [50.0, 70.0, 80.0, 100.0]),
])
.standardized("score", as: "score_z")
.minMaxScaled("score", to: 0.0...1.0, as: "score_minmax")

print(scores)
```

**Console Output**:
```text
      city  score    score_z  score_minmax
0    Seoul     50  -1.183216             0
1    Busan     70  -0.236643           0.4
2  Incheon     80   0.236643           0.6
3    Seoul    100   1.183216             1

[4 rows x 4 columns]
```

### 6.2 Discretization: Fixed-Edge (`binned`) & Quantile Binning (`quantileBinned`)

- **`binned(_:edges:labels:rightClosed:as:)`** (`pd.cut`): Assigns each number into intervals defined by strictly increasing `edges`. Values outside the outer edges become `.missing`.
- **`quantileBinned(_:bins:labels:as:)`** (`pd.qcut`): Splits a numeric column into `bins` equal-probability quantile buckets.

```swift
let binnedDF = try scores
    .select("city", "score")
    .binned(
        "score",
        edges: [0.0, 60.0, 85.0, 100.0],
        labels: ["Low", "Mid", "High"],
        rightClosed: true,
        as: "grade"
    )
    .quantileBinned(
        "score",
        bins: 2,
        labels: ["Bottom50%", "Top50%"],
        as: "half"
    )

print(binnedDF)
```

**Console Output**:
```text
      city  score  grade       half
0    Seoul     50    Low  Bottom50%
1    Busan     70    Mid  Bottom50%
2  Incheon     80    Mid     Top50%
3    Seoul    100   High     Top50%

[4 rows x 4 columns]
```

### 6.3 One-Hot Encoding (`oneHotEncoded` / `pd.get_dummies`)

`oneHotEncoded(_:prefix:dropFirst:includeMissing:dropOriginal:)` expands a categorical or string column into numeric `0.0` / `1.0` indicator columns in first-appearance category order:

- `dropFirst: true` omits the first category column to avoid multicollinearity in linear regression.
- `includeMissing: true` appends an additional `<prefix>_nan` indicator column.
- `dropOriginal: false` retains the original categorical column alongside the indicator columns.

```swift
let encoded = try binnedDF
    .select("city", "score")
    .oneHotEncoded("city", prefix: "city", dropFirst: false, dropOriginal: true)

print(encoded)
```

**Console Output**:
```text
   score  city_Seoul  city_Busan  city_Incheon
0     50           1           0             0
1     70           0           1             0
2     80           0           0             1
3    100           1           0             0

[4 rows x 4 columns]
```

---

> **Next Step**: Move on to **[Guide 05: Vectorized Math & Statistics →](05-Vectorized-Math-and-Statistics.md)** to explore element-wise arithmetic, mathematical functions, cumulative reductions, and Bandal's structured descriptive statistics APIs.
