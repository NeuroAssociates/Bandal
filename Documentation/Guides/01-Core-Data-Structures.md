# Guide 01: Core Data Structures & Swift Value Semantics

> **Language**: **English** | [한국어](01-Core-Data-Structures.ko.md)  
> **Navigation**: [API Reference](../API.md) | **Next**: [Guide 02: Data I/O & Inspection →](02-Data-IO-and-Inspection.md)

---

## Overview

If you have used Python's `pandas`, you already know the core mental model of tabular data analysis: a **`DataFrame`** is a 2D table of named columns, and a **`Series`** is a single 1D column vector.

**Bandal** brings that exact mental model to Swift, while leveraging Swift's static type safety, value semantics, and hardware-accelerated columnar storage (`mlx-swift`). This guide explains how `pandas` concepts map to Bandal's core types:

- **`DataValue`** — A type-safe cell value (`Double`, `String`, `Bool`, `Date`, or `.missing`) supporting direct Swift literals.
- **`DataType`** — Logical column data types (`.number`, `.string`, `.boolean`, `.date`, `.mixed`, `.unknown`).
- **`Series`** — A 1D named column backed by contiguous typed columnar buffers (`Float64Column`, `Int64Column`, `StringColumn`, `CategoricalColumn`, `BoolColumn`, `DateColumn`).
- **`DataFrame`** — An immutable 2D table of named columns with `O(1)` unchanged column sharing.
- **`DataRow`** — A read-only view of a single row with column-name and positional subscripting.
- **`DataFrameError`** — Structured, descriptive errors thrown when schemas, column names, or types do not match.

---

## 1. Mental Model Shift: Python `pandas` vs. Swift `Bandal`

| Concept | Python `pandas` | Swift `Bandal` | Why It Matters |
| --- | --- | --- | --- |
| **Mutability** | Mutable (`df["a"] = 1`, `inplace=True`) | Immutable value types (`let df2 = try df.withColumn(...)`) | Eliminates accidental side effects (like pandas' `SettingWithCopyWarning`). Unchanged columns are shared in `O(1)` memory. |
| **Missing Values** | `np.nan`, `None`, `pd.NA` | `DataValue.missing` (or literal `nil`) | Explicit, unified missing representation across numbers, strings, booleans, and dates. Tracked via a 64-bit `BitSet` null bitmap. |
| **Heterogeneous Cells** | `dtype: object` (boxed Python objects) | `DataValue` enum & specialized columnar storage | Pure numeric/string/bool/date columns are stored in unboxed contiguous buffers; `DataValue` provides ergonomic cell access. |
| **Error Handling** | Runtime `KeyError`, `ValueError` | Swift `throws` (`DataFrameError`) | Compiler enforces handling of missing column names or invalid type conversions via `try`. |
| **Console Output** | `print(df)`, `df.describe()` | `print(df)`, `print(try df.describeStructured())` | Every Bandal type and result struct conforms to `CustomStringConvertible` and provides `.describe()`. |

---

## 2. `DataValue`: Representing Individual Cells

In `pandas`, a cell can hold a float, int, string, bool, timestamp, or `NaN`/`None`. In Swift, collections must have a concrete type. Bandal solves this with `DataValue`:

```swift
public enum DataValue: Hashable, Sendable, Codable,
    CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable,
    ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByStringLiteral, ExpressibleByBooleanLiteral, ExpressibleByNilLiteral
{
    case number(Double)
    case string(String)
    case boolean(Bool)
    case date(Date)
    case missing
}
```

### 2.1 Creating `DataValue` Instances with Literals

Because `DataValue` conforms to Swift's `ExpressibleBy*Literal` protocols, you rarely need to write `.number(10.5)` or `.string("Seoul")`. You can write standard Swift literals directly inside `[DataValue]` arrays:

```swift
import Foundation
import Bandal

// Using Swift literals directly in a [DataValue] array
let cells: [DataValue] = [
    42,                         // .number(42.0) via ExpressibleByIntegerLiteral
    19.95,                      // .number(19.95) via ExpressibleByFloatLiteral
    "Seoul",                    // .string("Seoul") via ExpressibleByStringLiteral
    true,                       // .boolean(true) via ExpressibleByBooleanLiteral
    nil,                        // .missing via ExpressibleByNilLiteral
    .date(Date(timeIntervalSince1970: 1_700_000_000)) // Explicit Date case
]

print(cells)
// Output: [42.0, 19.95, Seoul, true, nil, 2023-11-14T22:13:20Z]
```

### 2.2 Extracting Typed Swift Values from `DataValue`

When reading a cell from a `DataFrame`, `Series`, or `DataRow`, use convenience optional properties instead of verbose `switch` statements:

| Property | Return Type | Description |
| --- | --- | --- |
| `value.numberValue` | `Double?` | Returns the `Double` if `.number`, otherwise `nil` |
| `value.stringValue` | `String?` | Returns the `String` if `.string`, otherwise `nil` |
| `value.boolValue` | `Bool?` | Returns the `Bool` if `.boolean`, otherwise `nil` |
| `value.dateValue` | `Date?` | Returns the `Date` if `.date`, otherwise `nil` |
| `value.isMissing` | `Bool` | `true` when `value == .missing` |
| `value.description` | `String` | Formatted text (`"nil"` for `.missing`, ISO-8601 for `.date`) |

```swift
let priceCell: DataValue = 1250.5
let missingCell: DataValue = nil

if let price = priceCell.numberValue {
    print("Price with tax: \(price * 1.1)")
}
print("Is missing? \(missingCell.isMissing)")
```

```text
Price with tax: 1375.5500000000002
Is missing? true
```

---

## 3. `DataType`: Column Type Inference

Every column in a `DataFrame` or `Series` has a logical `DataType`, inferred automatically from its non-missing values:

```swift
public enum DataType: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case number   // All non-missing values are .number
    case string   // All non-missing values are .string
    case boolean  // All non-missing values are .boolean
    case date     // All non-missing values are .date
    case mixed    // Contains two or more distinct value types (e.g., .number and .string)
    case unknown  // Column is empty (0 rows) or every row is .missing
}
```

> **Tip for `pandas` users**: In `pandas`, a numeric column with a missing value used to force `int64` into `float64`, and a string column with `None` became `object`. In Bandal, `.missing` values are tracked in a separate `BitSet` null bitmap and **never pollute the column's `DataType`**. A column of `[10, nil, 30]` still has `DataType.number`.

---

## 4. `Series`: 1D Named Column Vector

A `Series` represents a single named column of data, analogous to `pd.Series`. It conforms to `RandomAccessCollection` (where `Element == DataValue`, indexed by `0..<count`).

### 4.1 Creating a `Series`

**Pandas (Python)**
```python
s_num = pd.Series([10.0, 20.0, 30.0], name="revenue")
s_opt = pd.Series([10.0, None, 30.0], name="revenue")
s_mix = pd.Series(["A", "B", None], name="grade")
```

**Bandal (Swift)**
```swift
import Bandal

// 1. Fast unboxed numeric initializer ([Double])
let sNum = Series(name: "revenue", numbers: [10.0, 20.0, 30.0])

// 2. Optional numeric initializer ([Double?], where nil -> .missing)
let sOpt = Series(name: "revenue", values: [10.0, nil, 30.0])

// 3. General DataValue initializer ([DataValue] with literals)
let sMix = Series(name: "grade", values: ["A", "B", nil])

print(sOpt)
```

**Console Output (`print(sOpt)`)**:
```text
   revenue
0     10.0
1      nil
2     30.0
Name: revenue, dtype: number
```

### 4.2 Inspecting a `Series`

| Property / Subscript | Type | Pandas Equivalent | Description |
| --- | --- | --- | --- |
| `s.name` | `String` | `s.name` | Column name |
| `s.count` | `Int` | `len(s)` / `s.size` | Total number of rows (including `.missing`) |
| `s.isEmpty` | `Bool` | `s.empty` | `true` if `count == 0` |
| `s.nonMissingCount` | `Int` | `s.count()` | Number of non-missing elements |
| `s.missingCount` | `Int` | `s.isna().sum()` | Number of `.missing` elements |
| `s.dtype` | `DataType` | `s.dtype` | Logical type (`.number`, `.string`, etc.) |
| `s.values` | `[DataValue]` | `s.tolist()` | All values in row order |
| `s.numbers` | `[Double]` | `s.dropna().to_numpy()` | Non-missing numeric values |
| `s[index]` | `DataValue` | `s.iloc[i]` | Element at zero-based row `index` |

```swift
let scores = Series(name: "score", values: [85.0, 92.5, nil, 78.0, 92.5])

print("Count: \(scores.count), Non-missing: \(scores.nonMissingCount), Missing: \(scores.missingCount)")
print("Dtype: \(scores.dtype)")
print("First element: \(scores[0])")
print("Non-missing numbers: \(scores.numbers)")
print("Unique values: \(scores.unique())")
```

```text
Count: 5, Non-missing: 4, Missing: 1
Dtype: number
First element: 85.0
Non-missing numbers: [85.0, 92.5, 78.0, 92.5]
Unique values: [85.0, 92.5, 78.0]
```

---

## 5. `DataFrame`: 2D Column-Oriented Table

`DataFrame` is Bandal's primary data structure. Every column must have a unique, non-empty name, and all columns must have the same number of rows (`rowCount`).

### 5.1 Four Ways to Construct a `DataFrame` in Memory

#### A. Ordered Mixed-Type Columns (`[(name: String, values: [DataValue])]`) — *Most Common*
Preserves the exact column order you specify and supports any combination of numbers, strings, booleans, dates, and `nil` (`.missing`).

**Pandas (Python)**
```python
df = pd.DataFrame({
    "city": ["Seoul", "Busan", "Incheon", "Daegu"],
    "temp": [21.5, 24.0, 20.1, None],
    "rain": [False, True, False, False]
})
```

**Bandal (Swift)**
```swift
let df = try DataFrame(columns: [
    (name: "city", values: ["Seoul", "Busan", "Incheon", "Daegu"]),
    (name: "temp", values: [21.5, 24.0, 20.1, nil]),
    (name: "rain", values: [false, true, false, false]),
])
print(df)
```

**Console Output**:
```text
      city  temp   rain
0    Seoul  21.5  false
1    Busan    24   true
2  Incheon  20.1  false
3    Daegu   nil  false
```

#### B. Ordered Pure Numeric Columns (`[(name: String, values: [Double])]`)
Zero-boxing fast path when all columns are `Double` arrays without missing values:

```swift
let matrixDF = try DataFrame(numericColumns: [
    (name: "x", values: [1.0, 2.0, 3.0]),
    (name: "y", values: [10.0, 20.0, 30.0]),
])
print(matrixDF)
```

```text
   x   y
0  1  10
1  2  20
2  3  30
```

#### C. Dictionary of Numeric Columns (`[String: [Double]]`)
Convenient for quick numeric tables. Because Swift dictionaries are unordered, Bandal **sorts column names lexicographically** (`A` to `Z`) for deterministic behavior:

```swift
let dictDF = try DataFrame([
    "revenue": [100.0, 200.0, 150.0],
    "cost": [60.0, 110.0, 90.0],
])
print(dictDF.columnNames) // ["cost", "revenue"] (sorted alphabetically)
print(dictDF)
```

```text
["cost", "revenue"]
   cost  revenue
0    60      100
1   110      200
2    90      150
```

#### D. From `MLXVector` Columns (`[(name: String, vector: MLXVector)]`)
Constructs a `DataFrame` directly from `MLXVector` buffers (see [Guide 08](08-MLX-Acceleration-and-Workflows.md) for MLX tensor workflows):

```swift
let vec = MLXVector(values: [1.5, 0.0, 3.5], isMissing: [false, true, false])
let mlxDF = try DataFrame(mlxColumns: [
    (name: "signal", vector: vec)
])
print(mlxDF)
```

```text
   signal
0     1.5
1     nil
2     3.5
```

---

## 6. `DataRow`: Inspecting Individual Rows

When you access a single row via `df[row: index]`, Bandal returns a `DataRow` struct (analogous to `df.iloc[i]` in `pandas`). You can also retrieve a dictionary of column-to-value mappings via `try df.row(at: index)`.

```swift
public struct DataRow: Equatable, Hashable, Sendable, Codable, CustomStringConvertible {
    public let index: Int
    public let columnOrder: [String]
    public let values: [String: DataValue]

    public subscript(_ column: String) -> DataValue?
    public subscript(position: Int) -> DataValue
}
```

### Example: Accessing Row Values by Column Name or Position

```swift
let row0 = df[row: 0]
print(row0)
print("---")
print("City by name: \(row0["city"]?.stringValue ?? "Unknown")")
print("Temp by position 1: \(row0[1].numberValue ?? .nan)")
```

**Console Output**:
```text
city    Seoul
temp    21.5
rain    false
Row: 0, Columns: 3
---
City by name: Seoul
Temp by position 1: 21.5
```

---

## 7. Value Semantics & Functional Chaining (`pipe`)

In `pandas`, mutating a slice of a DataFrame often triggers `SettingWithCopyWarning` because it is unclear whether you are modifying a view or a copy.

In Bandal, **`DataFrame` and `Series` are Swift value types (`struct`)**. Every transformation returns a new `DataFrame` while sharing untouched underlying column storage (`AnyColumn` references) in `O(1)` time.

### Chaining Transformations Idiomatically

```swift
let raw = try DataFrame(columns: [
    (name: "product", values: ["Laptop", "Mouse", "Monitor", "Keyboard"]),
    (name: "price", values: [1200.0, 25.0, nil, 80.0]),
    (name: "qty", values: [2, 10, 3, 5]),
])

// Fluent multi-step pipeline
let processed = try raw
    .fillMissing(in: ["price"], with: 300.0)
    .combining("price", "qty", using: .multiply, as: "total")
    .query(.column("total", greaterThanOrEqual: 300.0))
    .sorted(by: [SortDescriptor("total", order: .descending)])

print(processed)
```

**Console Output**:
```text
    product  price  qty  total
0    Laptop   1200    2   2400
1   Monitor    300    3    900
2  Keyboard     80    5    400
```

You can also use `.pipe(_:)` (just like `df.pipe(fn)` in `pandas`) to insert reusable custom transformation functions into a method chain:

```swift
func addTax(_ frame: DataFrame) throws -> DataFrame {
    try frame.scaling("total", by: 1.1, using: .multiply, as: "total_with_tax")
}

let finalReport = try processed.pipe(addTax)
print(finalReport)
```

```text
    product  price  qty  total  total_with_tax
0    Laptop   1200    2   2400            2640
1   Monitor    300    3    900             990
2  Keyboard     80    5    400             440
```

---

## 8. Error Handling with `DataFrameError`

Operations that validate schemas, column names, or numeric invariants throw `DataFrameError`. It conforms to `LocalizedError` and `Equatable`, making both debugging and unit testing straightforward:

| `DataFrameError` Case | When It Is Thrown |
| --- | --- |
| `.emptyColumns` | Creating a `DataFrame` with `0` columns or calling a reduction with no target columns |
| `.emptyColumnName` | A column name is `""` |
| `.duplicateColumnName(String)` | Two columns share the same name (or `pivoted` encounters duplicate keys) |
| `.unequalColumnLengths(column:expected:actual:)` | A column or `BooleanMask` has a different length than `rowCount` |
| `.columnNotFound(String)` | Referencing a column name that does not exist |
| `.nonFiniteValue(column:row:)` | Passing `.nan` or `.infinity` into a numeric column |
| `.nonNumericColumn(String)` | Calling a numeric-only API (`column(_:)`, `standardized`, etc.) on a non-numeric or missing-containing column |
| `.invalidCSV(String)` | Malformed CSV quotes or inconsistent field counts |
| `.columnNameConflict(String)` | Joining frames that share non-key column names without suffixes, or adding an existing column name |
| `.incompatibleSchema(expected:actual:)` | `concatenatingRows` called on frames with mismatched column schemas |
| `.invalidConversion(column:row:type:)` | `castColumn` or `resample` fails to convert a cell to the target `DataType` |
| `.invalidWindowSize(Int)` | Window size `<= 0`, `alpha` outside `(0, 1]`, or invalid bin edges |

```swift
do {
    _ = try df.series("non_existent_column")
} catch let error as DataFrameError {
    print("Caught DataFrameError: \(error.localizedDescription)")
}
```

```text
Caught DataFrameError: Column 'non_existent_column' does not exist.
```

---

> **Next Step**: Proceed to **[Guide 02: Data I/O & Inspection →](02-Data-IO-and-Inspection.md)** to learn how to read and write CSV and JSON datasets, inspect schemas with `infoSummary()`, and customize console and Markdown table formatting.
