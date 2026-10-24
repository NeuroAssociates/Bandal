# Bandal API Reference

[한국어](API.ko.md)

Bandal is a Swift package for working with tabular data, modeled after Python's [pandas](https://pandas.pydata.org/) and designed according to the [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/). All data frames and result structs are immutable value types, conform to `Sendable`, `Equatable`, `Hashable`, `Codable`, `CustomStringConvertible`, `CustomDebugStringConvertible`, and `TextOutputStreamable`, and render pandas-like formatted text when passed to `print(...)`.

For numeric workloads, Bandal uses [MLX Swift](https://github.com/ml-explore/mlx-swift) as its vectorized numeric engine (`float64` on the MLX CPU stream by default, or `float32` on the GPU stream when requested via `ExecutionPolicy.gpu`) and falls back transparently to native Swift on small inputs or when MLX's Metal runtime is unavailable.

---

## Table of Contents

1. [Data Model (`DataFrame`, `Series`, `DataRow`, `DataValue`, `DataType`)](#1-data-model)
2. [Creating a DataFrame](#2-creating-a-dataframe)
3. [CSV & JSON Reading and Writing](#3-csv--json-reading-and-writing)
4. [Inspection, Rendering & Markdown Export](#4-inspection-rendering--markdown-export)
5. [Row, Column & Cell Selection](#5-row-column--cell-selection)
6. [Filtering & Querying (`filterRows`, `RowPredicate`, `query`)](#6-filtering--querying)
7. [Adding, Computing, Renaming & Casting Columns](#7-adding-computing-renaming--casting-columns)
8. [Date & String Column Operations](#8-date--string-column-operations)
9. [Missing Values](#9-missing-values)
10. [Duplicates, Sorting, Sampling & Row Concatenation](#10-duplicates-sorting-sampling--row-concatenation)
11. [Series & Column-wise Numeric Operations](#11-series--column-wise-numeric-operations)
12. [Descriptive Statistics & Structured Result Types](#12-descriptive-statistics--structured-result-types)
13. [Grouping (`groupBy`, `aggregate`, `transform`)](#13-grouping)
14. [Joining (`leftJoin`, `join`)](#14-joining)
15. [Reshaping (`melt`, `pivotTable`, `exploded`, `transposed`)](#15-reshaping)
16. [Window Functions (`rolling`, Time-based `rolling`, `expanding`)](#16-window-functions)
17. [MLX Runtime, `ExecutionPolicy` & `MLXVector` Interop](#17-mlx-runtime-executionpolicy--mlxvector-interop)
18. [Errors (`DataFrameError`)](#18-errors)

---

## 1. Data Model

### `DataFrame`

```swift
public struct DataFrame: Equatable, Hashable, Sendable, Codable,
    CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable
```

An ordered collection of named columns where every column has the same number of rows. Operations never mutate a `DataFrame` in place; they return a new `DataFrame` or throw `DataFrameError`.

### `Series`

```swift
public struct Series: Equatable, Hashable, Sendable, Codable,
    RandomAccessCollection, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable
```

A named one-dimensional column extracted via `try df.series("col")` or `df["col"]`. `Series` conforms to `RandomAccessCollection` (`Element == DataValue`, `Index == Int`) and supports vectorized arithmetic, cumulative reductions, ranking, quantile calculation, missing-value handling, and frequency counting.

### `DataRow`

```swift
public struct DataRow: Equatable, Hashable, Sendable, Codable,
    CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable
```

A single row view returned by `df.row(at:)`, `df[row: i]`, or `df.rows`.

| Member | Type | Description |
| --- | --- | --- |
| `index` | `Int` | Zero-based row index in the source frame. |
| `columnOrder` | `[String]` | Ordered column names. |
| `values` | `[String: DataValue]` | Column-to-cell dictionary. |
| `subscript(_ column: String)` | `DataValue?` | Cell value for `column`, or `nil` if the column does not exist. |
| `description` | `String` | Aligned two-column summary of column names and values, plus `Row: <index>`. |

```swift
let df = try DataFrame(columns: [
    (name: "city", values: [.string("Seoul"), .string("Busan")]),
    (name: "temp", values: [.number(21.5), .number(24.0)]),
])
if let firstRow = df[row: 0] {
    print(firstRow["city"]!) // Seoul
    print(firstRow)
}
```

### `DataValue`

```swift
public enum DataValue: Equatable, Hashable, Sendable, Codable,
    CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable,
    ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByStringLiteral, ExpressibleByBooleanLiteral, ExpressibleByNilLiteral
```

Represents a single cell value:

- `.number(Double)` — finite `Double` (`NaN` and `±infinity` are rejected with `.nonFiniteValue`).
- `.string(String)` — UTF-8 text.
- `.boolean(Bool)` — `true` or `false`.
- `.date(Date)` — `Foundation.Date` formatted in UTC ISO 8601 (`yyyy-MM-dd'T'HH:mm:ss'Z'`).
- `.missing` — missing value (printed as `NaN`, analogous to `NaN`/`None` in pandas).

Because `DataValue` conforms to Swift's literal protocols, you can write literals directly inside `[DataValue]` arrays:

```swift
let values: [DataValue] = [10, 20.5, "text", true, nil]
// Equivalent to [.number(10), .number(20.5), .string("text"), .boolean(true), .missing]
```

Typed convenience accessors on `DataValue`:

- `numberValue: Double?`
- `stringValue: String?`
- `boolValue: Bool?`
- `dateValue: Date?`
- `isMissing: Bool`

### `DataType`

```swift
public enum DataType: String, Equatable, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case number
    case string
    case boolean
    case date
    case mixed
    case unknown
}
```

Returned by `try df.dataType(of:)` and `df.dtypes` / `df.columnTypes()`:

- `.number`, `.string`, `.boolean`, `.date` when all non-missing values in the column share that type.
- `.mixed` when a column contains values of more than one concrete type.
- `.unknown` when every value in the column is `.missing` or the frame has 0 rows.

---

## 2. Creating a DataFrame

### From a dictionary of numeric columns

Column names are sorted lexicographically so the column order is deterministic across runs:

```swift
import Bandal

let weather = try DataFrame(columns: [
    "temperature": [18.0, 21.5, 19.0],
    "humidity": [0.45, 0.50, 0.48],
])
print(weather)
```

### From ordered numeric columns

Preserves the exact column order you provide:

```swift
let ordered = try DataFrame(columns: [
    (name: "z", values: [1.0, 2.0]),
    (name: "a", values: [3.0, 4.0]),
])
print(ordered.columnNames) // ["z", "a"]
```

### From ordered mixed-type `DataValue` columns

```swift
let mixed = try DataFrame(columns: [
    (name: "city", values: [.string("Seoul"), .string("Busan"), .missing]),
    (name: "temp", values: [.number(18.0), .missing, .number(22.5)]),
    (name: "active", values: [.boolean(true), .boolean(false), .missing]),
])
print(mixed)
```

### From MLX vectors (`MLXVector`)

See [Section 17](#17-mlx-runtime-executionpolicy--mlxvector-interop) for `try DataFrame(mlxColumns:)`.

---

## 3. CSV & JSON Reading and Writing

### Reading CSV (`init(contentsOfCSV:...)` and `CSVReadOptions`)

```swift
public struct CSVReadOptions: Equatable, Hashable, Sendable, Codable {
    public var delimiter: Character
    public var maximumRows: Int?
    public var selectedColumns: [String]?
    public var hasHeader: Bool
    public var columnNames: [String]?
    public var missingTokens: Set<String>
    public var infersNumbers: Bool
    public var columnTypes: [String: DataType]
    public var dateFormat: String?
}
```

Defaults:

- `delimiter = ","`
- `hasHeader = true`
- `missingTokens = ["", "NA", "N/A", "NaN", "null", "NULL", "nil"]`
- `infersNumbers = true` (strips `,` and leading `$` inside numeric tokens)

```swift
let df = try DataFrame(
    contentsOfCSV: url,
    options: CSVReadOptions(
        delimiter: ",",
        maximumRows: 1_000,
        selectedColumns: ["Country", "Sales", "Created"],
        columnTypes: ["Created": .date],
        dateFormat: "yyyy-MM-dd"
    )
)
```

Or use the shorthand initializer when you only need a row limit:

```swift
let preview = try DataFrame(contentsOfCSV: url, maximumRows: 10)
```

### Writing CSV (`writeCSV(to:delimiter:includeHeader:)`)

Streams rows directly to disk in 64 KB chunks with RFC 4180 quoting:

```swift
try df.writeCSV(to: outputURL, delimiter: ",", includeHeader: true)
```

### Reading and Writing JSON (`init(jsonData:)`, `init(contentsOfJSON:)`, `jsonData`, `writeJSON`)

Bandal reads and writes JSON arrays of row objects (`[{"col": value, ...}, ...]`). Numbers, strings, Booleans, `null` (`.missing`), and ISO 8601 (or custom `dateFormat`) date strings are supported.

```swift
let json = """
[
  {"city": "Seoul", "temp": 21.5, "capital": true},
  {"city": "Busan", "temp": null, "capital": false}
]
""".data(using: .utf8)!

let fromJSON = try DataFrame(jsonData: json)
print(fromJSON)

// Serialize back to Data or write to a file URL
let encodedData = try fromJSON.jsonData(prettyPrinted: true)
try fromJSON.writeJSON(to: jsonFileURL, prettyPrinted: true)
let reloaded = try DataFrame(contentsOfJSON: jsonFileURL)
```

---

## 4. Inspection, Rendering & Markdown Export

### Properties

| Property / Method | Return Type | Description |
| --- | --- | --- |
| `rowCount` | `Int` | Number of rows. |
| `columnNames` | `[String]` | Ordered column names. |
| `shape` | `[Int]` | `[rowCount, columnNames.count]`. |
| `isEmpty` | `Bool` | `true` when `rowCount == 0`. |
| `dtypes` / `columnTypes()` | `ColumnTypes` | Ordered inferred `DataType` per column. |
| `dataType(of:)` | `DataType` | Inferred `DataType` of one column (throws `.columnNotFound`). |

### Table Rendering (`description`, `render`, `RenderOptions`)

Passing a `DataFrame` to `print(df)` invokes `df.description`, which renders up to 20 rows and 10 columns with `…` truncation markers and a `[R rows x C columns]` footer.

Use `RenderOptions` for fine-grained control over row count, column count, and maximum cell width:

```swift
public struct RenderOptions: Equatable, Hashable, Sendable, Codable {
    public var maxRows: Int
    public var maxColumns: Int
    public var maxColumnWidth: Int?

    public init(maxRows: Int = 20, maxColumns: Int = 10, maxColumnWidth: Int? = nil)
}
```

```swift
print(df.render(maxRows: 6, maxColumns: 4))
print(df.render(options: RenderOptions(maxRows: 10, maxColumns: 8, maxColumnWidth: 24)))
```

### Markdown Export (`toMarkdown`)

Renders a GitHub-Flavored Markdown table suitable for README files, notebooks, or reports:

```swift
let md = df.toMarkdown(maxRows: 10, maxColumns: 6)
print(md)
// |  | city | temp |
// | --- | --- | --- |
// | 0 | Seoul | 21.5 |
// | 1 | Busan | NaN |
```

### Schema & Memory Overview (`info()` and `infoSummary()`)

```swift
print(df.info())
let info: DataFrameInfo = df.infoSummary()
print(info.rowCount, info.estimatedMemoryBytes)
print(info) // Same human-readable summary via CustomStringConvertible
```

---

## 5. Row, Column & Cell Selection

### Selecting and Dropping Columns

```swift
// Variadic or array column selection
let sub1 = try df.select("city", "temp")
let sub2 = try df.select(columns: ["city", "temp"])

// Select all columns of a given DataType
let numericOnly = try df.select(columnsOfType: .number)

// Drop columns by name
let withoutTemp = try df.drop(columns: ["temp"])
```

### Extracting Column Values and `Series`

```swift
// [Double] (throws .nonNumericColumn if any value is missing or non-numeric)
let numbers: [Double] = try df.column("temp")

// [DataValue] for any column type
let rawValues: [DataValue] = try df.values(in: "city")

// Series access (throwing or optional subscript)
let s1: Series = try df.series("temp")
let s2: Series? = df["temp"]

// Export entire DataFrame as [String: [DataValue]]
let dict: [String: [DataValue]] = df.toDictionary()
```

### Row & Cell Access (`row(at:)`, `rows`, Subscripts, `head`, `tail`, `largest`, `smallest`)

```swift
// Single row (DataRow?)
let firstRow = df.row(at: 0)
let sameRow = df[row: 0]

// All rows ([DataRow])
for row in df.rows {
    print(row.index, row["city"] ?? .missing)
}

// Contiguous row slice (DataFrame?)
if let slice = df[rows: 0..<2] {
    print(slice)
}

// Single cell subscript (DataValue?)
let cell = df["city", 0] // .string("Seoul")

// First / last N rows
let top3 = try df.head(3)
let bottom3 = try df.tail(3)

// Top / bottom N rows ordered by a column
let highest = try df.largest(2, by: "temp")
let lowest = try df.smallest(2, by: "temp")
```

---

## 6. Filtering & Querying

### Closure Filtering (`filterRows`)

```swift
let warm = try df.filterRows { row in
    guard case .number(let t) = row["temp"] else { return false }
    return t >= 20.0
}
```

### Composable Type-Safe Predicates (`RowPredicate` and `query(_ predicate:)`)

`RowPredicate` provides readable static builders and logical operators (`&&`, `||`, `!`, `.and(_:)`, `.or(_:)`, `.negated`):

| Builder | Description |
| --- | --- |
| `.column(name, equals: value)` | `row[name] == value` |
| `.column(name, notEquals: value)` | `row[name] != value` |
| `.column(name, greaterThan: x)` | Numeric `>` |
| `.column(name, greaterThanOrEqual: x)` | Numeric `>=` |
| `.column(name, lessThan: x)` | Numeric `<` |
| `.column(name, lessThanOrEqual: x)` | Numeric `<=` |
| `.column(name, between: a, and: b, inclusive: true)` | Numeric range check |
| `.column(name, in: Set<DataValue>)` | Set membership (`isin`) |
| `.column(name, contains: substring, caseSensitive: true)` | Substring match |
| `.isMissing(name)` / `.notMissing(name)` | Missingness check |

```swift
let predicate: RowPredicate =
    .column("temp", greaterThanOrEqual: 20)
    && .notMissing("city")
    && !.column("city", equals: "Jeju")

let filtered = try df.query(predicate)
```

### String Expression Queries (`query(_ expression:)`)

Supports `==`, `!=`, `>=`, `<=`, `>`, `<` combined with `&&` or `||`:

```swift
let result = try df.query("temp >= 20 && active == true")
let cities = try df.query("city == 'Seoul' || city == 'Busan'")
```

---

## 7. Adding, Computing, Renaming & Casting Columns

### Adding or Replacing a Column (`withColumn`)

```swift
// From an explicit [DataValue] array
let withCountry = try df.withColumn(
    name: "country",
    values: [.string("KR"), .string("KR"), .string("KR")]
)

// Computed row-by-row from a DataRow closure (like pandas df.assign)
let withFahrenheit = try df.withColumn(name: "temp_f") { row in
    guard let c = row["temp"]?.numberValue else { return .missing }
    return .number(c * 9 / 5 + 32)
}
```

### Mapping One Column (`mapped`)

```swift
let doubled = try df.mapped("temp") { value in
    guard case .number(let x) = value else { return value }
    return .number(x * 2)
}
```

### Renaming Columns (`renamedColumns`)

```swift
let renamed = try df.renamedColumns(["temp": "temperature_c"])
```

### Explicit Type Casting (`castColumn`)

Converts a column to `.number`, `.string`, `.boolean`, or `.date`. Throws `.invalidConversion(column:row:type:)` with the exact row index if a cell cannot be converted.

```swift
let casted = try rawFrame
    .castColumn("amount", to: .number)
    .castColumn("enabled", to: .boolean)
    .castColumn("created", to: .date, dateFormat: "yyyy/MM/dd")
```

---

## 8. Date & String Column Operations

### Extracting UTC Date Components (`DateComponent` & `dateComponent(_:from:as:)`)

`DateComponent` supports `.year`, `.month`, `.day`, `.weekday`, `.hour`, `.minute`, and `.second`:

```swift
let withYearAndMonth = try df
    .dateComponent(.year, from: "created", as: "year")
    .dateComponent(.month, from: "created") // defaults to "created_month"
```

### Vectorized String Column Operations

 analogous to pandas' `df["col"].str.*` accessor:

| Method | Description |
| --- | --- |
| `lowercased(_:as:)` | Converts strings to lowercase. |
| `uppercased(_:as:)` | Converts strings to uppercase. |
| `trimmed(_:as:)` | Strips leading/trailing whitespace and newlines. |
| `stringLength(_:as:)` | Character length as a numeric column (defaults to `"<col>_length"`). |
| `replacingSubstring(in:target:with:as:)` | Replaces occurrences of `target` with `replacement`. |
| `containsSubstring(_:_:caseSensitive:as:)` | Boolean column indicating whether each cell contains the substring (defaults to `"<col>_contains"`). |

```swift
let cleaned = try df
    .trimmed("city")
    .uppercased("city", as: "city_upper")
    .stringLength("city", as: "name_len")
    .replacingSubstring(in: "city", target: "SEOUL", with: "Seoul")
    .containsSubstring("city", "ul", caseSensitive: false, as: "has_ul")
```

---

## 9. Missing Values

| Method | Return Type | Description |
| --- | --- | --- |
| `isMissing(in:)` | `[Bool]` | Row-aligned Boolean array (`true` where the column is `.missing`). |
| `missingMask()` | `DataFrame` | Boolean `DataFrame` of the same shape (`true` where cells are `.missing`). |
| `missingCounts()` | `[String: Int]` | Dictionary of missing counts per column. |
| `missingSummary()` | `MissingCounts` | Ordered struct with `counts`, `totalMissing`, subscript access, and printable `description`. |
| `fillMissing(in:with:)` | `DataFrame` | Replaces `.missing` with a constant `DataValue` in selected (or all) columns. |
| `forwardFilled()` | `DataFrame` | Propagates the last valid observation forward (`ffill`). |
| `backwardFilled()` | `DataFrame` | Propagates the next valid observation backward (`bfill`). |
| `dropMissing(in:)` | `DataFrame` | Drops rows containing `.missing` in selected (or all) columns (`dropna`). |
| `replacing(_:with:in:)` | `DataFrame` | Replaces occurrences of any `DataValue` with another across selected (or all) columns. |

```swift
print(df.missingSummary())
// city    1
// temp    1
// active  1
// Total missing: 3 (rows: 3)

let filled = try df.fillMissing(in: ["temp"], with: .number(20.0))
let ffilled = try df.forwardFilled()
let bfilled = try df.backwardFilled()
let complete = try df.dropMissing()
```

---

## 10. Duplicates, Sorting, Sampling & Row Concatenation

### Duplicate Detection & Removal (`duplicatedRows`, `dropDuplicates`, `DuplicateKeep`)

`DuplicateKeep` cases: `.first` (default), `.last`, `.none`.

```swift
let mask: [Bool] = try df.duplicatedRows(in: ["city"], keep: .first)
let deduped = try df.dropDuplicates(in: ["city"], keep: .last)
```

### Multi-Column Sorting (`sorted(by:)`, `SortDescriptor`, `SortOrder`)

Missing values are always placed last, regardless of `.ascending` or `.descending`:

```swift
let sorted = try df.sorted(by: [
    SortDescriptor("city", order: .ascending),
    SortDescriptor("temp", order: .descending),
])
```

### Reproducible Random Sampling (`sample(_:seed:)`)

Samples up to `count` rows without replacement, preserving their original relative row order. Pass `seed` for deterministic results:

```swift
let sample = try df.sample(10, seed: 42)
```

### Vertical Concatenation (`DataFrame.concatenatingRows`)

```swift
let combined = try DataFrame.concatenatingRows([part1, part2])
```

---

## 11. Series & Column-wise Numeric Operations

### Working with `Series`

```swift
let s = try df.series("temp")
print(s)

// Reductions & quantiles
let sum: DataValue = s.reduced(using: .sum)
let q90: Double? = s.quantile(0.9)
let distinctCount: Int = s.uniqueCount()
let distinctValues: [DataValue] = s.unique()
let frequencies: DataFrame = try s.valueCounts()

// Vectorized arithmetic (+, -, *, / with another Series or Double scalar)
let centered = try s - 20.0
let scaled = try s * 1.8 + 32.0
let combined = try s + s

// Element-wise transformations
let absSeries = try s.abs()
let clipped = try s.clipped(lower: 0, upper: 30)
let rounded = try s.rounded(decimals: 1)
let shifted = s.shifted(by: 1)
let diff = try s.difference(periods: 1)
let pct = try s.percentChange(periods: 1)
let cumSum = try s.cumulative(using: .sum)
let ranked = try s.ranked(ascending: false)

// Missing values & filtering on Series
let noMissing = s.droppedMissing()
let filled = try s.filledMissing(with: .number(0))
let ffill = s.forwardFilled()
let bfill = s.backwardFilled()
let replaced = try s.replacing(.missing, with: .number(0))
let inSet = s.isin([.number(18.0), .number(21.5)])
```

### Column-wise Operations on `DataFrame`

| Method | Description |
| --- | --- |
| `combining(_:_:using:as:)` | Element-wise `.add`, `.subtract`, `.multiply`, or `.divide` between two numeric columns into `outputColumn` (MLX-accelerated on large columns). |
| `scaling(_:by:using:as:)` | Element-wise scalar `.add`, `.subtract`, `.multiply`, or `.divide` on a numeric column (MLX-accelerated on large columns). |
| `cumulative(_:using:)` | Running `.sum`, `.product`, `.minimum`, or `.maximum` (`CumulativeFunction`). |
| `shifted(_:by:)` | Shifts a column by `periods` rows, filling vacated positions with `.missing`. |
| `difference(_:periods:)` | First/k-th difference (`x[i] - x[i - periods]`). |
| `percentChange(_:periods:)` | Fractional change (`(x[i] - x[i - periods]) / x[i - periods]`). |
| `clipped(_:lower:upper:)` | Clamps numeric values to `lower...upper`. |
| `rounded(_:decimals:)` | Half-away-from-zero rounding to `decimals` places. |
| `abs(_:)` | Absolute value of a numeric column. |
| `ranked(_:ascending:as:)` | 1-based average-tie ranks. |
| `reduced(using:)` | Single-row `DataFrame` reducing every numeric column with an `AggregateFunction`. |
| `quantile(_:)` | Single-row `DataFrame` containing the `p` quantile (`0...1`) for every numeric column. |
| `covariance()` | Pairwise sample covariance matrix (`DataFrame`) across numeric columns (uses `MLX.matmul` on large complete frames). |
| `correlation()` | Pairwise Pearson correlation matrix (`DataFrame`) across numeric columns (uses `MLX.matmul` on large complete frames). |
| `valueCounts(of:)` | Two-column `DataFrame` (`value`, `count`) sorted by descending frequency. |
| `uniqueCounts()` | `[String: Int]` mapping each column name to its distinct non-missing count. |
| `uniqueSummary()` | `UniqueCounts` struct with `counts`, subscript access, and printable `description`. |

```swift
let enriched = try salesFrame
    .combining("revenue", "cost", using: .subtract, as: "profit")
    .scaling("profit", by: 100.0, using: .multiply, as: "profit_cents")
    .cumulative("revenue", using: .sum)
    .ranked("profit", ascending: false, as: "profit_rank")

let corrMatrix = try enriched.correlation()
print(corrMatrix)
```

---

## 12. Descriptive Statistics & Structured Result Types

Every statistical and inspection operation returns a Swift struct that both exposes typed properties for programmatic access and renders a pandas-style summary when printed.

| Method | Return Type | Key Properties & Methods |
| --- | --- | --- |
| `statistics()` | `DataFrameStatistics` | `columns: [String: ColumnStatistics]`, `rowCount`, `columnCount`, `subscript(_ column: String)`, `description` |
| `describeStructured(percentiles:include:)` | `DataFrameDescription` | `columnOrder`, `numeric: [String: ColumnStatistics]`, `nonNumeric: [String: NonNumericSummary]`, `percentileLabels`, `describe()`, `description` |
| `describe(percentiles:include:)` | `String` | Equivalent to `describeStructured(percentiles:include:).describe()` |
| `summaryTable()` | `DataFrameSummary` | `columnOrder`, `summaries: [String: ColumnSummary]`, `subscript`, `describe()`, `description` |
| `summary()` | `[String: ColumnSummary]` | Dictionary of `ColumnSummary` (`count`, `mean`, `standardDeviation`, `median`, `min`, `max`) |
| `infoSummary()` | `DataFrameInfo` | `rowCount`, `columns: [DataFrameInfo.Column]`, `estimatedMemoryBytes`, `describe()`, `description` |
| `missingSummary()` | `MissingCounts` | `columnOrder`, `counts: [String: Int]`, `totalMissing`, `rowCount`, `describe()`, `description` |
| `uniqueSummary()` | `UniqueCounts` | `columnOrder`, `counts: [String: Int]`, `rowCount`, `describe()`, `description` |
| `columnTypes()` / `dtypes` | `ColumnTypes` | `columnOrder`, `types: [String: DataType]`, `describe()`, `description` |

### Example: Structured Statistics and Custom Percentiles

```swift
let stats: DataFrameStatistics = df.statistics()
if let tempStats: ColumnStatistics = stats["temp"] {
    print("Mean:", tempStats.mean ?? .nan)
    print("IQR:", (tempStats.q3 ?? 0) - (tempStats.q1 ?? 0))
}

// Custom percentiles and type filtering (like pandas df.describe(percentiles=[0.1, 0.5, 0.9], include=['number']))
let desc: DataFrameDescription = df.describeStructured(
    percentiles: [0.1, 0.5, 0.9],
    include: [.number]
)
print(desc)
print(df.summaryTable())
print(df.dtypes)
print(df.uniqueSummary())
```

---

## 13. Grouping

### `groupBy`, `GroupedDataFrame`, `Aggregation`, `AggregateFunction`

`AggregateFunction` cases:

- `.sum`
- `.mean`
- `.min`
- `.max`
- `.count`
- `.median`
- `.variance` (sample variance, `n - 1`)
- `.standardDeviation` (sample standard deviation, `n - 1`)
- `.first`
- `.last`

Rows with `.missing` in any grouping key column are excluded from grouped output (matching pandas' default `dropna=True`). Groups are ordered deterministically.

```swift
// Single aggregation
let byRegion = try sales
    .groupBy("Region")
    .aggregate("Sales", using: .sum)

// Multiple keys and multiple aggregations
let detailed = try sales
    .groupBy("Region", "Product")
    .aggregate([
        Aggregation("Sales", using: .sum),
        Aggregation("Sales", using: .mean),
        Aggregation("Sales", using: .standardDeviation),
        Aggregation("Sales", using: .count),
    ])
print(detailed)

// Broadcast group results back to the original row order (like pandas groupby.transform)
let withGroupMean = try sales
    .groupBy("Region")
    .transform("Sales", using: .mean)
```

---

## 14. Joining

Supports `.left`, `.inner`, `.right`, and `.full` joins (`JoinKind`) on single keys or multiple differently-named keys with configurable column suffixes for overlapping non-key columns:

```swift
// Single-key join
let joined = try orders.join(customers, on: "customer_id", kind: .left)

// Multi-key join with distinct left/right key names and disambiguating suffixes
let multiJoined = try leftFrame.join(
    rightFrame,
    leftOn: ["region_id", "year"],
    rightOn: ["id", "fiscal_year"],
    kind: .inner,
    suffixes: ("_left", "_right")
)
```

---

## 15. Reshaping

### `melt` (Wide to Long)

```swift
let longFrame = try wideFrame.melt(
    idColumns: ["region"],
    variableName: "metric",
    valueName: "amount"
)
```

### `pivotTable` (Long to Wide with Aggregation)

```swift
let pivoted = try longFrame.pivotTable(
    index: ["region"],
    columns: "metric",
    values: "amount",
    using: .sum
)
```

### `exploded` (Split Delimited Strings into Rows)

Splits string values in `column` by `separator` (default `","`), duplicating all other columns for each resulting token (similar to `df.assign(...).explode()` in pandas):

```swift
let tags = try DataFrame(columns: [
    (name: "id", values: [.number(1), .number(2)]),
    (name: "tags", values: [.string("swift,mlx,data"), .string("pandas")]),
])
let exploded = try tags.exploded("tags", separator: ",")
print(exploded)
// 4 rows: (1, "swift"), (1, "mlx"), (1, "data"), (2, "pandas")
```

### `transposed` (Swap Rows and Columns)

Turns source column names into the first column (`indexColumnName`, default `"column"`) and creates one column per source row (`row_0`, `row_1`, ...):

```swift
let t = try df.transposed(indexColumnName: "feature", rowPrefix: "sample_")
print(t)
```

---

## 16. Window Functions

### Row-Count Rolling Window (`rolling(_:window:using:)`)

Applies any `AggregateFunction` over a trailing window of `window` rows. Positions before the first full window (`0 ..< window - 1`) produce `.missing`. Large `.sum`, `.mean`, and `.count` windows are vectorized with prefix sums on MLX.

```swift
let rolled = try df.rolling("temp", window: 3, using: .mean)
```

### Time-Based Rolling Window (`rolling(_:on:interval:using:)`)

Computes a trailing time-window aggregation for each row `i`, including all earlier rows `j <= i` whose timestamp in `dateColumn` falls within `(t_i - interval) ... t_i`:

```swift
let timeRolled = try events.rolling(
    "latency_ms",
    on: "timestamp",
    interval: 60.0, // trailing 60 seconds
    using: .mean
)
```

### Expanding Window (`expanding(_:using:)`)

Applies an `AggregateFunction` over `0...row` for every row:

```swift
let cumulativeMean = try df.expanding("temp", using: .mean)
```

---

## 17. MLX Runtime, `ExecutionPolicy` & `MLXVector` Interop

### How Bandal Uses MLX

Bandal's `NumericEngine` uses MLX Swift across:

- Descriptive moments (`summary`, `statistics`, `describe`, `describeStructured`)
- Single-column and whole-frame reductions (`sum`, `mean`, `min`, `max`, `count`, `variance`, `standardDeviation`)
- Grouped reductions (`groupBy.aggregate`, `groupBy.transform`, `pivotTable`)
- Rolling and expanding windows (`sum`, `mean`, `count`)
- Element-wise unary and binary operations (`combining`, `scaling`, `Series` `+`, `-`, `*`, `/`, `abs`, `clipped`, `rounded`, `difference`, `percentChange`, `cumulative`)
- Pairwise `covariance()` and `correlation()` matrices via `MLX.matmul`

### `ExecutionPolicy` & `BandalRuntime`

```swift
public enum ExecutionPolicy: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case automatic // MLX CPU stream (float64) above threshold; native Swift below
    case native    // Pure native Swift (Double)
    case cpu       // Force MLX CPU stream (float64) for all non-empty inputs
    case gpu       // MLX GPU stream (float32) when values are exact in Float32, else CPU float64
}
```

Inspect or scope the execution policy at runtime:

```swift
print(BandalRuntime.isMLXAvailable)
print(BandalRuntime.executionBackend)
print(BandalRuntime.executionPolicy)

// Run a block on the GPU stream or force CPU / native execution
let corr = try BandalRuntime.withExecutionPolicy(.gpu) {
    try largeFrame.correlation()
}
```

Set `BANDAL_DISABLE_MLX=1` in the process environment to force the native Swift backend globally.

### `MLXVector` Interop (`toMLXVector` & `init(mlxColumns:)`)

Bridge numeric columns directly to and from MLX `MLXArray` tensors alongside their missing-value masks:

```swift
let vector: MLXVector = try df.toMLXVector("temp")
print(vector)
// Inspect non-missing Double values or construct an MLXArray:
let array = vector.makeMLXArray() // MLXArray? (float64 on CPU or float32 on GPU)

// Reconstruct a DataFrame from MLXVectors
let roundTripped = try DataFrame(mlxColumns: [
    (name: "temp", vector: vector)
])
```

---

## 18. Errors

All throwing APIs throw `DataFrameError`, which conforms to `Error`, `Equatable`, `Hashable`, `Sendable`, `LocalizedError`, and `CustomStringConvertible`:

| Case | When Thrown |
| --- | --- |
| `.emptyColumns` | Attempting to create or select a frame with zero columns. |
| `.emptyColumnName` | A column name is `""`. |
| `.duplicateColumnName(String)` | Two columns share the same name. |
| `.unequalColumnLengths(column:expected:actual:)` | A column's length does not match the frame's row count. |
| `.columnNotFound(String)` | Referencing a column name that does not exist. |
| `.nonNumericColumn(String)` | Calling a numeric-only operation on a column with strings, Booleans, dates, or required non-missing numbers. |
| `.nonFiniteValue(column:row:)` | A `Double` is `NaN` or `±infinity`. |
| `.columnNameConflict(String)` | A join or transform would overwrite an existing column name. |
| `.invalidCSV(String)` | Malformed CSV/JSON input or invalid query/cast syntax. |
| `.invalidWindowSize(Int)` | Rolling window size `<= 0`. |
| `.incompatibleSchema(expected:actual:)` | `concatenatingRows` called with mismatched column names or order. |
| `.invalidConversion(column:row:type:)` | `castColumn`, `dateComponent`, or string operations encountered an incompatible value at `row`. |
