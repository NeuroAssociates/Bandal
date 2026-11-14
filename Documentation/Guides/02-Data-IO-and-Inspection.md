# Guide 02: Data I/O (CSV / JSON) & Schema Inspection

> **Language**: **English** | [한국어](02-Data-IO-and-Inspection.ko.md)  
> **Navigation**: [← Guide 01: Core Data Structures](01-Core-Data-Structures.md) | [API Reference](../API.md) | **Next**: [Guide 03: Indexing, Filtering & Querying →](03-Indexing-Filtering-and-Querying.md)

---

## Overview

Every data analysis workflow starts with loading external data and inspecting its shape, column types, missing values, and sample rows. In `pandas`, this means `pd.read_csv`, `pd.read_json`, `df.info()`, `df.dtypes`, and `df.to_markdown()`.

This guide covers Bandal's streaming CSV/JSON I/O engine and structured schema inspection tools:

- **Streaming CSV Import & Export** (`init(contentsOfCSV:options:)`, `CSVReadOptions`, `writeCSV`)
- **Automatic Number, Currency, Boolean, and Date Parsing**
- **JSON Import & Export** (`init(contentsOfJSON:dateFormat:)`, `init(jsonData:dateFormat:)`, `writeJSON`, `jsonData`)
- **Schema & Memory Inspection** (`shape`, `dtypes`, `columnTypes()`, `info()`, `infoSummary()`, `DataFrameInfo`)
- **Tabular & Markdown Rendering** (`render(options:)`, `RenderOptions`, `toMarkdown(maxRows:maxColumns:)`)

---

## 1. Quick Comparison: `pandas` vs. `Bandal`

| Task | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| Read CSV | `pd.read_csv("data.csv")` | `try DataFrame(contentsOfCSV: url)` |
| Read CSV with options | `pd.read_csv("data.csv", sep="\t", nrows=100, usecols=["a","b"], na_values=["N/A"], dtype={"id": str})` | `try DataFrame(contentsOfCSV: url, options: CSVReadOptions(...))` |
| Write CSV | `df.to_csv("out.csv", index=False)` | `try df.writeCSV(to: url, delimiter: ",", includeHeader: true)` |
| Read JSON | `pd.read_json("data.json")` | `try DataFrame(contentsOfJSON: url, dateFormat: nil)` |
| Write JSON | `df.to_json("out.json", orient="records")` | `try df.writeJSON(to: url, prettyPrinted: true)` |
| Inspect shape | `df.shape` | `df.shape` (`[rowCount, columnNames.count]`) |
| Inspect column types | `df.dtypes` | `df.dtypes` (returns `ColumnTypes`) |
| Inspect schema & memory | `df.info()` | `df.info()` (String) or `df.infoSummary()` (`DataFrameInfo`) |
| Render as Markdown | `df.to_markdown()` | `df.toMarkdown(maxRows: nil, maxColumns: nil)` |

---

## 2. Reading & Writing CSV Files

Bandal's CSV reader streams records incrementally in 64 KB chunks using RFC 4180 quoting rules. When you set `maximumRows`, Bandal stops reading the file as soon as the requested number of rows has been parsed—making it fast to preview multi-gigabyte CSV files.

### 2.1 Basic CSV Loading

```swift
import Foundation
import Bandal

let csvURL = URL(fileURLWithPath: "/path/to/sales.csv")

// Load entire CSV (comma-delimited, first row as header)
let df = try DataFrame(contentsOfCSV: csvURL)

// Preview only the first 500 rows of a TSV (tab-separated) file
let preview = try DataFrame(
    contentsOfCSV: URL(fileURLWithPath: "/path/to/logs.tsv"),
    delimiter: "\t",
    maximumRows: 500
)
```

### 2.2 Fine-Grained Control with `CSVReadOptions`

Real-world CSV files often contain custom missing-value markers (`"NA"`, `"NULL"`, `"-"`), leading-zero ID codes (`"00124"`) that must remain strings, formatted currencies (`"$1,250.50"`), or custom date strings. `CSVReadOptions` handles all of these in a single pass:

```swift
public struct CSVReadOptions: Equatable, Sendable, CustomStringConvertible {
    public var delimiter: Character           // Default: ","
    public var maximumRows: Int?              // Default: nil (read all rows)
    public var hasHeader: Bool                // Default: true
    public var columnNames: [String]?         // Default: nil (override or supply header)
    public var selectedColumns: [String]?     // Default: nil (projected columns only)
    public var missingTokens: Set<String>     // Default: [""]
    public var infersNumbers: Bool            // Default: true
    public var columnTypes: [String: DataType]// Default: [:] (explicit type overrides)
    public var dateFormat: String?            // Default: nil (e.g. "yyyy-MM-dd")
}
```

#### Automatic Numeric & Currency Parsing Rules
When `infersNumbers == true` (the default), Bandal automatically parses:
- Standard integers and floats (`42`, `-3.14`, `1.5e4`)
- Quoted thousands-separated numbers (`"1,250,000.75"` → `.number(1250000.75)`)
- Leading dollar signs (`"$1,250.50"` or `"-$45.00"` → `.number(1250.5)`, `.number(-45.0)`)

#### Complete `CSVReadOptions` Example

```swift
import Foundation
import Bandal

// Create a sample messy CSV file in a temporary directory
let rawCSV = """
order_id,order_date,region,amount,is_vip,notes
00101,2026-03-01,Seoul,"$1,250.50",true,Fast shipping
00102,2026-03-02,Busan,N/A,false,NULL
00103,2026-03-03,Seoul,"$890.00",true,-
"""

let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("orders_demo.csv")
try rawCSV.write(to: tempURL, atomically: true, encoding: .utf8)

let options = CSVReadOptions(
    delimiter: ",",
    maximumRows: nil,
    hasHeader: true,
    selectedColumns: ["order_id", "order_date", "region", "amount", "is_vip"], // Drop "notes" during load
    missingTokens: ["", "N/A", "NULL", "-"],
    infersNumbers: true,
    columnTypes: [
        "order_id": .string,      // Keep leading zeros ("00101") instead of parsing as 101.0
        "order_date": .date,      // Parse as Date
        "is_vip": .boolean        // Parse "true"/"false" as Bool
    ],
    dateFormat: "yyyy-MM-dd"      // UTC DateFormatter pattern for .date columns
)

let orders = try DataFrame(contentsOfCSV: tempURL, options: options)
print(orders)
print("\nColumn Types:")
print(orders.dtypes)
```

**Console Output**:
```text
  order_id            order_date  region  amount  is_vip
0    00101  2026-03-01T00:00:00Z   Seoul  1250.5    true
1    00102  2026-03-02T00:00:00Z   Busan     NaN   false
2    00103  2026-03-03T00:00:00Z   Seoul     890    true

[3 rows x 5 columns]

Column Types:
order_id      string
order_date    date
region        string
amount        number
is_vip        boolean
Columns: 5
```

### 2.3 Writing CSV Files

`writeCSV(to:delimiter:includeHeader:)` streams rows directly to disk in UTF-8. Missing values are written as empty fields (`""`), dates are formatted as UTC ISO-8601 strings, and fields containing delimiters, quotes, or newlines are automatically RFC 4180 quoted.

```swift
let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("cleaned_orders.csv")
try orders.writeCSV(to: outURL, delimiter: ",", includeHeader: true)
```

---

## 3. Reading & Writing JSON Data

Bandal supports the two most common JSON table orientations out of the box:
1. **Records orientation** (`[{"colA": 1, "colB": "x"}, {"colA": 2, "colB": "y"}]`)
2. **Column-arrays orientation** (`{"colA": [1, 2], "colB": ["x", "y"]}`)

JSON `null` values automatically become `.missing`. If you pass a `dateFormat` string (such as `"yyyy-MM-dd"`), strings matching that format are automatically parsed into `.date(Date)` cells.

```swift
let jsonText = """
[
  {"date": "2026-04-01", "sensor": "S1", "reading": 41.2},
  {"date": "2026-04-02", "sensor": "S1", "reading": null},
  {"date": "2026-04-03", "sensor": "S2", "reading": 39.8}
]
"""

let sensorDF = try DataFrame(
    jsonData: Data(jsonText.utf8),
    dateFormat: "yyyy-MM-dd"
)
print(sensorDF)

// Export back to pretty-printed JSON (records orientation)
let exportedData = try sensorDF.jsonData(prettyPrinted: true)
print(String(decoding: exportedData, as: UTF8.self))
```

**Console Output**:
```text
                   date  reading  sensor
0  2026-04-01T00:00:00Z     41.2      S1
1  2026-04-02T00:00:00Z      NaN      S1
2  2026-04-03T00:00:00Z     39.8      S2

[3 rows x 3 columns]
[
  {
    "date" : "2026-04-01T00:00:00Z",
    "reading" : 41.2,
    "sensor" : "S1"
  },
  {
    "date" : "2026-04-02T00:00:00Z",
    "reading" : null,
    "sensor" : "S1"
  },
  {
    "date" : "2026-04-03T00:00:00Z",
    "reading" : 39.8,
    "sensor" : "S2"
  }
]
```

> **Note on `Codable`**: `DataFrame`, `Series`, `DataRow`, and `DataValue` also conform to Swift's standard `Codable` protocol, so you can embed them directly inside your own `Codable` application structs.

---

## 4. Schema & Memory Inspection (`dtypes`, `info()`, `infoSummary()`)

One of Bandal's core design principles is that **inspection APIs return structured Swift types** that can be both printed directly as tables (`print(...)` or `.describe()`) and queried programmatically in Swift code.

### 4.1 Checking Dimensions and Column Types (`shape` and `dtypes`)

```swift
print("Rows: \(orders.rowCount), Columns: \(orders.columnNames.count)")
print("Shape: \(orders.shape)") // [3, 5]

// Structured ColumnTypes result
let dtypes: ColumnTypes = orders.dtypes
print(dtypes) // Prints aligned table

// Programmatic lookup via subscript or dictionary
if dtypes["amount"] == .number {
    print("The 'amount' column is ready for numerical analysis.")
}
```

### 4.2 `info()` vs. `infoSummary()` (`DataFrameInfo`)

- `df.info()` returns a formatted string identical in layout to `pandas`' `df.info()`.
- `df.infoSummary()` returns a structured `DataFrameInfo` instance containing `rowCount`, `columns: [DataFrameInfo.Column]`, and `estimatedMemoryBytes`.

```swift
// 1. Pandas-style info() text report
print(orders.info())

print("\n--- Structured Inspection ---")
// 2. Programmatic inspection via infoSummary()
let info: DataFrameInfo = orders.infoSummary()
print("Estimated payload memory: \(info.estimatedMemoryBytes) bytes")
for col in info.columns where col.nonMissingCount < info.rowCount {
    let missing = info.rowCount - col.nonMissingCount
    print("Column '\(col.name)' (\(col.type)) has \(missing) missing value(s).")
}
```

**Console Output**:
```text
DataFrame
RangeIndex: 3 entries, 0 to 2
Data columns (total 5 columns):
 #  Column      Non-Null Count  Dtype   
 0  order_id    3 non-null      object  
 1  order_date  3 non-null      datetime
 2  region      3 non-null      object  
 3  amount      2 non-null      float64 
 4  is_vip      3 non-null      bool    
---
dtypes: bool(1), datetime(1), float64(1), object(2)
memory usage: ~70 bytes (estimated payload)

--- Structured Inspection ---
Estimated payload memory: 70 bytes
Column 'amount' (number) has 1 missing value(s).
```

---

## 5. Customizing Display & Exporting Markdown Tables

### 5.1 Controlling Console Truncation with `RenderOptions`

By default, `print(df)` renders up to `10` rows and `8` columns, inserting `…` ellipses for larger frames. When you want to see more rows, fewer columns, or truncate wide text cells, use `df.render(options:)` or `df.render(maxRows:maxColumns:)`:

```swift
let customView = orders.render(
    options: RenderOptions(
        maxRows: 20,
        maxColumns: 10,
        maxColumnWidth: 12 // Truncate strings longer than 12 characters with "…"
    )
)
print(customView)
```

```text
  order_id    order_date  region  amount  is_vip
0    00101  2026-03-01T…   Seoul  1250.5    true
1    00102  2026-03-02T…   Busan     NaN   false
2    00103  2026-03-03T…   Seoul     890    true

[3 rows x 5 columns]
```

### 5.2 Generating GitHub-Flavored Markdown (`toMarkdown`)

To embed analysis results into reports, documentation, or LLM prompts, call `toMarkdown(maxRows:maxColumns:)`:

```swift
print(orders.toMarkdown())
```

**Console Output**:
```markdown
| order_id | order_date | region | amount | is_vip |
| --- | --- | --- | --- | --- |
| 00101 | 2026-03-01T00:00:00Z | Seoul | 1250.5 | true |
| 00102 | 2026-03-02T00:00:00Z | Busan | NaN | false |
| 00103 | 2026-03-03T00:00:00Z | Seoul | 890 | true |
```

---

> **Next Step**: Continue to **[Guide 03: Indexing, Filtering & Querying →](03-Indexing-Filtering-and-Querying.md)** to learn how to slice columns/rows and filter datasets using vectorized `BooleanMask`s, `RowPredicate`s, and string queries.
