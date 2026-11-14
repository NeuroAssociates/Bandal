# Guide 02: Data I/O (CSV / JSON) & Schema Inspection

Load and export streaming CSV and JSON files, inspect schemas and memory footprints with structured types, and format console or Markdown tables.

## Overview

Every data analysis workflow starts with loading external data and inspecting its shape, column types, missing values, and sample rows. In `pandas`, this means `pd.read_csv`, `pd.read_json`, `df.info()`, `df.dtypes`, and `df.to_markdown()`.

This guide covers Bandal's streaming CSV/JSON I/O engine and structured schema inspection tools:

- **Streaming CSV Import & Export** (``DataFrame/init(contentsOfCSV:options:)``, ``CSVReadOptions``, ``DataFrame/writeCSV(to:delimiter:includeHeader:)``)
- **Automatic Number, Currency, Boolean, and Date Parsing**
- **JSON Import & Export** (``DataFrame/init(contentsOfJSON:dateFormat:)``, ``DataFrame/init(jsonData:dateFormat:)``, ``DataFrame/writeJSON(to:prettyPrinted:)``, ``DataFrame/jsonData(prettyPrinted:)``)
- **Schema & Memory Inspection** (``DataFrame/shape``, ``DataFrame/dtypes``, ``DataFrame/columnTypes()``, ``DataFrame/info()``, ``DataFrame/infoSummary()``, ``DataFrameInfo``)
- **Tabular & Markdown Rendering** (``DataFrame/render(options:)``, ``RenderOptions``, ``DataFrame/toMarkdown(maxRows:maxColumns:)``)

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
| Inspect column types | `df.dtypes` | `df.dtypes` (returns ``ColumnTypes``) |
| Inspect schema & memory | `df.info()` | `df.info()` (`String`) or `df.infoSummary()` (``DataFrameInfo``) |
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

Real-world CSV files often contain custom missing-value markers (`"NA"`, `"NULL"`, `"-"`), leading-zero ID codes (`"00124"`) that must remain strings, formatted currencies (`"$1,250.50"`), or custom date strings. ``CSVReadOptions`` handles all of these in a single pass:

```swift
let options = CSVReadOptions(
    delimiter: ",",
    maximumRows: nil,
    hasHeader: true,
    selectedColumns: ["order_id", "order_date", "region", "amount", "is_vip"],
    missingTokens: ["", "N/A", "NULL", "-"],
    infersNumbers: true,
    columnTypes: [
        "order_id": .string,      // Keep leading zeros ("00101")
        "order_date": .date,      // Parse as Date
        "is_vip": .boolean        // Parse "true"/"false" as Bool
    ],
    dateFormat: "yyyy-MM-dd"
)
```

### 2.3 Writing CSV Files

``DataFrame/writeCSV(to:delimiter:includeHeader:)`` streams rows directly to disk in UTF-8. Missing values are written as empty fields (`""`), dates are formatted as UTC ISO-8601 strings, and fields containing delimiters, quotes, or newlines are automatically RFC 4180 quoted.

```swift
let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("cleaned_orders.csv")
try orders.writeCSV(to: outURL, delimiter: ",", includeHeader: true)
```

---

## 3. Reading & Writing JSON Data

Bandal supports the two most common JSON table orientations out of the box:
1. **Records orientation** (`[{"colA": 1, "colB": "x"}, {"colA": 2, "colB": "y"}]`)
2. **Column-arrays orientation** (`{"colA": [1, 2], "colB": ["x", "y"]}`)

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

let exportedData = try sensorDF.jsonData(prettyPrinted: true)
```

---

## 4. Schema & Memory Inspection (`dtypes`, `info()`, `infoSummary()`)

One of Bandal's core design principles is that **inspection APIs return structured Swift types** that can be both printed directly as tables (`print(...)` or `.describe()`) and queried programmatically in Swift code.

- `df.info()` returns a formatted string identical in layout to `pandas`' `df.info()`.
- `df.infoSummary()` returns a structured ``DataFrameInfo`` instance containing `rowCount`, `columns: [DataFrameInfo.Column]`, and `estimatedMemoryBytes`.

```swift
let info: DataFrameInfo = orders.infoSummary()
print("Estimated payload memory: \(info.estimatedMemoryBytes) bytes")
for col in info.columns where col.nonMissingCount < info.rowCount {
    let missing = info.rowCount - col.nonMissingCount
    print("Column '\(col.name)' (\(col.type)) has \(missing) missing value(s).")
}
```

---

## 5. Customizing Display & Exporting Markdown Tables

By default, `print(df)` renders up to `10` rows and `8` columns. Use ``DataFrame/render(options:)`` with ``RenderOptions`` to customize row/column limits and column widths, or ``DataFrame/toMarkdown(maxRows:maxColumns:)`` to generate GitHub-Flavored Markdown tables.

> Next Step: Continue to <doc:03-Indexing-Filtering-and-Querying> to learn how to slice columns/rows and filter datasets using vectorized ``BooleanMask``s, ``RowPredicate``s, and string queries.
