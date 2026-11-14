# Guide 03: Indexing, Slicing, Boolean Masks & Querying

Select and project columns, slice rows by index or extremes, and filter datasets using vectorized 64-bit `BooleanMask`s, composable `RowPredicate`s, or concise query strings.

## Overview

In `pandas`, analysts spend a large portion of their time selecting columns (`df[["a", "b"]]`), slicing rows (`df.iloc[0:5]`), and filtering datasets (`df[df["score"] >= 80]`, `df.query(...)`).

Bandal provides four complementary ways to select and filter data, ranging from zero-copy columnar projection to vectorized 64-bit bitmap masks (``BooleanMask``) and type-safe predicates (``RowPredicate``):

- **Column Selection & Dropping** (``DataFrame/select(_:)``, ``DataFrame/select(columnsOfType:)``, ``DataFrame/drop(columns:)``, ``DataFrame/series(_:)``, `df["col"]`)
- **Positional Row & Cell Slicing** (`df[row:]`, `df[rows:]`, `df["col", row]`, ``DataFrame/head(_:)``, ``DataFrame/tail(_:)``, ``DataFrame/largest(_:by:)``, ``DataFrame/smallest(_:by:)``, ``DataFrame/indexOfMax(in:)``, ``DataFrame/indexOfMin(in:)``)
- **Vectorized Boolean Mask Filtering** (``BooleanMask``, ``Series`` comparison operators, `&&`, `||`, `!`, `df[mask]`, ``DataFrame/filtered(by:)``)
- **Composable Type-Safe Predicates** (``RowPredicate`` + ``DataFrame/query(_:)-(RowPredicate)``)
- **Concise String Query Expressions** (`df.query("score >= 80 && active == true")`)
- **Custom Row-Closure Filtering** (``DataFrame/filterRows(_:)``)

---

## 1. Quick Comparison: `pandas` vs. `Bandal`

| Operation | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| Extract single column as `Series` | `df["price"]` | `try df.series("price")` or `df["price"]` (`Series?`) |
| Extract unboxed `[Double]` | `df["price"].to_numpy()` | `try df.column("price")` |
| Select subset of columns | `df[["city", "price"]]` | `try df.select("city", "price")` |
| Select columns by `dtype` | `df.select_dtypes(include="number")` | `try df.select(columnsOfType: .number)` |
| Drop columns | `df.drop(columns=["notes"])` | `try df.drop(columns: ["notes"])` |
| Single row by position | `df.iloc[2]` | `df[row: 2]` or `df.row(at: 2)` (`DataRow?`) |
| Row range slice | `df.iloc[1:4]` | `df[rows: 1..<4]` (`DataFrame?`) |
| Single cell access | `df.loc[0, "price"]` | `df["price", 0]` (`DataValue?`) |
| First / last $N$ rows | `df.head(5)` / `df.tail(5)` | `try df.head(5)` / `try df.tail(5)` |
| Top / bottom $N$ by column | `df.nlargest(3, "price")` / `df.nsmallest(3, "price")` | `try df.largest(3, by: "price")` / `try df.smallest(3, by: "price")` |
| Row index of max / min | `df["price"].idxmax()` / `idxmin()` | `try df.indexOfMax("price")` / `try df.indexOfMin("price")` |
| Vectorized boolean mask | `df[(df["price"] > 50) & (df["qty"] >= 2)]` | `let mask = try (df.series("price") > 50) && (df.series("qty") >= 2)`<br>`let sub = df[mask]` |

---

## 2. Selecting & Dropping Columns

```swift
import Bandal

let employees = try DataFrame(columns: [
    (name: "id", values: ["E01", "E02", "E03", "E04", "E05"]),
    (name: "dept", values: ["AI", "Sales", "AI", "Finance", "Sales"]),
    (name: "salary", values: [95000.0, 72000.0, 110000.0, 88000.0, 68000.0]),
    (name: "bonus", values: [12000.0, nil, 18000.0, 9000.0, 5000.0]),
    (name: "remote", values: [true, false, true, false, false]),
])

// Keep specific columns in the requested order
let summaryCols = try employees.select("id", "dept", "salary")

// Select all numeric columns automatically
let numericOnly = try employees.select(columnsOfType: .number)

// Drop unwanted columns
let withoutBonus = try employees.drop(columns: ["bonus", "remote"])
```

---

## 3. Positional Row & Cell Slicing

```swift
// Single row by zero-based index (returns DataRow?)
if let secondRow = employees[row: 1] {
    print("Row 1 dept: \(secondRow["dept"] ?? .missing)")
}

// Contiguous row range slice (returns DataFrame?)
if let middleSlice = employees[rows: 1..<4] {
    print(middleSlice)
}

// Top 2 highest-paid employees
let topEarners = try employees.largest(2, by: "salary")
```

---

## 4. Vectorized Boolean Mask Indexing (`BooleanMask`)

``BooleanMask`` is backed by a compact 64-bit word-packed bitmap (`BitSet`) that processes 64 rows per CPU instruction.

```swift
let salary = try employees.series("salary")
let dept = try employees.series("dept")

// Combine vectorized masks with &&, ||, !
let mask: BooleanMask = try (salary >= 85000) && dept.equals("AI")

let aiSeniors = employees[mask]
let aiSeniorsChecked = try employees.filtered(by: mask)
```

---

## 5. Type-Safe Composable Filtering with `RowPredicate`

When filtering directly on a ``DataFrame`` in a method chain without extracting intermediate ``Series`` variables, use ``DataFrame/query(_:)-(RowPredicate)`` with ``RowPredicate``:

```swift
let result = try employees.query(
    .notMissing("bonus")
    && (.column("dept", in: ["AI", "Finance"]) || .column("salary", greaterThanOrEqual: 70000))
    && !.column("id", equals: "E04")
)
```

---

## 6. String Query Expressions & Custom Row Closures

Just like `df.query("...")` in `pandas`, Bandal can parse concise filter strings directly via ``DataFrame/query(_:)-(String)``:

```swift
let queried = try employees.query("salary >= 80000 && remote == true && bonus != nil")
```

For custom multi-column row predicates, use ``DataFrame/filterRows(_:)``:

```swift
let highBonusRatio = try employees.filterRows { row in
    guard let sal = row["salary"]?.numberValue,
          let bon = row["bonus"]?.numberValue,
          sal > 0
    else {
        return false
    }
    return (bon / sal) >= 0.15
}
```

> Next Step: Proceed to <doc:04-Cleaning-and-Feature-Engineering> to learn how to impute missing values, deduplicate, sort, transform strings/dates, scale features, and one-hot encode categorical variables.
