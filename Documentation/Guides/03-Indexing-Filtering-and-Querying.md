# Guide 03: Indexing, Slicing, Boolean Masks & Querying

> **Language**: **English** | [한국어](03-Indexing-Filtering-and-Querying.ko.md)  
> **Navigation**: [← Guide 02: Data I/O & Inspection](02-Data-IO-and-Inspection.md) | [API Reference](../API.md) | **Next**: [Guide 04: Cleaning & Feature Engineering →](04-Cleaning-and-Feature-Engineering.md)

---

## Overview

In `pandas`, analysts spend a large portion of their time selecting columns (`df[["a", "b"]]`), slicing rows (`df.iloc[0:5]`), and filtering datasets (`df[df["score"] >= 80]`, `df.query(...)`).

Bandal provides four complementary ways to select and filter data, ranging from zero-copy columnar projection to vectorized 64-bit bitmap masks (`BooleanMask`) and type-safe predicates (`RowPredicate`).

In this guide you will learn:
- **Column Selection & Dropping** (`select`, `select(columnsOfType:)`, `drop(columns:)`, `series(_:)`, `df["col"]`)
- **Positional Row & Cell Slicing** (`df[row:]`, `df[rows:]`, `df["col", row]`, `head`, `tail`, `largest`, `smallest`, `indexOfMax`, `indexOfMin`)
- **Vectorized Boolean Mask Filtering** (`BooleanMask`, `Series` comparison operators, `&&`, `||`, `!`, `df[mask]`, `filtered(by:)`)
- **Composable Type-Safe Predicates** (`RowPredicate` + `df.query(_:)`)
- **Concise String Query Expressions** (`df.query("score >= 80 && active == true")`)
- **Custom Row-Closure Filtering** (`df.filterRows { row in ... }`)

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
| Membership check (`isin`) | `s.isin(["Seoul", "Busan"])` | `s.isin(["Seoul", "Busan"])` |
| Range check (`between`) | `s.between(10, 50)` | `s.between(lower: 10, upper: 50)` |
| Structured query | `df.query("price > 50 and region == 'Seoul'")` | `try df.query(.column("price", greaterThan: 50) && .column("region", equals: "Seoul"))`<br>or `try df.query("price > 50 && region == 'Seoul'")` |

---

## 2. Selecting & Dropping Columns

Let's start with a sample dataset of employee records:

```swift
import Bandal

let employees = try DataFrame(columns: [
    (name: "id", values: ["E01", "E02", "E03", "E04", "E05"]),
    (name: "dept", values: ["AI", "Sales", "AI", "Finance", "Sales"]),
    (name: "salary", values: [95000.0, 72000.0, 110000.0, 88000.0, 68000.0]),
    (name: "bonus", values: [12000.0, nil, 18000.0, 9000.0, 5000.0]),
    (name: "remote", values: [true, false, true, false, false]),
])
```

### 2.1 Extracting a Single Column (`series`, `subscript`, `values`, `column`)

Bandal provides four ways to access a single column depending on what you need:

```swift
// 1. Throwing Series extraction in O(1) (Recommended in pipelines)
let salarySeries: Series = try employees.series("salary")

// 2. Optional subscript (returns nil if column name does not exist)
if let deptSeries = employees["dept"] {
    print("Departments: \(deptSeries.unique())")
}

// 3. Raw [DataValue] array
let bonusCells: [DataValue] = try employees.values(in: "bonus")

// 4. Unboxed [Double] array (throws .nonNumericColumn if any cell is .missing or non-numeric)
let salaryDoubles: [Double] = try employees.column("salary")
```

### 2.2 Projecting or Dropping Multiple Columns (`select` & `drop`)

Because `DataFrame` shares unchanged column storage references, `select` and `drop` execute in `O(columns)` without copying row buffers:

```swift
// Keep specific columns in the requested order
let summaryCols = try employees.select("id", "dept", "salary")
print(summaryCols)

// Select all numeric columns automatically (like df.select_dtypes(include="number"))
let numericOnly = try employees.select(columnsOfType: .number)
print("\nNumeric columns: \(numericOnly.columnNames)")

// Drop unwanted columns
let withoutBonus = try employees.drop(columns: ["bonus", "remote"])
```

**Console Output**:
```text
    id     dept  salary
0  E01       AI   95000
1  E02    Sales   72000
2  E03       AI  110000
3  E04  Finance   88000
4  E05    Sales   68000

[5 rows x 3 columns]

Numeric columns: ["salary", "bonus"]
```

---

## 3. Positional Row & Cell Slicing (`iloc` & Extremes)

### 3.1 Row, Slice, and Cell Subscripts

```swift
// 1. Single row by zero-based index (returns DataRow?)
if let secondRow = employees[row: 1] {
    print("Row 1 dept: \(secondRow["dept"] ?? .missing)")
}

// 2. Contiguous row range slice (returns DataFrame?, clamped safely to 0..<rowCount)
if let middleSlice = employees[rows: 1..<4] {
    print(middleSlice)
}

// 3. Single cell lookup by (column, row) (returns DataValue?)
let cell: DataValue? = employees["salary", 2] // .number(110000.0)
```

### 3.2 `head`, `tail`, `largest`, `smallest`, `indexOfMax`, `indexOfMin`

```swift
// First 2 rows and last 2 rows
let firstTwo = try employees.head(2)
let lastTwo = try employees.tail(2)

// Top 2 highest-paid employees (pandas: df.nlargest(2, "salary"))
let topEarners = try employees.largest(2, by: "salary")
print("Top 2 Earners:")
print(topEarners)

// Row index of the maximum and minimum bonus (ignores .missing automatically)
if let maxBonusRow = try employees.indexOfMax("bonus"),
   let minBonusRow = try employees.indexOfMin("bonus") {
    print("\nMax bonus is at row \(maxBonusRow) (\(employees["id", maxBonusRow]!))")
    print("Min bonus is at row \(minBonusRow) (\(employees["id", minBonusRow]!))")
}
```

**Console Output**:
```text
Top 2 Earners:
    id  dept  salary  bonus  remote
0  E03    AI  110000  18000    true
1  E01    AI   95000  12000    true

[2 rows x 5 columns]

Max bonus is at row 2 (E03)
Min bonus is at row 4 (E05)
```

---

## 4. Vectorized Boolean Mask Indexing (`BooleanMask`)

In `pandas`, boolean indexing (`df[df["salary"] > 80000]`) is the most idiomatic way to filter data. Bandal implements this with **`BooleanMask`**, a compact 64-bit word-packed bitmap (`BitSet`) that processes 64 rows per CPU instruction.

### 4.1 Generating a `BooleanMask` from a `Series`

| `Series` Comparison / Method | Return Type | Missing-Value Behavior |
| --- | --- | --- |
| `series > threshold` (also `>=`, `<`, `<=`) | `BooleanMask` (throws if non-numeric) | `.missing` cells evaluate to `false` |
| `series.equals(value)` | `BooleanMask` | `series.equals(.missing)` matches `.missing` rows |
| `series.notEquals(value)` | `BooleanMask` | `.missing != nonMissing` evaluates to `true` |

### 4.2 Combining Masks (`&&`, `||`, `!`) and Slicing `df[mask]`

`BooleanMask` supports logical operators (`&&`, `||`, `!`) as well as method forms (`.and(_:)`, `.or(_:)`, `.negated`) and summary properties (`trueCount`, `falseCount`, `count`).

**Pandas (Python)**
```python
mask = (df["salary"] >= 85000) & (df["dept"] == "AI")
ai_seniors = df[mask]
```

**Bandal (Swift)**
```swift
let salary = try employees.series("salary")
let dept = try employees.series("dept")

// Combine vectorized masks with &&, ||, !
let mask: BooleanMask = try (salary >= 85000) && dept.equals("AI")

print("Matching rows: \(mask.trueCount) out of \(mask.count)")

// Option A: Subscript syntax (returns empty DataFrame if mask length mismatches)
let aiSeniors = employees[mask]
print(aiSeniors)

// Option B: Throwing method (throws .unequalColumnLengths if mask length mismatches)
let aiSeniorsChecked = try employees.filtered(by: mask)
```

**Console Output**:
```text
Matching rows: 2 out of 5
    id  dept  salary  bonus  remote
0  E01    AI   95000  12000    true
1  E03    AI  110000  18000    true

[2 rows x 5 columns]
```

### 4.3 `isin` and `between` on `Series`

`Series` also provides `.isin(_:)` and `.between(lower:upper:inclusive:)` which return `[Bool]` (readily convertible to `BooleanMask([Bool])`):

```swift
let targetDepts = BooleanMask(dept.isin(["AI", "Finance"]))
let midSalary = BooleanMask(salary.between(lower: 70000, upper: 100000, inclusive: true))

let filtered = employees[targetDepts && midSalary]
print(filtered)
```

```text
    id     dept  salary  bonus  remote
0  E01       AI   95000  12000    true
1  E04  Finance   88000   9000   false

[2 rows x 5 columns]
```

---

## 5. Type-Safe Composable Filtering with `RowPredicate`

When filtering directly on a `DataFrame` in a method chain without extracting intermediate `Series` variables, use **`df.query(_ predicate: RowPredicate)`**.

Under the hood, built-in `RowPredicate` constructors automatically compile down to direct columnar `BitSet` scans (`fastMask`), so they run at vectorized bitmap speed without materializing row dictionaries!

### 5.1 Built-in `RowPredicate` Constructors

| Predicate Constructor | Description |
| --- | --- |
| `.column(name, equals: value)` / `.column(name, notEquals: value)` | Exact equality / inequality against a `DataValue` |
| `.column(name, greaterThan: x)` / `.column(name, greaterThanOrEqual: x)` | Numeric `>` / `>=` comparison |
| `.column(name, lessThan: x)` / `.column(name, lessThanOrEqual: x)` | Numeric `<` / `<=` comparison |
| `.column(name, between: low, and: high, inclusive: true)` | Numeric range check |
| `.column(name, in: ["A", "B"])` | Set membership (`Set<DataValue>`) |
| `.column(name, contains: "sub", caseSensitive: true)` | Substring search on string columns |
| `.isMissing(name)` / `.notMissing(name)` | Null / non-null check (`O(1)` null bitmap lookup) |
| `p1 && p2`, `p1 || p2`, `!p1` | Logical `AND`, `OR`, `NOT` composition |

### 5.2 Example: Complex Multi-Condition `RowPredicate`

```swift
let result = try employees.query(
    .notMissing("bonus")
    && (.column("dept", in: ["AI", "Finance"]) || .column("salary", greaterThanOrEqual: 70000))
    && !.column("id", equals: "E04")
)
print(result)
```

**Console Output**:
```text
    id  dept  salary  bonus  remote
0  E01    AI   95000  12000    true
1  E03    AI  110000  18000    true

[2 rows x 5 columns]
```

---

## 6. String Query Expressions & Custom Row Closures

### 6.1 Concise String Queries (`df.query(_ expression: String)`)

Just like `df.query("...")` in `pandas`, Bandal can parse concise filter strings directly:
- **Comparison operators**: `==`, `!=`, `>=`, `<=`, `>`, `<`
- **Logical operators**: `&&` (or `and`), `||` (or `or`)
- **Right-hand literals**: Numbers (`85000`), Booleans (`true`, `false`), Missing (`nil`, `null`, `NaN`), or single/double-quoted strings (`'AI'`, `"Sales"`)

```swift
let queried = try employees.query("salary >= 80000 && remote == true && bonus != nil")
print(queried)
```

```text
    id  dept  salary  bonus  remote
0  E01    AI   95000  12000    true
1  E03    AI  110000  18000    true

[2 rows x 5 columns]
```

### 6.2 Arbitrary Swift Logic with `filterRows`

When a filter depends on comparing multiple columns within the same row or custom Swift logic, use `df.filterRows(_:)`:

```swift
// Keep rows where bonus is at least 15% of salary
let highBonusRatio = try employees.filterRows { row in
    guard let sal = row["salary"]?.numberValue,
          let bon = row["bonus"]?.numberValue,
          sal > 0
    else {
        return false
    }
    return (bon / sal) >= 0.15
}
print(highBonusRatio)
```

**Console Output**:
```text
    id  dept  salary  bonus  remote
0  E03    AI  110000  18000    true

[1 rows x 5 columns]
```

---

> **Next Step**: Proceed to **[Guide 04: Cleaning & Feature Engineering →](04-Cleaning-and-Feature-Engineering.md)** to learn how to impute missing values, deduplicate, sort, transform strings/dates, scale features, and one-hot encode categorical variables.
