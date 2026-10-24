# Changelog

## Unreleased

### Added
- MLX-backed numeric engine (`float64` on CPU stream by default, or `float32` on GPU stream via `ExecutionPolicy.gpu`) for statistics, grouped aggregation, rolling/expanding windows, element-wise math, cumulative reductions, and pairwise `covariance()` / `correlation()` (`MLX.matmul`), with automatic native fallback, `BANDAL_DISABLE_MLX`, `BandalRuntime`, `ExecutionPolicy`, and `MLXVector` (`toMLXVector`, `init(mlxColumns:)`).
- JSON reader and writer (`init(jsonData:dateFormat:)`, `init(contentsOfJSON:dateFormat:)`, `jsonData(prettyPrinted:)`, `writeJSON(to:prettyPrinted:)`).
- `DataRow`, row and cell accessors (`row(at:)`, `rows`, `df[row: i]`, `df[rows: a..<b]`, `df["col", i]`, `toDictionary()`, `select(columns:)`), and literal conformances (`ExpressibleByFloatLiteral`, `ExpressibleByIntegerLiteral`, `ExpressibleByStringLiteral`, `ExpressibleByBooleanLiteral`, `ExpressibleByNilLiteral`) on `DataValue`.
- `RowPredicate`, `query(_ predicate: RowPredicate)`, `query(_ expression: String)`, and `withColumn(name:computing:)`.
- `DateComponent` and `dateComponent(_:from:as:)`, plus string column operations (`lowercased`, `uppercased`, `trimmed`, `stringLength`, `replacingSubstring`, `containsSubstring`).
- `exploded(_:separator:)`, `transposed(indexColumnName:rowPrefix:)`, and time-based `rolling(_:on:interval:using:)`.
- `RenderOptions`, `render(options:)`, and `toMarkdown(maxRows:maxColumns:)`.
- `Series`, `drop(columns:)`, `largest`/`smallest`, `reduced(using:)`, `quantile`, `ranked`, `abs`, `combining`, `scaling`, `replacing`, `uniqueCounts()`, `uniqueSummary()`, `valueCounts(of:)`, `correlation()`, `covariance()`.
- `cumulative`, `shifted`, `difference`, `percentChange`, `clipped`, `rounded`, `forwardFilled`, `backwardFilled`, `missingMask`.
- Multi-key joins with `leftOn`/`rightOn` and suffixes.
- `AggregateFunction` cases `median`, `variance`, `standardDeviation`, `first`, `last`.
- Structured result types (`DataFrameInfo`, `DataFrameSummary`, `MissingCounts`, `UniqueCounts`, `ColumnTypes`, `DataFrameDescription`, `DataFrameStatistics`, `ColumnStatistics`, `NonNumericSummary`) and `description` / `Codable` / `Hashable` / `Sendable` conformances for all result types.
- `Scripts/test-with-mlx.sh`, CI workflow, `LICENSE`, and contributing guide.

### Changed
- `README.md` and `README.ko.md` expanded for open-source distribution with complete feature coverage and pandas comparison table.
- `Documentation/API.md` and `Documentation/API.ko.md` updated to document every public type and method with runnable examples.
