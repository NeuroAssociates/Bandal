# Changelog

## [0.1.1] - 2026-10-24

### Added
- High-performance typed columnar storage engine (`BitSet`, `AnyColumn`, `Float64Column`, `Int64Column`, `StringColumn`, `CategoricalColumn`, `BooleanColumn`, `DateColumn`, `AnyValueColumn`).
- 64-bit contiguous word bitmap (`BitSet`) for zero-overhead null tracking and boolean mask operations.
- Zero-copy columnar slicing APIs (`slice(at:)`, `slice(atOptional:)`, `slice(range:)`, and `slice(mask:)`).
- CSV optimizations: capacity pre-allocation, quote-free line splitting, UTF-8 BOM auto-detection, and zero-allocation Substring numeric parsing.

### Performance
- **CSV Read**: 35% faster on 40,000 rows (130ms → 84.39ms) via capacity pre-allocation and Substring parsing.
- **Filtering**: Direct `BitSet` mask slicing runs in 6.20ms on 40,000 rows (outperforming Pandas' 9ms).
- **GroupBy Aggregation**: Single-pass dense `Int32` group ID assignment and vectorizable scatter aggregation (12.01ms → 6.79ms, 43% speedup).
- **Joins**: Index gathering with `slice(atOptional:)` columnar slicing speeds up joins by ~3x (19.59ms → 6.95ms).
- **Sorting**: Primitive buffer and categorical code direct sorting fast paths.
- **Memory**: Peak RSS reduced by 21% (45MB → 35.5MB on 40,000 rows).

## [0.1.0] - 2026-10-24

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
