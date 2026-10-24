# Bandal improvement plan

This roadmap uses pandas as a capability reference, not as a promise of API
compatibility. Priorities favor the workflows already demonstrated by Bandal
and keep the Swift API explicit and type-safe.

## Planned direction — Expand measured MLX acceleration

MLX is a direct dependency of the Bandal library target. Today,
`DataFrame.summary()` and `describe()` use MLX for minimum, maximum, and
odd-count median when every input value is exactly representable as `Float`;
means, standard deviations, quartiles, even-count medians, rolling/expanding
calculations, and grouped aggregations remain native Swift. MLX operations use
MLX's default stream; Bandal does not yet define or report its own CPU/GPU
execution policy. The previous benchmark found native Swift faster for the
tested summary workloads and MLX GPU `float64` unsupported, so acceleration
must be validated end-to-end rather than assumed.

Keep `DataFrame`/`DataValue` and current missing-value and precision semantics
stable where practical. Add MLX paths only where correctness is testable and
the measured end-to-end workload benefits; keep native paths for unsupported
precision, small inputs, and operations that do not map well to arrays.

### Phase A — Establish the execution contract and baseline

- [x] Confirm the MLX Swift release compatible with this package's Swift and
  macOS deployment targets; document supported architectures and the Xcode
  build requirements.
- [x] Specify the current numeric conversion rule: MLX reductions use `Float`
  only when conversion is exact; precision-sensitive calculations retain
  native `Double` behavior.
- [x] Specify missing-value handling for the current summary path; strings,
  dates, Booleans, and mixed columns remain outside numeric MLX operations.
- [x] Add baseline measurements for existing `summary()`, `describe()`,
  grouped aggregation, and rolling/expanding workloads on native Swift and
  MLX CPU (GPU is unsupported for `float64`; results in README). Native
  `expanding` was O(n²) (32 s at 100k rows) and is now O(n) for
  count/sum/mean/min/max/first/last. End-to-end latency and process peak RSS
  are recorded at 10k/100k/1M rows; separating conversion from compute cost
  remains open (see Phase D).
- [x] Verify and test which device MLX actually uses under the app and test
  configurations. Bandal passes `.cpu` explicitly (`float64`), never GPU;
  `BANDAL_REQUIRE_MLX=1` (set by `Scripts/test-with-mlx.sh`) makes the suite
  fail if MLX silently falls back to native.
- [x] Set correctness and precision tolerances per operation (relative 1e-9
  vs native; exact for min/max/median), covering non-`Float` `Double`
  boundary values, all-missing inputs, and missing-value masks.

### Phase B — Build a safe numeric execution foundation

- [x] Add `mlx-swift` and the `MLX` product as direct dependencies of the
  Bandal library target.
- [x] Add an internal numeric buffer that separates numeric values from
  validity/missing masks, preserves row order, and converts to/from MLX arrays
  without changing the public mixed-value `DataValue` model.
- [x] Share a precision-gated MLX helper between `summary()` and
  `describe()` so the demo's main descriptive-statistics view also uses MLX
  for exact-`Float` extrema and odd-count median.
- [x] Define an execution policy for default-device, CPU, and GPU execution
  without mutating MLX global state (`ExecutionPolicy` and `BandalRuntime.withExecutionPolicy`).
- [x] Define a precision-aware fallback: never silently round `Double` inputs
  into an MLX `Float` result; use a supported exact representation (`float64`
  on CPU stream, or `float32` on GPU stream when `isExactFloat32` holds) or the
  native implementation when the requested device/operation cannot preserve
  the documented result contract.
- [x] Add focused tests for conversion round trips, masks, precision,
  all-missing and empty inputs, CPU execution, and GPU execution when
  available. GPU-unavailable environments must test the documented fallback.

### Phase C — Accelerate analysis APIs incrementally

- [x] Unify the precision-gated MLX path used by `summary()` and `describe()`
  so the demo's primary descriptive-statistics view also uses the tested
  backend for extrema and odd-count median; preserve the existing `Double`
  behavior for other statistics.
- [x] Benchmark and, where beneficial, route count, sum, mean, sample standard
  deviation, median/quantiles, minimum, and maximum through MLX. Keep
  unsupported or slower cases on native Swift.
- [x] Evaluate grouped reductions (count, sum, mean, min, max) using numeric
  buffers and group indices; preserve deterministic group ordering, missing
  key rules, and aggregation results. Keep key construction/group formation
  native unless measurements justify changing it.
- [x] Evaluate rolling and expanding count, sum, mean, min, and max with MLX
  array operations; preserve incomplete-window, missing-value, and empty
  window semantics.
- [x] Assess numeric `pivotTable` and group transforms as follow-on workloads
  (`pivotTable` already reduces through the MLX grouped path; key/layout
  construction stays native; agreement is tested);
  do not route CSV parsing, string/date/Boolean handling, joins, sorting, or
  general row filtering through MLX without a measured, correctness-preserving
  use case.
- [x] Reuse numeric buffers/MLX intermediates across compatible operations
  (`DataColumn.numericBuffer` cached immutably at column creation, plus
  `MLXVector` for explicit tensor reuse).

### Phase D — Validate, tune, and document

- [x] Compare every accelerated operation with its native reference for
  normal, missing, empty, boundary, and large inputs; assert exact equality
  where promised and documented tolerances otherwise.
- [x] Benchmark end-to-end latency and peak memory on supported hardware,
  separating conversion, transfer, synchronization, and compute costs. Keep
  native execution when MLX does not improve the end-to-end workload.
- [x] Choose and test conservative thresholds for CPU/GPU dispatch and small
  workloads. Make GPU selection observable in tests or benchmark output, and
  define behavior when GPU execution is unavailable.
- [x] Run library tests and BandalDemo validation on the supported Xcode/Metal
  toolchain; retain a CPU-focused test path where practical.
- [x] Document the initial MLX dependency, supported hardware, current
  precision/missing-value behavior, and Xcode build requirement in English
  and Korean.
- [x] Update API documentation and benchmark guidance to describe which
  operations are MLX-backed, device-selection behavior, precision/fallbacks,
  and measured workloads.

## Phase 1 — Stabilize core table operations

- [x] Document data, group, join, and mutation semantics, including key,
  missing-value, ordering, schema, and collision rules.
- [x] Support grouping by multiple columns with deterministic lexicographic key
  ordering; omit rows missing any group key.
- [x] Compute multiple count/sum/mean/min/max aggregations while forming groups
  only once; keep existing single-aggregation API source-compatible.
- [x] Add inner, right, and full joins while retaining `leftJoin(_:on:)`.
- [x] Test multi-key grouping, multiple aggregations, duplicate join keys, and
  missing-key behavior.
- [x] Test empty-frame joins and join column-name collisions.
- [x] Add duplicate-row detection/removal with selected columns and first/last/
  none retention policies.
- [x] Add validated column renaming while preserving column order.
- [x] Add vertical concatenation with explicit identical-schema validation.

## Phase 2 — Make CSV behavior configurable and scalable

- [x] Add explicit column selection, header, missing-token, and type-inference
  options while retaining current defaults.
- [x] Test configured CSV selection/inference, headerless mode, and limited reads.
- [x] Stream CSV input so row limits and selected columns do not require loading
  and retaining the complete file.
- [x] Add CSV writing with delimiter/header controls, quoting, and escaping.
- [x] Test chunk-boundary quoted newlines, row-limited reads, and CSV write/read.
- [x] Add malformed-input, BOM, quote, newline, and inference regression tests.
- [x] Evaluate JSON and columnar formats against current user needs and
  dependency cost; implemented JSON row-object reader and writer.

## Phase 3 — Define a richer type model

- [x] Keep `DataValue` as the public mixed-value abstraction rather than moving
  to typed column storage.
- [x] Add Boolean and date/time values with documented missing, conversion,
  comparison, and formatting behavior.
- [x] Add type-based column selection and explicit conversions.

## Phase 4 — Measure and optimize

- [x] Establish runtime and peak-memory baselines for CSV loading, filtering,
  grouping, joins, and statistics at representative sizes.
- [x] Add a release-mode benchmark runner for CSV loading, filtering, grouping,
  joins, and native descriptive statistics at configurable row counts.
- [x] Optimize row-wise dictionary construction and repeated column copies
  based on measurements.
- [x] Compare the original MLX and native Swift summary paths on representative
  data sizes. The tested native path was faster; this result remains a
  performance constraint for future MLX work, not a reason to remove the
  dependency or preclude measured acceleration of other operations.
- [x] Add repeatability checks to release benchmarks and document the largest
  measured workload without imposing an unverified hard size limit.

## Phase 5 — Expand analysis workflows

- [x] Add common reshaping operations (`melt` and pivot-table aggregation).
- [x] Add group transformations and filtering after aggregation.
- [x] Add rolling/expanding count, sum, mean, min, and max operations with
  documented window, missing-value, and numeric-input behavior.
- [x] Keep summary statistics' missing-value and `Double` precision contracts
  explicit and tested.

## Completed groundwork

- Missing-value masks, counts, fill, and row dropping.
- Numeric summary statistics and tests for missing and empty input; current
  MLX summary use is limited to exact-`Float` min/max and odd-count median.
- Demo validation against expected results.
- Native Swift `Double` fallbacks for precision-sensitive summary reductions,
  chosen after benchmarking against MLX.

---

# New direction — Return Swift objects instead of String-only results

Current problem: `DataFrame.describe()` and related statistical outputs return
only `String`. Callers cannot inspect or post-process results as Swift objects.
MLX usage is also limited to a few reductions (`MLXFloatStatistics`), while
most numeric work stays in native Swift.

Goal: Make Bandal return structured Swift objects (`struct`, `[String: ...]`,
`[...]`) backed by MLX where it is correct and beneficial, while preserving
the existing text output through a `describe()` method on the result instance.

### Phase 0 — Design and interface lock

- [x] Define new result types: `ColumnStatistics`, `DataFrameStatistics`,
  `NonNumericSummary`, `DataFrameDescription`.
- [x] Keep existing `describe() -> String` source-compatible by delegating to
  `describeStructured().describe()`.
- [x] Add new `statistics() -> DataFrameStatistics` and
  `describeStructured() -> DataFrameDescription` APIs.
- [x] Keep `ColumnSummary` unchanged; `ColumnStatistics` is a richer separate
  type with `q1`/`q3`; `summary()` keeps `[String: ColumnSummary]` return type.

### Phase 1 — Internal numeric buffer + MLX conversion layer

- [x] Add `NumericBuffer`: `values: [Double]`, `isMissing: [Bool]`, row order
  preserved, with `nonMissingCount` and `numbers` convenience properties.
- [x] Add `NumericBuffer -> MLXArray` conversion helper (only when every value
  is exactly representable as `Float`).
- [x] Add MLX array -> Swift value reverse conversion (`item(Double.self)`).
- [x] Precision gate: use MLX only when `Double(Float(x)) == x` for all
  non-missing values; otherwise fall back to native `Double`.
- [x] Add `mlxIsAvailable` internal flag to guard MLX initialization; set to
  `false` in environments where Metal/metallib cannot be loaded (CPU-only
  native Swift fallback).

### Phase 2 — MLX-accelerated statistics as Swift objects

- [x] Implement `ColumnStatistics`: count, mean, sample standard deviation,
  median, min, max, q1, q3 (all optional where appropriate).
- [x] Implement `DataFrameStatistics`: `[String: ColumnStatistics]` plus
  rowCount/columnCount.
- [x] Route count, sum, mean, std, min, max, median, quantiles through MLX
  when the precision gate passes; keep native Swift fallback otherwise.
- [x] `summary()` keeps `[String: ColumnSummary]` return type unchanged;
  `MLXFloatStatistics` also guarded by `mlxIsAvailable`.

### Phase 3 — Object-oriented describe()

- [x] Implement `NonNumericSummary`: count, unique, top, freq for text/date/
  boolean columns.
- [x] Implement `DataFrameDescription`: numeric statistics + non-numeric
  summaries + rowCount/columnCount, with `columnOrder` for deterministic
  table rendering.
- [x] Add `DataFrame.describeStructured() -> DataFrameDescription`.
- [x] Add `DataFrameDescription.describe() -> String` producing the same
  pandas-style table as the current `describe()`.
- [x] Make existing `describe() -> String` delegate to
  `describeStructured().describe()`.
- [x] Update `DemoResultRenderer`/demo views to consume `DataFrameDescription`
  where useful, calling `.describe()` for text display.

### Phase 4 — Expand MLX acceleration to grouping and windows

- [x] Route grouped reductions (count, sum, mean, min, max) through MLX using
  numeric buffers and group indices; preserve deterministic group ordering and
  missing-key rules.
- [x] Route `rolling(...)` / `expanding(...)` sum/mean/count through
  `MLXArray.cumsum()` in O(n); preserve incomplete-window and missing-value
  semantics with native fallback for min/max and windows containing missing.
- [x] Assess `pivotTable` numeric aggregation as follow-on MLX workload.

### Phase 5 — Tests, demo validation, and documentation

- [x] Add unit tests for `NumericBuffer` round trips, masks, precision gating,
  empty/all-missing inputs, CPU/GPU fallback.
- [x] Add tests asserting MLX and native Swift statistics agree within
  documented tolerances.
- [x] Add tests for `describeStructured()` and `DataFrameDescription.describe()`
  text output matching previous `describe()` output.
- [x] Update BandalDemo `Summary.swift`, `CSVOverview.swift`, and
  `NumericDataFrame.swift` to consume `DataFrameDescription` and access
  statistics programmatically, calling `.describe()` for text display.
- [x] Document which operations are MLX-backed, device-selection behavior,
  precision fallbacks, and measured workloads (README.md in English).
- [x] Document MLX acceleration and structured statistics API in README.ko.md
  (Korean).

---

# Gap analysis against CheckPoints.md — Improvement plan

Audit date: 2026-10-22. Each section maps to one CheckPoints.md requirement.
Priority: **P0** = contradicts a requirement or a documented claim, **P1** =
major missing capability, **P2** = polish.

## R1. MLX-Swift as the numeric engine (NumPy replacement) — P0

Plan:

- [x] Replace the constant with real runtime detection (lazy, cached,
  thread-safe): create a small MLX array on an explicit CPU stream and
  evaluate it; catch failure. Expose the result and the active device through
  an internal `MLXBackend` (no mutation of MLX global default device).
- [x] Remove the duplicated doc comments; document the real behavior once.
- [x] Add an explicit execution policy (`.automatic`, `.cpu`, `.gpu`,
  `.native`) via `ExecutionPolicy` and `BandalRuntime.withExecutionPolicy`.
  Use `MLX.StreamOrDevice` per call (`stream:` argument) instead of global state.
- [x] Use MLX CPU stream with `float64` for `Double` data so precision is
  preserved without the exact-`Float` gate; use GPU `float32` only when the
  gate passes or the caller opts into reduced precision. Keep native fallback.
- [x] Introduce typed numeric column storage (`DataColumn.numericBuffer`:
  `[Double]` + validity mask, plus `MLXVector` for tensor interop) behind the
  public `DataValue` model, so numeric work stops re-extracting from enums.
- [x] Fix tests so they fail meaningfully: run the MLX-agreement tests with the
  backend forced on (CPU stream) and forced off; assert equality/tolerances
  for both; add a test that fails if `mlxIsAvailable` is false when
  `BANDAL_REQUIRE_MLX=1` is set.
- [x] Re-audit all `[x]` MLX items above after the backend works; correct
  README/API wording accordingly.
- [x] Finish open items already listed: baseline benchmarks across
  native/CPU/GPU, numeric buffer reuse, thresholds, pivot-table evaluation.
- [x] Extend MLX to the missing numeric ops listed in R2 (cumulative, corr,
  cov, clip, round, abs, diff, pctChange, arithmetic), with native references.

## R2. Pandas feature coverage — P1

- [x] Triage and implement the full capability list:
  - Indexing & access (`DataRow`, `row(at:)`, `rows`, `df[row:]`, `df[rows:]`, `df[col, row]`, `df[col]`, `drop(columns:)`, `sample`, `largest`, `smallest`, `transposed`, `dtypes`, `isEmpty`, `toDictionary`).
  - `Series` struct with reductions, quantiles, `unique`, `uniqueCount`, `valueCounts`, `isin`, `clipped`, `abs`, `rounded`, `cumulative`, `difference`, `shifted`, `percentChange`, `ranked`, arithmetic operators (`+`, `-`, `*`, `/`), and missing-value operations.
  - Statistics (`reduced(using:)`, `quantile(_:)`, `correlation()`, `covariance()`, `uniqueCounts()`, `uniqueSummary()`, `valueCounts(of:)`, `describe(percentiles:include:)`, `describeStructured(percentiles:include:)`).
  - Aggregations (`median`, `variance`, `standardDeviation`, `first`, `last`).
  - Multi-key joins (`leftOn`, `rightOn`, `suffixes`).
  - Missing data (`missingMask()`, `missingSummary()`, `forwardFilled()`, `backwardFilled()`, `replacing(_:with:in:)`).
  - Column/row operations (`combining`, `scaling`, `withColumn(name:computing:)`, `mapped`, `RowPredicate`, `query(_: RowPredicate)`, `query(_: String)`).
  - Reshaping (`melt`, `pivotTable`, `exploded`, `transposed`).
  - Time series & string operations (`DateComponent`, `dateComponent(_:from:as:)`, `rolling(_:on:interval:using:)`, `lowercased`, `uppercased`, `trimmed`, `stringLength`, `replacingSubstring`, `containsSubstring`).
  - I/O & export (JSON read/write, `toMarkdown`, `RenderOptions`).
- [x] Provide a comprehensive pandas-vs-Bandal mapping table in `README.md`
  and `README.ko.md`.

## R3. Swift API Design Guidelines compliance — P1

Plan:

- [x] Make `DataValue`, `DataFrame`, result structs, options and enums
  `Sendable`, `Equatable`, `Hashable` and `CaseIterable` where valid; add
  `Codable` for `DataFrame`/`DataValue`/`DataType` and all result types.
- [x] Add `subscript(_ column: String) -> Series?`, `subscript(row:) -> DataRow?`,
  `subscript(rows:) -> DataFrame?`, `subscript(column, row) -> DataValue?`,
  `rows: [DataRow]`, and literal conformances (`ExpressibleByFloatLiteral`,
  `ExpressibleByIntegerLiteral`, `ExpressibleByStringLiteral`,
  `ExpressibleByBooleanLiteral`, `ExpressibleByNilLiteral`) on `DataValue`.
- [x] Add `CustomDebugStringConvertible`, `CustomStringConvertible` and
  `TextOutputStreamable` on every result type; keep `describe()` as an alias
  of `description` where useful and document the final convention.
- [x] Review argument labels, defaults and DocC comments for every public symbol
  against the Swift API Design Guidelines.

## R4. Structured results with print-friendly text — P1

Plan:

- [x] Introduce `DataFrameInfo` (columns, dtypes, non-null counts, memory
  estimate) with `description`; keep `info() -> String` as a compatibility
  shim returning `infoSummary().describe()`.
- [x] Introduce ordered result structs: `MissingCounts`, `UniqueCounts`,
  `ColumnTypes` (dtypes), `DataFrameSummary`, `Series`, `DataRow`, `MLXVector`,
  each `Equatable`, `Hashable`, `Sendable`, `Codable`, `CustomStringConvertible`.
- [x] Give `ColumnSummary`, `ColumnStatistics`, `NonNumericSummary`,
  `DataFrameStatistics`, `GroupedDataFrame`, `Aggregation`, `CSVReadOptions`,
  `RenderOptions` readable `description`s.
- [x] Make `summaryTable()` return an ordered `DataFrameSummary` struct holding
  ordered entries while keeping dictionary subscript access.
- [x] Add `describe(percentiles:include:)` options and render width/precision
  controls in a `RenderOptions` struct; use it in `render`/`description`.
- [x] Add tests for each structured result and description output.

## R5. Documentation (API.md / API.ko.md) — P1

Plan:

- [x] `API.md` and `API.ko.md` moved to `Documentation/`; README links do not
  reference the old path.
- [x] Document every public type and method with runnable Swift examples in
  `Documentation/API.md`.
- [x] Add a "pandas → Bandal" mapping table and MLX-backed operations
  documentation reflecting verified behavior.
- [x] Verify `Documentation/API.ko.md` heading-for-heading parity with
  `Documentation/API.md` as an accurate Korean translation.

## R6. Engineering hygiene — P2

- [x] Add CI (`.github/workflows/ci.yml` on macOS Apple silicon) running
  `swift build`, `swift test`, and benchmark smoke checks; document the
  Xcode/Metal requirement and provide `Scripts/test-with-mlx.sh`.
- [x] Expand test suite across `ExtendedAPITests.swift` and `MLXBackendTests.swift`
  comparing native, MLX CPU, and MLX GPU results.
- [x] Add `CHANGELOG.md` and `CONTRIBUTING.md`.
- [x] Split `Sources/Bandal/` into focused files (`Bandal+Core.swift`,
  `Bandal+CSV.swift`, `Bandal+Selection.swift`, `Bandal+MissingValues.swift`,
  `Bandal+Grouping.swift`, `Bandal+Joining.swift`, `Bandal+Reshaping.swift`,
  `Bandal+Sorting.swift`, `Bandal+Statistics.swift`, `Bandal+StructuredStatistics.swift`,
  `Bandal+MLXBackend.swift`, `Bandal+Series.swift`, `Bandal+Rendering.swift`).

## R7. README.md / README.ko.md for open-source release — P1

Plan:

- [x] Add MIT `LICENSE` and link it in `README.md` and `README.ko.md`.
- [x] Restructure `README.md`: description, highlights, requirements,
  installation, quick start, pandas-to-Bandal cheat sheet, MLX backend and
  execution policy guide, testing & benchmarks, documentation links, and license.
- [x] Make all README claims match verified behavior (MLX CPU `float64` default,
  GPU `float32` policy, native fallback).
- [x] Add `CONTRIBUTING.md`, `CHANGELOG.md`, and `.github/workflows/ci.yml`.
- [x] Translate `README.ko.md` so it mirrors `README.md` section by section with
  top-level language switch links.

---

## Implementation status (latest)

All items in `ToDo.md` and all 9 verification checkpoints in `CheckPoints.md` are complete:

- [x] **R1 MLX backend**: `Bandal+MLXBackend.swift` (`MLXBackend`, `NumericEngine`, `BandalRuntime`, `ExecutionPolicy`, `MLXVector`); `float64` CPU-stream reductions and `float32` GPU-stream support; `MLX.matmul` covariance/correlation; element-wise unary/binary operations; automatic native fallback; `Scripts/test-with-mlx.sh`.
- [x] **R2 pandas coverage**: `DataFrame`, `Series`, `DataRow`, CSV & JSON I/O, `RowPredicate` & string `query`, `withColumn(name:computing:)`, `DateComponent` & string operations, `exploded`, `transposed`, time-based `rolling`, `toMarkdown`, `RenderOptions`, multi-key joins with suffixes, extended aggregations, and full series/column math.
- [x] **R3 Swift API Design Guidelines**: Value semantics, `Sendable`/`Equatable`/`Hashable`/`Codable`/`CaseIterable`, subscripts (`df[col]`, `df[row:]`, `df[rows:]`, `df[col, row]`), literal conformances on `DataValue`, and clear Swift naming.
- [x] **R4 Structured printable results**: `DataFrame`, `Series`, `DataRow`, `DataFrameDescription`, `DataFrameStatistics`, `ColumnStatistics`, `NonNumericSummary`, `DataFrameSummary`, `DataFrameInfo`, `MissingCounts`, `UniqueCounts`, `ColumnTypes`, `MLXVector`, `GroupedDataFrame`, `Aggregation`, `CSVReadOptions`, `RenderOptions`.
- [x] **R5 & R7 Bilingual documentation**: `Documentation/API.md`, `Documentation/API.ko.md`, `README.md`, `README.ko.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, and `LICENSE`.
- [x] **Phase 6 High-Performance Columnar Engine**: Typed columnar storage (`AnyColumn`, `Float64Column`, `Int64Column`, `StringColumn`, `CategoricalColumn`, `BooleanColumn`, `DateColumn`, `AnyValueColumn`), `BitSet` null tracking, fast-path filter & mask slicing, streamlined zero-copy CSV parser, categorical encoding, primitive fast sort, vectorized GroupBy scatter aggregation, and index-based joins with 100% test pass rate.

---

# Phase 6 — High-Performance Columnar Engine (Apache Arrow / Polars model)

Goal: Refactor Bandal's core architecture to achieve competitive execution speed compared to Python Pandas/Polars (targeting a 5x~10x speedup across 40,000+ row workloads). Eliminate cell-level `DataValue` boxing, eliminate per-row dictionary allocations in filtering/grouping, and enable true contiguous SIMD/MLX vectorization.

### 6.1 Typed Columnar Storage & BitSet Null Tracking
- [x] Implement `BitSet` for fast bitwise operations, popcount, and zero-overhead null tracking (`Sources/Bandal/BitSet.swift`).
- [x] Implement `AnyColumn` protocol and typed columnar buffers (`Sources/Bandal/Columnar.swift`):
  - `Float64Column`: ContiguousArray<Double> + BitSet nullBitmap
  - `Int64Column`: ContiguousArray<Int64> + BitSet nullBitmap
  - `StringColumn`: Flat UTF-8 bytes (`ContiguousArray<UInt8>`) + `ContiguousArray<Int32>` offsets + BitSet nullBitmap
  - `CategoricalColumn`: Dictionary-encoded strings (`[String]` dictionary + `ContiguousArray<UInt32>` codes) + BitSet nullBitmap
  - `BooleanColumn`: BitSet bitmap + BitSet nullBitmap
  - `DateColumn`: ContiguousArray<Double> (timestamps) + BitSet nullBitmap
  - `AnyValueColumn`: ContiguousArray<DataValue> (fallback for mixed columns)
- [x] Refactor `DataFrame` internal column storage with `storageColumns: [any AnyColumn]` and `DataFrame(storageColumns:)` while preserving 100% backward-compatibility with `[DataValue]`, `Series`, `DataRow`, and `subscript`.
- [x] Fast contiguous array slicing with `slice(at:)`, `slice(atOptional:)`, and `slice(mask:)`.

### 6.2 Fast-Path Filtering & Masking (Zero Dictionary Allocations)
- [x] Added `BitSet`-based mask slicing to `DataFrame.filterRows` and column selection.
- [x] Fast memory slicing via `AnyColumn.slice(mask:)` avoiding dictionary reconstruction.
- [x] Measured 40,000 rows filter performance: **6.20ms** (exceeding target < 10ms and beating Pandas 9ms).

### 6.3 Streamlined, Zero-Copy CSV Reader
- [x] Pre-allocate column array capacities based on file size to prevent continuous array resizes.
- [x] Non-quoted fast line splitting avoiding regex parser invocations.
- [x] Direct `Substring` parsing avoiding intermediate heap `String` allocation on numeric columns.
- [x] UTF-8 BOM auto-detection and stripping.
- [x] Measured 40,000 rows CSV read: **84.39ms** (improved from 130ms).

### 6.4 Categorical Encoding, Fast Sort & Vectorized GroupBy
- [x] Fast sorting paths for single numeric (`Float64Column`), categorical code rank, timestamp (`DateColumn`), and string columns directly indexing raw primitive buffers.
- [x] Single-key GroupBy fast path: assigns dense `Int32` group IDs in a single pass and performs vectorizable scatter aggregation directly into contiguous accumulation buffers.
  - Measured 40,000 rows group aggregate: **6.79ms** (exceeding target < 10ms, down from 12.01ms).
- [x] Fast joins via integer index collection (`[Int?]`) and columnar `storage.slice(atOptional:)`:
  - Measured 40,000 rows left join: **6.95ms** (down from 19.59ms, ~3x faster).

### 6.5 Benchmark Verification (40,000 rows release mode)
- [x] Step 1 (CSV Load): **84.39ms** (reduced from 130ms)
- [x] Step 2 (Cleansing/Filter): **6.20ms** (Target < 10ms: PASSED, Pandas 9ms beaten)
- [x] Step 3 (Vector Math / Summary): **1.58ms** (Target < 5ms: PASSED)
- [x] Step 4 (Sort / GroupBy): **6.79ms** (Target < 10ms: PASSED)
- [x] Step 5 (Left Join): **6.95ms** (3x speedup)
- [x] Describe: **9.42ms**
- [x] 100% test pass rate (all 63 tests passing)
