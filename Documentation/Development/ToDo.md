# Bandal improvement plan

> Archive: every item below is complete. This file records development history; `AGENTS.md` (repository root) holds the current project requirements.

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

---

# Phase 7 — Zero-Boxing Columnar ↔ MLX Pipeline & Full Pandas API Completion

Goal: Eliminate all remaining `[DataValue]` boxing bottlenecks between `Columnar` storage and `MLX-Swift`, unlock full Metal GPU (`Float32`) and fused lazy graph execution in MLX, and complete advanced Pandas capabilities in idiomatic Swift per `AGENTS.md` (repository root).

### 7.1 Step 1 — Zero-Boxing Columnar ↔ MLX Pipeline & O(1) Column Replacement
- [x] **Unify `Float64Column` / `Int64Column` with `NumericBuffer`**:
  - Eliminate redundant `Array(floatCol.values)` and `[Bool]` mask copies during `DataColumn.init(storage:)`.
  - Bridge `Int64Column` seamlessly into numeric buffers and `Float64Column` views so integer columns also benefit from MLX acceleration.
  - Replace `[Bool]` missing-value loops with word-level `BitSet` operations (`bitwiseOr`, `bitwiseAnd`, `inverted`).
- [x] **Direct `Float64Column` ↔ `MLXArray` Kernels**:
  - Refactor `NumericEngine` operations (`clip`, `round`, `abs`, `cumulative`, `difference`, `percentChange`, `binaryArithmetic`, `scalarArithmetic`, `unaryMath`) to accept and return `Float64Column` (`ContiguousArray<Double>` + `BitSet`) directly via `withUnsafeBufferPointer`, avoiding `[DataValue]` boxing/unboxing.
- [x] **O(1) Unmodified Column Reuse on `DataFrame` Mutations**:
  - Refactor `withColumn`, `replacingColumn`, `select`, `drop(columns:)`, and `renamedColumns` to construct `DataFrame(dataColumns:)` / `DataFrame(storageColumns:)` directly without materializing `[DataValue]` or re-running `AnyColumnFactory.make` on untouched columns.
- [x] **Columnar-Backed `Series`**:
  - Back `Series` with `any AnyColumn` storage (computing `values: [DataValue]` only when explicitly requested) so `df["col"]` and `Series` arithmetic/comparisons run with zero `[DataValue]` boxing.

### 7.2 Step 2 — MLX Graph Fusion, Practical GPU (`Float32`) Acceleration & Vectorized Kernels
- [x] **Single-Pass MLX Graph Fusion (`MLX.eval` Batching)**:
  - Fuse `sum`, `variance`, `min`, `max`, and `sort` in `NumericEngine.moments(of:)` into a single `MLX.eval(...)` call instead of separate synchronous `.item()` / `.asArray()` round-trips.
  - Batch prefix-sum and cumulative evaluations into a single `MLX.eval` call.
- [x] **Practical Metal GPU (`Float32`) Execution Policy & Adaptive Thresholds**:
  - Add `ExecutionPolicy.gpuFloat32` to execute general finite floating-point workloads in `Float32` on the Metal GPU stream while preserving `.gpu`'s lossless `isExactFloat32` check.
- [x] **Vectorized Rolling Windows (`MLX.asStrided`), Columnar Query Masking, and Multi-Key GroupBy**:
  - Accelerate `rolling` `.sum`, `.mean`, `.min`, `.max`, `.variance`, and `.standardDeviation` on non-null columns using `MLX.asStrided` 2D sliding window views and axis reductions.
  - Add columnar `fastMask` evaluation to `RowPredicate` and `query(_: String)` so filtering operates directly on `Float64Column`, `StringColumn`, `CategoricalColumn`, `BooleanColumn`, and `BitSet` without per-row dictionary allocation.
  - Extend dense `Int32` group ID hashing and `Float64Column` output to multi-key `groupBy`.
- [x] **Direct CSV-to-Columnar Streaming Builder**:
  - Parse CSV fields directly into typed columnar builders (`CSVColumnBuilder`) during streaming reads to eliminate intermediate `[[DataValue]]` allocation and rescanning for numeric columns.

### 7.3 Step 3 — Complete Idiomatic Swift Pandas Capabilities & Structured Result Types
- [x] **Vectorized `BooleanMask` & `Series` Comparison Operators**:
  - Introduce `BooleanMask` (backed by `BitSet`, conforming to `CustomStringConvertible`, supporting `&&`, `||`, `!`).
  - Add comparison operators (`<`, `<=`, `>`, `>=`, `equals(_:)`, `notEquals(_:)`) on `Series` returning `BooleanMask`.
  - Add `df[mask]` subscript and `df.filtered(by: BooleanMask)` for fast vectorized boolean indexing.
- [x] **Mathematical, Transcendental & Standardization Operations**:
  - Add `UnaryMathFunction` (`.sqrt`, `.exp`, `.log`, `.log10`, `.log1p`, `.sin`, `.cos`, `.tan`, `.sign`), `applying(_: UnaryMathFunction)`, and `power(_ exponent: Double)` on `Series` and `DataFrame` (MLX-accelerated).
  - Add `standardized(_ column:)` (Z-score) and `minMaxScaled(_ column:)`.
- [x] **Binning, Interpolation, Conditional Replacement & Extrema Indexing**:
  - Add `binned(_:edges:labels:rightClosed:includeLowest:as:)` (`pd.cut`) and `quantileBinned(_:quantiles:labels:as:)` (`pd.qcut`).
  - Add `interpolated(_:)` for linear interpolation of missing numeric values (`df.interpolate`).
  - Add `where(_ mask: BooleanMask, otherwise replacement: DataValue)` (`df.where` / `np.where`).
  - Add `indexOfMax()` / `indexOfMin()` on `Series` and `DataFrame` (`idxmax` / `idxmin`).
- [x] **Exponentially Weighted Windows (`ewm`) & Rolling `minPeriods`**:
  - Add `exponentiallyWeighted(_:alpha:adjust:as:)` and `Series.exponentiallyWeightedMean(alpha:adjust:)`.
  - Add `minPeriods: Int? = nil` support to `rolling`.
- [x] **Advanced Reshaping, Grouping, Time-Series Resampling & Joins**:
  - Add non-aggregating `pivoted(index:columns:values:)` (`df.pivot`), `crossTabulated(index:columns:normalize:)` (`pd.crosstab`), and `oneHotEncoded(_:dropFirst:)` (`pd.get_dummies`).
  - Add `GroupedDataFrame.filter(_:)` (`groupby.filter`), custom output naming on `Aggregation` (`as alias:`), and `DataFrame.pipe(_:)`.
  - Add `joinAsOf(with:on:by:direction:tolerance:)` (`pd.merge_asof`) and `resample(on:interval:aggregations:)` (`df.resample`).
- [x] **Structured Result Types Consistency**:
  - Add `SeriesDescription` (`Series.describeStructured()`) and `QuantileTable` (`DataFrame.quantiles(_:)`), ensuring every statistical/inspection API returns a printable, structured Swift type.

### 7.4 Step 4 — Verification, Benchmarks & Bilingual Documentation Sync
- [x] Add comprehensive unit tests covering Phase 7 APIs and MLX CPU/GPU parity in `ExtendedAPITests.swift` and `MLXBackendTests.swift` (66/66 passing).
- [x] Update `CheckPoints.md` and `ToDo.md` with Phase 7 completion details.
- [x] Synchronize English and Korean documentation (`Documentation/API.md`, `Documentation/API.ko.md`, `README.md`, `README.ko.md`).

---

# Phase 8 — Large-Scale Benchmark Optimization (100k / 1M / 5M Rows vs Pandas)

Goal: Eliminate remaining `DataValue` boxing in grouping/joining/correlation, replace $O(N \log N)$ and $O(N \times W)$ kernels with $O(N)$ algorithms, and optimize memory bandwidth across all 9 benchmark workloads in `Scripts/benchmark.sh` (`filter`, `arithmetic_sqrt`, `stats_summary`, `rolling_mean_20`, `expanding_sum`, `groupby_sum`, `sort`, `left_join`, `correlation`) at 100k, 1M, and 5M rows.

### 8.1 Baseline Diagnosis (5,000,000 Rows, `Scripts/benchmark.sh`)

| Workload | Pandas (ms) | Bandal Best (ms) | Speedup vs Pandas | Root Cause of Bottleneck |
| :--- | ---: | ---: | ---: | :--- |
| `groupby_sum` | 82.50 | 1,130.50 | 0.07x | `CategoricalColumn` key unboxed to `DataValue` and hashed in `[DataValue: Int]` 5M times; `Float64Column` aggregation bypasses MLX and checks nulls row-by-row. |
| `left_join` | 217.42 | 1,066.74 | 0.20x | `values(in:)` unboxes 5M keys to `[DataValue]`; 1:1 left join re-allocates and copies all 5 left columns via `[Int?]`. |
| `correlation` | 114.23 | 148.18 (MLX) / 2,778.93 (Swift) | 0.77x / 0.04x | Native unboxes `[DataValue]` and computes all $4 \times 4 = 16$ pairs (including diagonal and lower triangle); MLX manually transposes $5\text{M} \times 4$ array on CPU and falls back from GPU due to `maxMagnitude: 1e4`. |
| `stats_summary` | 34.96 | 15.92 (GPU32) / 259.65 (Swift) / 533.68 (MLX-CPU) | 2.20x (GPU32) / 0.13x (CPU) | `moments(of:)` performs full $O(N \log N)$ sort of 5M doubles just to compute `median` and repeatedly scans 5M elements for `isFinite` and `isExactFloat32`. |
| `rolling_mean_20` | 32.92 | 40.70 (MLX) / 342.68 (Swift) | 0.81x / 0.10x | Native mode disables `prefixSums` and falls back to $O(N \times W)$ window loop; MLX `asStrided` reduces a 2D `[N, 20]` view in $O(N \times W)$ work instead of $O(N)$ sliding window accumulation. |
| `expanding_sum` | 30.22 | 20.11 (Swift) / 50.43 (MLX) | 1.50x / 0.60x | `DataFrame.prefixSums` boxes `NumericBuffer` into `[Double?]` (5M optionals) and `NumericEngine.prefixSums` maps it twice into `[Double]`. |
| `arithmetic_sqrt` | 4.56 | 25.00 (MLX) / 28.48 (Swift) | 0.18x / 0.16x | Chained `(a * b + c).sqrt()` allocates intermediate arrays, converts `MLXArray` ↔ `[Double]` ↔ `ContiguousArray<Double>` 3 times on CPU, and re-scans 5M elements for `isFinite` at each step. |
| `sort` | 478.83 | 805.59 | 0.59x | `sorted(by:)` checks `nullBitmap[i]` inside the $O(N \log N)$ comparator even when `!hasNulls`, suffers pointer-chasing cache misses, and `slice(at:)` appends element-by-element. |
| `filter` | 26.82 | 33.89 | 0.79x | `RowPredicate` sets `mask[i] = true` bit-by-bit and `slice(mask:)` iterates row-by-row instead of scanning 64-bit words with `trailingZeroBitCount` and writing into `unsafeUninitializedCapacity`. |

### 8.2 Step 1 — Zero-Boxing `groupby_sum` & `left_join` (`Bandal+Grouping.swift`, `Bandal+Joining.swift`)
- [x] **Categorical & Typed Column GroupBy Fast-Path**:
  - When grouping on a single `CategoricalColumn` (or `StringColumn` / `Int64Column` / `Float64Column`), build dense `Int32` group IDs directly from `codes` (`ContiguousArray<UInt32>`) and dictionary sort order in $O(K \log K)$ without `DataValue` boxing or 5M dictionary lookups.
  - Add branchless `withUnsafeBufferPointer` scatter reductions (`.sum`, `.mean`, `.min`, `.max`, `.count`) when neither key nor target column has nulls.
- [x] **Categorical Dictionary Mapping & 1:1 Left Join Zero-Copy Reuse**:
  - When joining on a `CategoricalColumn` against a unique right key column, hash only the right key column and map the left `dictionary` ($K$ entries) to right row indices (`codeToRightRow`), turning 5M string hash lookups into 100 lookups and an $O(N)$ array gather.
  - When `kind == .left` (or `.inner` with 100% match) and right keys are unique (1:1 row mapping), reuse all left `DataFrame` storage columns in $O(1)$ without slicing/copying, and gather only the newly joined right columns directly via `unsafeUninitializedCapacity`.
  - Add typed hash-join paths for `Int64Column`, `Float64Column`, and `StringColumn` keys without materializing `[DataValue]`.

### 8.3 Step 2 — $O(N)$ Sliding Windows, Unboxed Prefix Sums, Matrix Correlation & Quickselect Summary (`Bandal+Statistics.swift`, `Bandal+MLXBackend.swift`, `Bandal+Series.swift`)
- [x] **$O(N)$ Sliding Window `rolling` Kernel**:
  - Implement a direct single-pass $O(N)$ sliding-window accumulator (`runningSum += x[i] - x[i - w]`) on `Float64Column` for `.sum`, `.mean`, and `.count` (both native and MLX policies), eliminating the $O(N \times W)$ fallback in `.native` and the $20 \times N$ `asStrided` reduction overhead.
- [ ] **Zero-Boxing `expanding` / `prefixSums`**:
  - Refactor `DataFrame.prefixSums` and `NumericEngine.prefixSums` to accept `NumericBuffer` (`Float64Column`) directly instead of `[Double?]`.
  - Add a specialized tight loop for non-null `expanding(.sum)` / `expanding(.mean)` using `unsafeUninitializedCapacity`.
- [ ] **Upper-Triangular Native Correlation & Zero-Transpose MLX Correlation**:
  - Native `DataFrame.pairwise` (`correlation` / `covariance`): read `NumericBuffer` `ContiguousArray<Double>` buffers directly, precompute column means/stdDevs once ($M$ columns), compute dot products only for the upper triangle ($i < j$, $M(M-1)/2 = 6$ pairs instead of 16), and mirror symmetrically.
  - MLX `NumericEngine.covarianceOrCorrelationMatrix`: eliminate the CPU row-major manual transpose loop by stacking contiguous column buffers directly as `[M, N]` (`[columnCount, rowCount]`), centering along axis 1, and computing `MLX.matmul(centered, centered.transposed())`. Raise/normalize the GPU `Float32` magnitude gate so `gpuFloat32` executes on the Metal GPU.
- [ ] **$O(N)$ Quickselect Median & Single-Pass `summary()` Moments**:
  - Add `NumericEngine.summaryMoments(of: NumericBuffer)` for `Series.summary()` / `DataFrame.summary()`.
  - Skip redundant `allSatisfy(\.isFinite)` scans on pre-validated `Float64Column` buffers and skip `isExactFloat32` unless `.gpu` policy is active.
  - Compute `min`, `max`, `mean`, and `standardDeviation` in a single vectorized pass and compute `median` in $O(N)$ average time via Quickselect (`nth_element`) instead of full $O(N \log N)$ sort.

### 8.4 Step 3 — Vectorized Arithmetic, No-Null Sort & Word-Level BitSet Slicing (`Bandal+Series.swift`, `Bandal+Sorting.swift`, `Columnar.swift`, `BitSet.swift`, `Bandal+Selection.swift`)
- [ ] **Fast Vector Arithmetic & `sqrt` (`Accelerate` / Uninitialized Buffer Direct Write)**:
  - Eliminate intermediate `Array` → `ContiguousArray` copies and redundant 5M-element `isFinite` rescans in `binaryArithmeticColumn` and `unaryMathColumn`.
  - Use Apple `Accelerate` (`vDSP_vaddD`, `vDSP_vsubD`, `vDSP_vmulD`, `vDSP_vdivD`, `vvsqrt`) and `ContiguousArray(unsafeUninitializedCapacity:)` for non-null contiguous `Double` columns.
- [ ] **No-Null Specialized Sort & Fast `slice(at:)`**:
  - Specialize `DataFrame.sorted(by:)` when `!col.hasNulls` to sort without checking `nullBitmap` inside the comparator, and use radix/key-value pair sorting for single `Float64Column` / `CategoricalColumn` keys to eliminate pointer-chasing cache misses.
  - Optimize `slice(at:)` across `Float64Column`, `Int64Column`, and `CategoricalColumn` using `ContiguousArray(unsafeUninitializedCapacity:)` when `!hasNulls`.
- [ ] **64-Bit Word-Level `BitSet` Predicate Building & `slice(mask:)`**:
  - Build `BitSet` words directly in 64-element chunks in `RowPredicate` / `Series` comparisons instead of calling `mask[i] = true` per row.
  - Optimize `Float64Column.slice(mask:)`, `Int64Column.slice(mask:)`, and `CategoricalColumn.slice(mask:)` to allocate `unsafeUninitializedCapacity: mask.setBitCount` and iterate over non-zero 64-bit `words` using `word.trailingZeroBitCount` and `word &= word - 1`.

### 8.5 Step 4 — Verification & Re-Benchmarking
- [x] Verify all existing and new unit tests pass (`swift test` and `Scripts/test-with-mlx.sh`).
- [ ] Re-run `Scripts/benchmark.sh` across 100k, 1M, and 5M rows and verify `match: ok` checksum parity with `pandas` across all 4 execution policies (`native`, `cpu`, `gpu`, `gpu_f32`).

---

# Phase 9 — Close the Remaining Gaps vs Pandas (Latest `benchmark.sh` Results)

Goal: every workload reaches ≥ 1.0x vs pandas in all execution policies (`native`, `cpu`, `gpu`, `gpu_f32`) with `match: ok`. Per AGENTS.md §4, MLX stays the numeric engine where it wins; memory-bound or sequential kernels fall back to native Swift automatically. Public API signatures stay unchanged (AGENTS.md §2, §3, §5).

### 9.1 Baseline Diagnosis
| Workload | Rows | Result vs pandas | Root cause |
|---|---|---|---|
| `arithmetic_sqrt` | 1M / 5M | 0.74x–0.88x (all policies) | `multiply` → `add` → `sqrt` run as 3 passes, each allocating a `Float64Column` plus a `vDSP_sveD` finite-check pass in `unaryMathColumn`. Memory-bound; MLX adds no value. |
| `rolling_mean_20` | all | 0.49x–0.60x (MLX policies only) | `DataFrame.prefixSums(of:for:)` is routed through `NumericEngine.prefixSums(values:)`: boxes into `[Double?]`, two `map`s, two MLX `cumsum`s, copy back, `[Double]` → `[Int]` conversion. |
| `expanding_sum` | all | 0.54x–0.61x (MLX policies only) | Same MLX prefix-sum route, although the single-pass `runningColumn` already exists. |
| `stats_summary` | 100k–5M | 1.1x–1.6x (marginal) | Median uses a full sort; moments are computed in separate passes. |
| `rolling_mean_20` / `expanding_sum` | all | 1.4x / 2.5x (native) | Prefix arrays (`[Double]` + `[Int]`) allocated even without nulls. |

### 9.2 Step 1 — Native Scan Routing for `rolling` / `expanding` (`Bandal+Statistics.swift`, `Bandal+MLXBackend.swift`)
- [x] **Stop selecting MLX for sequential scans**:
  - In `DataFrame.rolling(_:window:minPeriods:using:)` and `DataFrame.expanding(_:using:)`, remove the `NumericEngine.prefixSums(values:)` branch for `.sum`, `.mean`, `.count`.
  - Delete `NumericEngine.prefixSums(values:)` and `nativePrefixSumMinimumCount` if no test or caller depends on them; otherwise keep it internal and unused by the automatic policy.
  - Keep MLX only for `rollingWindowNoNulls` (`.min`, `.max`, `.variance`, `.standardDeviation`), where strided reductions are not a sequential scan.
- [x] **Update doc comments** on `rolling` / `expanding` that currently state "Large inputs compute `sum`, `mean`, and `count` from MLX prefix sums".

### 9.3 Step 2 — Single-Pass Sliding `rolling` and `expanding` Kernels
- [x] **No-null fast path for `rolling(.sum / .mean / .count)`**:
  - One loop with a running sum/count: add `x[i]`, subtract `x[i - window]`; write directly into `ContiguousArray(unsafeUninitializedCapacity:)` through `withUnsafeBufferPointer`; no prefix arrays, no per-row bitmap checks.
  - Use Neumaier-compensated summation (or a periodic exact recompute every N rows) so results still match pandas checksums.
  - Rows before `minPeriods` are set in the null bitmap with word-level `BitSet` writes.
- [x] **Null-aware sliding path**: same single-pass accumulator that also tracks the valid count; replaces the prefix-array path so the `[Int]` counts array is never allocated.
- [x] **No-null fast path for `expanding(.sum / .mean / .count)`**: specialize `runningColumn` for `!hasNulls` (no `srcNulls[i]` reads, no min/max/first/last bookkeeping unless requested) with `unsafeUninitializedCapacity`; fall back to the existing generic loop when nulls are present or the result is non-finite.

### 9.4 Step 3 — Fused `arithmetic_sqrt` (`Bandal+Series.swift`, `Bandal+AdvancedAnalytics.swift`, `Bandal+MLXBackend.swift`)
- [x] **Single allocation for `a * b + c` → `sqrt`**:
  - For non-null, equal-length `Float64Column` operands use Accelerate (`vDSP_vmulD`, `vDSP_vaddD`, `vDSP_vmaD`) writing directly into the output buffer; evaluate `sqrt` in place with `vvsqrt` on the final buffer.
  - Consider an internal chained-expression fast path (no public API change) so intermediate `Series` results reuse their uniquely referenced buffer (`isKnownUniquelyReferenced`) and avoid new allocations.
- [x] **Remove redundant finite-check passes in `unaryMathColumn`**:
  - Replace the unconditional `vDSP_sveD` checksum with a domain-aware check (`sqrt`/`log` negative inputs → `.missing`) folded into the vector op, or run it only when the input is not already known finite.
  - Preserve semantics: domain violations (`sqrt(-1)`, `log(-1)`) must still become `.missing`, not `NaN`.
- [x] **Keep GPU `Float32` off the automatic path for light element-wise ops**: conversion cost (`Double` ↔ `Float`) dominates; only explicit `.gpu` / `.gpuFloat32` policies may use it.

### 9.5 Step 4 — Execution-Policy Audit (`Bandal+MLXBackend.swift`)
- [x] Review `MLXBackend.shouldUse(elementCount:minimumCount:)` call sites; ensure `.automatic` never selects MLX for scans (`cumsum`, `rolling`/`expanding` sum/mean/count) or light element-wise arithmetic.
- [x] Keep MLX for kernels where it wins (`stats_summary` at ≥ 1M, `correlation`, matmul, sort-assisted statistics).
- [ ] Document the selection rule next to `automaticMinimumCount` / `automaticComplexMinimumCount`.

### 9.6 Step 5 — `stats_summary` Small/Medium Row Counts (`Bandal+MLXBackend.swift`, `Bandal+Series.swift`)
Deferred: `stats_summary` already measures ≥ 1.03x vs pandas in every policy after Steps 1–3; revisit for further headroom.
- [ ] Complete the pending `NumericEngine.summaryMoments(of:)` item from Phase 8.3: single-pass `min`, `max`, `mean`, `standardDeviation` on `NumericBuffer`, Quickselect for `median`.
- [ ] Skip `allSatisfy(\.isFinite)` / `isExactFloat32` scans on pre-validated `Float64Column` buffers unless `.gpu` policy is active.

### 9.7 Step 6 — Tests
- [x] `StatisticsTests`: `rolling` and `expanding` (`sum`, `mean`, `count`) on ≥ 4,096 rows with and without nulls, `minPeriods` < `window`, and parity across `.native`, `.cpu`, `.gpu`, `.gpuFloat32` policies.
- [x] `StatisticsTests`: chained `multiply` → `add` → `sqrt` equals the unfused result, including negative inputs → `.missing`, and nulls propagate.
- [x] Regression test for compensated sliding sums (large magnitude differences, e.g. `1e12` followed by small values).

### 9.8 Step 7 — Verification, Benchmarks & Bilingual Documentation (AGENTS.md §6)
- [x] `swift test` and `Scripts/test-with-mlx.sh` pass.
- [x] Re-run `Scripts/benchmark.sh` (100k, 1M, 5M): all workloads ≥ 1.0x vs pandas, `match: ok`, MLX columns for `rolling_mean_20` / `expanding_sum` within ~10% of the Swift column.
- [x] Update `CHANGELOG.md`, `Documentation/API.md` ↔ `Documentation/API.ko.md`, and `README.md` ↔ `README.ko.md` (benchmark tables and execution-policy notes) in lockstep.
