# Agent Guidelines & Project Requirements

The project must be structured and maintained to satisfy the following requirements:

## 1. Distribution as a Swift Package
- The Bandal library is distributed as a standalone Swift Package.

## 2. Idiomatic Swift Implementation of Pandas
- Model the capabilities and design of Pandas as idiomatically Swift as possible:
  https://github.com/pandas-dev/pandas
- Maintain type safety, value semantics, and ergonomic Swift conventions throughout the library.

## 3. Swift API Design Guidelines Compliance
- All API design must strictly adhere to Apple's official guidelines:
  https://www.swift.org/documentation/api-design-guidelines/
- Ensure clear naming, explicit argument labels, and predictable behavior across all types and methods.

## 4. Accelerated Numerical Engine via MLX-Swift
- Leverage MLX-Swift to the maximum extent as the numerical computing engine:
  https://github.com/ml-explore/mlx-swift
  https://github.com/numpy/numpy
- Just as Pandas relies on NumPy for vectorized performance and numeric foundational routines, Bandal utilizes MLX-Swift.
- Provide automatic fallbacks to native Swift operations where MLX does not support specific types/devices or where native execution is more efficient.

## 5. Structured Swift Result Types & Printable Descriptions
- All API responses must return structured Swift instances (`struct`, typed collections, etc.) so callers can inspect and process results programmatically.
- Conforming to protocols such as `CustomStringConvertible`, result instances provide print-friendly, pandas-style tabular text representations via `description` and `.describe()`.

## 6. Bilingual Documentation Requirements
- Maintain developer documentation in `Documentation/API.md` (in English), listing all public APIs with runnable Swift code examples.
- Maintain `Documentation/API.ko.md` as a faithful, mirror Korean translation of `Documentation/API.md`.
- Maintain the topic-by-topic self-study guides in `Documentation/Guides/` in both English (`*.md`) and Korean (`*.ko.md`):
  1. `01-Core-Data-Structures.md` / `.ko.md` — Core types (`DataValue`, `DataType`, `Series`, `DataFrame`, `DataRow`, `pipe`, `DataFrameError`)
  2. `02-Data-IO-and-Inspection.md` / `.ko.md` — CSV/JSON I/O, schema inspection (`infoSummary`, `dtypes`), rendering (`RenderOptions`, `toMarkdown`)
  3. `03-Indexing-Filtering-and-Querying.md` / `.ko.md` — Column/row/cell selection, `BooleanMask`, `RowPredicate`, string `query`
  4. `04-Cleaning-and-Feature-Engineering.md` / `.ko.md` — Missing values, deduplication, sorting, sampling, string/date ops, scaling, binning, one-hot encoding
  5. `05-Vectorized-Math-and-Statistics.md` / `.ko.md` — Vectorized arithmetic, `UnaryMathFunction`, cumulative/lag/diff/rank, structured statistics, covariance/correlation
  6. `06-Grouping-Joining-and-Reshaping.md` / `.ko.md` — `groupBy` (`aggregate`, `transform`, `filter`), `join`/`leftJoin`, `concatenatingRows`, `melt`, `pivoted`, `pivotTable`, `crossTabulated`, `exploded`, `transposed`
  7. `07-Time-Series-and-Window-Functions.md` / `.ko.md` — Row/time `rolling`, `expanding`, `exponentiallyWeighted`, `joinAsOf`, `resample`
  8. `08-MLX-Acceleration-and-Workflows.md` / `.ko.md` — Columnar architecture, `ExecutionPolicy`, `BandalRuntime`, `MLXVector`/`MLXArray` interop, end-to-end workflows
- Whenever any public API, type, or behavior is added or modified, update the relevant guide(s) in `Documentation/Guides/` (both `.md` and `.ko.md` in lockstep) with side-by-side `pandas` comparisons, runnable Swift examples, and console outputs, in addition to updating `Documentation/API.md`, `Documentation/API.ko.md`, `README.md`, and `README.ko.md`.
- Maintain `README.md` in English as a comprehensive guide suitable for GitHub open-source release.
- Maintain `README.ko.md` as an exact Korean translation of `README.md`.
