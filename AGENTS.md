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
- Maintain `README.md` in English as a comprehensive guide suitable for GitHub open-source release.
- Maintain `README.ko.md` as an exact Korean translation of `README.md`.
