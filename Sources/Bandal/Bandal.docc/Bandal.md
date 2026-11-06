# ``Bandal``

Pandas-style tabular data analysis for Swift, accelerated by MLX.

## Overview

Bandal provides immutable, `Sendable` value types — ``DataFrame``, ``Series`` and ``DataRow`` — with CSV and JSON I/O, querying, grouping, joins, reshaping, window functions and statistics. Large numeric workloads run on MLX Swift and fall back to native Swift automatically.

Every result type conforms to `CustomStringConvertible` and prints a pandas-style table.

## Topics

### Essentials

- ``DataFrame``
- ``Series``
- ``DataRow``
- ``DataValue``
