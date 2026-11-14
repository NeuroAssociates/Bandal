# Guide 04: Data Cleaning, Missing Values & Feature Engineering

Diagnose and impute missing values, deduplicate and sort records, extract string and calendar features, and preprocess ML features with standardization, binning, and one-hot encoding.

## Overview

Before statistical modeling or machine learning, raw datasets require cleaning and feature engineering: diagnosing and imputing missing values, removing duplicate records, standardizing strings, extracting calendar features from timestamps, scaling numerical variables, discretizing continuous columns into bins, and one-hot encoding categorical columns.

This guide covers all of Bandal's data-wrangling and feature-engineering APIs:
- **Missing Value Diagnostics & Imputation** (``DataFrame/missingSummary()``, ``MissingCounts``, ``DataFrame/fillMissing(in:with:)``, ``DataFrame/forwardFilled()``, ``DataFrame/backwardFilled()``, ``DataFrame/interpolated(_:)``, ``DataFrame/dropMissing(in:)``, ``DataFrame/replacing(_:with:in:)``, ``Series/where(_:otherwise:)``)
- **Deduplication, Sorting & Reproducible Sampling** (``DataFrame/duplicatedRows(in:keep:)``, ``DataFrame/dropDuplicates(in:keep:)``, ``DataFrame/sorted(by:)``, ``DataFrame/sample(_:seed:)``)
- **Adding, Mapping, Renaming & Casting Columns** (``DataFrame/withColumn(name:computing:)``, ``DataFrame/mapped(_:_:)``, ``DataFrame/renamedColumns(_:)``, ``DataFrame/castColumn(_:to:dateFormat:)``)
- **Vectorized String & Date Feature Extraction** (``DataFrame/lowercased(_:as:)``, ``DataFrame/uppercased(_:as:)``, ``DataFrame/trimmed(_:as:)``, ``DataFrame/stringLength(_:as:)``, ``DataFrame/replacingSubstring(in:target:with:as:)``, ``DataFrame/containsSubstring(_:_:caseSensitive:as:)``, ``DataFrame/dateComponent(_:from:as:)``)
- **ML Feature Engineering** (``DataFrame/standardized(_:as:)``, ``DataFrame/minMaxScaled(_:as:)``, ``DataFrame/binned(_:edges:labels:rightClosed:includeLowest:as:)``, ``DataFrame/quantileBinned(_:quantiles:labels:as:)``, ``DataFrame/oneHotEncoded(_:dropFirst:)``)

---

## 1. Diagnosing & Handling Missing Values

``DataFrame/missingSummary()`` returns a structured ``MissingCounts`` instance that prints an aligned summary table while providing subscript access (`missing["price"]`) and `totalMissing`:

```swift
import Bandal

let sensorData = try DataFrame(columns: [
    (name: "hour", values: [1, 2, 3, 4, 5]),
    (name: "temp", values: [20.0, nil, nil, 26.0, 28.0]),
    (name: "status", values: ["OK", "WARN", nil, "OK", "ERR"]),
])

let missing: MissingCounts = sensorData.missingSummary()
print(missing)

// 1. Constant fill on specific column(s)
let filledStatus = try sensorData.fillMissing(in: ["status"], with: "UNKNOWN")

// 2. Forward fill across the frame
let ffilled = try sensorData.forwardFilled()

// 3. Linear interpolation on 'temp' (20.0 -> [22.0, 24.0] -> 26.0 -> 28.0)
let interpolated = try sensorData.interpolated("temp", as: "temp_linear")
```

---

## 2. Deduplication, Sorting & Sampling

``DuplicateKeep`` controls which occurrence of a duplicate key is preserved (`.first`, `.last`, or `.none`):

```swift
let logs = try DataFrame(columns: [
    (name: "user", values: ["u1", "u2", "u1", "u3", "u2"]),
    (name: "event", values: ["login", "view", "login", "buy", "buy"]),
    (name: "seq", values: [1, 2, 3, 4, 5]),
])

// Deduplicate by "user", keeping only each user's latest event (.last)
let latestPerUser = try logs.dropDuplicates(in: ["user"], keep: .last)

// Multi-column sort
let sortedLogs = try logs.sorted(by: [
    SortDescriptor("user", order: .ascending),
    SortDescriptor("seq", order: .descending),
])

// Reproducible random sampling
let threeRows = try logs.sample(3, seed: 2026)
let fortyPercent = try logs.sample(fraction: 0.4, seed: 2026)
```

---

## 3. Adding, Mapping, Renaming & Casting Columns

```swift
let rawProducts = try DataFrame(columns: [
    (name: "sku_code", values: [" ab-10 ", " cd-20 ", " ef-30 "]),
    (name: "raw_price", values: ["$120.00", "$250.50", "$80.00"]),
])

let transformed = try rawProducts
    .renamedColumns(["sku_code": "sku", "raw_price": "price"])
    .castColumn("price", to: .number)
    .mapped("price") { val in
        guard let p = val.numberValue else { return .missing }
        return .number(p * 1.1)
    }
    .withColumn(name: "tier") { row in
        guard let p = row["price"]?.numberValue else { return .missing }
        return p >= 200.0 ? "Premium" : "Standard"
    }
```

---

## 4. Feature Scaling, Discretization (Binning) & One-Hot Encoding

```swift
let scores = try DataFrame(columns: [
    (name: "city", values: ["Seoul", "Busan", "Incheon", "Seoul"]),
    (name: "score", values: [50.0, 70.0, 80.0, 100.0]),
])
.standardized("score", as: "score_z")
.minMaxScaled("score", to: 0.0...1.0, as: "score_minmax")
.binned(
    "score",
    edges: [0.0, 60.0, 85.0, 100.0],
    labels: ["Low", "Mid", "High"],
    rightClosed: true,
    as: "grade"
)
.oneHotEncoded("city", prefix: "city", dropFirst: false, dropOriginal: true)
```

> Next Step: Move on to <doc:05-Vectorized-Math-and-Statistics> to explore element-wise arithmetic, mathematical functions, cumulative reductions, and Bandal's structured descriptive statistics APIs.
