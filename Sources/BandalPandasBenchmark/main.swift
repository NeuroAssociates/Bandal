//
//  main.swift
//  BandalPandasBenchmark
//
//  Created by Ned Park on 11/6/26.
//
//  Bandal side of the pandas comparison (see benchmark.sh / benchmark.py).
//  Every workload mirrors benchmark.py. Only the computation is timed: data
//  generation, warm-up runs (MLX kernel compilation) and checksum evaluation
//  are excluded. Results are printed as JSON.
//
//  Usage: BandalPandasBenchmark [--rows 100000,1000000,5000000] [--repeat 5] [--warmup 2]
//

import Bandal
import Foundation

/// Command-line options controlling benchmark dataset sizes, repetition count, and warm-up iterations.
private struct Options {
    /// Row counts to benchmark.
    var rows = [100_000, 1_000_000, 5_000_000]
    /// Number of timed repetitions per workload.
    var repeats = 5
    /// Number of untimed warm-up runs before measurement.
    var warmup = 2

    /// Parses command-line arguments into benchmark options.
    ///
    /// - Parameter arguments: Full `CommandLine.arguments` array.
    /// - Throws: ``OptionError`` if a flag is unknown, missing a value, or invalid.
    init(arguments: [String]) throws {
        var iterator = arguments.dropFirst().makeIterator()
        while let flag = iterator.next() {
            guard let value = iterator.next() else { throw OptionError.missingValue(flag) }
            switch flag {
            case "--rows":
                let parsed = value.split(separator: ",").compactMap { Int($0) }
                guard !parsed.isEmpty, parsed.allSatisfy({ $0 > 0 }) else { throw OptionError.invalid(flag) }
                rows = parsed
            case "--repeat":
                guard let parsed = Int(value), parsed > 0 else { throw OptionError.invalid(flag) }
                repeats = parsed
            case "--warmup":
                guard let parsed = Int(value), parsed >= 0 else { throw OptionError.invalid(flag) }
                warmup = parsed
            default:
                throw OptionError.invalid(flag)
            }
        }
    }
}

/// Errors thrown when parsing command-line arguments for `BandalPandasBenchmark`.
private enum OptionError: Error {
    /// A flag requiring a value was provided without an argument.
    case missingValue(String)
    /// A flag or its value could not be parsed.
    case invalid(String)
}

/// A single timed workload measurement across repetitions.
private struct Measurement {
    /// Number of rows in the benchmarked dataset.
    let rows: Int
    /// Identifier of the benchmark workload.
    let workload: String
    /// Execution mode label (`native`, `mlx_cpu`, `mlx_gpu`, or `mlx_gpu_float32`).
    let mode: String
    /// Median execution time in milliseconds.
    let medianMilliseconds: Double
    /// Minimum execution time in milliseconds.
    let minimumMilliseconds: Double
    /// Numerical checksum used to verify parity across backends and pandas.
    let checksum: Double
}

/// Execution policies benchmarked for each workload.
private let modes: [(name: String, policy: ExecutionPolicy)] = [
    ("native", .native),
    ("mlx_cpu", .cpu),
    ("mlx_gpu", .gpu),
    ("mlx_gpu_float32", .gpuFloat32),
]

/// Monotonic clock used to time benchmark iterations.
private let clock = ContinuousClock()

/// Converts a `Duration` to milliseconds as a `Double`.
///
/// - Parameter duration: Elapsed duration to convert.
/// - Returns: Duration expressed in milliseconds.
private func milliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1_000
        + Double(duration.components.attoseconds) / 1_000_000_000_000_000
}

/// Times only `operation`; `checksum` runs after the clock stops.
///
/// - Parameters:
///   - rows: Number of rows in the dataset.
///   - workload: Name of the workload being measured.
///   - mode: Execution mode name and policy.
///   - options: Benchmark repetition and warm-up configuration.
///   - operation: Closure performing the timed workload.
///   - checksum: Closure computing a verification checksum from the result.
/// - Returns: A ``Measurement`` summarizing the timing and checksum.
private func measure<Result>(
    rows: Int,
    workload: String,
    mode: (name: String, policy: ExecutionPolicy),
    options: Options,
    operation: () throws -> Result,
    checksum: (Result) throws -> Double
) throws -> Measurement {
    try BandalRuntime.withExecutionPolicy(mode.policy) {
        for _ in 0..<options.warmup {
            _ = try operation()
        }
        var durations: [Double] = []
        var checksumValue = Double.nan
        for _ in 0..<options.repeats {
            let start = clock.now
            let result = try operation()
            let elapsed = start.duration(to: clock.now)
            durations.append(milliseconds(elapsed))
            checksumValue = try checksum(result)
        }
        durations.sort()
        return Measurement(
            rows: rows,
            workload: workload,
            mode: mode.name,
            medianMilliseconds: durations[durations.count / 2],
            minimumMilliseconds: durations[0],
            checksum: checksumValue
        )
    }
}

/// Computes the sum of an array of `Double` values.
///
/// - Parameter values: Numeric values to sum.
/// - Returns: Sum of all elements.
private func total(_ values: [Double]) -> Double {
    values.reduce(0, +)
}

/// Sums the numeric values in the last column of `frame`.
///
/// - Parameter frame: The data frame whose last column is summed.
/// - Returns: The sum of the last column, or `.nan` if the frame has no columns.
private func lastColumnTotal(_ frame: DataFrame) throws -> Double {
    guard let name = frame.columnNames.last else { return .nan }
    return total(try frame.series(name).numbers)
}

/// Deterministic integer-valued data (exact in `Float32`), identical to `benchmark.py`.
///
/// - Parameter rows: Number of rows to generate.
/// - Returns: A tuple containing the primary data frame and the 100-row lookup table.
private func makeData(rows: Int) throws -> (frame: DataFrame, lookup: DataFrame) {
    let a = (0..<rows).map { Double(($0 * 7_919) % 10_007) }
    let b = (0..<rows).map { Double(($0 * 104_729) % 10_009 + 1) }
    let c = (0..<rows).map { Double($0 % 1_000) }
    let d = (0..<rows).map { Double(($0 * 31) % 9_973) }
    let group = (0..<rows).map { DataValue.string("g\($0 % 100)") }
    let frame = try DataFrame(columns: [
        (name: "group", values: group),
        (name: "a", values: a.map(DataValue.number)),
        (name: "b", values: b.map(DataValue.number)),
        (name: "c", values: c.map(DataValue.number)),
        (name: "d", values: d.map(DataValue.number)),
    ])
    let lookup = try DataFrame(columns: [
        (name: "group", values: (0..<100).map { DataValue.string("g\($0)") }),
        (name: "category", values: (0..<100).map { DataValue.number(Double($0)) }),
    ])
    return (frame, lookup)
}

/// Runs all benchmark workloads for a given row count across all execution modes.
///
/// - Parameters:
///   - rows: Number of rows to generate and benchmark.
///   - options: Warm-up and repetition options.
/// - Returns: Collected measurements for every workload and execution mode.
private func runWorkloads(rows: Int, options: Options) throws -> [Measurement] {
    let (frame, lookup) = try makeData(rows: rows)
    let numeric = try frame.select("a", "b", "c", "d")
    let a = try frame.series("a")
    let b = try frame.series("b")
    let c = try frame.series("c")
    var results: [Measurement] = []

    for mode in modes {
        func run<Result>(
            _ workload: String,
            operation: () throws -> Result,
            checksum: (Result) throws -> Double
        ) throws {
            results.append(
                try measure(
                    rows: rows,
                    workload: workload,
                    mode: mode,
                    options: options,
                    operation: operation,
                    checksum: checksum
                )
            )
        }

        try run("stats_summary") {
            a.summary()
        } checksum: { summary in
            guard let summary, let mean = summary.mean, let std = summary.standardDeviation,
                  let median = summary.median, let minimum = summary.min, let maximum = summary.max
            else { return .nan }
            return mean + std + median + minimum + maximum
        }
        try run("sort") {
            try frame.sorted(by: [SortDescriptor("a")])
        } checksum: { sorted in
            let values = try sorted.series("a").numbers
            return values[0] + values[values.count / 2] + values[values.count - 1]
        }
        try run("rolling_mean_20") {
            try numeric.rolling("a", window: 20, using: .mean)
        } checksum: {
            try lastColumnTotal($0)
        }
        try run("expanding_sum") {
            try numeric.expanding("a", using: .sum)
        } checksum: {
            try lastColumnTotal($0)
        }
        try run("arithmetic_sqrt") {
            try a.applying(.multiply, other: b).applying(.add, other: c).applying(.sqrt)
        } checksum: {
            total($0.numbers)
        }
        try run("correlation") {
            try numeric.correlation()
        } checksum: { matrix in
            try matrix.columnNames.reduce(0) { $0 + total(try matrix.series($1).numbers) }
        }
        try run("filter") {
            try frame.query("a >= 5000")
        } checksum: {
            Double($0.rowCount) + total(try $0.series("b").numbers)
        }
        try run("groupby_sum") {
            try frame.groupBy("group").aggregate("c", using: .sum)
        } checksum: {
            Double($0.rowCount) + (try lastColumnTotal($0))
        }
        try run("left_join") {
            try frame.join(lookup, on: "group", kind: .left)
        } checksum: {
            Double($0.rowCount) + total(try $0.series("category").numbers)
        }
    }
    return results
}

/// Formats benchmark measurements as a JSON string.
///
/// - Parameters:
///   - measurements: Measurements to serialize.
///   - backend: Description of the active execution backend.
/// - Returns: Formatted JSON payload.
private func json(_ measurements: [Measurement], backend: String) -> String {
    let entries = measurements.map { m in
        "    {\"rows\": \(m.rows), \"workload\": \"\(m.workload)\", \"mode\": \"\(m.mode)\", "
            + "\"median_ms\": \(m.medianMilliseconds), \"min_ms\": \(m.minimumMilliseconds), "
            + "\"checksum\": \(m.checksum.isFinite ? "\(m.checksum)" : "null")}"
    }
    return "{\n  \"library\": \"bandal\",\n  \"backend\": \"\(backend)\",\n  \"results\": [\n"
        + entries.joined(separator: ",\n") + "\n  ]\n}"
}

do {
    let options = try Options(arguments: CommandLine.arguments)
    var all: [Measurement] = []
    for rows in options.rows {
        FileHandle.standardError.write(Data("bandal: \(rows) rows\n".utf8))
        all += try runWorkloads(rows: rows, options: options)
    }
    print(json(all, backend: BandalRuntime.executionBackend))
} catch {
    FileHandle.standardError.write(Data("Benchmark failed: \(error)\n".utf8))
    exit(1)
}
