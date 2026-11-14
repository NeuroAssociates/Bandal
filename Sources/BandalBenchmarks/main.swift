import Bandal
import Darwin
import Foundation

/// Number of timed repetitions executed for each benchmark operation.
private let repetitions = 3

/// Runs `operation` `repetitions` times, verifies checksum stability, and prints median duration and peak RSS.
///
/// - Parameters:
///   - name: Benchmark operation label.
///   - rows: Number of rows in the dataset.
///   - operation: Closure performing the workload and returning a deterministic checksum.
/// - Throws: ``BenchmarkError`` if the checksum is non-finite or varies across repetitions.
private func benchmark(
    _ name: String,
    rows: Int,
    operation: () throws -> Double
) throws {
    var durations: [Double] = []
    var checksumReference: Double?
    for _ in 0..<repetitions {
        let start = ProcessInfo.processInfo.systemUptime
        let checksum = try operation()
        guard checksum.isFinite else {
            throw BenchmarkError.nonFiniteChecksum(operation: name)
        }
        if let firstChecksum = checksumReference,
           abs(checksum - firstChecksum) > max(abs(firstChecksum), 1) * 1e-12
        {
            throw BenchmarkError.unstableChecksum(operation: name)
        }
        durations.append((ProcessInfo.processInfo.systemUptime - start) * 1_000)
        if checksumReference == nil {
            checksumReference = checksum
        }
    }

    let median = durations.sorted()[durations.count / 2]
    let peakMemoryMB = peakResidentMemoryMB()
    let checksum = checksumReference ?? .nan
    print(
        "\(name), rows=\(rows), median_ms=\(String(format: "%.2f", median)), "
            + "process_peak_rss_mb=\(String(format: "%.1f", peakMemoryMB)), "
            + "checksum=\(String(format: "%.6f", checksum))"
    )
}

/// Errors thrown during benchmark verification.
private enum BenchmarkError: Error {
    /// The benchmark produced a `NaN` or infinite checksum.
    case nonFiniteChecksum(operation: String)
    /// The benchmark produced inconsistent checksums across repetitions.
    case unstableChecksum(operation: String)
}

/// Computes a scalar checksum from a ``ColumnSummary``.
///
/// - Parameter summary: Summary to reduce.
/// - Returns: Sum of mean, standard deviation, median, minimum, and maximum, or `.nan` if unavailable.
private func summaryChecksum(_ summary: ColumnSummary?) -> Double {
    guard let summary,
          let mean = summary.mean,
          let standardDeviation = summary.standardDeviation,
          let median = summary.median,
          let minimum = summary.min,
          let maximum = summary.max
    else {
        return .nan
    }
    return mean + standardDeviation + median + minimum + maximum
}

/// Sums all `.number` elements in an array of ``DataValue``s.
///
/// - Parameter values: Values to accumulate.
/// - Returns: Sum of numeric cells.
private func numberTotal(_ values: [DataValue]) -> Double {
    values.reduce(0) { total, value in
        if case .number(let number) = value { total + number } else { total }
    }
}

/// Queries the current process peak resident set size (RSS) in megabytes.
///
/// - Returns: Peak RSS in megabytes, or `.nan` if `getrusage` fails.
private func peakResidentMemoryMB() -> Double {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return .nan }
    return Double(usage.ru_maxrss) / 1_048_576
}

/// Generates a synthetic ``DataFrame`` with `rowCount` rows for benchmarking.
///
/// - Parameter rowCount: Number of rows to generate.
/// - Returns: A three-column ``DataFrame`` (`group`, `value`, `id`).
private func makeFrame(rowCount: Int) throws -> DataFrame {
    try DataFrame(columns: [
        (
            name: "group",
            values: (0..<rowCount).map { .string("group-\($0 % 100)") }
        ),
        (
            name: "value",
            values: (0..<rowCount).map { .number(Double($0 % 1_000)) }
        ),
        (
            name: "id",
            values: (0..<rowCount).map { .number(Double($0)) }
        ),
    ])
}

/// Writes a temporary CSV file with `rowCount` rows for CSV I/O benchmarking.
///
/// - Parameter rowCount: Number of rows to write.
/// - Returns: File URL of the temporary CSV file.
private func makeCSV(rowCount: Int) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("bandal-benchmark-\(UUID().uuidString).csv")
    var contents = "group,value,id\n"
    for row in 0..<rowCount {
        contents += "group-\(row % 100),\(row % 1_000),\(row)\n"
    }
    try contents.write(to: url, atomically: true, encoding: .utf8)
    return url
}

/// Runs the full suite of benchmarks for `rowCount` rows.
///
/// - Parameter rowCount: Number of rows in the benchmark dataset.
private func runBenchmarks(rowCount: Int) throws {
    let data = try makeFrame(rowCount: rowCount)
    let lookup = try DataFrame(columns: [
        (
            name: "group",
            values: (0..<100).map { .string("group-\($0)") }
        ),
        (
            name: "category",
            values: (0..<100).map { .number(Double($0)) }
        ),
    ])
    let numericValues = try data.column("value")
    let statisticsData = try DataFrame(columns: ["value": numericValues])

    print("backend=\(BandalRuntime.executionBackend)")
    print("dataset rows=\(rowCount), columns=\(data.columnNames.count)")
    let csvURL = try makeCSV(rowCount: rowCount)
    defer { try? FileManager.default.removeItem(at: csvURL) }
    try benchmark("csv_read", rows: rowCount) {
        Double(try DataFrame(contentsOfCSV: csvURL).rowCount)
    }
    try benchmark("filter", rows: rowCount) {
        Double(try data.filterRows { row in
            guard case .number(let value) = row["value"] else { return false }
            return value >= 500
        }.rowCount)
    }
    try benchmark("group_aggregate", rows: rowCount) {
        Double(try data.groupBy("group").aggregate("value", using: .sum).rowCount)
    }
    try benchmark("left_join", rows: rowCount) {
        Double(try data.join(lookup, on: "group", kind: .left).rowCount)
    }
    try benchmark("describe", rows: rowCount) {
        Double(data.describe().utf8.count)
    }
    try benchmark("summary", rows: rowCount) {
        summaryChecksum(statisticsData.summary()["value"])
    }
    try benchmark("rolling_mean", rows: rowCount) {
        let result = try statisticsData.rolling("value", window: 20, using: .mean)
        return try numberTotal(result.values(in: result.columnNames.last ?? "value"))
    }
    try benchmark("expanding_sum", rows: rowCount) {
        let result = try statisticsData.expanding("value", using: .sum)
        return try numberTotal(result.values(in: result.columnNames.last ?? "value"))
    }
}

do {
    let requestedSizes = CommandLine.arguments.dropFirst().compactMap(Int.init)
    let rowCounts = requestedSizes.isEmpty ? [1_000, 10_000, 100_000] : requestedSizes
    guard rowCounts.allSatisfy({ $0 > 0 }) else {
        fputs("Row counts must be positive integers.\n", stderr)
        exit(EXIT_FAILURE)
    }
    for rowCount in rowCounts {
        try runBenchmarks(rowCount: rowCount)
    }
} catch {
    fputs("Benchmark failed: \(error)\n", stderr)
    exit(EXIT_FAILURE)
}
