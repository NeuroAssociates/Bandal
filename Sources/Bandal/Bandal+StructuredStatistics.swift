import Foundation

// MARK: - Internal numeric buffer

/// Internal numeric buffer that separates numeric values from validity/missing
/// masks, preserving row order.
///
/// This buffer is the bridge between Bandal's public mixed-value `DataValue`
/// model and the numeric kernels in `NumericEngine`. It is cached immutably on
/// `DataColumn` for columns containing only `.number` and `.missing` values.
struct NumericBuffer: Equatable, Hashable, Sendable {
    /// Underlying storage: one `Double` per row. Placeholder `0` is used for
    /// missing rows; callers must consult `isMissing` before reading.
    let values: [Double]
    /// Validity mask: `true` when the corresponding row is missing.
    let isMissing: [Bool]
    /// Pre-extracted non-missing numeric values in row order.
    let numbers: [Double]

    var count: Int { values.count }
    var nonMissingCount: Int { numbers.count }
    var hasMissing: Bool { numbers.count != values.count }

    var isExactFloat32: Bool {
        MLXBackend.isExactFloat32(numbers)
    }

    init(values: [Double], isMissing: [Bool]) {
        self.values = values
        self.isMissing = isMissing
        var compact: [Double] = []
        compact.reserveCapacity(values.count)
        for index in values.indices where !isMissing[index] {
            compact.append(values[index])
        }
        self.numbers = compact
    }

    init(numbers: [Double]) {
        self.values = numbers
        self.isMissing = Array(repeating: false, count: numbers.count)
        self.numbers = numbers
    }

    init?(dataValues: [DataValue]) {
        var values: [Double] = []
        var isMissing: [Bool] = []
        var compact: [Double] = []
        values.reserveCapacity(dataValues.count)
        isMissing.reserveCapacity(dataValues.count)
        compact.reserveCapacity(dataValues.count)

        for value in dataValues {
            switch value {
            case .number(let number):
                values.append(number)
                isMissing.append(false)
                compact.append(number)
            case .missing:
                values.append(0)
                isMissing.append(true)
            default:
                return nil
            }
        }
        self.values = values
        self.isMissing = isMissing
        self.numbers = compact
    }

    static func toDataValues(numbers: [Double], isMissing: [Bool]) -> [DataValue] {
        var result: [DataValue] = []
        result.reserveCapacity(numbers.count)
        for index in numbers.indices {
            if isMissing[index] {
                result.append(.missing)
            } else {
                let value = numbers[index]
                result.append(value.isFinite ? .number(value) : .missing)
            }
        }
        return result
    }
}

// MARK: - Public result types

/// Statistics for one numeric column.
///
/// Large columns are reduced through MLX in `float64` on the CPU stream;
/// small columns use native Swift. Both paths use `Double` precision, the
/// sample standard deviation (`n - 1`), and linear-interpolation quartiles.
public struct ColumnStatistics: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    public let count: Int
    public let mean: Double?
    public let standardDeviation: Double?
    public let median: Double?
    public let min: Double?
    public let max: Double?
    public let q1: Double?
    public let q3: Double?
    /// Additional requested percentiles keyed by formatted label (for example `"10%"` or `"90%"`).
    public let percentiles: [String: Double]

    /// Creates statistics from a numeric buffer.
    init(buffer: NumericBuffer, probabilities: [Double] = [0.25, 0.5, 0.75]) {
        let moments = NumericEngine.moments(of: buffer.numbers)
        self.count = moments.count
        self.mean = moments.mean
        self.standardDeviation = moments.standardDeviation
        self.median = moments.median
        self.min = moments.min
        self.max = moments.max
        self.q1 = moments.q1
        self.q3 = moments.q3
        if moments.sorted.isEmpty {
            self.percentiles = [:]
        } else {
            var map: [String: Double] = [:]
            for p in probabilities {
                let label = Self.percentileLabel(for: p)
                map[label] = NumericEngine.quantile(sorted: moments.sorted, probability: p)
            }
            self.percentiles = map
        }
    }

    static func percentileLabel(for probability: Double) -> String {
        let percentage = Swift.min(Swift.max(probability, 0), 1) * 100
        if percentage.rounded() == percentage {
            return "\(Int(percentage))%"
        }
        return "\(String(format: "%.4g", percentage))%"
    }
}

/// Summary for one non-numeric column (text, Boolean, date, or mixed).
public struct NonNumericSummary: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    public let count: Int
    public let unique: Int
    public let top: DataValue?
    public let freq: Int
}

/// Aggregated statistics for all numeric columns of a data frame.
public struct DataFrameStatistics: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    public let columns: [String: ColumnStatistics]
    public let rowCount: Int
    public let columnCount: Int

    public subscript(_ column: String) -> ColumnStatistics? {
        columns[column]
    }
}

/// Ordered per-column numeric summaries returned by ``DataFrame/summaryTable()``.
public struct DataFrameSummary: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    public let columnOrder: [String]
    public let summaries: [String: ColumnSummary]

    public subscript(_ column: String) -> ColumnSummary? {
        summaries[column]
    }

    public func describe() -> String {
        guard !columnOrder.isEmpty else { return "Empty DataFrameSummary" }
        let metricNames = ["count", "mean", "std", "min", "50%", "max"]
        let columns: [(header: String, values: [String])] = [
            (header: "", values: metricNames),
        ] + columnOrder.compactMap { name -> (header: String, values: [String])? in
            guard let s = summaries[name] else { return nil }
            return (
                header: name,
                values: [
                    String(s.count),
                    format(s.mean),
                    format(s.standardDeviation),
                    format(s.min),
                    format(s.median),
                    format(s.max),
                ]
            )
        }
        return renderAlignedTable(columns: columns)
    }

    public var description: String { describe() }
    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Ordered per-column missing-value counts returned by ``DataFrame/missingSummary()``.
public struct MissingCounts: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    public let columnOrder: [String]
    public let counts: [String: Int]
    public let rowCount: Int

    public subscript(_ column: String) -> Int? {
        counts[column]
    }

    public var totalMissing: Int {
        counts.values.reduce(0, +)
    }

    public func describe() -> String {
        let width = columnOrder.map(\.count).max() ?? 0
        var lines = columnOrder.map { name in
            let padded = name.padding(toLength: width, withPad: " ", startingAt: 0)
            return "\(padded)    \(counts[name] ?? 0)"
        }
        lines.append("Total missing: \(totalMissing) (rows: \(rowCount))")
        return lines.joined(separator: "\n")
    }

    public var description: String { describe() }
    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Ordered per-column data types returned by ``DataFrame/columnTypes()`` and ``DataFrame/dtypes``.
public struct ColumnTypes: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    public let columnOrder: [String]
    public let types: [String: DataType]

    public subscript(_ column: String) -> DataType? {
        types[column]
    }

    public func describe() -> String {
        let width = columnOrder.map(\.count).max() ?? 0
        var lines = columnOrder.map { name in
            let padded = name.padding(toLength: width, withPad: " ", startingAt: 0)
            return "\(padded)    \(types[name] ?? .unknown)"
        }
        lines.append("Columns: \(columnOrder.count)")
        return lines.joined(separator: "\n")
    }

    public var description: String { describe() }
    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Ordered per-column distinct non-missing counts returned by ``DataFrame/uniqueSummary()``.
public struct UniqueCounts: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    public let columnOrder: [String]
    public let counts: [String: Int]
    public let rowCount: Int

    public subscript(_ column: String) -> Int? {
        counts[column]
    }

    public func describe() -> String {
        let width = columnOrder.map(\.count).max() ?? 0
        var lines = columnOrder.map { name in
            let padded = name.padding(toLength: width, withPad: " ", startingAt: 0)
            return "\(padded)    \(counts[name] ?? 0)"
        }
        lines.append("Columns: \(columnOrder.count) (rows: \(rowCount))")
        return lines.joined(separator: "\n")
    }

    public var description: String { describe() }
    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Structured result of `describe()`, suitable for programmatic inspection
/// and text rendering via `describe()`.
///
/// Column order is preserved in `columnOrder` so that rendered tables are
/// deterministic regardless of dictionary iteration order.
public struct DataFrameDescription: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    public let columnOrder: [String]
    public let numeric: [String: ColumnStatistics]
    public let nonNumeric: [String: NonNumericSummary]
    public let rowCount: Int
    public let columnCount: Int
    public let percentileLabels: [String]

    public init(
        columnOrder: [String],
        numeric: [String: ColumnStatistics],
        nonNumeric: [String: NonNumericSummary],
        rowCount: Int,
        columnCount: Int,
        percentileLabels: [String] = ["25%", "50%", "75%"]
    ) {
        self.columnOrder = columnOrder
        self.numeric = numeric
        self.nonNumeric = nonNumeric
        self.rowCount = rowCount
        self.columnCount = columnCount
        self.percentileLabels = percentileLabels
    }
}

// MARK: - DataFrame extensions

extension DataFrame {
    /// Returns structured statistics for all numeric columns, using MLX
    /// (`float64`, CPU stream) for large columns and native Swift otherwise.
    ///
    /// Columns containing non-numeric values are excluded. Missing values are
    /// ignored in every reduction.
    public func statistics() -> DataFrameStatistics {
        var result: [String: ColumnStatistics] = [:]
        for column in columns {
            guard let buffer = column.numericBuffer, buffer.nonMissingCount > 0 else {
                continue
            }
            result[column.name] = ColumnStatistics(buffer: buffer)
        }
        return DataFrameStatistics(
            columns: result,
            rowCount: rowCount,
            columnCount: columnNames.count
        )
    }

    /// Returns a structured descriptive-statistics result covering both numeric
    /// and non-numeric columns.
    ///
    /// Use `describe()` on the returned instance to render a pandas-style text
    /// table identical to `DataFrame.describe()`.
    public func describeStructured(
        percentiles: [Double] = [0.25, 0.5, 0.75],
        include: Set<DataType>? = nil
    ) -> DataFrameDescription {
        var numeric: [String: ColumnStatistics] = [:]
        var nonNumeric: [String: NonNumericSummary] = [:]
        var includedOrder: [String] = []
        let labels = percentiles.map(ColumnStatistics.percentileLabel(for:))

        for column in columns {
            let inferred = (try? dataType(of: column.name)) ?? .unknown
            if let include, !include.contains(inferred) {
                continue
            }
            includedOrder.append(column.name)

            if let buffer = column.numericBuffer {
                guard buffer.nonMissingCount > 0 else {
                    nonNumeric[column.name] = NonNumericSummary(
                        count: 0, unique: 0, top: nil, freq: 0
                    )
                    continue
                }
                numeric[column.name] = ColumnStatistics(
                    buffer: buffer,
                    probabilities: percentiles
                )
            } else {
                let nonMissingValues = column.values.filter { $0 != .missing }
                let frequencies = nonMissingValues.reduce(into: [DataValue: Int]()) { counts, value in
                    counts[value, default: 0] += 1
                }
                let highestFrequency = frequencies.values.max() ?? 0
                let top = nonMissingValues.first { frequencies[$0] == highestFrequency }
                nonNumeric[column.name] = NonNumericSummary(
                    count: nonMissingValues.count,
                    unique: frequencies.count,
                    top: top,
                    freq: highestFrequency
                )
            }
        }

        return DataFrameDescription(
            columnOrder: includedOrder,
            numeric: numeric,
            nonNumeric: nonNumeric,
            rowCount: rowCount,
            columnCount: includedOrder.count,
            percentileLabels: labels
        )
    }

    /// Returns a structured, ordered summary table for all numeric columns.
    public func summaryTable() -> DataFrameSummary {
        let map = summary()
        let order = columnNames.filter { map[$0] != nil }
        return DataFrameSummary(columnOrder: order, summaries: map)
    }

    /// Returns a structured, ordered missing-count summary for all columns.
    public func missingSummary() -> MissingCounts {
        MissingCounts(
            columnOrder: columnNames,
            counts: missingCounts(),
            rowCount: rowCount
        )
    }

    /// Returns a structured, ordered distinct-value count summary for all columns.
    public func uniqueSummary() -> UniqueCounts {
        UniqueCounts(
            columnOrder: columnNames,
            counts: uniqueCounts(),
            rowCount: rowCount
        )
    }

    /// Ordered inferred data types for every column in the frame.
    public var dtypes: ColumnTypes { columnTypes() }

    /// Returns an ordered struct of inferred column types.
    public func columnTypes() -> ColumnTypes {
        var map: [String: DataType] = [:]
        for name in columnNames {
            map[name] = (try? dataType(of: name)) ?? .unknown
        }
        return ColumnTypes(columnOrder: columnNames, types: map)
    }

    /// Returns a numeric buffer for the named column, or throws if the column
    /// contains non-numeric values.
    func numericBuffer(in name: String) throws -> NumericBuffer {
        guard let column = columns.first(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        guard let buffer = column.numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        return buffer
    }
}

private func format(_ value: Double?) -> String {
    value.map { String(format: "%.6g", $0) } ?? "NaN"
}

func renderAlignedTable(columns: [(header: String, values: [String])]) -> String {
    let rowCount = columns.first?.values.count ?? 0
    let widths = columns.map { column in
        max(column.header.count, column.values.map(\.count).max() ?? 0)
    }
    var lines = [renderAlignedRow(columns.map(\.header), widths: widths)]
    for rowIndex in 0..<rowCount {
        lines.append(
            renderAlignedRow(
                columns.map { $0.values[rowIndex] },
                widths: widths
            )
        )
    }
    return lines.joined(separator: "\n")
}

private func renderAlignedRow(_ cells: [String], widths: [Int]) -> String {
    cells.enumerated().map { index, value in
        value.padding(toLength: widths[index], withPad: " ", startingAt: 0)
    }.joined(separator: "  ")
}

extension ColumnStatistics {
    public var description: String {
        "count \(count), mean \(format(mean)), std \(format(standardDeviation)), "
            + "min \(format(min)), 25% \(format(q1)), 50% \(format(median)), "
            + "75% \(format(q3)), max \(format(max))"
    }

    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

extension NonNumericSummary {
    public var description: String {
        "count \(count), unique \(unique), top \(top.map(String.init(describing:)) ?? "NaN"), freq \(freq)"
    }

    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

extension DataFrameStatistics {
    public var description: String {
        var lines = ["DataFrameStatistics: \(rowCount) rows x \(columnCount) columns"]
        for name in columns.keys.sorted() {
            lines.append("\(name): \(columns[name]!)")
        }
        return lines.joined(separator: "\n")
    }

    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

extension DataFrameDescription {
    public var description: String { describe() }
    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}
