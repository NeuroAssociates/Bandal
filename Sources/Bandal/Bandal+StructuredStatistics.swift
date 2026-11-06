import Foundation

// MARK: - Internal numeric buffer

/// Internal numeric buffer that separates numeric values from validity/missing
/// masks, preserving row order.
///
/// This buffer is the bridge between Bandal's public mixed-value `DataValue`
/// model and the numeric kernels in `NumericEngine`. It is cached immutably on
/// `DataColumn` for columns containing only `.number` and `.missing` values.
struct NumericBuffer: Equatable, Hashable, Sendable {
    /// Underlying contiguous storage: one `Double` per row. Placeholder `0` is
    /// used for missing rows; callers must consult `nullBitmap` / `isMissing`.
    let floatColumn: Float64Column
    let nonMissingCount: Int
    let hasMissing: Bool
    private let _cachedNumbers: [Double]?

    /// Non-missing numeric values in row order. Materialized lazily when not pre-cached.
    var numbers: [Double] {
        if let cached = _cachedNumbers {
            return cached
        }
        if !hasMissing {
            return Array(floatColumn.values)
        }
        var compact: [Double] = []
        compact.reserveCapacity(nonMissingCount)
        let vals = floatColumn.values
        let bm = floatColumn.nullBitmap
        for i in 0..<vals.count where !bm[i] {
            compact.append(vals[i])
        }
        return compact
    }

    var values: [Double] { Array(floatColumn.values) }
    var nullBitmap: BitSet { floatColumn.nullBitmap }
    var isMissing: [Bool] {
        let n = floatColumn.count
        let bm = floatColumn.nullBitmap
        if bm.allFalse {
            return Array(repeating: false, count: n)
        }
        return (0..<n).map { bm[$0] }
    }

    var count: Int { floatColumn.count }

    var isExactFloat32: Bool {
        if !hasMissing {
            return MLXBackend.isExactFloat32(floatColumn.values)
        }
        return MLXBackend.isExactFloat32(numbers)
    }

    init(floatColumn: Float64Column) {
        self.floatColumn = floatColumn
        let nullCount = floatColumn.nullBitmap.setBitCount
        self.nonMissingCount = floatColumn.count - nullCount
        self.hasMissing = nullCount != 0
        self._cachedNumbers = nil
    }

    init(intColumn: Int64Column) {
        let count = intColumn.count
        let bm = intColumn.nullBitmap
        let hasNulls = intColumn.hasNulls
        let dblVals = ContiguousArray<Double>(unsafeUninitializedCapacity: count) { buf, initCount in
            intColumn.values.withUnsafeBufferPointer { intBuf in
                guard let inBase = intBuf.baseAddress, let outBase = buf.baseAddress else { return }
                for i in 0..<count {
                    outBase[i] = Double(inBase[i])
                }
            }
            initCount = count
        }
        self.floatColumn = Float64Column(name: intColumn.name, values: dblVals, nullBitmap: bm)
        let nullCount = hasNulls ? bm.setBitCount : 0
        self.nonMissingCount = count - nullCount
        self.hasMissing = nullCount != 0
        self._cachedNumbers = nil
    }

    init(values: [Double], isMissing: [Bool]) {
        var bm = BitSet(count: values.count)
        var compact: [Double] = []
        compact.reserveCapacity(values.count)
        for index in values.indices {
            if isMissing[index] {
                bm[index] = true
            } else {
                compact.append(values[index])
            }
        }
        self.floatColumn = Float64Column(name: "", values: ContiguousArray(values), nullBitmap: bm)
        self.nonMissingCount = compact.count
        self.hasMissing = compact.count != values.count
        self._cachedNumbers = compact
    }

    init(numbers: [Double], name: String = "") {
        self.floatColumn = Float64Column(name: name, numbers: numbers)
        self.nonMissingCount = numbers.count
        self.hasMissing = false
        self._cachedNumbers = numbers
    }

    init?(dataValues: [DataValue], name: String = "") {
        var vals = ContiguousArray<Double>()
        vals.reserveCapacity(dataValues.count)
        var bm = BitSet(count: dataValues.count)
        var validCount = 0

        for (index, value) in dataValues.enumerated() {
            switch value {
            case .number(let number):
                vals.append(number)
                validCount += 1
            case .missing:
                vals.append(0)
                bm[index] = true
            default:
                return nil
            }
        }
        self.floatColumn = Float64Column(name: name, values: vals, nullBitmap: bm)
        self.nonMissingCount = validCount
        self.hasMissing = validCount != dataValues.count
        self._cachedNumbers = nil
    }

    static func == (lhs: NumericBuffer, rhs: NumericBuffer) -> Bool {
        lhs.floatColumn.values == rhs.floatColumn.values
            && lhs.floatColumn.nullBitmap == rhs.floatColumn.nullBitmap
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(floatColumn.values)
        hasher.combine(floatColumn.nullBitmap)
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

    static func makeColumn(name: String, numbers: [Double], isMissing: [Bool]) -> Float64Column {
        var vals = ContiguousArray<Double>()
        vals.reserveCapacity(numbers.count)
        var bm = BitSet(count: numbers.count)
        for index in numbers.indices {
            if isMissing[index] || !numbers[index].isFinite {
                vals.append(0.0)
                bm[index] = true
            } else {
                vals.append(numbers[index])
            }
        }
        return Float64Column(name: name, values: vals, nullBitmap: bm)
    }

    static func makeColumn(name: String, values: ContiguousArray<Double>, nullBitmap: BitSet) -> Float64Column {
        let count = values.count
        if !nullBitmap.mayHaveTrueBits {
            var anyNonFinite = false
            values.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                for i in 0..<count {
                    if !base[i].isFinite {
                        anyNonFinite = true
                        break
                    }
                }
            }
            if !anyNonFinite {
                return Float64Column(name: name, values: values, nullBitmap: nullBitmap)
            }
        }
        var vals = values
        var bm = nullBitmap
        for index in 0..<count where !bm[index] && !vals[index].isFinite {
            vals[index] = 0.0
            bm[index] = true
        }
        return Float64Column(name: name, values: vals, nullBitmap: bm)
    }
}

// MARK: - Public result types

/// Statistics for one numeric column.
///
/// Large columns are reduced through MLX in `float64` on the CPU stream;
/// small columns use native Swift. Both paths use `Double` precision, the
/// sample standard deviation (`n - 1`), and linear-interpolation quartiles.
public struct ColumnStatistics: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// The number of numeric, non-missing values.
    public let count: Int
    /// The arithmetic mean, or `nil` when there are no numeric values.
    public let mean: Double?
    /// The sample standard deviation, or `nil` for fewer than two numeric values.
    public let standardDeviation: Double?
    /// The median, or `nil` when there are no numeric values.
    public let median: Double?
    /// The smallest value, or `nil` when there are no numeric values.
    public let min: Double?
    /// The largest value, or `nil` when there are no numeric values.
    public let max: Double?
    /// The first quartile, or `nil` when there are no numeric values.
    public let q1: Double?
    /// The third quartile, or `nil` when there are no numeric values.
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
    /// The number of non-missing values.
    public let count: Int
    /// The number of distinct non-missing values.
    public let unique: Int
    /// The most frequent value, or `nil` when there are no non-missing values.
    public let top: DataValue?
    /// The number of occurrences of ``top``.
    public let freq: Int
}

/// Aggregated statistics for all numeric columns of a data frame.
public struct DataFrameStatistics: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Statistics keyed by column name.
    public let columns: [String: ColumnStatistics]
    /// The number of rows in the source frame.
    public let rowCount: Int
    /// The number of columns in the source frame.
    public let columnCount: Int

    /// Returns the statistics for a column, or `nil` if it has none.
    public subscript(_ column: String) -> ColumnStatistics? {
        columns[column]
    }
}

/// Ordered per-column numeric summaries returned by ``DataFrame/summaryTable()``.
public struct DataFrameSummary: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Column names in display order.
    public let columnOrder: [String]
    /// Summaries keyed by column name.
    public let summaries: [String: ColumnSummary]

    /// Returns the summary for a column, or `nil` if it has none.
    public subscript(_ column: String) -> ColumnSummary? {
        summaries[column]
    }

    /// Renders the summaries as an aligned table with one column per data-frame column.
    ///
    /// - Returns: The table text, or `"Empty DataFrameSummary"` when there are no columns.
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

    /// The table rendered by ``describe()``.
    public var description: String { describe() }
    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Ordered per-column missing-value counts returned by ``DataFrame/missingSummary()``.
public struct MissingCounts: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Column names in display order.
    public let columnOrder: [String]
    /// Missing-value counts keyed by column name.
    public let counts: [String: Int]
    /// The number of rows in the source frame.
    public let rowCount: Int

    /// Returns the missing-value count for a column, or `nil` if the column is unknown.
    public subscript(_ column: String) -> Int? {
        counts[column]
    }

    /// The total number of missing values across all columns.
    public var totalMissing: Int {
        counts.values.reduce(0, +)
    }

    /// Renders one line per column followed by a total-missing footer.
    public func describe() -> String {
        let width = columnOrder.map(\.count).max() ?? 0
        var lines = columnOrder.map { name in
            let padded = name.padding(toLength: width, withPad: " ", startingAt: 0)
            return "\(padded)    \(counts[name] ?? 0)"
        }
        lines.append("Total missing: \(totalMissing) (rows: \(rowCount))")
        return lines.joined(separator: "\n")
    }

    /// The text rendered by ``describe()``.
    public var description: String { describe() }
    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Ordered per-column data types returned by ``DataFrame/columnTypes()`` and ``DataFrame/dtypes``.
public struct ColumnTypes: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Column names in display order.
    public let columnOrder: [String]
    /// Data types keyed by column name.
    public let types: [String: DataType]

    /// Returns the type of a column, or `nil` if the column is unknown.
    public subscript(_ column: String) -> DataType? {
        types[column]
    }

    /// Renders one line per column followed by a column-count footer.
    public func describe() -> String {
        let width = columnOrder.map(\.count).max() ?? 0
        var lines = columnOrder.map { name in
            let padded = name.padding(toLength: width, withPad: " ", startingAt: 0)
            return "\(padded)    \(types[name] ?? .unknown)"
        }
        lines.append("Columns: \(columnOrder.count)")
        return lines.joined(separator: "\n")
    }

    /// The text rendered by ``describe()``.
    public var description: String { describe() }
    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Ordered per-column distinct non-missing counts returned by ``DataFrame/uniqueSummary()``.
public struct UniqueCounts: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Column names in display order.
    public let columnOrder: [String]
    /// Distinct non-missing value counts keyed by column name.
    public let counts: [String: Int]
    /// The number of rows in the source frame.
    public let rowCount: Int

    /// Returns the distinct-value count for a column, or `nil` if the column is unknown.
    public subscript(_ column: String) -> Int? {
        counts[column]
    }

    /// Renders one line per column followed by a column and row count footer.
    public func describe() -> String {
        let width = columnOrder.map(\.count).max() ?? 0
        var lines = columnOrder.map { name in
            let padded = name.padding(toLength: width, withPad: " ", startingAt: 0)
            return "\(padded)    \(counts[name] ?? 0)"
        }
        lines.append("Columns: \(columnOrder.count) (rows: \(rowCount))")
        return lines.joined(separator: "\n")
    }

    /// The text rendered by ``describe()``.
    public var description: String { describe() }
    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
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
    /// Column names in display order.
    public let columnOrder: [String]
    /// Statistics for numeric columns keyed by column name.
    public let numeric: [String: ColumnStatistics]
    /// Summaries for non-numeric columns keyed by column name.
    public let nonNumeric: [String: NonNumericSummary]
    /// The number of rows in the source frame.
    public let rowCount: Int
    /// The number of columns in the source frame.
    public let columnCount: Int
    /// Labels of the percentiles reported, such as `"25%"`.
    public let percentileLabels: [String]

    /// Creates a description.
    ///
    /// - Parameters:
    ///   - columnOrder: Column names in display order.
    ///   - numeric: Statistics for numeric columns.
    ///   - nonNumeric: Summaries for non-numeric columns.
    ///   - rowCount: The number of rows in the source frame.
    ///   - columnCount: The number of columns in the source frame.
    ///   - percentileLabels: Labels of the percentiles reported.
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
    /// A one-line summary of the statistics; unavailable values render as `NaN`.
    public var description: String {
        "count \(count), mean \(format(mean)), std \(format(standardDeviation)), "
            + "min \(format(min)), 25% \(format(q1)), 50% \(format(median)), "
            + "75% \(format(q3)), max \(format(max))"
    }

    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

extension NonNumericSummary {
    /// A one-line summary of count, unique values, top value, and its frequency.
    public var description: String {
        "count \(count), unique \(unique), top \(top.map(String.init(describing:)) ?? "NaN"), freq \(freq)"
    }

    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

extension DataFrameStatistics {
    /// A header line followed by one line per column, sorted by column name.
    public var description: String {
        var lines = ["DataFrameStatistics: \(rowCount) rows x \(columnCount) columns"]
        for name in columns.keys.sorted() {
            lines.append("\(name): \(columns[name]!)")
        }
        return lines.joined(separator: "\n")
    }

    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

extension DataFrameDescription {
    /// The table rendered by `describe()`.
    public var description: String { describe() }
    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}
