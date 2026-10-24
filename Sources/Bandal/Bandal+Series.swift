import Foundation

/// A single named column of a ``DataFrame``.
///
/// `Series` is a value type. Its `description` prints an index-labelled table
/// in the same style as ``DataFrame/description``.
public struct Series: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// The column name.
    public let name: String
    /// The values in row order.
    public let values: [DataValue]
    /// Cached numeric buffer when all non-missing values are `.number`.
    let numericBuffer: NumericBuffer?

    /// Creates a series from a name and values.
    public init(name: String, values: [DataValue]) {
        self.name = name
        self.values = values
        self.numericBuffer = NumericBuffer(dataValues: values)
    }

    /// Creates a numeric series from `Double` values.
    public init(name: String, numbers: [Double]) {
        self.name = name
        self.values = numbers.map(DataValue.number)
        self.numericBuffer = NumericBuffer(numbers: numbers)
    }

    init(column: DataColumn) {
        self.name = column.name
        self.values = column.values
        self.numericBuffer = column.numericBuffer
    }

    enum CodingKeys: String, CodingKey {
        case name
        case values
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedName = try container.decode(String.self, forKey: .name)
        let decodedValues = try container.decode([DataValue].self, forKey: .values)
        self.init(name: decodedName, values: decodedValues)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(values, forKey: .values)
    }

    public static func == (lhs: Series, rhs: Series) -> Bool {
        lhs.name == rhs.name && lhs.values == rhs.values
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(name)
        hasher.combine(values)
    }

    /// The number of rows, including missing values.
    public var count: Int { values.count }

    /// Whether the series has zero elements.
    public var isEmpty: Bool { values.isEmpty }

    /// Number of `.missing` elements.
    public var missingCount: Int {
        if let numericBuffer {
            return numericBuffer.count - numericBuffer.nonMissingCount
        }
        return values.reduce(0) { $0 + ($1 == .missing ? 1 : 0) }
    }

    /// Number of non-missing elements.
    public var nonMissingCount: Int {
        count - missingCount
    }

    /// The inferred data type of the series, ignoring `.missing` values.
    public var dtype: DataType {
        var types = Set<DataType>()
        for value in values where value != .missing {
            switch value {
            case .number: types.insert(.number)
            case .string: types.insert(.string)
            case .boolean: types.insert(.boolean)
            case .date: types.insert(.date)
            case .missing: break
            }
        }
        guard !types.isEmpty else { return .unknown }
        guard types.count == 1, let single = types.first else { return .mixed }
        return single
    }

    /// Non-missing numeric values; other values are ignored.
    public var numbers: [Double] {
        if let numericBuffer { return numericBuffer.numbers }
        return values.compactMap { value in
            if case .number(let number) = value { return number }
            return nil
        }
    }

    /// Accesses the element at `index`.
    public subscript(index: Int) -> DataValue {
        values[index]
    }

    /// Reduces the numeric values with `function`; `.missing` when undefined.
    public func reduced(using function: AggregateFunction) -> DataValue {
        switch function {
        case .count:
            return .number(Double(nonMissingCount))
        case .first:
            return values.first { $0 != .missing } ?? .missing
        case .last:
            return values.last { $0 != .missing } ?? .missing
        default:
            return DataFrame.reduce(numbers, function).map(DataValue.number) ?? .missing
        }
    }

    /// Sum of non-missing numeric values (`nil` if no numeric values).
    public func sum() -> Double? { DataFrame.reduce(numbers, .sum) }

    /// Arithmetic mean of non-missing numeric values (`nil` if no numeric values).
    public func mean() -> Double? { DataFrame.reduce(numbers, .mean) }

    /// Minimum of non-missing numeric values (`nil` if no numeric values).
    public func min() -> Double? { DataFrame.reduce(numbers, .min) }

    /// Maximum of non-missing numeric values (`nil` if no numeric values).
    public func max() -> Double? { DataFrame.reduce(numbers, .max) }

    /// Median of non-missing numeric values (`nil` if no numeric values).
    public func median() -> Double? { DataFrame.reduce(numbers, .median) }

    /// Sample variance (`ddof = 1`) of non-missing numeric values (`nil` if `< 2` values).
    public func variance() -> Double? { DataFrame.reduce(numbers, .variance) }

    /// Sample standard deviation (`ddof = 1`) of non-missing numeric values (`nil` if `< 2` values).
    public func standardDeviation() -> Double? { DataFrame.reduce(numbers, .standardDeviation) }

    /// Linear-interpolation quantile at `probability` in `0...1` (`nil` if no numeric values).
    public func quantile(_ probability: Double) -> Double? {
        let nums = numbers
        guard !nums.isEmpty else { return nil }
        let sorted = MLXDeviceVector(nums)?.sortedValues() ?? nums.sorted()
        return NumericEngine.quantile(sorted: sorted, probability: probability)
    }

    /// Most frequent non-missing values in first-appearance order.
    public func mode() -> [DataValue] {
        var counts: [DataValue: Int] = [:]
        for value in values where value != .missing {
            counts[value, default: 0] += 1
        }
        guard let highest = counts.values.max(), highest > 0 else { return [] }
        return unique().filter { counts[$0] == highest }
    }

    /// Summary statistics when the series is numeric.
    public func summary() -> ColumnSummary? {
        guard let buffer = numericBuffer, !buffer.numbers.isEmpty else { return nil }
        let moments = NumericEngine.moments(of: buffer.numbers)
        return ColumnSummary(
            count: moments.count,
            mean: moments.mean,
            standardDeviation: moments.standardDeviation,
            median: moments.median,
            min: moments.min,
            max: moments.max
        )
    }

    /// Structured statistics (including quartiles) when the series is numeric.
    public func statistics() -> ColumnStatistics? {
        guard let buffer = numericBuffer, !buffer.numbers.isEmpty else { return nil }
        return ColumnStatistics(buffer: buffer)
    }

    /// Renders a Pandas-style descriptive summary of the series.
    public func describe() -> String {
        if let stats = statistics() {
            func f(_ v: Double?) -> String { v.map { String(format: "%.6g", $0) } ?? "NaN" }
            let rows: [(String, String)] = [
                ("count", String(stats.count)),
                ("mean", f(stats.mean)),
                ("std", f(stats.standardDeviation)),
                ("min", f(stats.min)),
                ("25%", f(stats.q1)),
                ("50%", f(stats.median)),
                ("75%", f(stats.q3)),
                ("max", f(stats.max)),
            ]
            return DataFrame.renderTable(
                rowLabels: rows.map(\.0),
                columnNames: [name],
                values: rows.map { [$0.1] }
            ) + "\nName: \(name), dtype: float64"
        } else {
            let nonMissing = values.filter { $0 != .missing }
            var counts: [DataValue: Int] = [:]
            for v in nonMissing { counts[v, default: 0] += 1 }
            let topFreq = counts.values.max() ?? 0
            let topVal = nonMissing.first { counts[$0] == topFreq }
            let rows: [(String, String)] = [
                ("count", String(nonMissing.count)),
                ("unique", String(counts.count)),
                ("top", topVal.map(DataFrame.displayValue) ?? "NaN"),
                ("freq", String(topFreq)),
            ]
            return DataFrame.renderTable(
                rowLabels: rows.map(\.0),
                columnNames: [name],
                values: rows.map { [$0.1] }
            ) + "\nName: \(name), dtype: \(dtype)"
        }
    }

    /// Distinct non-missing values in first-appearance order.
    public func unique() -> [DataValue] {
        var seen = Set<DataValue>()
        return values.filter { $0 != .missing && seen.insert($0).inserted }
    }

    /// The number of distinct non-missing values.
    public func uniqueCount() -> Int { unique().count }

    /// Frequencies of non-missing values, most frequent first.
    ///
    /// Ties keep first-appearance order. The result has `value` and `count`
    /// columns.
    public func valueCounts() throws -> DataFrame {
        var counts: [DataValue: Int] = [:]
        for value in values where value != .missing { counts[value, default: 0] += 1 }
        let ordered = unique().enumerated()
            .sorted { lhs, rhs in
                let l = counts[lhs.element, default: 0], r = counts[rhs.element, default: 0]
                return l != r ? l > r : lhs.offset < rhs.offset
            }
            .map(\.element)
        return try DataFrame(columns: [
            (name: "value", values: ordered),
            (name: "count", values: ordered.map { .number(Double(counts[$0, default: 0])) }),
        ])
    }

    /// Returns a Boolean mask indicating whether each element belongs to `candidates`.
    public func isin(_ candidates: Set<DataValue>) -> [Bool] {
        values.map { candidates.contains($0) }
    }

    /// Returns a Boolean mask indicating whether each numeric element lies in `lower...upper`.
    public func between(lower: Double, upper: Double, inclusive: Bool = true) -> [Bool] {
        values.map { value in
            guard case .number(let number) = value else { return false }
            return inclusive
                ? (number >= lower && number <= upper)
                : (number > lower && number < upper)
        }
    }

    /// Returns a new series with each element transformed by `transform`.
    public func mapped(_ transform: (DataValue) throws -> DataValue) rethrows -> Series {
        Series(name: name, values: try values.map(transform))
    }

    /// Returns a copy with numeric values clamped to `lower...upper`.
    public func clipped(lower: Double? = nil, upper: Double? = nil) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let mlxResult = NumericEngine.clip(buffer, lower: lower, upper: upper) {
            return Series(name: name, values: mlxResult)
        }
        let output = values.map { value -> DataValue in
            guard case .number(var number) = value else { return .missing }
            if let lower { number = Swift.max(number, lower) }
            if let upper { number = Swift.min(number, upper) }
            return .number(number)
        }
        return Series(name: name, values: output)
    }

    /// Returns a copy with numeric values rounded to `decimals` decimal places.
    public func rounded(decimals: Int = 0) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let mlxResult = NumericEngine.round(buffer, decimals: decimals) {
            return Series(name: name, values: mlxResult)
        }
        let scale = pow(10.0, Double(decimals))
        let output = values.map { value -> DataValue in
            guard case .number(let number) = value else { return .missing }
            return .number((number * scale).rounded(.toNearestOrEven) / scale)
        }
        return Series(name: name, values: output)
    }

    /// Returns a copy with the absolute value of each numeric element.
    public func abs() throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let mlxResult = NumericEngine.abs(buffer) {
            return Series(name: name, values: mlxResult)
        }
        let output = values.map { value -> DataValue in
            guard case .number(let number) = value else { return .missing }
            return .number(Swift.abs(number))
        }
        return Series(name: name, values: output)
    }

    /// Returns a copy with cumulative `sum`, `product`, `minimum`, or `maximum`.
    public func cumulative(using function: CumulativeFunction) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let mlxResult = NumericEngine.cumulative(buffer, using: function) {
            return Series(name: name, values: mlxResult)
        }
        var state: Double?
        let output = values.map { value -> DataValue in
            guard case .number(let number) = value else { return .missing }
            switch (function, state) {
            case (_, nil): state = number
            case (.sum, let s?): state = s + number
            case (.product, let s?): state = s * number
            case (.minimum, let s?): state = Swift.min(s, number)
            case (.maximum, let s?): state = Swift.max(s, number)
            }
            return .number(state ?? number)
        }
        return Series(name: name, values: output)
    }

    /// Returns a copy shifted by `periods` positions, filling vacated entries with `fill`.
    public func shifted(by periods: Int = 1, fillWith fill: DataValue = .missing) -> Series {
        let output = values.indices.map { row -> DataValue in
            let index = row - periods
            return values.indices.contains(index) ? values[index] : fill
        }
        return Series(name: name, values: output)
    }

    /// Returns the difference from `periods` elements earlier.
    public func difference(periods: Int = 1) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let mlxResult = NumericEngine.difference(buffer, periods: periods) {
            return Series(name: name, values: mlxResult)
        }
        let output = values.indices.map { row -> DataValue in
            let index = row - periods
            guard values.indices.contains(index),
                  case .number(let current) = values[row],
                  case .number(let prior) = values[index]
            else {
                return .missing
            }
            return .number(current - prior)
        }
        return Series(name: name, values: output)
    }

    /// Returns the fractional change from `periods` elements earlier.
    public func percentChange(periods: Int = 1) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let mlxResult = NumericEngine.percentChange(buffer, periods: periods) {
            return Series(name: name, values: mlxResult)
        }
        let output = values.indices.map { row -> DataValue in
            let index = row - periods
            guard values.indices.contains(index),
                  case .number(let current) = values[row],
                  case .number(let prior) = values[index],
                  prior != 0
            else {
                return .missing
            }
            return .number(current / prior - 1)
        }
        return Series(name: name, values: output)
    }

    /// Computes average-tie ranks (1-based) for numeric values, keeping `.missing` as `.missing`.
    public func ranked(ascending: Bool = true) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        var indexed: [(row: Int, value: Double)] = []
        indexed.reserveCapacity(buffer.nonMissingCount)
        for row in 0..<buffer.count where !buffer.isMissing[row] {
            indexed.append((row: row, value: buffer.values[row]))
        }
        indexed.sort { lhs, rhs in
            ascending ? lhs.value < rhs.value : lhs.value > rhs.value
        }

        var output = Array(repeating: DataValue.missing, count: buffer.count)
        var start = 0
        while start < indexed.count {
            var end = start + 1
            while end < indexed.count && indexed[end].value == indexed[start].value {
                end += 1
            }
            // 1-based average rank of positions (start + 1)...end
            let avgRank = Double(start + 1 + end) / 2.0
            for k in start..<end {
                output[indexed[k].row] = .number(avgRank)
            }
            start = end
        }
        return Series(name: "\(name)_rank", values: output)
    }

    // MARK: Vectorized arithmetic

    /// Applies an arithmetic operation with a scalar `Double` to every element.
    public func applying(_ operation: ArithmeticOperator, scalar: Double) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let mlxResult = NumericEngine.scalarArithmetic(buffer: buffer, scalar: scalar, operation: operation) {
            return Series(name: name, values: mlxResult)
        }
        let output = values.map { value -> DataValue in
            guard case .number(let number) = value else { return .missing }
            let result: Double
            switch operation {
            case .add: result = number + scalar
            case .subtract: result = number - scalar
            case .multiply: result = number * scalar
            case .divide:
                guard scalar != 0 else { return .missing }
                result = number / scalar
            }
            return result.isFinite ? .number(result) : .missing
        }
        return Series(name: name, values: output)
    }

    /// Applies an element-wise arithmetic operation with another `Series` of equal length.
    public func applying(_ operation: ArithmeticOperator, other: Series) throws -> Series {
        guard count == other.count else {
            throw DataFrameError.unequalColumnLengths(
                column: other.name,
                expected: count,
                actual: other.count
            )
        }
        guard let leftBuffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        guard let rightBuffer = other.numericBuffer else {
            throw DataFrameError.nonNumericColumn(other.name)
        }
        if let mlxResult = NumericEngine.binaryArithmetic(left: leftBuffer, right: rightBuffer, operation: operation) {
            return Series(name: name, values: mlxResult)
        }
        let output = (0..<count).map { row -> DataValue in
            guard !leftBuffer.isMissing[row], !rightBuffer.isMissing[row] else {
                return .missing
            }
            let a = leftBuffer.values[row]
            let b = rightBuffer.values[row]
            let result: Double
            switch operation {
            case .add: result = a + b
            case .subtract: result = a - b
            case .multiply: result = a * b
            case .divide:
                guard b != 0 else { return .missing }
                result = a / b
            }
            return result.isFinite ? .number(result) : .missing
        }
        return Series(name: name, values: output)
    }

    public static func + (lhs: Series, rhs: Series) throws -> Series {
        try lhs.applying(.add, other: rhs)
    }

    public static func - (lhs: Series, rhs: Series) throws -> Series {
        try lhs.applying(.subtract, other: rhs)
    }

    public static func * (lhs: Series, rhs: Series) throws -> Series {
        try lhs.applying(.multiply, other: rhs)
    }

    public static func / (lhs: Series, rhs: Series) throws -> Series {
        try lhs.applying(.divide, other: rhs)
    }

    public static func + (lhs: Series, rhs: Double) throws -> Series {
        try lhs.applying(.add, scalar: rhs)
    }

    public static func - (lhs: Series, rhs: Double) throws -> Series {
        try lhs.applying(.subtract, scalar: rhs)
    }

    public static func * (lhs: Series, rhs: Double) throws -> Series {
        try lhs.applying(.multiply, scalar: rhs)
    }

    public static func / (lhs: Series, rhs: Double) throws -> Series {
        try lhs.applying(.divide, scalar: rhs)
    }

    // MARK: Missing value helpers

    /// Returns a copy where `.missing` elements are replaced with `replacement`.
    public func fillMissing(with replacement: DataValue) throws -> Series {
        if case .number(let n) = replacement, !n.isFinite {
            throw DataFrameError.nonFiniteValue(column: name, row: 0)
        }
        return Series(name: name, values: values.map { $0 == .missing ? replacement : $0 })
    }

    /// Returns a copy with `.missing` elements removed.
    public func dropMissing() -> Series {
        Series(name: name, values: values.filter { $0 != .missing })
    }

    /// Returns a copy where `.missing` elements are filled with the closest earlier non-missing value.
    public func forwardFilled() -> Series {
        var last: DataValue = .missing
        return Series(name: name, values: values.map { value in
            if value == .missing { return last }
            last = value
            return value
        })
    }

    /// Returns a copy where `.missing` elements are filled with the closest later non-missing value.
    public func backwardFilled() -> Series {
        var next: DataValue = .missing
        let filled = values.reversed().map { value -> DataValue in
            if value == .missing { return next }
            next = value
            return value
        }
        return Series(name: name, values: Array(filled.reversed()))
    }

    public var description: String {
        let labels = values.indices.map(String.init)
        let cells = values.map { [DataFrame.displayValue($0)] }
        return DataFrame.renderTable(rowLabels: labels, columnNames: [name], values: cells)
            + "\nName: \(name), Length: \(count)"
    }

    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Element-wise cumulative operations.
public enum CumulativeFunction: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case sum, product, minimum, maximum

    public var description: String { rawValue }
}

extension DataFrame {
    /// Returns one column as a ``Series``.
    public func series(_ name: String) throws -> Series {
        guard let column = columns.first(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        return Series(column: column)
    }

    /// Returns one column as a ``Series``, or `nil` when it does not exist.
    public subscript(name: String) -> Series? {
        try? series(name)
    }

    /// Returns a copy without the named columns.
    public func drop(columns names: [String]) throws -> DataFrame {
        for name in names where !columnNames.contains(name) {
            throw DataFrameError.columnNotFound(name)
        }
        let kept = columns.filter { !names.contains($0.name) }
        guard !kept.isEmpty else { throw DataFrameError.emptyColumns }
        return try DataFrame(columns: kept.map { (name: $0.name, values: $0.values) })
    }

    /// Returns the `count` rows with the largest values in `column`.
    public func largest(_ count: Int, by column: String) throws -> DataFrame {
        try sorted(by: [SortDescriptor(column, order: .descending)]).head(count)
    }

    /// Returns the `count` rows with the smallest values in `column`.
    public func smallest(_ count: Int, by column: String) throws -> DataFrame {
        try sorted(by: [SortDescriptor(column, order: .ascending)]).head(count)
    }

    /// Reduces every numeric column to one row using `function`.
    ///
    /// The result has a single row; non-numeric columns are omitted.
    public func reduced(using function: AggregateFunction) throws -> DataFrame {
        let numeric = try columnNames.filter { try dataType(of: $0) == .number }
        guard !numeric.isEmpty else { throw DataFrameError.emptyColumns }
        return try DataFrame(columns: numeric.map { name in
            (name: name, values: [try series(name).reduced(using: function)])
        })
    }

    /// Computes the given `probability` quantile (`0...1`) for every numeric column
    /// as a single-row `DataFrame`.
    public func quantile(_ probability: Double) throws -> DataFrame {
        let numeric = try columnNames.filter { try dataType(of: $0) == .number }
        guard !numeric.isEmpty else { throw DataFrameError.emptyColumns }
        return try DataFrame(columns: numeric.map { name in
            let q = try series(name).quantile(probability)
            return (name: name, values: [q.map(DataValue.number) ?? .missing])
        })
    }

    /// Pairwise sample covariance of numeric columns.
    ///
    /// Large frames compute the covariance matrix via MLX (`float64` on CPU
    /// stream); rows where either column is missing are ignored for that pair.
    public func covariance() throws -> DataFrame {
        let numericColumns = columns.filter { (try? dataType(of: $0.name)) == .number }
        guard !numericColumns.isEmpty else { throw DataFrameError.emptyColumns }
        let buffers = numericColumns.compactMap(\.numericBuffer)
        if buffers.count == numericColumns.count,
           let matrix = NumericEngine.covarianceOrCorrelationMatrix(
               buffers: buffers,
               rowCount: rowCount,
               correlation: false
           )
        {
            return try DataFrame(columns: numericColumns.indices.map { j in
                (
                    name: numericColumns[j].name,
                    values: (0..<numericColumns.count).map { i in
                        matrix[i][j].map(DataValue.number) ?? .missing
                    }
                )
            })
        }
        return try pairwise { x, y in
            guard x.count > 1 else { return nil }
            let mx = x.reduce(0, +) / Double(x.count), my = y.reduce(0, +) / Double(y.count)
            return zip(x, y).reduce(0) { $0 + ($1.0 - mx) * ($1.1 - my) } / Double(x.count - 1)
        }
    }

    /// Pairwise Pearson correlation of numeric columns.
    ///
    /// Large frames compute the correlation matrix via MLX (`float64` on CPU
    /// stream); rows where either column is missing are ignored for that pair.
    public func correlation() throws -> DataFrame {
        let numericColumns = columns.filter { (try? dataType(of: $0.name)) == .number }
        guard !numericColumns.isEmpty else { throw DataFrameError.emptyColumns }
        let buffers = numericColumns.compactMap(\.numericBuffer)
        if buffers.count == numericColumns.count,
           let matrix = NumericEngine.covarianceOrCorrelationMatrix(
               buffers: buffers,
               rowCount: rowCount,
               correlation: true
           )
        {
            return try DataFrame(columns: numericColumns.indices.map { j in
                (
                    name: numericColumns[j].name,
                    values: (0..<numericColumns.count).map { i in
                        matrix[i][j].map(DataValue.number) ?? .missing
                    }
                )
            })
        }
        return try pairwise { x, y in
            guard x.count > 1 else { return nil }
            let mx = x.reduce(0, +) / Double(x.count), my = y.reduce(0, +) / Double(y.count)
            let sxy = zip(x, y).reduce(0) { $0 + ($1.0 - mx) * ($1.1 - my) }
            let sxx = x.reduce(0) { $0 + ($1 - mx) * ($1 - mx) }
            let syy = y.reduce(0) { $0 + ($1 - my) * ($1 - my) }
            let denominator = (sxx * syy).squareRoot()
            return denominator > 0 ? sxy / denominator : nil
        }
    }

    private func pairwise(_ body: ([Double], [Double]) -> Double?) throws -> DataFrame {
        let numeric = try columnNames.filter { try dataType(of: $0) == .number }
        guard !numeric.isEmpty else { throw DataFrameError.emptyColumns }
        let data = try numeric.map { try values(in: $0) }
        var result: [(name: String, values: [DataValue])] = []
        for (j, name) in numeric.enumerated() {
            var column: [DataValue] = []
            for i in numeric.indices {
                var x: [Double] = [], y: [Double] = []
                for row in 0..<rowCount {
                    if case .number(let a) = data[i][row], case .number(let b) = data[j][row] {
                        x.append(a)
                        y.append(b)
                    }
                }
                column.append(body(x, y).map(DataValue.number) ?? .missing)
            }
            result.append((name: name, values: column))
        }
        return try DataFrame(columns: result)
    }

    /// Replaces a numeric column with its running total, product, minimum, or maximum.
    ///
    /// Missing values stay missing and do not reset the accumulator. Large
    /// columns run through MLX (`cumsum`, `cumprod`, `cummin`, `cummax`) on the
    /// CPU stream in `float64`.
    public func cumulative(_ column: String, using function: CumulativeFunction) throws -> DataFrame {
        let updated = try series(column).cumulative(using: function)
        return try replacingColumn(column, with: updated.values)
    }

    /// Returns a copy where `column` is shifted down by `periods` rows
    /// (up when negative), filling vacated rows with missing values.
    public func shifted(_ column: String, by periods: Int = 1) throws -> DataFrame {
        let updated = try series(column).shifted(by: periods)
        return try replacingColumn(column, with: updated.values)
    }

    /// Returns a copy where `column` holds the difference from `periods` rows earlier.
    public func difference(_ column: String, periods: Int = 1) throws -> DataFrame {
        let updated = try series(column).difference(periods: periods)
        return try replacingColumn(column, with: updated.values)
    }

    /// Returns a copy where `column` holds the fractional change from `periods` rows earlier.
    public func percentChange(_ column: String, periods: Int = 1) throws -> DataFrame {
        let updated = try series(column).percentChange(periods: periods)
        return try replacingColumn(column, with: updated.values)
    }

    /// Limits numeric values of `column` to `lower...upper`; either bound may be `nil`.
    public func clipped(_ column: String, lower: Double? = nil, upper: Double? = nil) throws -> DataFrame {
        let updated = try series(column).clipped(lower: lower, upper: upper)
        return try replacingColumn(column, with: updated.values)
    }

    /// Rounds numeric values of `column` to `decimals` places.
    public func rounded(_ column: String, decimals: Int = 0) throws -> DataFrame {
        let updated = try series(column).rounded(decimals: decimals)
        return try replacingColumn(column, with: updated.values)
    }

    /// Replaces a numeric column with the absolute value of each entry.
    public func abs(_ column: String) throws -> DataFrame {
        let updated = try series(column).abs()
        return try replacingColumn(column, with: updated.values)
    }

    /// Computes average-tie ranks (1-based) for `column`.
    ///
    /// When `outputColumn` is `nil`, the ranked values replace `column`;
    /// otherwise they are appended or replace `outputColumn`.
    public func ranked(
        _ column: String,
        ascending: Bool = true,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let ranks = try series(column).ranked(ascending: ascending)
        let target = outputColumn ?? column
        return try withColumn(name: target, values: ranks.values)
    }

    /// Combines two numeric columns element-wise with `operation` (+, -, *, /)
    /// and writes the result to `outputColumn`.
    public func combining(
        _ leftColumn: String,
        _ rightColumn: String,
        using operation: ArithmeticOperator,
        as outputColumn: String
    ) throws -> DataFrame {
        let lhs = try series(leftColumn)
        let rhs = try series(rightColumn)
        let combined = try lhs.applying(operation, other: rhs)
        return try withColumn(name: outputColumn, values: combined.values)
    }

    /// Applies a scalar arithmetic operation to `column` and writes the result
    /// to `outputColumn` (or replaces `column` when `outputColumn` is `nil`).
    public func scaling(
        _ column: String,
        by scalar: Double,
        using operation: ArithmeticOperator = .multiply,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let updated = try series(column).applying(operation, scalar: scalar)
        return try withColumn(name: outputColumn ?? column, values: updated.values)
    }

    /// Number of distinct non-missing values in each column (similar to `df.nunique()`).
    public func uniqueCounts() -> [String: Int] {
        Dictionary(
            uniqueKeysWithValues: columns.map { column in
                (column.name, Series(column: column).uniqueCount())
            }
        )
    }

    /// Frequencies of non-missing values in `column`, most frequent first.
    public func valueCounts(of column: String) throws -> DataFrame {
        try series(column).valueCounts()
    }

    /// Replaces occurrences of `target` with `replacement` in the specified
    /// `columns` (or across all columns when `columns` is `nil`).
    public func replacing(
        _ target: DataValue,
        with replacement: DataValue,
        in targetColumns: [String]? = nil
    ) throws -> DataFrame {
        if case .number(let n) = replacement, !n.isFinite {
            throw DataFrameError.nonFiniteValue(column: targetColumns?.first ?? columnNames.first ?? "", row: 0)
        }
        if let targetColumns {
            for name in targetColumns where !columnNames.contains(name) {
                throw DataFrameError.columnNotFound(name)
            }
        }
        let scope = targetColumns.map(Set.init)
        return try DataFrame(columns: columns.map { col in
            guard scope == nil || scope!.contains(col.name) else {
                return (name: col.name, values: col.values)
            }
            return (
                name: col.name,
                values: col.values.map { $0 == target ? replacement : $0 }
            )
        })
    }

    /// Fills each missing value with the closest earlier non-missing value.
    public func forwardFilled() throws -> DataFrame {
        try DataFrame(columns: columns.map { column in
            var last: DataValue = .missing
            return (name: column.name, values: column.values.map { value in
                if value == .missing { return last }
                last = value
                return value
            })
        })
    }

    /// Fills each missing value with the closest later non-missing value.
    public func backwardFilled() throws -> DataFrame {
        try DataFrame(columns: columns.map { column in
            var next: DataValue = .missing
            let filled = column.values.reversed().map { value -> DataValue in
                if value == .missing { return next }
                next = value
                return value
            }
            return (name: column.name, values: Array(filled.reversed()))
        })
    }

    /// Returns a Boolean frame that is `true` where the source is missing.
    public func missingMask() throws -> DataFrame {
        try DataFrame(columns: columns.map { column in
            (name: column.name, values: column.values.map { DataValue.boolean($0 == .missing) })
        })
    }

    /// Returns a structured version of ``info()``.
    public func infoSummary() -> DataFrameInfo {
        var totalMemory = 0
        let columnInfos = columns.map { column -> DataFrameInfo.Column in
            let nonMissing = column.values.filter { $0 != .missing }.count
            for value in column.values {
                switch value {
                case .number, .date: totalMemory += 8
                case .boolean: totalMemory += 1
                case .string(let s): totalMemory += s.utf8.count
                case .missing: break
                }
            }
            return DataFrameInfo.Column(
                name: column.name,
                nonMissingCount: nonMissing,
                type: (try? dataType(of: column.name)) ?? .unknown
            )
        }
        return DataFrameInfo(
            rowCount: rowCount,
            columns: columnInfos,
            estimatedMemoryBytes: totalMemory
        )
    }

    func replacingColumn(_ name: String, with values: [DataValue]) throws -> DataFrame {
        guard columnNames.contains(name) else { throw DataFrameError.columnNotFound(name) }
        return try DataFrame(columns: columns.map {
            (name: $0.name, values: $0.name == name ? values : $0.values)
        })
    }
}

/// Structured result of ``DataFrame/infoSummary()``.
public struct DataFrameInfo: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Per-column information.
    public struct Column: Equatable, Hashable, Sendable, Codable {
        public let name: String
        public let nonMissingCount: Int
        public let type: DataType

        public init(name: String, nonMissingCount: Int, type: DataType) {
            self.name = name
            self.nonMissingCount = nonMissingCount
            self.type = type
        }
    }

    public let rowCount: Int
    public let columns: [Column]
    public let estimatedMemoryBytes: Int

    public init(
        rowCount: Int,
        columns: [Column],
        estimatedMemoryBytes: Int = 0
    ) {
        self.rowCount = rowCount
        self.columns = columns
        self.estimatedMemoryBytes = estimatedMemoryBytes
    }

    public func describe() -> String {
        var lines = ["DataFrame: \(rowCount) rows x \(columns.count) columns"]
        for (index, column) in columns.enumerated() {
            lines.append("\(index)  \(column.name)  \(column.nonMissingCount) non-null  \(column.type)")
        }
        return lines.joined(separator: "\n")
    }

    public var description: String { describe() }
    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}
