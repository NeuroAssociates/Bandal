import Accelerate
import Foundation

/// A single named column of a ``DataFrame``.
///
/// `Series` is a value type backed by typed columnar storage (`AnyColumn`).
/// Its `description` prints an index-labelled table in the same style as
/// ``DataFrame/description``.
public struct Series: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// The column name.
    public let name: String
    /// Typed columnar storage.
    let storage: any AnyColumn
    /// Cached numeric buffer when all non-missing values are `.number`.
    let numericBuffer: NumericBuffer?
    /// Optional cached `[DataValue]` array when initialized directly from `DataValue` elements.
    private let _cachedValues: [DataValue]?

    /// The values in row order.
    public var values: [DataValue] {
        _cachedValues ?? storage.toDataValues()
    }

    /// Creates a series from a name and values.
    ///
    /// - Parameters:
    ///   - name: Name of the series.
    ///   - values: Ordered ``DataValue`` elements.
    public init(name: String, values: [DataValue]) {
        self.name = name
        self._cachedValues = values
        let built = AnyColumnFactory.make(name: name, values: values)
        self.storage = built
        if let floatCol = built as? Float64Column {
            self.numericBuffer = NumericBuffer(floatColumn: floatCol)
        } else if let intCol = built as? Int64Column {
            self.numericBuffer = NumericBuffer(intColumn: intCol)
        } else if let anyCol = built as? AnyValueColumn, built.nullCount == built.count {
            let emptyFloat = Float64Column(
                name: name,
                values: ContiguousArray(repeating: 0.0, count: anyCol.count),
                nullBitmap: BitSet(count: anyCol.count, repeating: true)
            )
            self.numericBuffer = NumericBuffer(floatColumn: emptyFloat)
        } else {
            self.numericBuffer = nil
        }
    }

    /// Creates a numeric series from `Double` values without boxing into `[DataValue]`.
    ///
    /// - Parameters:
    ///   - name: Name of the series.
    ///   - numbers: Ordered finite `Double` values.
    public init(name: String, numbers: [Double]) {
        let floatCol = Float64Column(
            name: name,
            values: ContiguousArray(numbers),
            nullBitmap: BitSet(count: numbers.count)
        )
        self.name = name
        self.storage = floatCol
        self.numericBuffer = NumericBuffer(floatColumn: floatCol)
        self._cachedValues = nil
    }

    /// Creates a numeric series from optional `Double?` values (`nil` becomes `.missing`).
    ///
    /// - Parameters:
    ///   - name: Name of the series.
    ///   - values: Ordered optional `Double` values where `nil` or non-finite values become `.missing`.
    public init(name: String, values: [Double?]) {
        var arr = ContiguousArray<Double>(repeating: 0.0, count: values.count)
        var nulls = BitSet(count: values.count)
        for i in 0..<values.count {
            if let v = values[i], v.isFinite {
                arr[i] = v
            } else {
                nulls[i] = true
            }
        }
        let floatCol = Float64Column(name: name, values: arr, nullBitmap: nulls)
        self.name = name
        self.storage = floatCol
        self.numericBuffer = NumericBuffer(floatColumn: floatCol)
        self._cachedValues = nil
    }

    /// Creates a ``Series`` directly from typed columnar storage and an optional pre-built numeric buffer.
    init(storage: any AnyColumn, numericBuffer: NumericBuffer? = nil, cachedValues: [DataValue]? = nil) {
        self.name = storage.name
        self.storage = storage
        self._cachedValues = cachedValues
        if let numericBuffer {
            self.numericBuffer = numericBuffer
        } else if let floatCol = storage as? Float64Column {
            self.numericBuffer = NumericBuffer(floatColumn: floatCol)
        } else if let intCol = storage as? Int64Column {
            self.numericBuffer = NumericBuffer(intColumn: intCol)
        } else if let anyCol = storage as? AnyValueColumn, storage.nullCount == storage.count {
            let emptyFloat = Float64Column(
                name: storage.name,
                values: ContiguousArray(repeating: 0.0, count: anyCol.count),
                nullBitmap: BitSet(count: anyCol.count, repeating: true)
            )
            self.numericBuffer = NumericBuffer(floatColumn: emptyFloat)
        } else {
            self.numericBuffer = nil
        }
    }

    /// Wraps an internal ``DataColumn`` as a ``Series`` in $O(1)$ without copying storage.
    init(column: DataColumn) {
        self.name = column.name
        self.storage = column.storage
        self.numericBuffer = column.numericBuffer
        self._cachedValues = nil
    }

    /// Returns a copy of the series with its column name changed to `newName`.
    func renamed(to newName: String) -> Series {
        guard newName != name else { return self }
        let newStorage = storage.renamed(to: newName)
        let newBuffer: NumericBuffer?
        if let numericBuffer {
            let fc = numericBuffer.floatColumn
            let renamedFloat = Float64Column(name: newName, values: fc.values, nullBitmap: fc.nullBitmap)
            newBuffer = NumericBuffer(floatColumn: renamedFloat)
        } else {
            newBuffer = nil
        }
        return Series(storage: newStorage, numericBuffer: newBuffer, cachedValues: _cachedValues)
    }

    /// Coding keys used for `Codable` serialization of ``Series``.
    enum CodingKeys: String, CodingKey {
        case name
        case values
    }

    /// Creates a series by decoding its `name` and `values`.
    ///
    /// - Parameter decoder: The decoder to read data from.
    /// - Throws: A decoding error if either key is missing or malformed.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedName = try container.decode(String.self, forKey: .name)
        let decodedValues = try container.decode([DataValue].self, forKey: .values)
        self.init(name: decodedName, values: decodedValues)
    }

    /// Encodes the series' `name` and `values`.
    ///
    /// - Parameter encoder: The encoder to write data to.
    /// - Throws: An encoding error if a value cannot be encoded.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(values, forKey: .values)
    }

    /// Returns a Boolean value indicating whether two series have the same name and equal values.
    public static func == (lhs: Series, rhs: Series) -> Bool {
        guard lhs.name == rhs.name, lhs.count == rhs.count else { return false }
        if let lf = lhs.storage as? Float64Column, let rf = rhs.storage as? Float64Column {
            return lf.nullBitmap == rf.nullBitmap && lf.values == rf.values
        }
        return lhs.values == rhs.values
    }

    /// Hashes the series' name and values into the given hasher.
    ///
    /// - Parameter hasher: The hasher to use when combining the components of this instance.
    public func hash(into hasher: inout Hasher) {
        hasher.combine(name)
        hasher.combine(values)
    }

    /// The number of rows, including missing values.
    public var count: Int { storage.count }

    /// Whether the series has zero elements.
    public var isEmpty: Bool { storage.count == 0 }

    /// Number of `.missing` elements.
    public var missingCount: Int {
        storage.nullCount
    }

    /// Number of non-missing elements.
    public var nonMissingCount: Int {
        storage.count - storage.nullCount
    }

    /// The inferred data type of the series, ignoring `.missing` values.
    public var dtype: DataType {
        if storage.nullCount == storage.count {
            return .unknown
        }
        if storage.dataType != .mixed && storage.dataType != .unknown {
            return storage.dataType
        }
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
    ///
    /// - Parameter index: Zero-based element position.
    /// - Returns: The ``DataValue`` at `index`.
    public subscript(index: Int) -> DataValue {
        storage.value(at: index)
    }

    /// Reduces the numeric values with `function`; `.missing` when undefined.
    ///
    /// - Parameter function: The ``AggregateFunction`` to apply.
    /// - Returns: The reduced ``DataValue``, or `.missing` if undefined.
    public func reduced(using function: AggregateFunction) -> DataValue {
        switch function {
        case .count:
            return .number(Double(nonMissingCount))
        case .first:
            for i in 0..<count where !storage.isNull(at: i) {
                return storage.value(at: i)
            }
            return .missing
        case .last:
            for i in (0..<count).reversed() where !storage.isNull(at: i) {
                return storage.value(at: i)
            }
            return .missing
        default:
            return DataFrame.reduce(numbers, function).map(DataValue.number) ?? .missing
        }
    }

    /// Sum of non-missing numeric values (`nil` if no numeric values).
    ///
    /// - Returns: The sum of all non-missing numbers, or `nil` when empty.
    public func sum() -> Double? { DataFrame.reduce(numbers, .sum) }

    /// Arithmetic mean of non-missing numeric values (`nil` if no numeric values).
    ///
    /// - Returns: The arithmetic mean, or `nil` when empty.
    public func mean() -> Double? { DataFrame.reduce(numbers, .mean) }

    /// Minimum of non-missing numeric values (`nil` if no numeric values).
    ///
    /// - Returns: The minimum value, or `nil` when empty.
    public func min() -> Double? { DataFrame.reduce(numbers, .min) }

    /// Maximum of non-missing numeric values (`nil` if no numeric values).
    ///
    /// - Returns: The maximum value, or `nil` when empty.
    public func max() -> Double? { DataFrame.reduce(numbers, .max) }

    /// Median of non-missing numeric values (`nil` if no numeric values).
    ///
    /// - Returns: The 50th percentile, or `nil` when empty.
    public func median() -> Double? { DataFrame.reduce(numbers, .median) }

    /// Sample variance (`ddof = 1`) of non-missing numeric values (`nil` if `< 2` values).
    ///
    /// - Returns: The unbiased sample variance, or `nil` when fewer than 2 numeric values exist.
    public func variance() -> Double? { DataFrame.reduce(numbers, .variance) }

    /// Sample standard deviation (`ddof = 1`) of non-missing numeric values (`nil` if `< 2` values).
    ///
    /// - Returns: The unbiased sample standard deviation, or `nil` when fewer than 2 numeric values exist.
    public func standardDeviation() -> Double? { DataFrame.reduce(numbers, .standardDeviation) }

    /// Linear-interpolation quantile at `probability` in `0...1` (`nil` if no numeric values).
    ///
    /// - Parameter probability: Fractional probability in `0...1`.
    /// - Returns: The interpolated quantile value, or `nil` when no numeric values exist.
    public func quantile(_ probability: Double) -> Double? {
        let nums = numbers
        guard !nums.isEmpty else { return nil }
        let sorted = MLXDeviceVector(nums, minimumCount: MLXBackend.automaticComplexMinimumCount)?.sortedValues() ?? nums.sorted()
        return NumericEngine.quantile(sorted: sorted, probability: probability)
    }

    /// Most frequent non-missing values in first-appearance order.
    ///
    /// - Returns: An array of the modal ``DataValue`` elements (multiple elements when tied).
    public func mode() -> [DataValue] {
        var counts: [DataValue: Int] = [:]
        for value in values where value != .missing {
            counts[value, default: 0] += 1
        }
        guard let highest = counts.values.max(), highest > 0 else { return [] }
        return unique().filter { counts[$0] == highest }
    }

    /// Summary statistics when the series is numeric.
    ///
    /// - Returns: A ``ColumnSummary``, or `nil` if the series is non-numeric or has no values.
    public func summary() -> ColumnSummary? {
        guard let buffer = numericBuffer, buffer.nonMissingCount > 0 else { return nil }
        return NumericEngine.summaryMoments(of: buffer)
    }

    /// Structured statistics (including quartiles) when the series is numeric.
    ///
    /// - Returns: A ``ColumnStatistics`` instance, or `nil` if the series is non-numeric or has no values.
    public func statistics() -> ColumnStatistics? {
        guard let buffer = numericBuffer, buffer.nonMissingCount > 0 else { return nil }
        return ColumnStatistics(buffer: buffer)
    }

    /// Renders a Pandas-style descriptive summary of the series.
    ///
    /// - Returns: A formatted summary table string.
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
    ///
    /// - Returns: Ordered unique ``DataValue`` elements.
    public func unique() -> [DataValue] {
        var seen = Set<DataValue>()
        return values.filter { $0 != .missing && seen.insert($0).inserted }
    }

    /// The number of distinct non-missing values.
    ///
    /// - Returns: Count of distinct non-missing values.
    public func uniqueCount() -> Int {
        if let catCol = storage as? CategoricalColumn {
            if !catCol.hasNulls {
                return catCol.dictionary.count
            }
            var seen = Set<UInt32>()
            let codes = catCol.codes
            let nulls = catCol.nullBitmap
            for i in 0..<codes.count where !nulls[i] {
                seen.insert(codes[i])
            }
            return seen.count
        }
        return unique().count
    }

    /// Frequencies of non-missing values, most frequent first.
    ///
    /// Ties keep first-appearance order. The result has `value` and `count`
    /// columns.
    ///
    /// - Returns: A two-column ``DataFrame`` (`value` and `count`).
    /// - Throws: ``DataFrameError`` if constructing the result frame fails.
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
    ///
    /// - Parameter candidates: Set of target ``DataValue`` values.
    /// - Returns: An array of Booleans of length ``count``.
    public func isin(_ candidates: Set<DataValue>) -> [Bool] {
        (0..<count).map { candidates.contains(storage.value(at: $0)) }
    }

    /// Returns a Boolean mask indicating whether each numeric element lies in `lower...upper`.
    ///
    /// - Parameters:
    ///   - lower: Lower bound of the interval.
    ///   - upper: Upper bound of the interval.
    ///   - inclusive: When `true`, includes `lower` and `upper`; when `false`, uses strict inequalities.
    /// - Returns: An array of Booleans of length ``count``.
    public func between(lower: Double, upper: Double, inclusive: Bool = true) -> [Bool] {
        if let buffer = numericBuffer {
            let vals = buffer.floatColumn.values
            let nulls = buffer.floatColumn.nullBitmap
            return (0..<count).map { i in
                guard !nulls[i] else { return false }
                let number = vals[i]
                return inclusive
                    ? (number >= lower && number <= upper)
                    : (number > lower && number < upper)
            }
        }
        return values.map { value in
            guard case .number(let number) = value else { return false }
            return inclusive
                ? (number >= lower && number <= upper)
                : (number > lower && number < upper)
        }
    }

    /// Returns a new series with each element transformed by `transform`.
    ///
    /// - Parameter transform: Closure mapping each ``DataValue`` to a new ``DataValue``.
    /// - Returns: A transformed ``Series`` with the same name.
    public func mapped(_ transform: (DataValue) throws -> DataValue) rethrows -> Series {
        Series(name: name, values: try values.map(transform))
    }

    /// Returns a copy with numeric values clamped to `lower...upper`.
    ///
    /// - Parameters:
    ///   - lower: Optional minimum bound.
    ///   - upper: Optional maximum bound.
    /// - Returns: A new ``Series`` with clamped numeric values.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func clipped(lower: Double? = nil, upper: Double? = nil) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let col = NumericEngine.clipColumn(buffer, name: name, lower: lower, upper: upper) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        var outVals = buffer.floatColumn.values
        let nulls = buffer.floatColumn.nullBitmap
        for i in 0..<outVals.count where !nulls[i] {
            var v = outVals[i]
            if let lower { v = Swift.max(v, lower) }
            if let upper { v = Swift.min(v, upper) }
            outVals[i] = v
        }
        let col = Float64Column(name: name, values: outVals, nullBitmap: nulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Returns a copy with numeric values rounded to `decimals` decimal places.
    ///
    /// - Parameter decimals: Number of fractional decimal digits to round to. Defaults to `0`.
    /// - Returns: A new ``Series`` with rounded values.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func rounded(decimals: Int = 0) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let col = NumericEngine.roundColumn(buffer, name: name, decimals: decimals) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let scale = pow(10.0, Double(decimals))
        var outVals = buffer.floatColumn.values
        let nulls = buffer.floatColumn.nullBitmap
        for i in 0..<outVals.count where !nulls[i] {
            outVals[i] = (outVals[i] * scale).rounded(.toNearestOrEven) / scale
        }
        let col = Float64Column(name: name, values: outVals, nullBitmap: nulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Returns a copy with the absolute value of each numeric element.
    ///
    /// - Returns: A new ``Series`` containing non-negative magnitudes.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func abs() throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let col = NumericEngine.absColumn(buffer, name: name) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        var outVals = buffer.floatColumn.values
        let nulls = buffer.floatColumn.nullBitmap
        for i in 0..<outVals.count where !nulls[i] {
            outVals[i] = Swift.abs(outVals[i])
        }
        let col = Float64Column(name: name, values: outVals, nullBitmap: nulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Returns a copy with cumulative `sum`, `product`, `minimum`, or `maximum`.
    ///
    /// - Parameter function: The ``CumulativeFunction`` to evaluate.
    /// - Returns: A new ``Series`` with cumulative values.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func cumulative(using function: CumulativeFunction) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let col = NumericEngine.cumulativeColumn(buffer, name: name, using: function) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        var outVals = buffer.floatColumn.values
        var outNulls = buffer.floatColumn.nullBitmap
        var state: Double?
        for i in 0..<outVals.count where !outNulls[i] {
            let number = outVals[i]
            switch (function, state) {
            case (_, nil): state = number
            case (.sum, let s?): state = s + number
            case (.product, let s?): state = s * number
            case (.minimum, let s?): state = Swift.min(s, number)
            case (.maximum, let s?): state = Swift.max(s, number)
            }
            let val = state ?? number
            if val.isFinite {
                outVals[i] = val
            } else {
                outVals[i] = 0.0
                outNulls[i] = true
            }
        }
        let col = Float64Column(name: name, values: outVals, nullBitmap: outNulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Returns a copy shifted by `periods` positions, filling vacated entries with `fill`.
    ///
    /// - Parameters:
    ///   - periods: Number of positions to shift (positive shifts down, negative shifts up). Defaults to `1`.
    ///   - fill: Value used to populate vacated positions. Defaults to `.missing`.
    /// - Returns: A shifted ``Series``.
    public func shifted(by periods: Int = 1, fillWith fill: DataValue = .missing) -> Series {
        if let buffer = numericBuffer, fill == .missing {
            let srcVals = buffer.floatColumn.values
            let srcNulls = buffer.floatColumn.nullBitmap
            let n = srcVals.count
            var outVals = ContiguousArray<Double>(repeating: 0.0, count: n)
            var outNulls = BitSet(count: n, repeating: true)
            for row in 0..<n {
                let idx = row - periods
                if (0..<n).contains(idx) && !srcNulls[idx] {
                    outVals[row] = srcVals[idx]
                    outNulls[row] = false
                }
            }
            let col = Float64Column(name: name, values: outVals, nullBitmap: outNulls)
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let vals = values
        let output = vals.indices.map { row -> DataValue in
            let index = row - periods
            return vals.indices.contains(index) ? vals[index] : fill
        }
        return Series(name: name, values: output)
    }

    /// Returns the difference from `periods` elements earlier.
    ///
    /// - Parameter periods: Lag offset in rows. Defaults to `1`.
    /// - Returns: A new ``Series`` holding first differences `x[t] - x[t - periods]`.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func difference(periods: Int = 1) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let col = NumericEngine.differenceColumn(buffer, name: name, periods: periods) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let srcVals = buffer.floatColumn.values
        let srcNulls = buffer.floatColumn.nullBitmap
        let n = srcVals.count
        var outVals = ContiguousArray<Double>(repeating: 0.0, count: n)
        var outNulls = BitSet(count: n, repeating: true)
        for row in 0..<n {
            let index = row - periods
            guard (0..<n).contains(index), !srcNulls[row], !srcNulls[index] else { continue }
            let d = srcVals[row] - srcVals[index]
            if d.isFinite {
                outVals[row] = d
                outNulls[row] = false
            }
        }
        let col = Float64Column(name: name, values: outVals, nullBitmap: outNulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Returns the fractional change from `periods` elements earlier.
    ///
    /// - Parameter periods: Lag offset in rows. Defaults to `1`.
    /// - Returns: A new ``Series`` holding fractional changes `x[t] / x[t - periods] - 1`.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func percentChange(periods: Int = 1) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let col = NumericEngine.percentChangeColumn(buffer, name: name, periods: periods) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let srcVals = buffer.floatColumn.values
        let srcNulls = buffer.floatColumn.nullBitmap
        let n = srcVals.count
        var outVals = ContiguousArray<Double>(repeating: 0.0, count: n)
        var outNulls = BitSet(count: n, repeating: true)
        for row in 0..<n {
            let index = row - periods
            guard (0..<n).contains(index), !srcNulls[row], !srcNulls[index], srcVals[index] != 0 else {
                continue
            }
            let pct = srcVals[row] / srcVals[index] - 1.0
            if pct.isFinite {
                outVals[row] = pct
                outNulls[row] = false
            }
        }
        let col = Float64Column(name: name, values: outVals, nullBitmap: outNulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Computes average-tie ranks (1-based) for numeric values, keeping `.missing` as `.missing`.
    ///
    /// - Parameter ascending: When `true`, smallest values receive rank `1.0`; when `false`, largest values receive rank `1.0`.
    /// - Returns: A new ``Series`` containing 1-based average-tie ranks.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func ranked(ascending: Bool = true) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        let srcVals = buffer.floatColumn.values
        let srcNulls = buffer.floatColumn.nullBitmap
        var indexed: [(row: Int, value: Double)] = []
        indexed.reserveCapacity(buffer.nonMissingCount)
        for row in 0..<buffer.count where !srcNulls[row] {
            indexed.append((row: row, value: srcVals[row]))
        }
        indexed.sort { lhs, rhs in
            ascending ? lhs.value < rhs.value : lhs.value > rhs.value
        }

        var outVals = ContiguousArray<Double>(repeating: 0.0, count: buffer.count)
        let outNulls = srcNulls
        var start = 0
        while start < indexed.count {
            var end = start + 1
            while end < indexed.count && indexed[end].value == indexed[start].value {
                end += 1
            }
            let avgRank = Double(start + 1 + end) / 2.0
            for k in start..<end {
                outVals[indexed[k].row] = avgRank
            }
            start = end
        }
        let col = Float64Column(name: "\(name)_rank", values: outVals, nullBitmap: outNulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    // MARK: Vectorized arithmetic

    /// Applies an arithmetic operation with a scalar `Double` to every element.
    ///
    /// - Parameters:
    ///   - operation: The ``ArithmeticOperator`` to apply.
    ///   - scalar: Right-hand scalar operand.
    /// - Returns: A new ``Series`` with the result of the scalar arithmetic operation.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func applying(_ operation: ArithmeticOperator, scalar: Double) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let col = NumericEngine.scalarArithmeticColumn(buffer: buffer, name: name, scalar: scalar, operation: operation) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let n = buffer.count
        if !scalar.isFinite || (operation == .divide && scalar == 0) {
            let col = Float64Column(
                name: name,
                values: ContiguousArray(repeating: 0.0, count: n),
                nullBitmap: BitSet(count: n, repeating: true)
            )
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        if n > 0 {
            var checkSum = Double.nan
            let raw = ContiguousArray<Double>(unsafeUninitializedCapacity: n) { outBuf, initCount in
                buffer.floatColumn.values.withUnsafeBufferPointer { inBuf in
                    guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                    let len = vDSP_Length(n)
                    var s = scalar
                    switch operation {
                    case .add:
                        vDSP_vsaddD(inBase, 1, &s, outBase, 1, len)
                    case .subtract:
                        var negS = -scalar
                        vDSP_vsaddD(inBase, 1, &negS, outBase, 1, len)
                    case .multiply:
                        vDSP_vsmulD(inBase, 1, &s, outBase, 1, len)
                    case .divide:
                        vDSP_vsdivD(inBase, 1, &s, outBase, 1, len)
                    }
                    vDSP_sveD(outBase, 1, &checkSum, len)
                }
                initCount = n
            }
            let nulls = buffer.floatColumn.nullBitmap
            if !nulls.mayHaveTrueBits && checkSum.isFinite {
                let col = Float64Column(name: name, values: raw, nullBitmap: nulls)
                return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
            }
            let col = NumericBuffer.makeColumn(name: name, values: raw, nullBitmap: nulls)
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let col = Float64Column(name: name, values: [], nullBitmap: BitSet())
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Applies an element-wise arithmetic operation with another `Series` of equal length.
    ///
    /// - Parameters:
    ///   - operation: The ``ArithmeticOperator`` to apply.
    ///   - other: Right-hand ``Series`` operand of equal length.
    /// - Returns: A new ``Series`` with the element-wise result.
    /// - Throws: ``DataFrameError`` if lengths differ or either series is non-numeric.
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
        if let col = NumericEngine.binaryArithmeticColumn(left: leftBuffer, right: rightBuffer, name: name, operation: operation) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let n = count
        if n == 0 {
            let col = Float64Column(name: name, values: [], nullBitmap: BitSet())
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let col = NumericEngine.nativeBinaryArithmeticColumn(
            left: leftBuffer,
            right: rightBuffer,
            name: name,
            operation: operation
        )
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Returns a series made by adding corresponding elements of two series.
    ///
    /// - Throws: An error if the series differ in length or contain non-numeric values.
    public static func + (lhs: Series, rhs: Series) throws -> Series {
        try lhs.applying(.add, other: rhs)
    }

    /// Returns a series made by subtracting corresponding elements of two series.
    ///
    /// - Throws: An error if the series differ in length or contain non-numeric values.
    public static func - (lhs: Series, rhs: Series) throws -> Series {
        try lhs.applying(.subtract, other: rhs)
    }

    /// Returns a series made by multiplying corresponding elements of two series.
    ///
    /// - Throws: An error if the series differ in length or contain non-numeric values.
    public static func * (lhs: Series, rhs: Series) throws -> Series {
        try lhs.applying(.multiply, other: rhs)
    }

    /// Returns a series made by dividing corresponding elements of two series.
    ///
    /// - Throws: An error if the series differ in length or contain non-numeric values.
    public static func / (lhs: Series, rhs: Series) throws -> Series {
        try lhs.applying(.divide, other: rhs)
    }

    /// Returns a series made by adding a scalar to every element.
    ///
    /// - Throws: An error if the series contains non-numeric values.
    public static func + (lhs: Series, rhs: Double) throws -> Series {
        try lhs.applying(.add, scalar: rhs)
    }

    /// Returns a series made by subtracting a scalar from every element.
    ///
    /// - Throws: An error if the series contains non-numeric values.
    public static func - (lhs: Series, rhs: Double) throws -> Series {
        try lhs.applying(.subtract, scalar: rhs)
    }

    /// Returns a series made by multiplying every element by a scalar (`rhs`).
    ///
    /// - Throws: An error if the series contains non-numeric values.
    public static func * (lhs: Series, rhs: Double) throws -> Series {
        try lhs.applying(.multiply, scalar: rhs)
    }

    /// Returns a series made by dividing every element by a scalar (`rhs`).
    ///
    /// - Throws: An error if the series contains non-numeric values.
    public static func / (lhs: Series, rhs: Double) throws -> Series {
        try lhs.applying(.divide, scalar: rhs)
    }

    // MARK: Missing value helpers

    /// Returns a copy where `.missing` elements are replaced with `replacement`.
    ///
    /// - Parameter replacement: Replacement ``DataValue`` for missing entries.
    /// - Returns: A new ``Series`` with missing values replaced.
    /// - Throws: ``DataFrameError/nonFiniteValue(column:row:)`` if `replacement` is a non-finite number.
    public func fillMissing(with replacement: DataValue) throws -> Series {
        if case .number(let n) = replacement, !n.isFinite {
            throw DataFrameError.nonFiniteValue(column: name, row: 0)
        }
        if !storage.hasNulls || replacement == .missing {
            return self
        }
        if let buffer = numericBuffer, case .number(let n) = replacement {
            var outVals = buffer.floatColumn.values
            let nulls = buffer.floatColumn.nullBitmap
            for i in 0..<outVals.count where nulls[i] {
                outVals[i] = n
            }
            let col = Float64Column(name: name, values: outVals, nullBitmap: BitSet(count: outVals.count))
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        return Series(name: name, values: values.map { $0 == .missing ? replacement : $0 })
    }

    /// Returns a copy with `.missing` elements removed.
    ///
    /// - Returns: A compacted ``Series`` containing only non-missing elements.
    public func dropMissing() -> Series {
        if !storage.hasNulls {
            return self
        }
        let filteredStorage = storage.slice(mask: storage.nullBitmap.inverted())
        return Series(storage: filteredStorage)
    }

    /// Returns a copy where `.missing` elements are filled with the closest earlier non-missing value.
    ///
    /// - Returns: A forward-filled ``Series``.
    public func forwardFilled() -> Series {
        if !storage.hasNulls {
            return self
        }
        if let buffer = numericBuffer {
            var outVals = buffer.floatColumn.values
            var outNulls = buffer.floatColumn.nullBitmap
            var last: Double?
            for i in 0..<outVals.count {
                if outNulls[i] {
                    if let last {
                        outVals[i] = last
                        outNulls[i] = false
                    }
                } else {
                    last = outVals[i]
                }
            }
            let col = Float64Column(name: name, values: outVals, nullBitmap: outNulls)
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        var last: DataValue = .missing
        return Series(name: name, values: values.map { value in
            if value == .missing { return last }
            last = value
            return value
        })
    }

    /// Returns a copy where `.missing` elements are filled with the closest later non-missing value.
    ///
    /// - Returns: A backward-filled ``Series``.
    public func backwardFilled() -> Series {
        if !storage.hasNulls {
            return self
        }
        if let buffer = numericBuffer {
            var outVals = buffer.floatColumn.values
            var outNulls = buffer.floatColumn.nullBitmap
            var next: Double?
            for i in (0..<outVals.count).reversed() {
                if outNulls[i] {
                    if let next {
                        outVals[i] = next
                        outNulls[i] = false
                    }
                } else {
                    next = outVals[i]
                }
            }
            let col = Float64Column(name: name, values: outVals, nullBitmap: outNulls)
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        var next: DataValue = .missing
        let filled = values.reversed().map { value -> DataValue in
            if value == .missing { return next }
            next = value
            return value
        }
        return Series(name: name, values: Array(filled.reversed()))
    }

    /// A pandas-style tabular text representation including name and length.
    public var description: String {
        let vals = values
        let labels = vals.indices.map(String.init)
        let cells = vals.map { [DataFrame.displayValue($0)] }
        return DataFrame.renderTable(rowLabels: labels, columnNames: [name], values: cells)
            + "\nName: \(name), Length: \(count)"
    }

    /// A debugging representation; identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    ///
    /// - Parameter target: Destination text output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Element-wise cumulative operations.
public enum CumulativeFunction: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    /// Running cumulative sum (`cumsum`).
    case sum
    /// Running cumulative product (`cumprod`).
    case product
    /// Running cumulative minimum (`cummin`).
    case minimum
    /// Running cumulative maximum (`cummax`).
    case maximum

    /// The lowercase name of the function.
    public var description: String { rawValue }
}

extension DataFrame {
    /// Returns one column as a ``Series`` in $O(1)$ without unboxing column storage.
    ///
    /// - Parameter name: Name of the column to retrieve.
    /// - Returns: A ``Series`` backed by the column's underlying storage.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if `name` does not exist.
    public func series(_ name: String) throws -> Series {
        guard let column = columns.first(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        return Series(column: column)
    }

    /// Returns one column as a ``Series``, or `nil` when it does not exist.
    ///
    /// - Parameter name: Name of the column to retrieve.
    /// - Returns: The requested ``Series``, or `nil` if not found.
    public subscript(name: String) -> Series? {
        try? series(name)
    }

    /// Returns a copy with a new or replaced column from a ``Series``.
    ///
    /// - Parameter series: The ``Series`` to insert or replace by its ``Series/name``.
    /// - Returns: A new ``DataFrame`` containing `series`.
    /// - Throws: ``DataFrameError`` if the series length does not match ``rowCount``.
    public func withColumn(_ series: Series) throws -> DataFrame {
        try withColumn(storage: series.storage)
    }

    /// Returns a copy with a new or replaced numeric column from `Double` values.
    ///
    /// - Parameters:
    ///   - name: Name of the column to insert or replace.
    ///   - numbers: Finite `Double` values for each row.
    /// - Returns: A new ``DataFrame`` containing the numeric column.
    /// - Throws: ``DataFrameError`` if `numbers` contains non-finite values or has an invalid length.
    public func withColumn(name: String, numbers: [Double]) throws -> DataFrame {
        for (row, num) in numbers.enumerated() where !num.isFinite {
            throw DataFrameError.nonFiniteValue(column: name, row: row)
        }
        let col = Float64Column(
            name: name,
            values: ContiguousArray(numbers),
            nullBitmap: BitSet(count: numbers.count)
        )
        return try withColumn(storage: col)
    }

    /// Returns a copy without the named columns.
    ///
    /// - Parameter names: Names of the columns to remove.
    /// - Returns: A new ``DataFrame`` with `names` removed.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if any name does not exist, or ``DataFrameError/emptyColumns`` if all columns are dropped.
    public func drop(columns names: [String]) throws -> DataFrame {
        for name in names where !columnNames.contains(name) {
            throw DataFrameError.columnNotFound(name)
        }
        let dropped = Set(names)
        let kept = columns.filter { !dropped.contains($0.name) }
        guard !kept.isEmpty else { throw DataFrameError.emptyColumns }
        return try DataFrame(dataColumns: kept)
    }

    /// Returns the `count` rows with the largest values in `column`.
    ///
    /// - Parameters:
    ///   - count: Maximum number of top rows to return.
    ///   - column: Name of the column to sort by in descending order.
    /// - Returns: A new ``DataFrame`` containing up to `count` largest rows.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if `column` does not exist.
    public func largest(_ count: Int, by column: String) throws -> DataFrame {
        try sorted(by: [SortDescriptor(column, order: .descending)]).head(count)
    }

    /// Returns the `count` rows with the smallest values in `column`.
    ///
    /// - Parameters:
    ///   - count: Maximum number of bottom rows to return.
    ///   - column: Name of the column to sort by in ascending order.
    /// - Returns: A new ``DataFrame`` containing up to `count` smallest rows.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if `column` does not exist.
    public func smallest(_ count: Int, by column: String) throws -> DataFrame {
        try sorted(by: [SortDescriptor(column, order: .ascending)]).head(count)
    }

    /// Reduces every numeric column to one row using `function`.
    ///
    /// The result has a single row; non-numeric columns are omitted.
    ///
    /// - Parameter function: The ``AggregateFunction`` to apply to each numeric column.
    /// - Returns: A single-row ``DataFrame`` containing the reduced numeric columns.
    /// - Throws: ``DataFrameError/emptyColumns`` if the frame has no numeric columns.
    public func reduced(using function: AggregateFunction) throws -> DataFrame {
        let numeric = try columnNames.filter { try dataType(of: $0) == .number }
        guard !numeric.isEmpty else { throw DataFrameError.emptyColumns }
        return try DataFrame(columns: numeric.map { name in
            (name: name, values: [try series(name).reduced(using: function)])
        })
    }

    /// Computes the given `probability` quantile (`0...1`) for every numeric column
    /// as a single-row `DataFrame`.
    ///
    /// - Parameter probability: Fractional probability in `0...1`.
    /// - Returns: A single-row ``DataFrame`` containing the quantile for each numeric column.
    /// - Throws: ``DataFrameError/emptyColumns`` if the frame has no numeric columns.
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
    ///
    /// - Returns: A square ``DataFrame`` whose columns and rows correspond to the numeric columns.
    /// - Throws: ``DataFrameError/emptyColumns`` if the frame has no numeric columns.
    public func covariance() throws -> DataFrame {
        try computePairwiseMatrix(correlation: false)
    }

    /// Pairwise Pearson correlation of numeric columns.
    ///
    /// Large frames compute the correlation matrix via MLX (`float64` on CPU
    /// stream); rows where either column is missing are ignored for that pair.
    ///
    /// - Returns: A square ``DataFrame`` whose columns and rows correspond to the numeric columns.
    /// - Throws: ``DataFrameError/emptyColumns`` if the frame has no numeric columns.
    public func correlation() throws -> DataFrame {
        try computePairwiseMatrix(correlation: true)
    }

    /// Computes either the pairwise sample covariance or Pearson correlation matrix across all numeric columns.
    private func computePairwiseMatrix(correlation: Bool) throws -> DataFrame {
        let numericColumns = columns.filter { (try? dataType(of: $0.name)) == .number }
        guard !numericColumns.isEmpty else { throw DataFrameError.emptyColumns }
        let buffers = numericColumns.compactMap(\.numericBuffer)
        let columnCount = numericColumns.count

        if buffers.count == columnCount {
            if let matrix = NumericEngine.covarianceOrCorrelationMatrix(
                buffers: buffers,
                rowCount: rowCount,
                correlation: correlation
            ) {
                return try DataFrame(columns: numericColumns.indices.map { j in
                    (
                        name: numericColumns[j].name,
                        values: (0..<columnCount).map { i in
                            matrix[i][j].map(DataValue.number) ?? .missing
                        }
                    )
                })
            }

            // Native upper-triangular columnar fast path (zero DataValue boxing).
            var matrix = Array(
                repeating: Array(repeating: Double?.none, count: columnCount),
                count: columnCount
            )

            if rowCount >= 2 && buffers.allSatisfy({ !$0.hasMissing }) {
                let len = vDSP_Length(rowCount)
                let invDenom = 1.0 / Double(rowCount - 1)
                var centeredColumns: [ContiguousArray<Double>] = []
                centeredColumns.reserveCapacity(columnCount)

                for col in 0..<columnCount {
                    let centered = ContiguousArray<Double>(unsafeUninitializedCapacity: rowCount) { outBuf, initCount in
                        buffers[col].floatColumn.values.withUnsafeBufferPointer { inBuf in
                            guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                            var mean = 0.0
                            vDSP_meanvD(inBase, 1, &mean, len)
                            var negMean = -mean
                            vDSP_vsaddD(inBase, 1, &negMean, outBase, 1, len)
                        }
                        initCount = rowCount
                    }
                    centeredColumns.append(centered)
                }

                var covMatrix = [Double](repeating: 0.0, count: columnCount * columnCount)
                for i in 0..<columnCount {
                    centeredColumns[i].withUnsafeBufferPointer { iBuf in
                        guard let iBase = iBuf.baseAddress else { return }
                        for j in i..<columnCount {
                            centeredColumns[j].withUnsafeBufferPointer { jBuf in
                                guard let jBase = jBuf.baseAddress else { return }
                                var dot = 0.0
                                vDSP_dotprD(iBase, 1, jBase, 1, &dot, len)
                                let cov = dot * invDenom
                                covMatrix[i * columnCount + j] = cov
                                covMatrix[j * columnCount + i] = cov
                            }
                        }
                    }
                }

                if !correlation {
                    for i in 0..<columnCount {
                        for j in 0..<columnCount {
                            let v = covMatrix[i * columnCount + j]
                            matrix[i][j] = v.isFinite ? v : nil
                        }
                    }
                } else {
                    let diagStds = (0..<columnCount).map { i -> Double in
                        let v = covMatrix[i * columnCount + i]
                        return v > 0 ? v.squareRoot() : 0
                    }
                    for i in 0..<columnCount {
                        for j in 0..<columnCount {
                            let denom = diagStds[i] * diagStds[j]
                            guard denom > 0 else {
                                matrix[i][j] = nil
                                continue
                            }
                            if i == j {
                                matrix[i][j] = 1.0
                            } else {
                                let corr = covMatrix[i * columnCount + j] / denom
                                matrix[i][j] = corr.isFinite ? Swift.min(Swift.max(corr, -1.0), 1.0) : nil
                            }
                        }
                    }
                }
            } else if rowCount >= 2 {
                for i in 0..<columnCount {
                    let lVals = buffers[i].floatColumn.values
                    let lNulls = buffers[i].floatColumn.nullBitmap
                    for j in i..<columnCount {
                        let rVals = buffers[j].floatColumn.values
                        let rNulls = buffers[j].floatColumn.nullBitmap
                        var count = 0
                        var sumX = 0.0
                        var sumY = 0.0
                        for row in 0..<rowCount where !lNulls[row] && !rNulls[row] {
                            sumX += lVals[row]
                            sumY += rVals[row]
                            count += 1
                        }
                        guard count >= 2 else { continue }
                        let mx = sumX / Double(count)
                        let my = sumY / Double(count)
                        var sxy = 0.0
                        var sxx = 0.0
                        var syy = 0.0
                        for row in 0..<rowCount where !lNulls[row] && !rNulls[row] {
                            let dx = lVals[row] - mx
                            let dy = rVals[row] - my
                            sxy += dx * dy
                            if correlation {
                                sxx += dx * dx
                                syy += dy * dy
                            }
                        }
                        let val: Double?
                        if correlation {
                            let denom = (sxx * syy).squareRoot()
                            val = denom > 0 ? Swift.min(Swift.max(sxy / denom, -1.0), 1.0) : nil
                        } else {
                            val = sxy / Double(count - 1)
                        }
                        matrix[i][j] = val
                        matrix[j][i] = val
                    }
                }
            }

            return try DataFrame(columns: numericColumns.indices.map { j in
                (
                    name: numericColumns[j].name,
                    values: (0..<columnCount).map { i in
                        matrix[i][j].map(DataValue.number) ?? .missing
                    }
                )
            })
        }

        return try pairwise { x, y in
            guard x.count > 1 else { return nil }
            let mx = x.reduce(0, +) / Double(x.count), my = y.reduce(0, +) / Double(y.count)
            let sxy = zip(x, y).reduce(0) { $0 + ($1.0 - mx) * ($1.1 - my) }
            if !correlation {
                return sxy / Double(x.count - 1)
            }
            let sxx = x.reduce(0) { $0 + ($1 - mx) * ($1 - mx) }
            let syy = y.reduce(0) { $0 + ($1 - my) * ($1 - my) }
            let denominator = (sxx * syy).squareRoot()
            return denominator > 0 ? sxy / denominator : nil
        }
    }

    /// Evaluates `body` over all pairs of numeric columns using aligned non-missing values.
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
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column to transform.
    ///   - function: The ``CumulativeFunction`` to apply.
    /// - Returns: A new ``DataFrame`` with `column` replaced by its cumulative values.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func cumulative(_ column: String, using function: CumulativeFunction) throws -> DataFrame {
        let updated = try series(column).cumulative(using: function)
        return try replacingColumn(column, withStorage: updated.storage)
    }

    /// Returns a copy where `column` is shifted down by `periods` rows
    /// (up when negative), filling vacated rows with missing values.
    ///
    /// - Parameters:
    ///   - column: Name of the column to shift.
    ///   - periods: Number of rows to shift. Defaults to `1`.
    /// - Returns: A new ``DataFrame`` with `column` shifted.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if `column` does not exist.
    public func shifted(_ column: String, by periods: Int = 1) throws -> DataFrame {
        let updated = try series(column).shifted(by: periods)
        return try replacingColumn(column, withStorage: updated.storage)
    }

    /// Returns a copy where `column` holds the difference from `periods` rows earlier.
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column.
    ///   - periods: Lag offset in rows. Defaults to `1`.
    /// - Returns: A new ``DataFrame`` with `column` replaced by its differences.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func difference(_ column: String, periods: Int = 1) throws -> DataFrame {
        let updated = try series(column).difference(periods: periods)
        return try replacingColumn(column, withStorage: updated.storage)
    }

    /// Returns a copy where `column` holds the fractional change from `periods` rows earlier.
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column.
    ///   - periods: Lag offset in rows. Defaults to `1`.
    /// - Returns: A new ``DataFrame`` with `column` replaced by its fractional changes.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func percentChange(_ column: String, periods: Int = 1) throws -> DataFrame {
        let updated = try series(column).percentChange(periods: periods)
        return try replacingColumn(column, withStorage: updated.storage)
    }

    /// Limits numeric values of `column` to `lower...upper`; either bound may be `nil`.
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column to clip.
    ///   - lower: Optional minimum bound.
    ///   - upper: Optional maximum bound.
    /// - Returns: A new ``DataFrame`` with `column` clipped.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func clipped(_ column: String, lower: Double? = nil, upper: Double? = nil) throws -> DataFrame {
        let updated = try series(column).clipped(lower: lower, upper: upper)
        return try replacingColumn(column, withStorage: updated.storage)
    }

    /// Rounds numeric values of `column` to `decimals` places.
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column to round.
    ///   - decimals: Number of fractional decimal digits. Defaults to `0`.
    /// - Returns: A new ``DataFrame`` with `column` rounded.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func rounded(_ column: String, decimals: Int = 0) throws -> DataFrame {
        let updated = try series(column).rounded(decimals: decimals)
        return try replacingColumn(column, withStorage: updated.storage)
    }

    /// Replaces a numeric column with the absolute value of each entry.
    ///
    /// - Parameter column: Name of the numeric column.
    /// - Returns: A new ``DataFrame`` with `column` replaced by its absolute values.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func abs(_ column: String) throws -> DataFrame {
        let updated = try series(column).abs()
        return try replacingColumn(column, withStorage: updated.storage)
    }

    /// Computes average-tie ranks (1-based) for `column`.
    ///
    /// When `outputColumn` is `nil`, the ranked values replace `column`;
    /// otherwise they are appended or replace `outputColumn`.
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column to rank.
    ///   - ascending: When `true`, smallest values receive rank `1.0`.
    ///   - outputColumn: Optional destination column name; `nil` replaces `column` in place.
    /// - Returns: A new ``DataFrame`` containing the ranked column.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func ranked(
        _ column: String,
        ascending: Bool = true,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let ranks = try series(column).ranked(ascending: ascending)
        let target = outputColumn ?? column
        return try withColumn(storage: ranks.storage.renamed(to: target))
    }

    /// Combines two numeric columns element-wise with `operation` (+, -, *, /)
    /// and writes the result to `outputColumn`.
    ///
    /// - Parameters:
    ///   - leftColumn: Name of the left numeric column.
    ///   - rightColumn: Name of the right numeric column.
    ///   - operation: The ``ArithmeticOperator`` to apply element-wise.
    ///   - outputColumn: Name of the output column to insert or replace.
    /// - Returns: A new ``DataFrame`` with `outputColumn` containing the combined values.
    /// - Throws: ``DataFrameError`` if either input column is missing or non-numeric.
    public func combining(
        _ leftColumn: String,
        _ rightColumn: String,
        using operation: ArithmeticOperator,
        as outputColumn: String
    ) throws -> DataFrame {
        let lhs = try series(leftColumn)
        let rhs = try series(rightColumn)
        let combined = try lhs.applying(operation, other: rhs)
        return try withColumn(storage: combined.storage.renamed(to: outputColumn))
    }

    /// Applies a scalar arithmetic operation to `column` and writes the result
    /// to `outputColumn` (or replaces `column` when `outputColumn` is `nil`).
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column to scale.
    ///   - scalar: Right-hand scalar operand.
    ///   - operation: The ``ArithmeticOperator`` to apply. Defaults to `.multiply`.
    ///   - outputColumn: Optional destination column name; `nil` replaces `column` in place.
    /// - Returns: A new ``DataFrame`` containing the scaled column.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func scaling(
        _ column: String,
        by scalar: Double,
        using operation: ArithmeticOperator = .multiply,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let updated = try series(column).applying(operation, scalar: scalar)
        let target = outputColumn ?? column
        return try withColumn(storage: updated.storage.renamed(to: target))
    }

    /// Number of distinct non-missing values in each column (similar to `df.nunique()`).
    ///
    /// - Returns: A dictionary mapping each column name to its distinct non-missing count.
    public func uniqueCounts() -> [String: Int] {
        Dictionary(
            uniqueKeysWithValues: columns.map { column in
                (column.name, Series(column: column).uniqueCount())
            }
        )
    }

    /// Frequencies of non-missing values in `column`, most frequent first.
    ///
    /// - Parameter column: Name of the column to count values in.
    /// - Returns: A two-column ``DataFrame`` (`value` and `count`).
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if `column` does not exist.
    public func valueCounts(of column: String) throws -> DataFrame {
        try series(column).valueCounts()
    }

    /// Replaces occurrences of `target` with `replacement` in the specified
    /// `columns` (or across all columns when `columns` is `nil`).
    ///
    /// - Parameters:
    ///   - target: Value to search for.
    ///   - replacement: Replacement ``DataValue``.
    ///   - targetColumns: Optional subset of column names to modify; `nil` modifies all columns.
    /// - Returns: A new ``DataFrame`` with matching values replaced.
    /// - Throws: ``DataFrameError`` if `replacement` is a non-finite number or any column in `targetColumns` does not exist.
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
        let updated: [DataColumn] = columns.map { col in
            guard scope == nil || scope!.contains(col.name) else {
                return col
            }
            return DataColumn(
                name: col.name,
                values: col.values.map { $0 == target ? replacement : $0 }
            )
        }
        return try DataFrame(dataColumns: updated)
    }

    /// Fills each missing value with the closest earlier non-missing value.
    ///
    /// - Returns: A forward-filled ``DataFrame``.
    /// - Throws: ``DataFrameError`` if constructing the result frame fails.
    public func forwardFilled() throws -> DataFrame {
        let updated = columns.map { column -> DataColumn in
            guard column.storage.hasNulls else { return column }
            return DataColumn(storage: Series(column: column).forwardFilled().storage)
        }
        return try DataFrame(dataColumns: updated)
    }

    /// Fills each missing value with the closest later non-missing value.
    ///
    /// - Returns: A backward-filled ``DataFrame``.
    /// - Throws: ``DataFrameError`` if constructing the result frame fails.
    public func backwardFilled() throws -> DataFrame {
        let updated = columns.map { column -> DataColumn in
            guard column.storage.hasNulls else { return column }
            return DataColumn(storage: Series(column: column).backwardFilled().storage)
        }
        return try DataFrame(dataColumns: updated)
    }

    /// Returns a Boolean frame that is `true` where the source is missing.
    ///
    /// - Returns: A ``DataFrame`` of `.boolean` columns indicating missingness at each cell.
    /// - Throws: ``DataFrameError`` if constructing the result frame fails.
    public func missingMask() throws -> DataFrame {
        try DataFrame(storageColumns: columns.map { column in
            BooleanColumn(
                name: column.name,
                bitmap: column.storage.nullBitmap,
                nullBitmap: BitSet(count: rowCount)
            )
        })
    }

    /// Returns a structured version of ``info()``.
    ///
    /// - Returns: A ``DataFrameInfo`` value describing row count, columns, and estimated memory usage.
    public func infoSummary() -> DataFrameInfo {
        var totalMemory = 0
        let columnInfos = columns.map { column -> DataFrameInfo.Column in
            let nonMissing = column.storage.count - column.storage.nullCount
            if column.numericBuffer != nil || column.storage is DateColumn {
                totalMemory += nonMissing * 8
            } else if column.storage is BooleanColumn {
                totalMemory += nonMissing
            } else {
                for value in column.values {
                    switch value {
                    case .number, .date: totalMemory += 8
                    case .boolean: totalMemory += 1
                    case .string(let s): totalMemory += s.utf8.count
                    case .missing: break
                    }
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

    /// Replaces the column named `name` with `values` in place.
    func replacingColumn(_ name: String, with values: [DataValue]) throws -> DataFrame {
        guard let idx = columns.firstIndex(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        var updated = columns
        updated[idx] = DataColumn(name: name, values: values)
        return try DataFrame(dataColumns: updated)
    }

    /// Replaces the column named `name` with typed columnar `storage` in place.
    func replacingColumn(_ name: String, withStorage storage: any AnyColumn) throws -> DataFrame {
        guard let idx = columns.firstIndex(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        var updated = columns
        updated[idx] = DataColumn(storage: storage.renamed(to: name))
        return try DataFrame(dataColumns: updated)
    }
}

/// Structured result of ``DataFrame/infoSummary()``.
public struct DataFrameInfo: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Per-column information.
    public struct Column: Equatable, Hashable, Sendable, Codable {
        /// The column name.
        public let name: String
        /// The number of non-missing values in the column.
        public let nonMissingCount: Int
        /// The column's data type.
        public let type: DataType

        /// Creates per-column information.
        ///
        /// - Parameters:
        ///   - name: The column name.
        ///   - nonMissingCount: Number of non-missing values in the column.
        ///   - type: The inferred ``DataType`` of the column.
        public init(name: String, nonMissingCount: Int, type: DataType) {
            self.name = name
            self.nonMissingCount = nonMissingCount
            self.type = type
        }
    }

    /// The number of rows in the frame.
    public let rowCount: Int
    /// Per-column information, in column order.
    public let columns: [Column]
    /// The estimated memory footprint of the frame, in bytes.
    public let estimatedMemoryBytes: Int

    /// Creates a frame information summary.
    ///
    /// - Parameters:
    ///   - rowCount: The number of rows.
    ///   - columns: Per-column information.
    ///   - estimatedMemoryBytes: The estimated memory use in bytes.
    public init(
        rowCount: Int,
        columns: [Column],
        estimatedMemoryBytes: Int = 0
    ) {
        self.rowCount = rowCount
        self.columns = columns
        self.estimatedMemoryBytes = estimatedMemoryBytes
    }

    /// Renders a multi-line summary of the row count and each column's name, non-null count, and type.
    ///
    /// - Returns: A formatted plain-text summary.
    public func describe() -> String {
        var lines = ["DataFrame: \(rowCount) rows x \(columns.count) columns"]
        for (index, column) in columns.enumerated() {
            lines.append("\(index)  \(column.name)  \(column.nonMissingCount) non-null  \(column.type)")
        }
        return lines.joined(separator: "\n")
    }

    /// A text summary; equivalent to ``describe()``.
    public var description: String { describe() }
    /// A debugging representation; identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    ///
    /// - Parameter target: Destination text output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}
