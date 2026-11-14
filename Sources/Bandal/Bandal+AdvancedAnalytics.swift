import Accelerate
import Foundation

// MARK: - Structured Series Description & Quantile Table

/// Structured descriptive summary of a ``Series`` (returned by ``Series/describeStructured()``).
public struct SeriesDescription: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// The name of the described series.
    public let name: String
    /// The logical data type of the series.
    public let dtype: DataType
    /// The number of non-missing values.
    public let count: Int
    /// The arithmetic mean, or `nil` for non-numeric series.
    public let mean: Double?
    /// The sample standard deviation, or `nil` when not computed.
    public let standardDeviation: Double?
    /// The minimum value, or `nil` for non-numeric series.
    public let min: Double?
    /// The 25th percentile, or `nil` for non-numeric series.
    public let q1: Double?
    /// The median (50th percentile), or `nil` for non-numeric series.
    public let median: Double?
    /// The 75th percentile, or `nil` for non-numeric series.
    public let q3: Double?
    /// The maximum value, or `nil` for non-numeric series.
    public let max: Double?
    /// The number of distinct non-missing values, or `nil` for numeric series.
    public let unique: Int?
    /// The most frequent value, or `nil` for numeric series.
    public let top: DataValue?
    /// The occurrence count of ``top``, or `nil` for numeric series.
    public let frequency: Int?

    /// Creates a series description from precomputed statistics.
    ///
    /// Statistics that do not apply to the series' data type default to `nil`.
    ///
    /// - Parameters:
    ///   - name: Name of the described series.
    ///   - dtype: Inferred ``DataType`` of the series.
    ///   - count: Number of non-missing values.
    ///   - mean: Arithmetic mean for numeric series.
    ///   - standardDeviation: Sample standard deviation for numeric series.
    ///   - min: Minimum value for numeric series.
    ///   - q1: First quartile (25th percentile) for numeric series.
    ///   - median: Median (50th percentile) for numeric series.
    ///   - q3: Third quartile (75th percentile) for numeric series.
    ///   - max: Maximum value for numeric series.
    ///   - unique: Distinct non-missing count for non-numeric series.
    ///   - top: Most frequent value for non-numeric series.
    ///   - frequency: Occurrence count of `top` for non-numeric series.
    public init(
        name: String,
        dtype: DataType,
        count: Int,
        mean: Double? = nil,
        standardDeviation: Double? = nil,
        min: Double? = nil,
        q1: Double? = nil,
        median: Double? = nil,
        q3: Double? = nil,
        max: Double? = nil,
        unique: Int? = nil,
        top: DataValue? = nil,
        frequency: Int? = nil
    ) {
        self.name = name
        self.dtype = dtype
        self.count = count
        self.mean = mean
        self.standardDeviation = standardDeviation
        self.min = min
        self.q1 = q1
        self.median = median
        self.q3 = q3
        self.max = max
        self.unique = unique
        self.top = top
        self.frequency = frequency
    }

    /// Renders the description as a pandas-style table.
    ///
    /// - Returns: Numeric statistics (count, mean, std, min, quartiles, max) for numeric series; otherwise count, unique, top, and freq.
    public func describe() -> String {
        func f(_ v: Double?) -> String { v.map { String(format: "%.6g", $0) } ?? "NaN" }
        if dtype == .number {
            let rows: [(String, String)] = [
                ("count", String(count)),
                ("mean", f(mean)),
                ("std", f(standardDeviation)),
                ("min", f(min)),
                ("25%", f(q1)),
                ("50%", f(median)),
                ("75%", f(q3)),
                ("max", f(max)),
            ]
            return DataFrame.renderTable(
                rowLabels: rows.map(\.0),
                columnNames: [name],
                values: rows.map { [$0.1] }
            ) + "\nName: \(name), dtype: float64"
        } else {
            let rows: [(String, String)] = [
                ("count", String(count)),
                ("unique", unique.map(String.init) ?? "NaN"),
                ("top", top.map(DataFrame.displayValue) ?? "NaN"),
                ("freq", frequency.map(String.init) ?? "NaN"),
            ]
            return DataFrame.renderTable(
                rowLabels: rows.map(\.0),
                columnNames: [name],
                values: rows.map { [$0.1] }
            ) + "\nName: \(name), dtype: \(dtype)"
        }
    }

    /// A pandas-style tabular text representation; equivalent to ``describe()``.
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

/// Structured result of multi-quantile computation across numeric columns (returned by ``DataFrame/quantiles(_:)``).
public struct QuantileTable: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// The probabilities (rows) at which quantiles were computed.
    public let probabilities: [Double]
    /// The names of the numeric columns, in table order.
    public let columnNames: [String]
    /// Row-major matrix where `values[pIndex][colIndex]` is the quantile at `probabilities[pIndex]` for `columnNames[colIndex]`.
    public let values: [[Double?]]

    /// Creates a quantile table.
    ///
    /// - Parameters:
    ///   - probabilities: The probabilities labeling each row.
    ///   - columnNames: The column names labeling each column.
    ///   - values: Row-major quantile values, indexed `[probabilityIndex][columnIndex]`; `nil` marks an undefined quantile.
    public init(probabilities: [Double], columnNames: [String], values: [[Double?]]) {
        self.probabilities = probabilities
        self.columnNames = columnNames
        self.values = values
    }

    /// Returns the quantile for a probability and column.
    ///
    /// - Parameters:
    ///   - probability: A probability present in ``probabilities`` (matched within `1e-12`).
    ///   - column: A name present in ``columnNames``.
    /// - Returns: The quantile value, or `nil` if either key is not found or the value is undefined.
    ///
    /// - Complexity: $O(p + c)$ for `p` probabilities and `c` columns.
    public subscript(probability: Double, column: String) -> Double? {
        guard let pIdx = probabilities.firstIndex(where: { Swift.abs($0 - probability) < 1e-12 }),
              let cIdx = columnNames.firstIndex(of: column)
        else {
            return nil
        }
        return values[pIdx][cIdx]
    }

    /// Converts the table to a ``DataFrame``.
    ///
    /// The result has a leading `quantile` column holding the probabilities, followed by one column per entry in ``columnNames``; undefined quantiles become missing values.
    ///
    /// - Throws: An error if the data frame cannot be constructed.
    /// - Returns: A data frame with one row per probability.
    public func toDataFrame() throws -> DataFrame {
        var cols: [(name: String, values: [DataValue])] = [
            (name: "quantile", values: probabilities.map(DataValue.number))
        ]
        for (cIdx, name) in columnNames.enumerated() {
            let colVals = probabilities.indices.map { pIdx in
                values[pIdx][cIdx].map(DataValue.number) ?? .missing
            }
            cols.append((name: name, values: colVals))
        }
        return try DataFrame(columns: cols)
    }

    /// Renders the table as pandas-style text with probabilities as row labels.
    ///
    /// - Returns: A formatted plain-text quantile table.
    public func describe() -> String {
        let rowLabels = probabilities.map { String(format: "%.4g", $0) }
        let formattedRows = values.map { row in
            row.map { cell in
                cell.map { String(format: "%.6g", $0) } ?? "NaN"
            }
        }
        return DataFrame.renderTable(
            rowLabels: rowLabels,
            columnNames: columnNames,
            values: formattedRows
        )
    }

    /// A pandas-style tabular text representation; equivalent to ``describe()``.
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

// MARK: - Direction for As-Of Joins

/// Matching direction for ``DataFrame/joinAsOf(with:on:by:direction:tolerance:)``.
public enum AsOfDirection: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    /// Matches the largest right key `<=` the left key (default in `pd.merge_asof`).
    case backward
    /// Matches the smallest right key `>=` the left key.
    case forward
    /// Matches the right key with the smallest absolute distance to the left key.
    case nearest

    /// The lowercase name of the direction.
    public var description: String { rawValue }
}

// MARK: - Series Vectorized Comparisons, Math, Binning, Interpolation & EWM

extension Series {
    /// Returns a structured ``SeriesDescription`` summarizing this series.
    ///
    /// - Returns: A ``SeriesDescription`` instance containing numeric or non-numeric summary metrics.
    public func describeStructured() -> SeriesDescription {
        if let stats = statistics() {
            return SeriesDescription(
                name: name,
                dtype: .number,
                count: stats.count,
                mean: stats.mean,
                standardDeviation: stats.standardDeviation,
                min: stats.min,
                q1: stats.q1,
                median: stats.median,
                q3: stats.q3,
                max: stats.max
            )
        } else {
            let nonMissing = values.filter { $0 != .missing }
            var counts: [DataValue: Int] = [:]
            for v in nonMissing { counts[v, default: 0] += 1 }
            let topFreq = counts.values.max() ?? 0
            let topVal = nonMissing.first { counts[$0] == topFreq }
            return SeriesDescription(
                name: name,
                dtype: dtype,
                count: nonMissing.count,
                unique: counts.count,
                top: topVal,
                frequency: nonMissing.isEmpty ? nil : topFreq
            )
        }
    }

    // MARK: Vectorized BooleanMask Comparisons

    /// Element-wise `>` comparison against a numeric scalar. Missing cells evaluate to `false`.
    public static func > (lhs: Series, rhs: Double) -> BooleanMask {
        lhs.compareScalar(rhs) { $0 > $1 }
    }

    /// Element-wise `>=` comparison against a numeric scalar. Missing cells evaluate to `false`.
    public static func >= (lhs: Series, rhs: Double) -> BooleanMask {
        lhs.compareScalar(rhs) { $0 >= $1 }
    }

    /// Element-wise `<` comparison against a numeric scalar. Missing cells evaluate to `false`.
    public static func < (lhs: Series, rhs: Double) -> BooleanMask {
        lhs.compareScalar(rhs) { $0 < $1 }
    }

    /// Element-wise `<=` comparison against a numeric scalar. Missing cells evaluate to `false`.
    public static func <= (lhs: Series, rhs: Double) -> BooleanMask {
        lhs.compareScalar(rhs) { $0 <= $1 }
    }

    /// Element-wise equality comparison against a numeric scalar.
    ///
    /// - Parameter scalar: Target `Double` value.
    /// - Returns: A ``BooleanMask`` marking matching rows.
    public func equals(_ scalar: Double) -> BooleanMask {
        compareScalar(scalar) { $0 == $1 }
    }

    /// Element-wise equality comparison against a `String`.
    ///
    /// - Parameter string: Target `String` value.
    /// - Returns: A ``BooleanMask`` marking matching rows.
    public func equals(_ string: String) -> BooleanMask {
        equals(.string(string))
    }

    /// Element-wise equality comparison against a `Bool`.
    ///
    /// - Parameter boolean: Target `Bool` value.
    /// - Returns: A ``BooleanMask`` marking matching rows.
    public func equals(_ boolean: Bool) -> BooleanMask {
        equals(.boolean(boolean))
    }

    /// Element-wise equality comparison against a `DataValue`.
    ///
    /// - Parameter target: Target ``DataValue``.
    /// - Returns: A ``BooleanMask`` marking matching rows.
    public func equals(_ target: DataValue) -> BooleanMask {
        var bm = BitSet(count: count)
        if target == .missing {
            return BooleanMask(bitSet: storage.nullBitmap)
        }
        if case .number(let n) = target, numericBuffer != nil {
            return compareScalar(n) { $0 == $1 }
        }
        for i in 0..<count where storage.value(at: i) == target {
            bm[i] = true
        }
        return BooleanMask(bitSet: bm)
    }

    /// Element-wise inequality comparison against a `DataValue`.
    ///
    /// - Parameter target: Target ``DataValue``.
    /// - Returns: A ``BooleanMask`` marking rows not equal to `target`.
    public func notEquals(_ target: DataValue) -> BooleanMask {
        !equals(target)
    }

    /// Evaluates `predicate(element, rhs)` across all numeric elements, packing the result into a 64-bit word ``BooleanMask``.
    private func compareScalar(_ rhs: Double, _ predicate: (Double, Double) -> Bool) -> BooleanMask {
        let n = count
        if n == 0 {
            return BooleanMask(bitSet: BitSet())
        }
        if let buffer = numericBuffer {
            let wordCount = (n + 63) >> 6
            let hasNulls = buffer.hasMissing
            let nullWords = buffer.floatColumn.nullBitmap.words
            var anyTrue = false
            let words = ContiguousArray<UInt64>(unsafeUninitializedCapacity: wordCount) { outBuf, initCount in
                buffer.floatColumn.values.withUnsafeBufferPointer { valBuf in
                    guard let valBase = valBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                    let fullWords = n >> 6
                    if !hasNulls {
                        for w in 0..<fullWords {
                            let baseIdx = w << 6
                            var word: UInt64 = 0
                            for bit in 0..<64 {
                                if predicate(valBase[baseIdx + bit], rhs) {
                                    word |= (1 &<< UInt64(bit))
                                }
                            }
                            if word != 0 { anyTrue = true }
                            outBase[w] = word
                        }
                        let rem = n & 63
                        if rem > 0 {
                            let baseIdx = fullWords << 6
                            var word: UInt64 = 0
                            for bit in 0..<rem {
                                if predicate(valBase[baseIdx + bit], rhs) {
                                    word |= (1 &<< UInt64(bit))
                                }
                            }
                            if word != 0 { anyTrue = true }
                            outBase[fullWords] = word
                        }
                    } else {
                        for w in 0..<fullWords {
                            let baseIdx = w << 6
                            let nWord = nullWords[w]
                            var word: UInt64 = 0
                            for bit in 0..<64 {
                                let mask = (1 as UInt64) &<< UInt64(bit)
                                if (nWord & mask) == 0 && predicate(valBase[baseIdx + bit], rhs) {
                                    word |= mask
                                }
                            }
                            if word != 0 { anyTrue = true }
                            outBase[w] = word
                        }
                        let rem = n & 63
                        if rem > 0 {
                            let baseIdx = fullWords << 6
                            let nWord = nullWords[fullWords]
                            var word: UInt64 = 0
                            for bit in 0..<rem {
                                let mask = (1 as UInt64) &<< UInt64(bit)
                                if (nWord & mask) == 0 && predicate(valBase[baseIdx + bit], rhs) {
                                    word |= mask
                                }
                            }
                            if word != 0 { anyTrue = true }
                            outBase[fullWords] = word
                        }
                    }
                }
                initCount = wordCount
            }
            return BooleanMask(bitSet: BitSet(words: words, count: n, mayHaveTrueBits: anyTrue))
        }
        var bm = BitSet(count: n)
        for i in 0..<n {
            if case .number(let v) = storage.value(at: i), predicate(v, rhs) {
                bm[i] = true
            }
        }
        return BooleanMask(bitSet: bm)
    }

    // MARK: Unary Math & Power Functions

    /// Applies a unary mathematical function (`sqrt`, `exp`, `log`, `log10`, `log1p`, `sin`, `cos`, `tan`, `sign`) element-wise.
    ///
    /// Domain violations (such as `log(-1)` or `sqrt(-1)`) become `.missing` rather than `NaN`.
    ///
    /// - Parameter function: The ``UnaryMathFunction`` to evaluate.
    /// - Returns: A new ``Series`` with transformed numeric values.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func applying(_ function: UnaryMathFunction) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let col = NumericEngine.unaryMathColumn(buffer: buffer, name: name, function: function) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let col = NumericEngine.nativeUnaryMathColumn(buffer: buffer, name: name, function: function)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Raises every numeric element to `exponent`. Non-finite results become `.missing`.
    ///
    /// - Parameter exponent: Power exponent.
    /// - Returns: A new ``Series`` containing `x ** exponent`.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func power(_ exponent: Double) throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if let col = NumericEngine.powerColumn(buffer: buffer, name: name, exponent: exponent) {
            return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
        }
        let srcVals = buffer.floatColumn.values
        var outVals = ContiguousArray<Double>(repeating: 0.0, count: count)
        var outNulls = buffer.floatColumn.nullBitmap
        for i in 0..<count where !outNulls[i] {
            let y = Foundation.pow(srcVals[i], exponent)
            if y.isFinite {
                outVals[i] = y
            } else {
                outNulls[i] = true
            }
        }
        let col = Float64Column(name: name, values: outVals, nullBitmap: outNulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    /// Standardizes numeric values to zero mean and unit sample standard deviation (`z = (x - mean) / std`).
    ///
    /// - Returns: A standardized ``Series`` (or all `.missing` when standard deviation is zero or undefined).
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func standardized() throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        let moments = NumericEngine.moments(of: buffer.numbers)
        guard let mean = moments.mean, let std = moments.standardDeviation, std > 0 else {
            let allMissing = Float64Column(
                name: name,
                values: ContiguousArray(repeating: 0.0, count: count),
                nullBitmap: BitSet(count: count, repeating: true)
            )
            return Series(storage: allMissing, numericBuffer: NumericBuffer(floatColumn: allMissing))
        }
        let centered = try applying(.subtract, scalar: mean)
        return try centered.applying(.divide, scalar: std)
    }

    /// Scales numeric values linearly into `0...1` (`(x - min) / (max - min)`).
    ///
    /// - Returns: A min-max scaled ``Series`` in `0...1` (or all `.missing` when `max == min`).
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func minMaxScaled() throws -> Series {
        guard numericBuffer != nil else {
            throw DataFrameError.nonNumericColumn(name)
        }
        guard let minVal = min(), let maxVal = max(), maxVal > minVal else {
            let allMissing = Float64Column(
                name: name,
                values: ContiguousArray(repeating: 0.0, count: count),
                nullBitmap: BitSet(count: count, repeating: true)
            )
            return Series(storage: allMissing, numericBuffer: NumericBuffer(floatColumn: allMissing))
        }
        let shifted = try applying(.subtract, scalar: minVal)
        return try shifted.applying(.divide, scalar: maxVal - minVal)
    }

    // MARK: Index of Min / Max (idxmax / idxmin)

    /// Returns the row index of the first occurrence of the maximum numeric value, or `nil` if empty.
    ///
    /// - Returns: Zero-based row index of the maximum value, or `nil`.
    public func indexOfMax() -> Int? {
        guard let buffer = numericBuffer else { return nil }
        let vals = buffer.floatColumn.values
        let nulls = buffer.floatColumn.nullBitmap
        var bestIndex: Int?
        var bestValue = -Double.infinity
        for i in 0..<count where !nulls[i] {
            if vals[i] > bestValue {
                bestValue = vals[i]
                bestIndex = i
            }
        }
        return bestIndex
    }

    /// Returns the row index of the first occurrence of the minimum numeric value, or `nil` if empty.
    ///
    /// - Returns: Zero-based row index of the minimum value, or `nil`.
    public func indexOfMin() -> Int? {
        guard let buffer = numericBuffer else { return nil }
        let vals = buffer.floatColumn.values
        let nulls = buffer.floatColumn.nullBitmap
        var bestIndex: Int?
        var bestValue = Double.infinity
        for i in 0..<count where !nulls[i] {
            if vals[i] < bestValue {
                bestValue = vals[i]
                bestIndex = i
            }
        }
        return bestIndex
    }

    // MARK: Conditional Replacement (where / mask)

    /// Keeps elements where `mask` is `true` and replaces elements where `mask` is `false` with `replacement`.
    ///
    /// - Parameters:
    ///   - mask: Boolean mask of length ``count``.
    ///   - replacement: Value used when `mask[i]` is `false`. Defaults to `.missing`.
    /// - Returns: A new ``Series`` with masked replacements applied.
    /// - Throws: ``DataFrameError`` if `mask.count != count` or `replacement` is a non-finite number.
    public func `where`(_ mask: BooleanMask, otherwise replacement: DataValue = .missing) throws -> Series {
        guard mask.count == count else {
            throw DataFrameError.unequalColumnLengths(column: name, expected: count, actual: mask.count)
        }
        if case .number(let n) = replacement, !n.isFinite {
            throw DataFrameError.nonFiniteValue(column: name, row: 0)
        }
        if let buffer = numericBuffer {
            if replacement == .missing {
                let newNulls = buffer.floatColumn.nullBitmap.bitwiseOr(mask.bitSet.inverted())
                var newVals = buffer.floatColumn.values
                for i in 0..<count where newNulls[i] {
                    newVals[i] = 0.0
                }
                let col = Float64Column(name: name, values: newVals, nullBitmap: newNulls)
                return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
            } else if case .number(let repNum) = replacement {
                var newVals = buffer.floatColumn.values
                var newNulls = buffer.floatColumn.nullBitmap
                for i in 0..<count where !mask[i] {
                    newVals[i] = repNum
                    newNulls[i] = false
                }
                let col = Float64Column(name: name, values: newVals, nullBitmap: newNulls)
                return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
            }
        }
        let out = (0..<count).map { i in
            mask[i] ? storage.value(at: i) : replacement
        }
        return Series(name: name, values: out)
    }

    // MARK: Linear Interpolation

    /// Fills `.missing` interior gaps between numeric values via linear interpolation.
    ///
    /// Leading and trailing `.missing` values remain `.missing`.
    ///
    /// - Returns: A new ``Series`` with interior missing gaps linearly interpolated.
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series is not numeric.
    public func interpolated() throws -> Series {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        if !buffer.hasMissing {
            return self
        }
        var outVals = buffer.floatColumn.values
        var outNulls = buffer.floatColumn.nullBitmap
        let n = count
        var i = 0
        while i < n {
            if !outNulls[i] {
                let startIdx = i
                let startVal = outVals[startIdx]
                var j = i + 1
                while j < n && outNulls[j] {
                    j += 1
                }
                if j > startIdx + 1 && j < n {
                    let endVal = outVals[j]
                    let span = Double(j - startIdx)
                    for k in (startIdx + 1)..<j {
                        let t = Double(k - startIdx) / span
                        outVals[k] = startVal + (endVal - startVal) * t
                        outNulls[k] = false
                    }
                }
                i = j
            } else {
                i += 1
            }
        }
        let col = Float64Column(name: name, values: outVals, nullBitmap: outNulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    // MARK: Exponentially Weighted Moving Average (ewm)

    /// Computes the exponentially weighted moving average with smoothing factor `alpha` in `(0, 1]`.
    ///
    /// When `adjust == true` (the Pandas default), weights are normalized by decaying powers of `1 - alpha`;
    /// when `adjust == false`, the recursive formula `y_t = (1 - alpha) * y_{t-1} + alpha * x_t` is used.
    ///
    /// - Parameters:
    ///   - alpha: Smoothing factor in `(0, 1]`.
    ///   - adjust: Whether to divide by the decaying normalization factor in beginning periods. Defaults to `true`.
    /// - Returns: A new ``Series`` containing the exponentially weighted moving average.
    /// - Throws: ``DataFrameError`` if `alpha` is outside `(0, 1]` or the series is not numeric.
    public func exponentiallyWeightedMean(alpha: Double, adjust: Bool = true) throws -> Series {
        guard alpha > 0, alpha <= 1, alpha.isFinite else {
            throw DataFrameError.invalidWindowSize(0)
        }
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        let srcVals = buffer.floatColumn.values
        let srcNulls = buffer.floatColumn.nullBitmap
        var outVals = ContiguousArray<Double>(repeating: 0.0, count: count)
        var outNulls = BitSet(count: count)
        let decay = 1.0 - alpha

        var weightedSum = 0.0
        var weightTotal = 0.0
        var recursiveMean: Double?

        for i in 0..<count {
            if srcNulls[i] {
                if let current = recursiveMean {
                    outVals[i] = current
                } else {
                    outNulls[i] = true
                }
                continue
            }
            let x = srcVals[i]
            if adjust {
                weightedSum = x + decay * weightedSum
                weightTotal = 1.0 + decay * weightTotal
                let y = weightedSum / weightTotal
                outVals[i] = y
                recursiveMean = y
            } else {
                if let prev = recursiveMean {
                    let y = decay * prev + alpha * x
                    outVals[i] = y
                    recursiveMean = y
                } else {
                    outVals[i] = x
                    recursiveMean = x
                }
            }
        }
        let col = Float64Column(name: name, values: outVals, nullBitmap: outNulls)
        return Series(storage: col, numericBuffer: NumericBuffer(floatColumn: col))
    }

    // MARK: Binning / Discretization (cut / qcut)

    /// Discretizes numeric values into intervals defined by strictly increasing `edges` (analogous to `pd.cut`).
    ///
    /// By default (`rightClosed == true`), bin `k` is `(edges[k], edges[k + 1]]` (with `edges[0]` included when `includeLowest == true`).
    ///
    /// - Parameters:
    ///   - edges: Strictly increasing bin boundary values (at least 2).
    ///   - labels: Optional custom bin labels of length `edges.count - 1`.
    ///   - rightClosed: Whether intervals are closed on the right. Defaults to `true`.
    ///   - includeLowest: Whether the first interval includes the minimum edge. Defaults to `true`.
    /// - Returns: A `.string` ``Series`` containing bin labels (or `.missing` for out-of-range values).
    /// - Throws: ``DataFrameError`` if `edges` is not strictly increasing, `labels` has the wrong count, or the series is non-numeric.
    public func binned(
        edges: [Double],
        labels: [String]? = nil,
        rightClosed: Bool = true,
        includeLowest: Bool = true
    ) throws -> Series {
        guard edges.count >= 2 else {
            throw DataFrameError.invalidWindowSize(edges.count)
        }
        for i in 1..<edges.count where edges[i] <= edges[i - 1] {
            throw DataFrameError.invalidWindowSize(i)
        }
        let binCount = edges.count - 1
        if let labels, labels.count != binCount {
            throw DataFrameError.unequalColumnLengths(column: name, expected: binCount, actual: labels.count)
        }
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        let resolvedLabels: [String] = labels ?? (0..<binCount).map { k in
            let lo = String(format: "%.6g", edges[k])
            let hi = String(format: "%.6g", edges[k + 1])
            if rightClosed {
                let leftBracket = (k == 0 && includeLowest) ? "[" : "("
                return "\(leftBracket)\(lo), \(hi)]"
            } else {
                let rightBracket = (k == binCount - 1 && includeLowest) ? "]" : ")"
                return "[\(lo), \(hi)\(rightBracket)"
            }
        }

        let vals = buffer.floatColumn.values
        let nulls = buffer.floatColumn.nullBitmap
        var out: [DataValue] = Array(repeating: .missing, count: count)
        for i in 0..<count where !nulls[i] {
            let x = vals[i]
            var matchedBin: Int?
            for k in 0..<binCount {
                let lo = edges[k]
                let hi = edges[k + 1]
                let inBin: Bool
                if rightClosed {
                    if k == 0 && includeLowest {
                        inBin = x >= lo && x <= hi
                    } else {
                        inBin = x > lo && x <= hi
                    }
                } else {
                    if k == binCount - 1 && includeLowest {
                        inBin = x >= lo && x <= hi
                    } else {
                        inBin = x >= lo && x < hi
                    }
                }
                if inBin {
                    matchedBin = k
                    break
                }
            }
            if let k = matchedBin {
                out[i] = .string(resolvedLabels[k])
            }
        }
        return Series(name: name, values: out)
    }

    /// Discretizes numeric values into `quantiles` equal-sized quantile buckets (analogous to `pd.qcut`).
    ///
    /// - Parameters:
    ///   - quantiles: Number of quantile buckets (`>= 1`).
    ///   - labels: Optional custom bucket labels.
    /// - Returns: A `.string` ``Series`` containing quantile bucket labels.
    /// - Throws: ``DataFrameError`` if `quantiles < 1`, `labels` count mismatches, or the series is non-numeric.
    public func quantileBinned(quantiles: Int, labels: [String]? = nil) throws -> Series {
        guard quantiles >= 1 else {
            throw DataFrameError.invalidWindowSize(quantiles)
        }
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        let nums = buffer.numbers
        guard !nums.isEmpty else {
            return Series(name: name, values: Array(repeating: .missing, count: count))
        }
        let sorted = MLXDeviceVector(nums, minimumCount: MLXBackend.automaticComplexMinimumCount)?.sortedValues() ?? nums.sorted()
        var edges: [Double] = []
        edges.reserveCapacity(quantiles + 1)
        for k in 0...quantiles {
            let p = Double(k) / Double(quantiles)
            edges.append(NumericEngine.quantile(sorted: sorted, probability: p))
        }
        var uniqueEdges: [Double] = []
        for e in edges {
            if let last = uniqueEdges.last, e <= last {
                continue
            }
            uniqueEdges.append(e)
        }
        guard uniqueEdges.count >= 2 else {
            let label = labels?.first ?? "[\(String(format: "%.6g", sorted.first!)), \(String(format: "%.6g", sorted.last!))]"
            let nulls = buffer.floatColumn.nullBitmap
            return Series(
                name: name,
                values: (0..<count).map { nulls[$0] ? .missing : .string(label) }
            )
        }
        return try binned(edges: uniqueEdges, labels: labels, rightClosed: true, includeLowest: true)
    }
}

// MARK: - DataFrame Advanced Analytics, Reshaping, Time Series & Functional Chaining

extension DataFrame {
    // MARK: Vectorized BooleanMask Indexing

    /// Filters rows using a ``BooleanMask``.
    ///
    /// - Parameter mask: A ``BooleanMask`` whose length equals ``rowCount``.
    /// - Returns: A new ``DataFrame`` containing only rows where `mask` is `true`.
    /// - Throws: ``DataFrameError/unequalColumnLengths(column:expected:actual:)`` if `mask.count != rowCount`.
    public func filtered(by mask: BooleanMask) throws -> DataFrame {
        guard mask.count == rowCount else {
            throw DataFrameError.unequalColumnLengths(
                column: "mask",
                expected: rowCount,
                actual: mask.count
            )
        }
        return try rows(matching: mask.bitSet)
    }

    /// Subscript overload returning rows where `mask` is `true` (or an empty 0-row frame if `mask` mismatches).
    ///
    /// - Parameter mask: A ``BooleanMask`` to filter rows by.
    /// - Returns: A filtered ``DataFrame``.
    public subscript(mask: BooleanMask) -> DataFrame {
        (try? filtered(by: mask)) ?? (try! head(0))
    }

    /// Evaluates multiple quantiles across all numeric columns in a single pass per column.
    ///
    /// - Parameter probabilities: Fractional probabilities in `0...1`.
    /// - Returns: A ``QuantileTable`` containing quantiles for all numeric columns.
    /// - Throws: ``DataFrameError/emptyColumns`` if the frame has no numeric columns.
    public func quantiles(_ probabilities: [Double]) throws -> QuantileTable {
        let numericCols = columns.filter { (try? dataType(of: $0.name)) == .number }
        guard !numericCols.isEmpty else { throw DataFrameError.emptyColumns }
        let sortedArrays: [[Double]] = numericCols.map { col in
            guard let nums = col.numericBuffer?.numbers, !nums.isEmpty else { return [] }
            return MLXDeviceVector(nums, minimumCount: MLXBackend.automaticComplexMinimumCount)?.sortedValues() ?? nums.sorted()
        }
        let tableValues: [[Double?]] = probabilities.map { p in
            sortedArrays.map { sorted in
                sorted.isEmpty ? nil : NumericEngine.quantile(sorted: sorted, probability: p)
            }
        }
        return QuantileTable(
            probabilities: probabilities,
            columnNames: numericCols.map(\.name),
            values: tableValues
        )
    }

    // MARK: Unary Math, Power & Feature Scaling on DataFrame

    /// Applies a unary mathematical function (`sqrt`, `exp`, `log`, `log10`, `log1p`, `sin`, `cos`, `tan`, `sign`) to `column`.
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column to transform.
    ///   - function: The ``UnaryMathFunction`` to apply.
    ///   - outputColumn: Optional output column name; `nil` replaces `column` in place.
    /// - Returns: A new ``DataFrame`` containing the transformed column.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func applying(
        _ column: String,
        function: UnaryMathFunction,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let updated = try series(column).applying(function)
        let target = outputColumn ?? column
        return try withColumn(storage: updated.storage.renamed(to: target))
    }

    /// Raises `column` to `exponent` and stores the result in `outputColumn` (or replaces `column`).
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column.
    ///   - exponent: Power exponent.
    ///   - outputColumn: Optional output column name; `nil` replaces `column` in place.
    /// - Returns: A new ``DataFrame`` containing the exponentiated column.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func power(
        _ column: String,
        exponent: Double,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let updated = try series(column).power(exponent)
        let target = outputColumn ?? column
        return try withColumn(storage: updated.storage.renamed(to: target))
    }

    /// Standardizes `column` to zero mean and unit standard deviation (`(x - mean) / std`).
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column.
    ///   - outputColumn: Optional output column name; `nil` replaces `column` in place.
    /// - Returns: A new ``DataFrame`` containing the standardized column.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func standardized(
        _ column: String,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let updated = try series(column).standardized()
        let target = outputColumn ?? column
        return try withColumn(storage: updated.storage.renamed(to: target))
    }

    /// Scales `column` linearly into `0...1` (`(x - min) / (max - min)`).
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column.
    ///   - outputColumn: Optional output column name; `nil` replaces `column` in place.
    /// - Returns: A new ``DataFrame`` containing the min-max scaled column.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func minMaxScaled(
        _ column: String,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let updated = try series(column).minMaxScaled()
        let target = outputColumn ?? column
        return try withColumn(storage: updated.storage.renamed(to: target))
    }

    // MARK: Binning, Interpolation, Conditional Replacement & EWM on DataFrame

    /// Discretizes `column` into intervals defined by `edges` (analogous to `pd.cut`).
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column to bin.
    ///   - edges: Strictly increasing bin boundary values.
    ///   - labels: Optional custom bin labels.
    ///   - rightClosed: Whether intervals are closed on the right. Defaults to `true`.
    ///   - includeLowest: Whether the first interval includes the minimum edge. Defaults to `true`.
    ///   - outputColumn: Optional output column name; defaults to `"\(column)_bin"`.
    /// - Returns: A new ``DataFrame`` with the binned column added or replaced.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric, or `edges` / `labels` are invalid.
    public func binned(
        _ column: String,
        edges: [Double],
        labels: [String]? = nil,
        rightClosed: Bool = true,
        includeLowest: Bool = true,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let updated = try series(column).binned(
            edges: edges,
            labels: labels,
            rightClosed: rightClosed,
            includeLowest: includeLowest
        )
        let target = outputColumn ?? "\(column)_bin"
        return try withColumn(storage: updated.storage.renamed(to: target))
    }

    /// Discretizes `column` into `quantiles` equal-sized quantile buckets (analogous to `pd.qcut`).
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column to bin.
    ///   - quantiles: Number of quantile buckets (`>= 1`).
    ///   - labels: Optional custom bucket labels.
    ///   - outputColumn: Optional output column name; defaults to `"\(column)_qbin"`.
    /// - Returns: A new ``DataFrame`` with the quantile-binned column added or replaced.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric, or `quantiles` is invalid.
    public func quantileBinned(
        _ column: String,
        quantiles: Int,
        labels: [String]? = nil,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let updated = try series(column).quantileBinned(quantiles: quantiles, labels: labels)
        let target = outputColumn ?? "\(column)_qbin"
        return try withColumn(storage: updated.storage.renamed(to: target))
    }

    /// Linearly interpolates interior `.missing` values in `column` (or across all numeric columns when `column == nil`).
    ///
    /// - Parameter column: Optional numeric column name; `nil` interpolates every numeric column in the frame.
    /// - Returns: A new ``DataFrame`` with interior missing gaps linearly interpolated.
    /// - Throws: ``DataFrameError`` if `column` is specified but missing or non-numeric.
    public func interpolated(_ column: String? = nil) throws -> DataFrame {
        if let column {
            let updated = try series(column).interpolated()
            return try replacingColumn(column, withStorage: updated.storage)
        }
        let updatedCols = try columns.map { col -> DataColumn in
            guard col.numericBuffer != nil, col.storage.hasNulls else { return col }
            let interp = try Series(column: col).interpolated()
            return DataColumn(storage: interp.storage)
        }
        return try DataFrame(dataColumns: updatedCols)
    }

    /// Adds an exponentially weighted moving average column `column_ewm`.
    ///
    /// - Parameters:
    ///   - column: Name of the numeric column.
    ///   - alpha: Smoothing factor in `(0, 1]`.
    ///   - adjust: Whether to divide by the decaying normalization factor. Defaults to `true`.
    ///   - outputColumn: Optional output column name; defaults to `"\(column)_ewm"`.
    /// - Returns: A new ``DataFrame`` containing the EWM column.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric, `alpha` is invalid, or the default output name conflicts.
    public func exponentiallyWeighted(
        _ column: String,
        alpha: Double,
        adjust: Bool = true,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let ewm = try series(column).exponentiallyWeightedMean(alpha: alpha, adjust: adjust)
        let target = outputColumn ?? "\(column)_ewm"
        guard !columnNames.contains(target) || outputColumn != nil else {
            throw DataFrameError.columnNameConflict(target)
        }
        return try withColumn(storage: ewm.storage.renamed(to: target))
    }

    /// Returns the row index of the first maximum numeric value in `column`.
    ///
    /// - Parameter column: Name of the numeric column.
    /// - Returns: Zero-based row index of the maximum value, or `nil` if the column has no numeric values.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func indexOfMax(in column: String) throws -> Int? {
        let s = try series(column)
        guard s.numericBuffer != nil else {
            throw DataFrameError.nonNumericColumn(column)
        }
        return s.indexOfMax()
    }

    /// Returns the row index of the first minimum numeric value in `column`.
    ///
    /// - Parameter column: Name of the numeric column.
    /// - Returns: Zero-based row index of the minimum value, or `nil` if the column has no numeric values.
    /// - Throws: ``DataFrameError`` if `column` is missing or non-numeric.
    public func indexOfMin(in column: String) throws -> Int? {
        let s = try series(column)
        guard s.numericBuffer != nil else {
            throw DataFrameError.nonNumericColumn(column)
        }
        return s.indexOfMin()
    }

    // MARK: Exact Pivot, Cross-Tabulation, One-Hot Encoding & Pipe

    /// Reshapes unique `(index, columns)` pairs into a wide table without aggregation (analogous to `df.pivot`).
    ///
    /// - Parameters:
    ///   - index: Column name whose distinct values form the output rows.
    ///   - columnKey: Column name whose distinct values form the output columns.
    ///   - valueKey: Column name supplying cell values.
    /// - Returns: A reshaped wide ``DataFrame``.
    /// - Throws: ``DataFrameError/duplicateColumnName(_:)`` if duplicate `(index, columns)` pairs exist, or ``DataFrameError/columnNotFound(_:)`` if any column is missing.
    public func pivoted(
        index: String,
        columns columnKey: String,
        values valueKey: String
    ) throws -> DataFrame {
        let indexVals = try values(in: index)
        let colVals = try values(in: columnKey)
        let valVals = try values(in: valueKey)

        var rowKeys: [DataValue] = []
        var rowIndexByKey: [DataValue: Int] = [:]
        var pivotColNames: [String] = []
        var colIndexByName: [String: Int] = [:]

        for row in 0..<rowCount {
            let rKey = indexVals[row]
            let cVal = colVals[row]
            guard rKey != .missing, cVal != .missing else { continue }
            let cName = Self.displayValue(cVal)
            if rowIndexByKey[rKey] == nil {
                rowIndexByKey[rKey] = rowKeys.count
                rowKeys.append(rKey)
            }
            if colIndexByName[cName] == nil {
                colIndexByName[cName] = pivotColNames.count
                pivotColNames.append(cName)
            }
        }

        var grid = Array(
            repeating: Array(repeating: DataValue.missing, count: rowKeys.count),
            count: pivotColNames.count
        )
        var seenPairs = Set<String>()
        for row in 0..<rowCount {
            let rKey = indexVals[row]
            let cVal = colVals[row]
            guard rKey != .missing, cVal != .missing else { continue }
            let cName = Self.displayValue(cVal)
            let pairKey = "\(Self.displayValue(rKey))|\(cName)"
            guard seenPairs.insert(pairKey).inserted else {
                throw DataFrameError.duplicateColumnName(pairKey)
            }
            let rIdx = rowIndexByKey[rKey]!
            let cIdx = colIndexByName[cName]!
            grid[cIdx][rIdx] = valVals[row]
        }

        var output: [(name: String, values: [DataValue])] = [(name: index, values: rowKeys)]
        for (cIdx, name) in pivotColNames.enumerated() {
            output.append((name: name, values: grid[cIdx]))
        }
        return try DataFrame(columns: output)
    }

    /// Computes a contingency frequency table (or normalized proportions) of two columns (analogous to `pd.crosstab`).
    ///
    /// - Parameters:
    ///   - index: Column name whose distinct values form the table rows.
    ///   - columnKey: Column name whose distinct values form the table columns.
    ///   - normalize: When `true`, divides all counts by the grand total so cells sum to `1.0`. Defaults to `false`.
    /// - Returns: A contingency ``DataFrame``.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if either column does not exist.
    public func crossTabulated(
        index: String,
        columns columnKey: String,
        normalize: Bool = false
    ) throws -> DataFrame {
        let indexVals = try values(in: index)
        let colVals = try values(in: columnKey)

        var rowKeys: [DataValue] = []
        var rowIndexByKey: [DataValue: Int] = [:]
        var colNames: [String] = []
        var colIndexByName: [String: Int] = [:]

        for row in 0..<rowCount {
            let rKey = indexVals[row]
            let cVal = colVals[row]
            guard rKey != .missing, cVal != .missing else { continue }
            let cName = Self.displayValue(cVal)
            if rowIndexByKey[rKey] == nil {
                rowIndexByKey[rKey] = rowKeys.count
                rowKeys.append(rKey)
            }
            if colIndexByName[cName] == nil {
                colIndexByName[cName] = colNames.count
                colNames.append(cName)
            }
        }

        var counts = Array(
            repeating: Array(repeating: 0.0, count: rowKeys.count),
            count: colNames.count
        )
        var total = 0.0
        for row in 0..<rowCount {
            let rKey = indexVals[row]
            let cVal = colVals[row]
            guard rKey != .missing, cVal != .missing else { continue }
            let cName = Self.displayValue(cVal)
            counts[colIndexByName[cName]!][rowIndexByKey[rKey]!] += 1.0
            total += 1.0
        }

        let denom = (normalize && total > 0) ? total : 1.0
        var outCols: [any AnyColumn] = [AnyColumnFactory.make(name: index, values: rowKeys)]
        for (cIdx, name) in colNames.enumerated() {
            let scaled = ContiguousArray(counts[cIdx].map { $0 / denom })
            outCols.append(Float64Column(name: name, values: scaled, nullBitmap: BitSet(count: rowKeys.count)))
        }
        return try DataFrame(storageColumns: outCols)
    }

    /// Expands categorical or string columns into binary `0.0` / `1.0` indicator columns (analogous to `pd.get_dummies`).
    ///
    /// - Parameters:
    ///   - targetColumns: Names of the columns to expand into one-hot indicator columns.
    ///   - dropFirst: When `true`, omits the first category of each column to avoid multicollinearity. Defaults to `false`.
    /// - Returns: A new ``DataFrame`` with `targetColumns` replaced by binary indicator columns.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if any target column does not exist.
    public func oneHotEncoded(
        _ targetColumns: [String],
        dropFirst: Bool = false
    ) throws -> DataFrame {
        for name in targetColumns where !columnNames.contains(name) {
            throw DataFrameError.columnNotFound(name)
        }
        let targetSet = Set(targetColumns)
        var outColumns: [DataColumn] = []

        for col in columns {
            guard targetSet.contains(col.name) else {
                outColumns.append(col)
                continue
            }
            let s = Series(column: col)
            var categories = s.unique()
            if dropFirst, !categories.isEmpty {
                categories.removeFirst()
            }
            let colVals = col.values
            for cat in categories {
                let indicatorName = "\(col.name)_\(Self.displayValue(cat))"
                var indicator = ContiguousArray<Double>(repeating: 0.0, count: rowCount)
                for row in 0..<rowCount where colVals[row] == cat {
                    indicator[row] = 1.0
                }
                let floatCol = Float64Column(
                    name: indicatorName,
                    values: indicator,
                    nullBitmap: BitSet(count: rowCount)
                )
                outColumns.append(DataColumn(storage: floatCol))
            }
        }
        guard !outColumns.isEmpty else { throw DataFrameError.emptyColumns }
        return try DataFrame(dataColumns: outColumns)
    }

    /// Applies `transform` to `self` for fluent functional method chaining (analogous to `df.pipe`).
    ///
    /// - Parameter transform: Closure receiving `self` and returning a transformed value `R`.
    /// - Returns: The result returned by `transform(self)`.
    public func pipe<R>(_ transform: (DataFrame) throws -> R) rethrows -> R {
        try transform(self)
    }

    // MARK: Time-Series As-Of Join (merge_asof) & Resampling (resample)

    /// Performs an ordered as-of join with `other` on `key` (analogous to `pd.merge_asof`).
    ///
    /// For each row in `self`, finds the matching row in `other` according to `direction`
    /// (`.backward`, `.forward`, or `.nearest`), optionally Partitioned by exact matches on `by`
    /// and bounded by `tolerance` (in numeric units or seconds for `.date` keys).
    ///
    /// - Parameters:
    ///   - other: Right-hand ``DataFrame`` to match against.
    ///   - key: Name of the numeric or `.date` key column present in both frames.
    ///   - groupColumn: Optional column name requiring an exact key match before performing the as-of search.
    ///   - direction: Matching direction (`.backward`, `.forward`, or `.nearest`). Defaults to `.backward`.
    ///   - tolerance: Optional maximum allowed distance between matched keys.
    /// - Returns: A joined ``DataFrame`` preserving the row order of `self`.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if `key` or `groupColumn` is missing.
    public func joinAsOf(
        with other: DataFrame,
        on key: String,
        by groupColumn: String? = nil,
        direction: AsOfDirection = .backward,
        tolerance: Double? = nil
    ) throws -> DataFrame {
        let leftKeyVals = try values(in: key)
        let rightKeyVals = try other.values(in: key)
        let leftByVals = try groupColumn.map { try values(in: $0) }
        let rightByVals = try groupColumn.map { try other.values(in: $0) }

        func numericTimestamp(_ value: DataValue) -> Double? {
            switch value {
            case .number(let n): return n
            case .date(let d): return d.timeIntervalSince1970
            default: return nil
            }
        }

        // Index right rows by group key and sort by numeric timestamp
        var rightCandidatesByGroup: [DataValue: [(row: Int, ts: Double)]] = [:]
        for rRow in 0..<other.rowCount {
            guard let ts = numericTimestamp(rightKeyVals[rRow]) else { continue }
            let gKey = rightByVals?[rRow] ?? .string("__all__")
            guard gKey != .missing else { continue }
            rightCandidatesByGroup[gKey, default: []].append((row: rRow, ts: ts))
        }
        for k in rightCandidatesByGroup.keys {
            rightCandidatesByGroup[k]?.sort { $0.ts < $1.ts }
        }

        var matchedRightRow = [Int?](repeating: nil, count: rowCount)
        for lRow in 0..<rowCount {
            guard let lTs = numericTimestamp(leftKeyVals[lRow]) else { continue }
            let gKey = leftByVals?[lRow] ?? .string("__all__")
            guard gKey != .missing, let candidates = rightCandidatesByGroup[gKey], !candidates.isEmpty else {
                continue
            }

            // Binary search for first candidate with ts >= lTs
            var lo = 0
            var hi = candidates.count
            while lo < hi {
                let mid = (lo + hi) / 2
                if candidates[mid].ts < lTs {
                    lo = mid + 1
                } else {
                    hi = mid
                }
            }

            let chosen: (row: Int, ts: Double)?
            switch direction {
            case .backward:
                if lo < candidates.count && candidates[lo].ts == lTs {
                    var idx = lo
                    while idx + 1 < candidates.count && candidates[idx + 1].ts == lTs {
                        idx += 1
                    }
                    chosen = candidates[idx]
                } else if lo > 0 {
                    chosen = candidates[lo - 1]
                } else {
                    chosen = nil
                }
            case .forward:
                chosen = lo < candidates.count ? candidates[lo] : nil
            case .nearest:
                let before = lo > 0 ? candidates[lo - 1] : nil
                let after = lo < candidates.count ? candidates[lo] : nil
                switch (before, after) {
                case (let b?, let a?):
                    chosen = Swift.abs(lTs - b.ts) <= Swift.abs(a.ts - lTs) ? b : a
                case (let b?, nil):
                    chosen = b
                case (nil, let a?):
                    chosen = a
                case (nil, nil):
                    chosen = nil
                }
            }

            if let match = chosen {
                if let tolerance, Swift.abs(match.ts - lTs) > tolerance {
                    continue
                }
                matchedRightRow[lRow] = match.row
            }
        }

        var outputCols: [(name: String, values: [DataValue])] = columns.map { ($0.name, $0.values) }
        let leftNames = Set(columnNames)
        for rCol in other.columns {
            if rCol.name == key || rCol.name == groupColumn { continue }
            let outName = leftNames.contains(rCol.name) ? "\(rCol.name)_right" : rCol.name
            let rVals = rCol.values
            let colOut = matchedRightRow.map { rIdx in
                rIdx.map { rVals[$0] } ?? .missing
            }
            outputCols.append((name: outName, values: colOut))
        }
        return try DataFrame(columns: outputCols)
    }

    /// Buckets rows into fixed `interval` time windows along `dateColumn` and aggregates numeric columns (analogous to `df.resample`).
    ///
    /// - Parameters:
    ///   - dateColumn: Name of the `.date` column to bucket by.
    ///   - interval: Positive bucket duration in seconds.
    ///   - aggregations: List of ``Aggregation`` specifications to evaluate per time bucket.
    /// - Returns: A resampled and aggregated ``DataFrame``.
    /// - Throws: ``DataFrameError`` if `interval` is not positive/finite, `aggregations` is empty, or `dateColumn` contains non-date values.
    public func resample(
        on dateColumn: String,
        interval: TimeInterval,
        aggregations: [Aggregation]
    ) throws -> DataFrame {
        guard interval > 0, interval.isFinite else {
            throw DataFrameError.invalidWindowSize(Int(interval))
        }
        guard !aggregations.isEmpty else {
            throw DataFrameError.emptyColumns
        }
        let dateVals = try values(in: dateColumn)
        var bucketDates: [DataValue] = []
        bucketDates.reserveCapacity(rowCount)

        for (row, v) in dateVals.enumerated() {
            switch v {
            case .date(let d):
                let bucketTs = Foundation.floor(d.timeIntervalSince1970 / interval) * interval
                bucketDates.append(.date(Date(timeIntervalSince1970: bucketTs)))
            case .missing:
                bucketDates.append(.missing)
            default:
                throw DataFrameError.invalidConversion(column: dateColumn, row: row, type: .date)
            }
        }

        let bucketed = try replacingColumn(dateColumn, with: bucketDates)
        return try bucketed.groupBy(dateColumn).aggregate(aggregations)
    }
}
