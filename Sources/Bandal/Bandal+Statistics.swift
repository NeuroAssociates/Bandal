import Foundation

extension DataFrame {
    /// Computes count, mean, sample standard deviation, median, minimum, and maximum.
    ///
    /// Missing values are ignored. Columns containing non-numeric values are excluded.
    /// Large columns are reduced through MLX in `float64` on the CPU stream, so
    /// results keep `Double` precision; small columns use native Swift. See
    /// ``statistics()`` for quartiles, ``summaryTable()`` for an ordered struct
    /// result, and ``describeStructured(percentiles:include:)`` for text output.
    public func summary() -> [String: ColumnSummary] {
        Dictionary(
            uniqueKeysWithValues: columns.compactMap { column in
                guard let buffer = column.numericBuffer else { return nil }
                let values = buffer.numbers
                guard !values.isEmpty else {
                    guard column.storage.count == 0 else {
                        return nil
                    }
                    return (
                        column.name,
                        ColumnSummary(
                            count: 0,
                            mean: nil,
                            standardDeviation: nil,
                            median: nil,
                            min: nil,
                            max: nil
                        )
                    )
                }

                let moments = NumericEngine.moments(of: values)
                return (
                    column.name,
                    ColumnSummary(
                        count: moments.count,
                        mean: moments.mean,
                        standardDeviation: moments.standardDeviation,
                        median: moments.median,
                        min: moments.min,
                        max: moments.max
                    )
                )
            }
        )
    }

    /// Adds a trailing-window aggregation to a copy of the data frame.
    ///
    /// By default (`minPeriods == nil`), the first `window - 1` results are
    /// missing unless `minPeriods` is set to a smaller positive integer.
    /// Missing values inside windows are ignored; a window with fewer than
    /// `minPeriods` non-missing numeric values produces missing (except `count`
    /// when the window has at least `minPeriods` total rows). The output column
    /// is named `column_operation_rolling_window`.
    ///
    /// Large inputs compute `sum`, `mean`, and `count` from MLX prefix sums in
    /// O(n) and non-null `min`, `max`, `variance`, and `standardDeviation` via
    /// 2D strided MLX reductions (`MLX.asStrided`).
    public func rolling(
        _ column: String,
        window: Int,
        minPeriods: Int? = nil,
        using operation: AggregateFunction
    ) throws -> DataFrame {
        guard window > 0 else { throw DataFrameError.invalidWindowSize(window) }
        let requiredPeriods = minPeriods ?? window
        guard requiredPeriods > 0, requiredPeriods <= window else {
            throw DataFrameError.invalidWindowSize(requiredPeriods)
        }
        let outputName = "\(column)_\(operation.name)_rolling_\(window)"
        guard !columnNames.contains(outputName) else {
            throw DataFrameError.columnNameConflict(outputName)
        }

        let buffer = try numericBuffer(in: column)
        let srcVals = buffer.floatColumn.values
        let srcNulls = buffer.floatColumn.nullBitmap
        let n = buffer.count

        if let stridedCol = NumericEngine.rollingWindowNoNulls(
            buffer.floatColumn.renamed(to: outputName) as! Float64Column,
            window: window,
            minPeriods: requiredPeriods,
            using: operation
        ) {
            return try withColumn(storage: stridedCol)
        }

        if let prefix = prefixSums(of: buffer, for: operation) {
            var outVals = ContiguousArray<Double>(repeating: 0.0, count: n)
            var outNulls = BitSet(count: n)
            for row in 0..<n {
                let windowRows = Swift.min(row + 1, window)
                guard windowRows >= requiredPeriods else {
                    outNulls[row] = true
                    continue
                }
                let start = row - window
                let validCount = prefix.counts[row] - (start >= 0 ? prefix.counts[start] : 0)
                if operation == .count {
                    outVals[row] = Double(validCount)
                } else if validCount > 0 && (minPeriods == nil || validCount >= requiredPeriods) {
                    let s = prefix.sums[row] - (start >= 0 ? prefix.sums[start] : 0)
                    let val = operation == .sum ? s : s / Double(validCount)
                    if val.isFinite {
                        outVals[row] = val
                    } else {
                        outNulls[row] = true
                    }
                } else {
                    outNulls[row] = true
                }
            }
            return try withColumn(storage: Float64Column(name: outputName, values: outVals, nullBitmap: outNulls))
        }

        var outVals = ContiguousArray<Double>(repeating: 0.0, count: n)
        var outNulls = BitSet(count: n)
        var windowBuffer: [Double] = []
        windowBuffer.reserveCapacity(window)

        for row in 0..<n {
            let windowRows = Swift.min(row + 1, window)
            guard windowRows >= requiredPeriods else {
                outNulls[row] = true
                continue
            }
            let start = Swift.max(0, row + 1 - window)
            windowBuffer.removeAll(keepingCapacity: true)
            for idx in start...row where !srcNulls[idx] {
                windowBuffer.append(srcVals[idx])
            }
            if operation == .count {
                outVals[row] = Double(windowBuffer.count)
            } else if !windowBuffer.isEmpty,
                      (minPeriods == nil || windowBuffer.count >= requiredPeriods),
                      let reduced = Self.reduce(windowBuffer, operation),
                      reduced.isFinite
            {
                outVals[row] = reduced
            } else {
                outNulls[row] = true
            }
        }
        return try withColumn(storage: Float64Column(name: outputName, values: outVals, nullBitmap: outNulls))
    }

    /// Adds a trailing time-window aggregation over a `.date` column to a copy
    /// of the data frame.
    ///
    /// For each row `i` with timestamp `t_i`, the window includes every row `j <= i`
    /// whose timestamp `t_j` falls in `(t_i - interval, t_i]`. Rows with a
    /// missing timestamp in `dateColumn` receive `.missing`.
    public func rolling(
        _ valueColumn: String,
        on dateColumn: String,
        interval: TimeInterval,
        using operation: AggregateFunction
    ) throws -> DataFrame {
        guard interval > 0, interval.isFinite else {
            throw DataFrameError.invalidWindowSize(Int(interval))
        }
        let intervalLabel = interval.rounded() == interval
            ? "\(Int(interval))s"
            : "\(String(format: "%.4g", interval))s"
        let outputName = "\(valueColumn)_\(operation.name)_rolling_\(intervalLabel)"
        guard !columnNames.contains(outputName) else {
            throw DataFrameError.columnNameConflict(outputName)
        }

        let buffer = try numericBuffer(in: valueColumn)
        let valVals = buffer.floatColumn.values
        let valNulls = buffer.floatColumn.nullBitmap
        let dateSource = try values(in: dateColumn)
        var timestamps: [Date?] = []
        timestamps.reserveCapacity(rowCount)
        for (row, item) in dateSource.enumerated() {
            switch item {
            case .date(let date):
                timestamps.append(date)
            case .missing:
                timestamps.append(nil)
            default:
                throw DataFrameError.invalidConversion(column: dateColumn, row: row, type: .date)
            }
        }

        var outVals = ContiguousArray<Double>(repeating: 0.0, count: rowCount)
        var outNulls = BitSet(count: rowCount, repeating: true)
        var numbers: [Double] = []
        for row in 0..<rowCount {
            guard let current = timestamps[row] else { continue }
            let windowStart = current.addingTimeInterval(-interval)
            numbers.removeAll(keepingCapacity: true)
            for index in 0...row {
                guard let ts = timestamps[index],
                      ts > windowStart,
                      ts <= current,
                      !valNulls[index]
                else {
                    continue
                }
                numbers.append(valVals[index])
            }
            if operation == .count {
                outVals[row] = Double(numbers.count)
                outNulls[row] = false
            } else if !numbers.isEmpty, let reduced = Self.reduce(numbers, operation), reduced.isFinite {
                outVals[row] = reduced
                outNulls[row] = false
            }
        }
        return try withColumn(storage: Float64Column(name: outputName, values: outVals, nullBitmap: outNulls))
    }

    /// Adds an aggregation over each prefix of a column to a copy of the frame.
    ///
    /// Each result includes the current row and all preceding rows. Missing
    /// values are ignored; a prefix with no numeric values produces zero for
    /// `count` and missing otherwise. The output column is named
    /// `column_operation_expanding`.
    ///
    /// Large inputs compute `sum`, `mean`, and `count` from MLX prefix sums.
    /// Native `count`, `sum`, `mean`, `min`, `max`, `first`, and `last` run in
    /// O(n); `median`, `variance`, and `standardDeviation` are O(n²).
    public func expanding(
        _ column: String,
        using operation: AggregateFunction
    ) throws -> DataFrame {
        let outputName = "\(column)_\(operation.name)_expanding"
        guard !columnNames.contains(outputName) else {
            throw DataFrameError.columnNameConflict(outputName)
        }

        let buffer = try numericBuffer(in: column)
        let n = buffer.count
        if let prefix = prefixSums(of: buffer, for: operation) {
            var outVals = ContiguousArray<Double>(repeating: 0.0, count: n)
            var outNulls = BitSet(count: n)
            for row in 0..<n {
                let count = prefix.counts[row]
                if operation == .count {
                    outVals[row] = Double(count)
                } else if count > 0 {
                    let s = prefix.sums[row]
                    let v = operation == .sum ? s : s / Double(count)
                    if v.isFinite {
                        outVals[row] = v
                    } else {
                        outNulls[row] = true
                    }
                } else {
                    outNulls[row] = true
                }
            }
            return try withColumn(storage: Float64Column(name: outputName, values: outVals, nullBitmap: outNulls))
        }
        if let runningCol = Self.runningColumn(buffer, name: outputName, operation: operation) {
            return try withColumn(storage: runningCol)
        }
        let source = try values(in: column)
        let results = source.indices.map { row in
            aggregateWindow(source, range: 0..<(row + 1), operation: operation)
        }
        return try withColumn(name: outputName, values: results)
    }

    /// O(n) native expanding results for operations with a running state.
    private static func runningColumn(
        _ buffer: NumericBuffer,
        name: String,
        operation: AggregateFunction
    ) -> Float64Column? {
        switch operation {
        case .count, .sum, .mean, .min, .max, .first, .last: break
        case .median, .variance, .standardDeviation: return nil
        }
        let srcVals = buffer.floatColumn.values
        let srcNulls = buffer.floatColumn.nullBitmap
        let n = buffer.count
        var outVals = ContiguousArray<Double>(repeating: 0.0, count: n)
        var outNulls = BitSet(count: n)
        var count = 0
        var sum = 0.0
        var minimum = Double.infinity
        var maximum = -Double.infinity
        var first: Double?
        var last: Double?

        for i in 0..<n {
            if !srcNulls[i] {
                let number = srcVals[i]
                count += 1
                sum += number
                minimum = Swift.min(minimum, number)
                maximum = Swift.max(maximum, number)
                first = first ?? number
                last = number
            }
            if operation == .count {
                outVals[i] = Double(count)
                continue
            }
            guard count > 0 else {
                outNulls[i] = true
                continue
            }
            switch operation {
            case .sum: outVals[i] = sum
            case .mean: outVals[i] = sum / Double(count)
            case .min: outVals[i] = minimum
            case .max: outVals[i] = maximum
            case .first:
                if let first { outVals[i] = first } else { outNulls[i] = true }
            default:
                if let last { outVals[i] = last } else { outNulls[i] = true }
            }
        }
        return Float64Column(name: name, values: outVals, nullBitmap: outNulls)
    }

    /// Prefix sums for `sum`, `mean`, and `count` when MLX is selected.
    private func prefixSums(
        of buffer: NumericBuffer,
        for operation: AggregateFunction
    ) -> (sums: [Double], counts: [Int])? {
        guard operation == .sum || operation == .mean || operation == .count else {
            return nil
        }
        let srcVals = buffer.floatColumn.values
        let srcNulls = buffer.floatColumn.nullBitmap
        let numbers = (0..<buffer.count).map { index -> Double? in
            srcNulls[index] ? nil : srcVals[index]
        }
        return NumericEngine.prefixSums(values: numbers)
    }

    func aggregateWindow(
        _ values: [DataValue],
        range: Range<Int>,
        operation: AggregateFunction
    ) -> DataValue {
        var numbers: [Double] = []
        for row in range {
            guard case .number(let value) = values[row] else { continue }
            numbers.append(value)
        }
        if case .count = operation {
            return .number(Double(numbers.count))
        }
        guard !numbers.isEmpty else { return .missing }
        let reduced = Self.reduce(numbers, operation)
        return reduced.map(DataValue.number) ?? .missing
    }

    /// Reduces non-missing numbers, routing large inputs through `NumericEngine` (MLX)
    /// and small inputs through native Swift.
    static func reduce(_ numbers: [Double], _ operation: AggregateFunction) -> Double? {
        NumericEngine.reduce(numbers, using: operation)
    }
}
