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
                    guard column.values.isEmpty else {
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
    /// The first `window - 1` results are missing because the window is not
    /// complete. Missing values inside complete windows are ignored; a window
    /// with no numeric values produces zero for `count` and missing otherwise.
    /// The output column is named `column_operation_rolling_window`.
    ///
    /// Large inputs compute `sum`, `mean`, and `count` from MLX prefix sums in
    /// O(n); results can differ from the native direct-window sums by rounding
    /// error proportional to the magnitude of the running total. The other
    /// operations always use native Swift.
    public func rolling(
        _ column: String,
        window: Int,
        using operation: AggregateFunction
    ) throws -> DataFrame {
        guard window > 0 else { throw DataFrameError.invalidWindowSize(window) }
        let outputName = "\(column)_\(operation.name)_rolling_\(window)"
        guard !columnNames.contains(outputName) else {
            throw DataFrameError.columnNameConflict(outputName)
        }

        let buffer = try numericBuffer(in: column)
        let source = try values(in: column)
        if let prefix = prefixSums(of: buffer, for: operation) {
            let results = source.indices.map { row -> DataValue in
                guard row + 1 >= window else { return .missing }
                return Self.windowResult(
                    prefix, operation: operation, from: row - window, to: row)
            }
            return try withColumn(name: outputName, values: results)
        }

        let results = source.indices.map { row -> DataValue in
            guard row + 1 >= window else { return .missing }
            return aggregateWindow(
                source,
                range: (row + 1 - window)..<(row + 1),
                operation: operation
            )
        }
        return try withColumn(name: outputName, values: results)
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

        _ = try numericBuffer(in: valueColumn)
        let valueSource = try values(in: valueColumn)
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

        var results = Array(repeating: DataValue.missing, count: rowCount)
        for row in 0..<rowCount {
            guard let current = timestamps[row] else { continue }
            let windowStart = current.addingTimeInterval(-interval)
            var numbers: [Double] = []
            for index in 0...row {
                guard let ts = timestamps[index],
                      ts > windowStart,
                      ts <= current,
                      case .number(let val) = valueSource[index]
                else {
                    continue
                }
                numbers.append(val)
            }
            if operation == .count {
                results[row] = .number(Double(numbers.count))
            } else if !numbers.isEmpty, let reduced = Self.reduce(numbers, operation) {
                results[row] = .number(reduced)
            }
        }
        return try withColumn(name: outputName, values: results)
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
        let source = try values(in: column)
        if let prefix = prefixSums(of: buffer, for: operation) {
            let results = source.indices.map { row in
                Self.windowResult(prefix, operation: operation, from: -1, to: row)
            }
            return try withColumn(name: outputName, values: results)
        }
        if let results = Self.runningResults(source, operation: operation) {
            return try withColumn(name: outputName, values: results)
        }
        let results = source.indices.map { row in
            aggregateWindow(source, range: 0..<(row + 1), operation: operation)
        }
        return try withColumn(name: outputName, values: results)
    }

    /// O(n) native expanding results for operations with a running state.
    ///
    /// Accumulates left to right, so values equal `reduce` over each prefix.
    /// Returns `nil` for median, variance, and standard deviation.
    private static func runningResults(
        _ source: [DataValue],
        operation: AggregateFunction
    ) -> [DataValue]? {
        switch operation {
        case .count, .sum, .mean, .min, .max, .first, .last: break
        case .median, .variance, .standardDeviation: return nil
        }
        var count = 0
        var sum = 0.0
        var minimum = Double.infinity
        var maximum = -Double.infinity
        var first: Double?
        var last: Double?
        return source.map { value in
            if case .number(let number) = value {
                count += 1
                sum += number
                minimum = Swift.min(minimum, number)
                maximum = Swift.max(maximum, number)
                first = first ?? number
                last = number
            }
            if operation == .count { return .number(Double(count)) }
            guard count > 0 else { return .missing }
            switch operation {
            case .sum: return .number(sum)
            case .mean: return .number(sum / Double(count))
            case .min: return .number(minimum)
            case .max: return .number(maximum)
            case .first: return first.map(DataValue.number) ?? .missing
            default: return last.map(DataValue.number) ?? .missing
            }
        }
    }

    /// Prefix sums for `sum`, `mean`, and `count` when MLX is selected.
    private func prefixSums(
        of buffer: NumericBuffer,
        for operation: AggregateFunction
    ) -> (sums: [Double], counts: [Int])? {
        guard operation == .sum || operation == .mean || operation == .count else {
            return nil
        }
        let numbers = buffer.values.indices.map { index -> Double? in
            buffer.isMissing[index] ? nil : buffer.values[index]
        }
        return NumericEngine.prefixSums(values: numbers)
    }

    /// Result for the rows `(start, end]` of the prefix arrays; a negative
    /// `start` means the window begins at the first row.
    private static func windowResult(
        _ prefix: (sums: [Double], counts: [Int]),
        operation: AggregateFunction,
        from start: Int,
        to end: Int
    ) -> DataValue {
        let count = prefix.counts[end] - (start >= 0 ? prefix.counts[start] : 0)
        if operation == .count { return .number(Double(count)) }
        guard count > 0 else { return .missing }
        let sum = prefix.sums[end] - (start >= 0 ? prefix.sums[start] : 0)
        return .number(operation == .sum ? sum : sum / Double(count))
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
