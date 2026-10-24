import Foundation

extension DataFrame {
    /// Groups rows by one or more columns for subsequent aggregation.
    ///
    /// Rows with a missing value in any grouping column are omitted. Group keys
    /// are sorted lexicographically in the same order as `names`.
    public func groupBy(_ names: String...) throws -> GroupedDataFrame {
        try groupBy(names)
    }

    /// Groups rows by a list of columns for subsequent aggregation.
    public func groupBy(_ names: [String]) throws -> GroupedDataFrame {
        guard !names.isEmpty else { throw DataFrameError.emptyColumns }
        var uniqueNames = Set<String>()
        for name in names {
            guard uniqueNames.insert(name).inserted else {
                throw DataFrameError.duplicateColumnName(name)
            }
            _ = try values(in: name)
        }
        return GroupedDataFrame(dataFrame: self, keys: names)
    }

    func aggregate(
        keys: [String],
        aggregations: [Aggregation]
    ) throws -> DataFrame {
        guard !aggregations.isEmpty else { throw DataFrameError.emptyColumns }
        let keyColumns = try keys.map { try values(in: $0) }
        let aggregationColumns = try aggregations.map { aggregation in
            try values(in: aggregation.column)
        }

        var groups: [GroupKey: [Int]] = [:]
        for row in 0..<rowCount {
            let values = keyColumns.map { $0[row] }
            guard !values.contains(.missing) else { continue }
            groups[GroupKey(values: values), default: []].append(row)
        }
        let sortedKeys = groups.keys.sorted(by: groupKeyIsOrderedBefore)
        var outputColumns = keys.enumerated().map { index, name in
            (name: name, values: sortedKeys.map { $0.values[index] })
        }

        let groupRows = sortedKeys.map { groups[$0, default: []] }
        for (aggregationIndex, aggregation) in aggregations.enumerated() {
            let name = "\(aggregation.column)_\(aggregation.function.name)"
            let results = try Self.reduceGroups(
                aggregationColumns[aggregationIndex],
                columnName: aggregation.column,
                groups: groupRows,
                function: aggregation.function
            )
            outputColumns.append((name: name, values: results))
        }

        return try DataFrame(columns: outputColumns)
    }

    /// Reduces one column for every group (row indices per group, in output order).
    ///
    /// Numeric reductions of many rows run through MLX with a single scatter
    /// over group indices; small inputs and `median` use native Swift.
    static func reduceGroups(
        _ column: [DataValue],
        columnName: String,
        groups: [[Int]],
        function: AggregateFunction
    ) throws -> [DataValue] {
        switch function {
        case .count:
            return groups.map { rows in
                .number(Double(rows.reduce(0) { $0 + (column[$1] == .missing ? 0 : 1) }))
            }
        case .first:
            return groups.map { rows in
                rows.lazy.map { column[$0] }.first { $0 != .missing } ?? .missing
            }
        case .last:
            return groups.map { rows in
                rows.reversed().lazy.map { column[$0] }.first { $0 != .missing } ?? .missing
            }
        default:
            break
        }

        var numbers: [[Double]] = []
        numbers.reserveCapacity(groups.count)
        for rows in groups {
            var groupNumbers: [Double] = []
            groupNumbers.reserveCapacity(rows.count)
            for row in rows {
                switch column[row] {
                case .number(let value): groupNumbers.append(value)
                case .missing: break
                default: throw DataFrameError.nonNumericColumn(columnName)
                }
            }
            numbers.append(groupNumbers)
        }

        let rowCount = groups.reduce(0) { $0 + $1.count }
        if MLXBackend.shouldUse(elementCount: rowCount) {
            var flat: [Double?] = []
            var ids: [Int] = []
            flat.reserveCapacity(rowCount)
            ids.reserveCapacity(rowCount)
            for (group, rows) in groups.enumerated() {
                for row in rows {
                    if case .number(let value) = column[row] {
                        flat.append(value)
                    } else {
                        flat.append(nil)
                    }
                    ids.append(group)
                }
            }
            if let reduced = NumericEngine.groupedReduction(
                values: flat,
                groupIDs: ids,
                groupCount: groups.count,
                function: function
            ) {
                return reduced.map { $0.map(DataValue.number) ?? .missing }
            }
        }

        return numbers.map { groupNumbers in
            reduce(groupNumbers, function).map(DataValue.number) ?? .missing
        }
    }
}

/// A grouped view of a data frame.
public struct GroupedDataFrame: CustomStringConvertible {
    public var description: String {
        "GroupedDataFrame(keys: \(keys.joined(separator: ", ")), rows: \(dataFrame.rowCount))"
    }

    private let dataFrame: DataFrame
    private let keys: [String]

    fileprivate init(dataFrame: DataFrame, keys: [String]) {
        self.dataFrame = dataFrame
        self.keys = keys
    }

    /// Aggregates one column by group and returns the group key and result column.
    public func aggregate(
        _ column: String,
        using operation: AggregateFunction
    ) throws -> DataFrame {
        try aggregate([Aggregation(column, using: operation)])
    }

    /// Computes several aggregations in one grouping pass.
    public func aggregate(_ aggregations: [Aggregation]) throws -> DataFrame {
        try dataFrame.aggregate(keys: keys, aggregations: aggregations)
    }

    /// Broadcasts a grouped aggregation back to each source row.
    ///
    /// Rows with a missing grouping key receive a missing result. The result
    /// column is named `column_operation`; an existing column with that name
    /// causes an error rather than being overwritten.
    public func transform(
        _ column: String,
        using operation: AggregateFunction
    ) throws -> DataFrame {
        let outputName = "\(column)_\(operation.name)"
        guard !dataFrame.columnNames.contains(outputName) else {
            throw DataFrameError.columnNameConflict(outputName)
        }

        let aggregated = try dataFrame.aggregate(
            keys: keys,
            aggregations: [Aggregation(column, using: operation)]
        )
        let resultValues = try aggregated.values(in: outputName)
        let aggregatedKeys = try keys.map { try aggregated.values(in: $0) }
        var resultsByKey: [[DataValue]: DataValue] = [:]
        for row in 0..<aggregated.rowCount {
            resultsByKey[aggregatedKeys.map { $0[row] }] = resultValues[row]
        }

        let sourceKeys = try keys.map { try dataFrame.values(in: $0) }
        let transformedValues = (0..<dataFrame.rowCount).map { row -> DataValue in
            let groupKey = sourceKeys.map { $0[row] }
            guard !groupKey.contains(.missing) else { return .missing }
            return resultsByKey[groupKey] ?? .missing
        }
        return try dataFrame.withColumn(name: outputName, values: transformedValues)
    }
}

private struct GroupKey: Hashable {
    let values: [DataValue]
}

private func groupKeyIsOrderedBefore(_ lhs: GroupKey, _ rhs: GroupKey) -> Bool {
    for (left, right) in zip(lhs.values, rhs.values) {
        guard left != right else { continue }
        return isOrderedBefore(left, right)
    }
    return false
}

extension AggregateFunction {
    var name: String {
        switch self {
        case .count: "count"
        case .sum: "sum"
        case .mean: "mean"
        case .min: "min"
        case .max: "max"
        case .median: "median"
        case .variance: "variance"
        case .standardDeviation: "std"
        case .first: "first"
        case .last: "last"
        }
    }
}
