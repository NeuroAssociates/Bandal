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
            guard columnNames.contains(name) else {
                throw DataFrameError.columnNotFound(name)
            }
        }
        return GroupedDataFrame(dataFrame: self, keys: names)
    }

    func aggregate(
        keys: [String],
        aggregations: [Aggregation]
    ) throws -> DataFrame {
        guard !aggregations.isEmpty else { throw DataFrameError.emptyColumns }

        var groupIds = ContiguousArray<Int32>(repeating: -1, count: rowCount)
        let groupCount: Int
        var outputColumns: [any AnyColumn] = []

        if keys.count == 1, let keyCol = columns.first(where: { $0.name == keys[0] }) {
            var groupIndexByKey: [DataValue: Int] = [:]
            var uniqueKeyValues: [DataValue] = []

            for row in 0..<rowCount {
                let v = keyCol.storage.value(at: row)
                if v == .missing { continue }
                if let gid = groupIndexByKey[v] {
                    groupIds[row] = Int32(gid)
                } else {
                    let gid = uniqueKeyValues.count
                    groupIndexByKey[v] = gid
                    uniqueKeyValues.append(v)
                    groupIds[row] = Int32(gid)
                }
            }

            let sortedGroupIndices = (0..<uniqueKeyValues.count).sorted {
                isOrderedBefore(uniqueKeyValues[$0], uniqueKeyValues[$1])
            }
            var remap = [Int32](repeating: -1, count: uniqueKeyValues.count)
            for (sortedIdx, oldIdx) in sortedGroupIndices.enumerated() {
                remap[oldIdx] = Int32(sortedIdx)
            }
            let sortedKeys = sortedGroupIndices.map { uniqueKeyValues[$0] }
            groupCount = sortedKeys.count

            for row in 0..<rowCount {
                let old = groupIds[row]
                if old >= 0 {
                    groupIds[row] = remap[Int(old)]
                }
            }

            outputColumns.append(AnyColumnFactory.make(name: keys[0], values: sortedKeys))
        } else {
            let keyStorages = try keys.map { name -> any AnyColumn in
                guard let col = columns.first(where: { $0.name == name }) else {
                    throw DataFrameError.columnNotFound(name)
                }
                return col.storage
            }
            var groupIndexByKey: [GroupKey: Int] = [:]
            var uniqueKeys: [GroupKey] = []

            for row in 0..<rowCount {
                var hasMissing = false
                for col in keyStorages where col.isNull(at: row) {
                    hasMissing = true
                    break
                }
                if hasMissing { continue }
                let keyVals = keyStorages.map { $0.value(at: row) }
                let gk = GroupKey(values: keyVals)
                if let gid = groupIndexByKey[gk] {
                    groupIds[row] = Int32(gid)
                } else {
                    let gid = uniqueKeys.count
                    groupIndexByKey[gk] = gid
                    uniqueKeys.append(gk)
                    groupIds[row] = Int32(gid)
                }
            }

            let sortedGroupIndices = (0..<uniqueKeys.count).sorted {
                groupKeyIsOrderedBefore(uniqueKeys[$0], uniqueKeys[$1])
            }
            var remap = [Int32](repeating: -1, count: uniqueKeys.count)
            for (sortedIdx, oldIdx) in sortedGroupIndices.enumerated() {
                remap[oldIdx] = Int32(sortedIdx)
            }
            let sortedKeys = sortedGroupIndices.map { uniqueKeys[$0] }
            groupCount = sortedKeys.count

            for row in 0..<rowCount {
                let old = groupIds[row]
                if old >= 0 {
                    groupIds[row] = remap[Int(old)]
                }
            }

            for (index, name) in keys.enumerated() {
                outputColumns.append(
                    AnyColumnFactory.make(name: name, values: sortedKeys.map { $0.values[index] })
                )
            }
        }

        var groupRows: [[Int]]? = nil

        for aggregation in aggregations {
            let name = aggregation.alias ?? "\(aggregation.column)_\(aggregation.function.name)"
            guard let targetCol = columns.first(where: { $0.name == aggregation.column }) else {
                throw DataFrameError.columnNotFound(aggregation.column)
            }

            if let buffer = targetCol.numericBuffer {
                let fCol = buffer.floatColumn
                switch aggregation.function {
                case .sum:
                    var sums = ContiguousArray<Double>(repeating: 0.0, count: groupCount)
                    var counts = ContiguousArray<Int>(repeating: 0, count: groupCount)
                    let vals = fCol.values
                    let nulls = fCol.nullBitmap
                    for row in 0..<rowCount {
                        let gid = Int(groupIds[row])
                        if gid >= 0 && !nulls[row] {
                            sums[gid] += vals[row]
                            counts[gid] += 1
                        }
                    }
                    var outNulls = BitSet(count: groupCount)
                    for gid in 0..<groupCount where counts[gid] == 0 {
                        outNulls[gid] = true
                    }
                    outputColumns.append(Float64Column(name: name, values: sums, nullBitmap: outNulls))
                    continue
                case .count:
                    var counts = ContiguousArray<Double>(repeating: 0.0, count: groupCount)
                    let nulls = fCol.nullBitmap
                    for row in 0..<rowCount {
                        let gid = Int(groupIds[row])
                        if gid >= 0 && !nulls[row] {
                            counts[gid] += 1.0
                        }
                    }
                    outputColumns.append(
                        Float64Column(name: name, values: counts, nullBitmap: BitSet(count: groupCount))
                    )
                    continue
                case .mean:
                    var sums = ContiguousArray<Double>(repeating: 0.0, count: groupCount)
                    var counts = ContiguousArray<Int>(repeating: 0, count: groupCount)
                    let vals = fCol.values
                    let nulls = fCol.nullBitmap
                    for row in 0..<rowCount {
                        let gid = Int(groupIds[row])
                        if gid >= 0 && !nulls[row] {
                            sums[gid] += vals[row]
                            counts[gid] += 1
                        }
                    }
                    var outVals = ContiguousArray<Double>(repeating: 0.0, count: groupCount)
                    var outNulls = BitSet(count: groupCount)
                    for gid in 0..<groupCount {
                        if counts[gid] > 0 {
                            outVals[gid] = sums[gid] / Double(counts[gid])
                        } else {
                            outNulls[gid] = true
                        }
                    }
                    outputColumns.append(Float64Column(name: name, values: outVals, nullBitmap: outNulls))
                    continue
                case .min:
                    var mins = ContiguousArray<Double>(repeating: .infinity, count: groupCount)
                    var hasVal = BitSet(count: groupCount)
                    let vals = fCol.values
                    let nulls = fCol.nullBitmap
                    for row in 0..<rowCount {
                        let gid = Int(groupIds[row])
                        if gid >= 0 && !nulls[row] {
                            let v = vals[row]
                            if v < mins[gid] { mins[gid] = v }
                            hasVal[gid] = true
                        }
                    }
                    for gid in 0..<groupCount where !hasVal[gid] {
                        mins[gid] = 0.0
                    }
                    outputColumns.append(
                        Float64Column(name: name, values: mins, nullBitmap: hasVal.inverted())
                    )
                    continue
                case .max:
                    var maxs = ContiguousArray<Double>(repeating: -.infinity, count: groupCount)
                    var hasVal = BitSet(count: groupCount)
                    let vals = fCol.values
                    let nulls = fCol.nullBitmap
                    for row in 0..<rowCount {
                        let gid = Int(groupIds[row])
                        if gid >= 0 && !nulls[row] {
                            let v = vals[row]
                            if v > maxs[gid] { maxs[gid] = v }
                            hasVal[gid] = true
                        }
                    }
                    for gid in 0..<groupCount where !hasVal[gid] {
                        maxs[gid] = 0.0
                    }
                    outputColumns.append(
                        Float64Column(name: name, values: maxs, nullBitmap: hasVal.inverted())
                    )
                    continue
                default:
                    break
                }
            }

            if groupRows == nil {
                var rowsByGroup = [[Int]](repeating: [], count: groupCount)
                for row in 0..<rowCount {
                    let gid = Int(groupIds[row])
                    if gid >= 0 {
                        rowsByGroup[gid].append(row)
                    }
                }
                groupRows = rowsByGroup
            }

            let targetValues = targetCol.values
            let results = try Self.reduceGroups(
                targetValues,
                columnName: aggregation.column,
                groups: groupRows!,
                function: aggregation.function
            )
            outputColumns.append(AnyColumnFactory.make(name: name, values: results))
        }

        return try DataFrame(storageColumns: outputColumns)
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
    /// A summary listing the grouping keys and row count.
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

    /// Filters groups using `predicate`, returning rows belonging to groups for
    /// which `predicate` returns `true` in their original row order (analogous
    /// to `df.groupby(...).filter(...)`).
    public func filter(
        _ predicate: (DataFrame) throws -> Bool
    ) throws -> DataFrame {
        let keyStorages = try keys.map { name -> any AnyColumn in
            guard let col = dataFrame.columns.first(where: { $0.name == name }) else {
                throw DataFrameError.columnNotFound(name)
            }
            return col.storage
        }
        var groups: [GroupKey: [Int]] = [:]
        var orderedGroupKeys: [GroupKey] = []
        for row in 0..<dataFrame.rowCount {
            var hasMissing = false
            for col in keyStorages where col.isNull(at: row) {
                hasMissing = true
                break
            }
            if hasMissing { continue }
            let gk = GroupKey(values: keyStorages.map { $0.value(at: row) })
            if groups[gk] == nil {
                orderedGroupKeys.append(gk)
                groups[gk] = [row]
            } else {
                groups[gk]?.append(row)
            }
        }

        var keepMask = BitSet(count: dataFrame.rowCount)
        for gk in orderedGroupKeys {
            guard let indices = groups[gk] else { continue }
            let subFrame = try dataFrame.rows(at: indices)
            if try predicate(subFrame) {
                for idx in indices {
                    keepMask[idx] = true
                }
            }
        }
        return try dataFrame.rows(matching: keepMask)
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
