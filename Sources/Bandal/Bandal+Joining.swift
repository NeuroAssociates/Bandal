extension DataFrame {
    /// Performs a left join using values in the named key column.
    public func leftJoin(_ other: DataFrame, on key: String) throws -> DataFrame {
        try join(other, on: key, kind: .left)
    }

    /// Joins two frames on a key column.
    ///
    /// Missing keys never match. Left and full joins retain unmatched left rows;
    /// right and full joins append unmatched right rows after left-side results.
    public func join(
        _ other: DataFrame,
        on key: String,
        kind: JoinKind
    ) throws -> DataFrame {
        let leftKeys = try values(in: key)
        let rightKeys = try other.values(in: key)
        for name in other.columnNames where name != key && columnNames.contains(name) {
            throw DataFrameError.columnNameConflict(name)
        }

        var rightRowsByKey: [DataValue: [Int]] = [:]
        for (index, value) in rightKeys.enumerated() where value != .missing {
            rightRowsByKey[value, default: []].append(index)
        }

        let additionalColumns = other.columns.filter { $0.name != key }
        var leftIndices: [Int?] = []
        var rightIndices: [Int?] = []
        leftIndices.reserveCapacity(rowCount)
        rightIndices.reserveCapacity(rowCount)

        var matchedRightRows = Set<Int>()
        for leftIndex in 0..<rowCount {
            let leftKey = leftKeys[leftIndex]
            if leftKey == .missing {
                if kind == .left || kind == .full {
                    leftIndices.append(leftIndex)
                    rightIndices.append(nil)
                }
                continue
            }
            let matches = rightRowsByKey[leftKey] ?? []
            if matches.isEmpty {
                guard kind == .left || kind == .full else { continue }
                leftIndices.append(leftIndex)
                rightIndices.append(nil)
            } else {
                for rightIndex in matches {
                    matchedRightRows.insert(rightIndex)
                    leftIndices.append(leftIndex)
                    rightIndices.append(rightIndex)
                }
            }
        }

        if kind == .right || kind == .full {
            for rightIndex in rightKeys.indices where !matchedRightRows.contains(rightIndex) {
                leftIndices.append(nil)
                rightIndices.append(rightIndex)
            }
        }

        var storageCols: [any AnyColumn] = []
        storageCols.reserveCapacity(columns.count + additionalColumns.count)

        for col in columns {
            if col.name == key && (kind == .right || kind == .full) {
                var vals: [DataValue] = []
                vals.reserveCapacity(leftIndices.count)
                for i in 0..<leftIndices.count {
                    if let lIdx = leftIndices[i] {
                        vals.append(col.storage.value(at: lIdx))
                    } else if let rIdx = rightIndices[i] {
                        vals.append(rightKeys[rIdx])
                    } else {
                        vals.append(.missing)
                    }
                }
                storageCols.append(AnyColumnFactory.make(name: col.name, values: vals))
            } else {
                storageCols.append(col.storage.slice(atOptional: leftIndices))
            }
        }

        for col in additionalColumns {
            storageCols.append(col.storage.slice(atOptional: rightIndices))
        }

        return try DataFrame(storageColumns: storageCols)
    }
}

extension DataFrame {
    /// Joins two frames on one or more key columns that may be named differently.
    ///
    /// Key columns whose left and right names are equal appear once. Other
    /// columns present on both sides receive `suffixes` instead of throwing
    /// `columnNameConflict`. Missing keys never match. Right and full joins
    /// append unmatched right rows after the left-side results.
    public func join(
        _ other: DataFrame,
        leftOn: [String],
        rightOn: [String],
        kind: JoinKind = .inner,
        suffixes: (left: String, right: String) = ("_x", "_y")
    ) throws -> DataFrame {
        guard !leftOn.isEmpty, leftOn.count == rightOn.count else {
            throw DataFrameError.incompatibleSchema(expected: leftOn, actual: rightOn)
        }
        let leftKeyColumns = try leftOn.map { try values(in: $0) }
        let rightKeyColumns = try rightOn.map { try other.values(in: $0) }
        let sharedKeys = Set(zip(leftOn, rightOn).filter { $0 == $1 }.map(\.0))

        let rightExtras = other.columns.filter { !(rightOn.contains($0.name) && sharedKeys.contains($0.name)) }
        let leftNames = Set(columnNames)
        let rightNames = Set(rightExtras.map(\.name))
        let leftOutputNames = columnNames.map { rightNames.contains($0) ? $0 + suffixes.left : $0 }
        let rightOutputNames = rightExtras.map { leftNames.contains($0.name) ? $0.name + suffixes.right : $0.name }

        func key(_ columns: [[DataValue]], _ row: Int) -> [DataValue]? {
            let parts = columns.map { $0[row] }
            return parts.contains(.missing) ? nil : parts
        }

        var rightRows: [[DataValue]: [Int]] = [:]
        for row in 0..<other.rowCount {
            if let k = key(rightKeyColumns, row) { rightRows[k, default: []].append(row) }
        }

        var result = (leftOutputNames + rightOutputNames).map { (name: $0, values: [DataValue]()) }
        func emit(left: Int?, right: Int?) {
            for (index, column) in columns.enumerated() {
                var value: DataValue = left.map { column.values[$0] } ?? .missing
                if left == nil, let right, let position = leftOn.firstIndex(of: column.name) {
                    value = other.columns.first { $0.name == rightOn[position] }?.values[right] ?? .missing
                }
                result[index].values.append(value)
            }
            for (offset, column) in rightExtras.enumerated() {
                result[columns.count + offset].values.append(right.map { column.values[$0] } ?? .missing)
            }
        }

        var matched = Set<Int>()
        for row in 0..<rowCount {
            let matches = key(leftKeyColumns, row).flatMap { rightRows[$0] } ?? []
            if matches.isEmpty {
                if kind == .left || kind == .full { emit(left: row, right: nil) }
            } else {
                for match in matches {
                    matched.insert(match)
                    emit(left: row, right: match)
                }
            }
        }
        if kind == .right || kind == .full {
            for row in 0..<other.rowCount where !matched.contains(row) { emit(left: nil, right: row) }
        }
        let names = result.map(\.name)
        guard Set(names).count == names.count else {
            throw DataFrameError.duplicateColumnName(names.first { name in names.filter { $0 == name }.count > 1 } ?? "")
        }
        return try DataFrame(columns: result)
    }
}
