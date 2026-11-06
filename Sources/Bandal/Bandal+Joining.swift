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
        guard let leftKeyCol = columns.first(where: { $0.name == key }) else {
            throw DataFrameError.columnNotFound(key)
        }
        guard let rightKeyCol = other.columns.first(where: { $0.name == key }) else {
            throw DataFrameError.columnNotFound(key)
        }
        for name in other.columnNames where name != key && columnNames.contains(name) {
            throw DataFrameError.columnNameConflict(name)
        }

        let additionalColumns = other.columns.filter { $0.name != key }

        // Fast-path 1: Left join with CategoricalColumn on the left and String/Categorical key on the right.
        if kind == .left,
           let leftCat = leftKeyCol.storage as? CategoricalColumn,
           let fastResult = try joinLeftCategorical(
               other: other,
               leftCat: leftCat,
               rightKeyStorage: rightKeyCol.storage,
               additionalColumns: additionalColumns
           )
        {
            return fastResult
        }

        // Fast-path 2: Typed 1-to-1 / many-to-1 Left Join for StringColumn, Int64Column, or Float64Column keys.
        if kind == .left,
           let fastResult = try joinLeftTypedUniqueRight(
               leftKeyStorage: leftKeyCol.storage,
               rightKeyStorage: rightKeyCol.storage,
               additionalColumns: additionalColumns
           )
        {
            return fastResult
        }

        let leftKeys = leftKeyCol.values
        let rightKeys = rightKeyCol.values

        var rightRowsByKey: [DataValue: [Int]] = [:]
        for (index, value) in rightKeys.enumerated() where value != .missing {
            rightRowsByKey[value, default: []].append(index)
        }

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

    /// Zero-copy left-column reuse and dictionary-level right lookup when the left key is a
    /// `CategoricalColumn` and the right key column has unique non-null strings.
    private func joinLeftCategorical(
        other: DataFrame,
        leftCat: CategoricalColumn,
        rightKeyStorage: any AnyColumn,
        additionalColumns: [DataColumn]
    ) throws -> DataFrame? {
        var rightRowByString: [String: Int] = [:]
        rightRowByString.reserveCapacity(other.rowCount)

        if let rightStr = rightKeyStorage as? StringColumn {
            let rStrings = rightStr.stringData
            let rNulls = rightStr.nullBitmap
            let rHasNulls = rightStr.hasNulls
            for r in 0..<rightStr.count {
                if rHasNulls && rNulls[r] { continue }
                let s = rStrings[r]
                if rightRowByString.updateValue(r, forKey: s) != nil {
                    return nil // Duplicate key on right; fall back to multi-match join
                }
            }
        } else if let rightCat = rightKeyStorage as? CategoricalColumn {
            let rDict = rightCat.dictionary
            let rCodes = rightCat.codes
            let rNulls = rightCat.nullBitmap
            let rHasNulls = rightCat.hasNulls
            for r in 0..<rightCat.count {
                if rHasNulls && rNulls[r] { continue }
                let s = rDict[Int(rCodes[r])]
                if rightRowByString.updateValue(r, forKey: s) != nil {
                    return nil
                }
            }
        } else {
            return nil
        }

        let dict = leftCat.dictionary
        let dictCount = dict.count
        let codes = leftCat.codes
        let leftHasNulls = leftCat.hasNulls
        let leftNulls = leftCat.nullBitmap
        let nRows = rowCount

        var codeToRightRow = ContiguousArray<Int32>(repeating: -1, count: dictCount)
        for (c, keyStr) in dict.enumerated() {
            if let rIdx = rightRowByString[keyStr] {
                codeToRightRow[c] = Int32(rIdx)
            }
        }

        // Reuse all left DataColumns in O(1) since every left row maps 1-to-1 in order.
        var outDataColumns = columns
        outDataColumns.reserveCapacity(columns.count + additionalColumns.count)

        for rightCol in additionalColumns {
            if let rFloat = rightCol.numericBuffer?.floatColumn {
                var codeToVal = ContiguousArray<Double>(repeating: 0.0, count: dictCount)
                var codeIsNull = ContiguousArray<Bool>(repeating: false, count: dictCount)
                var anyCodeNull = false
                for c in 0..<dictCount {
                    let rIdx = Int(codeToRightRow[c])
                    if rIdx >= 0 && !rFloat.nullBitmap[rIdx] {
                        codeToVal[c] = rFloat.values[rIdx]
                    } else {
                        codeIsNull[c] = true
                        anyCodeNull = true
                    }
                }

                if !leftHasNulls && !anyCodeNull && dictCount > 0 {
                    let outVals = ContiguousArray<Double>(unsafeUninitializedCapacity: nRows) { buffer, initializedCount in
                        codes.withUnsafeBufferPointer { codeBuf in
                            codeToVal.withUnsafeBufferPointer { mapBuf in
                                guard let cBase = codeBuf.baseAddress,
                                      let mBase = mapBuf.baseAddress,
                                      let outBase = buffer.baseAddress else { return }
                                for i in 0..<nRows {
                                    outBase[i] = mBase[Int(cBase[i])]
                                }
                            }
                        }
                        initializedCount = nRows
                    }
                    let newCol = Float64Column(
                        name: rightCol.name,
                        values: outVals,
                        nullBitmap: BitSet(count: nRows)
                    )
                    outDataColumns.append(DataColumn(storage: newCol))
                } else {
                    var outNulls = BitSet(count: nRows)
                    let outVals = ContiguousArray<Double>(unsafeUninitializedCapacity: nRows) { buffer, initializedCount in
                        guard let outBase = buffer.baseAddress else { return }
                        for i in 0..<nRows {
                            if leftHasNulls && leftNulls[i] {
                                outBase[i] = 0.0
                                outNulls[i] = true
                            } else {
                                let c = Int(codes[i])
                                if codeIsNull[c] {
                                    outBase[i] = 0.0
                                    outNulls[i] = true
                                } else {
                                    outBase[i] = codeToVal[c]
                                }
                            }
                        }
                        initializedCount = nRows
                    }
                    let newCol = Float64Column(name: rightCol.name, values: outVals, nullBitmap: outNulls)
                    outDataColumns.append(DataColumn(storage: newCol))
                }
            } else if let rStr = rightCol.storage as? StringColumn {
                var codeToStr = ContiguousArray<String>(repeating: "", count: dictCount)
                var codeIsNull = ContiguousArray<Bool>(repeating: false, count: dictCount)
                var anyCodeNull = false
                for c in 0..<dictCount {
                    let rIdx = Int(codeToRightRow[c])
                    if rIdx >= 0 && !rStr.nullBitmap[rIdx] {
                        codeToStr[c] = rStr.stringData[rIdx]
                    } else {
                        codeIsNull[c] = true
                        anyCodeNull = true
                    }
                }
                var outStrings = ContiguousArray<String>()
                outStrings.reserveCapacity(nRows)
                var outNulls = BitSet(count: nRows)
                if !leftHasNulls && !anyCodeNull {
                    for i in 0..<nRows {
                        outStrings.append(codeToStr[Int(codes[i])])
                    }
                } else {
                    for i in 0..<nRows {
                        if leftHasNulls && leftNulls[i] {
                            outStrings.append("")
                            outNulls[i] = true
                        } else {
                            let c = Int(codes[i])
                            if codeIsNull[c] {
                                outStrings.append("")
                                outNulls[i] = true
                            } else {
                                outStrings.append(codeToStr[c])
                            }
                        }
                    }
                }
                let newCol = StringColumn(name: rightCol.name, stringData: outStrings, nullBitmap: outNulls)
                outDataColumns.append(DataColumn(storage: newCol))
            } else {
                var rightIndices: [Int?] = []
                rightIndices.reserveCapacity(nRows)
                for i in 0..<nRows {
                    if leftHasNulls && leftNulls[i] {
                        rightIndices.append(nil)
                    } else {
                        let rIdx = Int(codeToRightRow[Int(codes[i])])
                        rightIndices.append(rIdx >= 0 ? rIdx : nil)
                    }
                }
                let sliced = rightCol.storage.slice(atOptional: rightIndices)
                outDataColumns.append(DataColumn(storage: sliced))
            }
        }

        return try DataFrame(dataColumns: outDataColumns)
    }

    /// Typed 1-to-1 / many-to-1 Left Join when right keys are unique, reusing left columns in O(1).
    private func joinLeftTypedUniqueRight(
        leftKeyStorage: any AnyColumn,
        rightKeyStorage: any AnyColumn,
        additionalColumns: [DataColumn]
    ) throws -> DataFrame? {
        let nRows = rowCount
        var rightIndices: [Int?] = []

        if let lStr = leftKeyStorage as? StringColumn, let rStr = rightKeyStorage as? StringColumn {
            var map: [String: Int] = [:]
            map.reserveCapacity(rStr.count)
            let rHasNulls = rStr.hasNulls
            for r in 0..<rStr.count {
                if rHasNulls && rStr.nullBitmap[r] { continue }
                if map.updateValue(r, forKey: rStr.stringData[r]) != nil { return nil }
            }
            rightIndices.reserveCapacity(nRows)
            let lHasNulls = lStr.hasNulls
            for i in 0..<nRows {
                if lHasNulls && lStr.nullBitmap[i] {
                    rightIndices.append(nil)
                } else {
                    rightIndices.append(map[lStr.stringData[i]])
                }
            }
        } else if let lFlt = leftKeyStorage as? Float64Column, let rFlt = rightKeyStorage as? Float64Column {
            var map: [Double: Int] = [:]
            map.reserveCapacity(rFlt.count)
            let rHasNulls = rFlt.hasNulls
            for r in 0..<rFlt.count {
                if rHasNulls && rFlt.nullBitmap[r] { continue }
                if map.updateValue(r, forKey: rFlt.values[r]) != nil { return nil }
            }
            rightIndices.reserveCapacity(nRows)
            let lHasNulls = lFlt.hasNulls
            for i in 0..<nRows {
                if lHasNulls && lFlt.nullBitmap[i] {
                    rightIndices.append(nil)
                } else {
                    rightIndices.append(map[lFlt.values[i]])
                }
            }
        } else if let lInt = leftKeyStorage as? Int64Column, let rInt = rightKeyStorage as? Int64Column {
            var map: [Int64: Int] = [:]
            map.reserveCapacity(rInt.count)
            let rHasNulls = rInt.hasNulls
            for r in 0..<rInt.count {
                if rHasNulls && rInt.nullBitmap[r] { continue }
                if map.updateValue(r, forKey: rInt.values[r]) != nil { return nil }
            }
            rightIndices.reserveCapacity(nRows)
            let lHasNulls = lInt.hasNulls
            for i in 0..<nRows {
                if lHasNulls && lInt.nullBitmap[i] {
                    rightIndices.append(nil)
                } else {
                    rightIndices.append(map[lInt.values[i]])
                }
            }
        } else {
            return nil
        }

        var outDataColumns = columns
        outDataColumns.reserveCapacity(columns.count + additionalColumns.count)
        for rightCol in additionalColumns {
            let sliced = rightCol.storage.slice(atOptional: rightIndices)
            outDataColumns.append(DataColumn(storage: sliced))
        }
        return try DataFrame(dataColumns: outDataColumns)
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
