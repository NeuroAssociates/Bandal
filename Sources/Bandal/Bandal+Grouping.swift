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

        // Fast-path: Single CategoricalColumn key with direct dictionary-code accumulation.
        if keys.count == 1,
           let keyCol = columns.first(where: { $0.name == keys[0] }),
           let catCol = keyCol.storage as? CategoricalColumn,
           rowCount > 0,
           !catCol.dictionary.isEmpty,
           let fastResult = try aggregateCategoricalKey(
               keyName: keys[0],
               catCol: catCol,
               aggregations: aggregations
           )
        {
            return fastResult
        }

        var groupIds = ContiguousArray<Int32>(repeating: -1, count: rowCount)
        let groupCount: Int
        var outputColumns: [any AnyColumn] = []
        var anyKeyHasMissing = false

        if keys.count == 1, let keyCol = columns.first(where: { $0.name == keys[0] }) {
            if let strCol = keyCol.storage as? StringColumn {
                let strings = strCol.stringData
                let nulls = strCol.nullBitmap
                let hasNulls = strCol.hasNulls
                anyKeyHasMissing = hasNulls
                var groupIndexByKey: [String: Int32] = [:]
                var uniqueStrings: [String] = []
                for row in 0..<rowCount {
                    if hasNulls && nulls[row] { continue }
                    let s = strings[row]
                    if let gid = groupIndexByKey[s] {
                        groupIds[row] = gid
                    } else {
                        let gid = Int32(uniqueStrings.count)
                        groupIndexByKey[s] = gid
                        uniqueStrings.append(s)
                        groupIds[row] = gid
                    }
                }
                let sortedIndices = (0..<uniqueStrings.count).sorted { uniqueStrings[$0] < uniqueStrings[$1] }
                var remap = ContiguousArray<Int32>(repeating: -1, count: uniqueStrings.count)
                for (sortedIdx, oldIdx) in sortedIndices.enumerated() {
                    remap[oldIdx] = Int32(sortedIdx)
                }
                let sortedStrings = sortedIndices.map { uniqueStrings[$0] }
                groupCount = sortedStrings.count
                for row in 0..<rowCount {
                    let old = groupIds[row]
                    if old >= 0 { groupIds[row] = remap[Int(old)] }
                }
                if groupCount > 0 {
                    outputColumns.append(StringColumn(name: keys[0], strings: sortedStrings))
                } else {
                    outputColumns.append(AnyColumnFactory.make(name: keys[0], values: []))
                }
            } else if let intCol = keyCol.storage as? Int64Column {
                let vals = intCol.values
                let nulls = intCol.nullBitmap
                let hasNulls = intCol.hasNulls
                anyKeyHasMissing = hasNulls
                var groupIndexByKey: [Int64: Int32] = [:]
                var uniqueVals: [Int64] = []
                for row in 0..<rowCount {
                    if hasNulls && nulls[row] { continue }
                    let v = vals[row]
                    if let gid = groupIndexByKey[v] {
                        groupIds[row] = gid
                    } else {
                        let gid = Int32(uniqueVals.count)
                        groupIndexByKey[v] = gid
                        uniqueVals.append(v)
                        groupIds[row] = gid
                    }
                }
                let sortedIndices = (0..<uniqueVals.count).sorted { uniqueVals[$0] < uniqueVals[$1] }
                var remap = ContiguousArray<Int32>(repeating: -1, count: uniqueVals.count)
                for (sortedIdx, oldIdx) in sortedIndices.enumerated() {
                    remap[oldIdx] = Int32(sortedIdx)
                }
                let sortedDoubles = sortedIndices.map { Double(uniqueVals[$0]) }
                groupCount = sortedDoubles.count
                for row in 0..<rowCount {
                    let old = groupIds[row]
                    if old >= 0 { groupIds[row] = remap[Int(old)] }
                }
                if groupCount > 0 {
                    outputColumns.append(Float64Column(name: keys[0], numbers: sortedDoubles))
                } else {
                    outputColumns.append(AnyColumnFactory.make(name: keys[0], values: []))
                }
            } else if let fltCol = keyCol.storage as? Float64Column {
                let vals = fltCol.values
                let nulls = fltCol.nullBitmap
                let hasNulls = fltCol.hasNulls
                anyKeyHasMissing = hasNulls
                var groupIndexByKey: [Double: Int32] = [:]
                var uniqueVals: [Double] = []
                for row in 0..<rowCount {
                    if hasNulls && nulls[row] { continue }
                    let v = vals[row]
                    if let gid = groupIndexByKey[v] {
                        groupIds[row] = gid
                    } else {
                        let gid = Int32(uniqueVals.count)
                        groupIndexByKey[v] = gid
                        uniqueVals.append(v)
                        groupIds[row] = gid
                    }
                }
                let sortedIndices = (0..<uniqueVals.count).sorted { uniqueVals[$0] < uniqueVals[$1] }
                var remap = ContiguousArray<Int32>(repeating: -1, count: uniqueVals.count)
                for (sortedIdx, oldIdx) in sortedIndices.enumerated() {
                    remap[oldIdx] = Int32(sortedIdx)
                }
                let sortedDoubles = sortedIndices.map { uniqueVals[$0] }
                groupCount = sortedDoubles.count
                for row in 0..<rowCount {
                    let old = groupIds[row]
                    if old >= 0 { groupIds[row] = remap[Int(old)] }
                }
                if groupCount > 0 {
                    outputColumns.append(Float64Column(name: keys[0], numbers: sortedDoubles))
                } else {
                    outputColumns.append(AnyColumnFactory.make(name: keys[0], values: []))
                }
            } else {
                var groupIndexByKey: [DataValue: Int] = [:]
                var uniqueKeyValues: [DataValue] = []

                for row in 0..<rowCount {
                    let v = keyCol.storage.value(at: row)
                    if v == .missing {
                        anyKeyHasMissing = true
                        continue
                    }
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
            }
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
                if hasMissing {
                    anyKeyHasMissing = true
                    continue
                }
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
                let noNullFast = !anyKeyHasMissing && !fCol.hasNulls
                switch aggregation.function {
                case .sum:
                    var sums = ContiguousArray<Double>(repeating: 0.0, count: groupCount)
                    if noNullFast {
                        groupIds.withUnsafeBufferPointer { gidBuf in
                            fCol.values.withUnsafeBufferPointer { valBuf in
                                sums.withUnsafeMutableBufferPointer { sumBuf in
                                    guard let gBase = gidBuf.baseAddress,
                                          let vBase = valBuf.baseAddress,
                                          let sBase = sumBuf.baseAddress else { return }
                                    for row in 0..<rowCount {
                                        sBase[Int(gBase[row])] += vBase[row]
                                    }
                                }
                            }
                        }
                        outputColumns.append(
                            Float64Column(name: name, values: sums, nullBitmap: BitSet(count: groupCount))
                        )
                    } else {
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
                    }
                    continue
                case .count:
                    var counts = ContiguousArray<Double>(repeating: 0.0, count: groupCount)
                    if noNullFast {
                        groupIds.withUnsafeBufferPointer { gidBuf in
                            counts.withUnsafeMutableBufferPointer { cntBuf in
                                guard let gBase = gidBuf.baseAddress,
                                      let cBase = cntBuf.baseAddress else { return }
                                for row in 0..<rowCount {
                                    cBase[Int(gBase[row])] += 1.0
                                }
                            }
                        }
                    } else {
                        let nulls = fCol.nullBitmap
                        for row in 0..<rowCount {
                            let gid = Int(groupIds[row])
                            if gid >= 0 && !nulls[row] {
                                counts[gid] += 1.0
                            }
                        }
                    }
                    outputColumns.append(
                        Float64Column(name: name, values: counts, nullBitmap: BitSet(count: groupCount))
                    )
                    continue
                case .mean:
                    var sums = ContiguousArray<Double>(repeating: 0.0, count: groupCount)
                    var counts = ContiguousArray<Int>(repeating: 0, count: groupCount)
                    if noNullFast {
                        groupIds.withUnsafeBufferPointer { gidBuf in
                            fCol.values.withUnsafeBufferPointer { valBuf in
                                sums.withUnsafeMutableBufferPointer { sumBuf in
                                    counts.withUnsafeMutableBufferPointer { cntBuf in
                                        guard let gBase = gidBuf.baseAddress,
                                              let vBase = valBuf.baseAddress,
                                              let sBase = sumBuf.baseAddress,
                                              let cBase = cntBuf.baseAddress else { return }
                                        for row in 0..<rowCount {
                                            let gid = Int(gBase[row])
                                            sBase[gid] += vBase[row]
                                            cBase[gid] &+= 1
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        let vals = fCol.values
                        let nulls = fCol.nullBitmap
                        for row in 0..<rowCount {
                            let gid = Int(groupIds[row])
                            if gid >= 0 && !nulls[row] {
                                sums[gid] += vals[row]
                                counts[gid] += 1
                            }
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

    /// Direct-code single-pass GroupBy for a single `CategoricalColumn` key when all
    /// aggregations are numeric `.sum`, `.count`, `.mean`, `.min`, or `.max`.
    private func aggregateCategoricalKey(
        keyName: String,
        catCol: CategoricalColumn,
        aggregations: [Aggregation]
    ) throws -> DataFrame? {
        let supportedFunctions: Set<AggregateFunction> = [.sum, .count, .mean, .min, .max]
        var resolvedTargets: [(name: String, function: AggregateFunction, floatCol: Float64Column)] = []
        resolvedTargets.reserveCapacity(aggregations.count)

        for agg in aggregations {
            guard supportedFunctions.contains(agg.function) else { return nil }
            guard let targetCol = columns.first(where: { $0.name == agg.column }) else {
                throw DataFrameError.columnNotFound(agg.column)
            }
            guard let buffer = targetCol.numericBuffer else { return nil }
            let outName = agg.alias ?? "\(agg.column)_\(agg.function.name)"
            resolvedTargets.append((name: outName, function: agg.function, floatCol: buffer.floatColumn))
        }

        let dict = catCol.dictionary
        let dictCount = dict.count
        let codes = catCol.codes
        let keyHasNulls = catCol.hasNulls
        let keyNulls = catCol.nullBitmap
        let nRows = rowCount

        // Specialized ultra-fast path: single aggregation, no nulls in key or value column.
        if resolvedTargets.count == 1, !keyHasNulls, !resolvedTargets[0].floatCol.hasNulls {
            let target = resolvedTargets[0]
            var sumsByCode = ContiguousArray<Double>(repeating: 0.0, count: dictCount)
            var countsByCode = ContiguousArray<Int>(repeating: 0, count: dictCount)

            switch target.function {
            case .sum, .mean:
                codes.withUnsafeBufferPointer { codeBuf in
                    target.floatCol.values.withUnsafeBufferPointer { valBuf in
                        sumsByCode.withUnsafeMutableBufferPointer { sumBuf in
                            countsByCode.withUnsafeMutableBufferPointer { cntBuf in
                                guard let cBase = codeBuf.baseAddress,
                                      let vBase = valBuf.baseAddress,
                                      let sBase = sumBuf.baseAddress,
                                      let nBase = cntBuf.baseAddress else { return }
                                for i in 0..<nRows {
                                    let c = Int(cBase[i])
                                    sBase[c] += vBase[i]
                                    nBase[c] &+= 1
                                }
                            }
                        }
                    }
                }
            case .count:
                codes.withUnsafeBufferPointer { codeBuf in
                    countsByCode.withUnsafeMutableBufferPointer { cntBuf in
                        guard let cBase = codeBuf.baseAddress,
                              let nBase = cntBuf.baseAddress else { return }
                        for i in 0..<nRows {
                            nBase[Int(cBase[i])] &+= 1
                        }
                    }
                }
            case .min:
                sumsByCode = ContiguousArray<Double>(repeating: .infinity, count: dictCount)
                codes.withUnsafeBufferPointer { codeBuf in
                    target.floatCol.values.withUnsafeBufferPointer { valBuf in
                        sumsByCode.withUnsafeMutableBufferPointer { minBuf in
                            countsByCode.withUnsafeMutableBufferPointer { cntBuf in
                                guard let cBase = codeBuf.baseAddress,
                                      let vBase = valBuf.baseAddress,
                                      let mBase = minBuf.baseAddress,
                                      let nBase = cntBuf.baseAddress else { return }
                                for i in 0..<nRows {
                                    let c = Int(cBase[i])
                                    let v = vBase[i]
                                    if v < mBase[c] { mBase[c] = v }
                                    nBase[c] &+= 1
                                }
                            }
                        }
                    }
                }
            case .max:
                sumsByCode = ContiguousArray<Double>(repeating: -.infinity, count: dictCount)
                codes.withUnsafeBufferPointer { codeBuf in
                    target.floatCol.values.withUnsafeBufferPointer { valBuf in
                        sumsByCode.withUnsafeMutableBufferPointer { maxBuf in
                            countsByCode.withUnsafeMutableBufferPointer { cntBuf in
                                guard let cBase = codeBuf.baseAddress,
                                      let vBase = valBuf.baseAddress,
                                      let mBase = maxBuf.baseAddress,
                                      let nBase = cntBuf.baseAddress else { return }
                                for i in 0..<nRows {
                                    let c = Int(cBase[i])
                                    let v = vBase[i]
                                    if v > mBase[c] { mBase[c] = v }
                                    nBase[c] &+= 1
                                }
                            }
                        }
                    }
                }
            default:
                return nil
            }

            var activeCodes: [Int] = []
            activeCodes.reserveCapacity(dictCount)
            for c in 0..<dictCount where countsByCode[c] > 0 {
                activeCodes.append(c)
            }
            activeCodes.sort { dict[$0] < dict[$1] }
            let groupCount = activeCodes.count
            guard groupCount > 0 else { return nil }

            let sortedKeys = activeCodes.map { dict[$0] }
            var outVals = ContiguousArray<Double>(repeating: 0.0, count: groupCount)
            switch target.function {
            case .sum, .min, .max:
                for (gid, c) in activeCodes.enumerated() {
                    outVals[gid] = sumsByCode[c]
                }
            case .mean:
                for (gid, c) in activeCodes.enumerated() {
                    outVals[gid] = sumsByCode[c] / Double(countsByCode[c])
                }
            case .count:
                for (gid, c) in activeCodes.enumerated() {
                    outVals[gid] = Double(countsByCode[c])
                }
            default:
                break
            }

            let outKeyCol = StringColumn(name: keyName, strings: sortedKeys)
            let outValCol = Float64Column(
                name: target.name,
                values: outVals,
                nullBitmap: BitSet(count: groupCount)
            )
            return try DataFrame(storageColumns: [outKeyCol, outValCol])
        }

        // General CategoricalColumn fast path (supports nulls and multiple aggregations)
        var keyCounts = ContiguousArray<Int>(repeating: 0, count: dictCount)
        if keyHasNulls {
            for i in 0..<nRows where !keyNulls[i] {
                keyCounts[Int(codes[i])] &+= 1
            }
        } else {
            codes.withUnsafeBufferPointer { codeBuf in
                keyCounts.withUnsafeMutableBufferPointer { cntBuf in
                    guard let cBase = codeBuf.baseAddress, let nBase = cntBuf.baseAddress else { return }
                    for i in 0..<nRows {
                        nBase[Int(cBase[i])] &+= 1
                    }
                }
            }
        }

        var activeCodes: [Int] = []
        activeCodes.reserveCapacity(dictCount)
        for c in 0..<dictCount where keyCounts[c] > 0 {
            activeCodes.append(c)
        }
        guard !activeCodes.isEmpty else { return nil }
        activeCodes.sort { dict[$0] < dict[$1] }
        let groupCount = activeCodes.count

        var outputColumns: [any AnyColumn] = [
            StringColumn(name: keyName, strings: activeCodes.map { dict[$0] })
        ]
        outputColumns.reserveCapacity(1 + resolvedTargets.count)

        for target in resolvedTargets {
            let fCol = target.floatCol
            let valHasNulls = fCol.hasNulls
            let valNulls = fCol.nullBitmap
            let vals = fCol.values

            var accByCode = ContiguousArray<Double>(
                repeating: target.function == .min ? .infinity : (target.function == .max ? -.infinity : 0.0),
                count: dictCount
            )
            var validCountsByCode = ContiguousArray<Int>(repeating: 0, count: dictCount)

            if !keyHasNulls && !valHasNulls {
                validCountsByCode = keyCounts
                switch target.function {
                case .sum, .mean:
                    codes.withUnsafeBufferPointer { codeBuf in
                        vals.withUnsafeBufferPointer { valBuf in
                            accByCode.withUnsafeMutableBufferPointer { accBuf in
                                guard let cBase = codeBuf.baseAddress,
                                      let vBase = valBuf.baseAddress,
                                      let aBase = accBuf.baseAddress else { return }
                                for i in 0..<nRows {
                                    aBase[Int(cBase[i])] += vBase[i]
                                }
                            }
                        }
                    }
                case .count:
                    break
                case .min:
                    codes.withUnsafeBufferPointer { codeBuf in
                        vals.withUnsafeBufferPointer { valBuf in
                            accByCode.withUnsafeMutableBufferPointer { accBuf in
                                guard let cBase = codeBuf.baseAddress,
                                      let vBase = valBuf.baseAddress,
                                      let aBase = accBuf.baseAddress else { return }
                                for i in 0..<nRows {
                                    let c = Int(cBase[i])
                                    let v = vBase[i]
                                    if v < aBase[c] { aBase[c] = v }
                                }
                            }
                        }
                    }
                case .max:
                    codes.withUnsafeBufferPointer { codeBuf in
                        vals.withUnsafeBufferPointer { valBuf in
                            accByCode.withUnsafeMutableBufferPointer { accBuf in
                                guard let cBase = codeBuf.baseAddress,
                                      let vBase = valBuf.baseAddress,
                                      let aBase = accBuf.baseAddress else { return }
                                for i in 0..<nRows {
                                    let c = Int(cBase[i])
                                    let v = vBase[i]
                                    if v > aBase[c] { aBase[c] = v }
                                }
                            }
                        }
                    }
                default:
                    return nil
                }
            } else {
                for i in 0..<nRows {
                    if keyHasNulls && keyNulls[i] { continue }
                    if valHasNulls && valNulls[i] { continue }
                    let c = Int(codes[i])
                    let v = vals[i]
                    validCountsByCode[c] &+= 1
                    switch target.function {
                    case .sum, .mean:
                        accByCode[c] += v
                    case .min:
                        if v < accByCode[c] { accByCode[c] = v }
                    case .max:
                        if v > accByCode[c] { accByCode[c] = v }
                    case .count:
                        break
                    default:
                        break
                    }
                }
            }

            var outVals = ContiguousArray<Double>(repeating: 0.0, count: groupCount)
            var outNulls = BitSet(count: groupCount)
            for (gid, c) in activeCodes.enumerated() {
                let cnt = validCountsByCode[c]
                switch target.function {
                case .count:
                    outVals[gid] = Double(cnt)
                case .sum, .min, .max:
                    if cnt > 0 {
                        outVals[gid] = accByCode[c]
                    } else {
                        outNulls[gid] = true
                    }
                case .mean:
                    if cnt > 0 {
                        outVals[gid] = accByCode[c] / Double(cnt)
                    } else {
                        outNulls[gid] = true
                    }
                default:
                    break
                }
            }
            outputColumns.append(Float64Column(name: target.name, values: outVals, nullBitmap: outNulls))
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
            if let reduced = NumericEngine.reduceGroups(
                values: flat,
                groupIds: ids,
                groupCount: groups.count,
                using: function
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
