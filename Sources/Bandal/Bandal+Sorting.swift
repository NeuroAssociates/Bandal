import Foundation

extension DataFrame {
    /// Returns a new frame sorted by one or more columns.
    ///
    /// Sort keys are applied in order, ties preserve their input order, and
    /// missing values remain last in both ascending and descending order.
    public func sorted(by descriptors: [SortDescriptor]) throws -> DataFrame {
        guard !descriptors.isEmpty else { return self }
        guard rowCount > 1 else {
            for desc in descriptors where !columnNames.contains(desc.column) {
                throw DataFrameError.columnNotFound(desc.column)
            }
            return self
        }

        if descriptors.count == 1 {
            let desc = descriptors[0]
            guard let col = columns.first(where: { $0.name == desc.column }) else {
                throw DataFrameError.columnNotFound(desc.column)
            }
            let ascending = desc.order == .ascending

            if let fCol = col.storage as? Float64Column {
                if rowCount >= 512 {
                    let indices = Self.radixSortFloat64Indices(
                        values: fCol.values,
                        nullBitmap: fCol.nullBitmap,
                        ascending: ascending
                    )
                    return try rows(at: indices)
                }
                var indices = Array(0..<rowCount)
                let vals = fCol.values
                let nulls = fCol.nullBitmap
                indices.sort { i, j in
                    let ni = nulls[i]
                    let nj = nulls[j]
                    if ni != nj { return !ni }
                    if ni { return i < j }
                    let vi = vals[i]
                    let vj = vals[j]
                    if vi != vj { return ascending ? vi < vj : vi > vj }
                    return i < j
                }
                return try rows(at: indices)
            } else if let iCol = col.storage as? Int64Column {
                if rowCount >= 512 {
                    let indices = Self.radixSortInt64Indices(
                        values: iCol.values,
                        nullBitmap: iCol.nullBitmap,
                        ascending: ascending
                    )
                    return try rows(at: indices)
                }
                var indices = Array(0..<rowCount)
                let vals = iCol.values
                let nulls = iCol.nullBitmap
                indices.sort { i, j in
                    let ni = nulls[i]
                    let nj = nulls[j]
                    if ni != nj { return !ni }
                    if ni { return i < j }
                    let vi = vals[i]
                    let vj = vals[j]
                    if vi != vj { return ascending ? vi < vj : vi > vj }
                    return i < j
                }
                return try rows(at: indices)
            } else if let catCol = col.storage as? CategoricalColumn {
                let dict = catCol.dictionary
                let dictCount = dict.count
                let sortedDictIndices = (0..<dictCount).sorted {
                    ascending ? (dict[$0] < dict[$1]) : (dict[$0] > dict[$1])
                }
                var rank = [Int](repeating: 0, count: dictCount)
                for (r, dIdx) in sortedDictIndices.enumerated() {
                    rank[dIdx] = r
                }
                let codes = catCol.codes
                let nulls = catCol.nullBitmap
                let hasNulls = catCol.hasNulls
                let bucketCount = dictCount + 1
                var counts = [Int](repeating: 0, count: bucketCount)
                if !hasNulls {
                    for i in 0..<rowCount {
                        counts[rank[Int(codes[i])]] += 1
                    }
                } else {
                    for i in 0..<rowCount {
                        let b = nulls[i] ? dictCount : rank[Int(codes[i])]
                        counts[b] += 1
                    }
                }
                var offsets = [Int](repeating: 0, count: bucketCount)
                var running = 0
                for b in 0..<bucketCount {
                    offsets[b] = running
                    running += counts[b]
                }
                let indices = [Int](unsafeUninitializedCapacity: rowCount) { outBuf, initCount in
                    guard let outBase = outBuf.baseAddress else { return }
                    if !hasNulls {
                        for i in 0..<rowCount {
                            let b = rank[Int(codes[i])]
                            outBase[offsets[b]] = i
                            offsets[b] += 1
                        }
                    } else {
                        for i in 0..<rowCount {
                            let b = nulls[i] ? dictCount : rank[Int(codes[i])]
                            outBase[offsets[b]] = i
                            offsets[b] += 1
                        }
                    }
                    initCount = rowCount
                }
                return try rows(at: indices)
            } else if let dateCol = col.storage as? DateColumn {
                if rowCount >= 512 {
                    let indices = Self.radixSortFloat64Indices(
                        values: dateCol.timeIntervals,
                        nullBitmap: dateCol.nullBitmap,
                        ascending: ascending
                    )
                    return try rows(at: indices)
                }
                var indices = Array(0..<rowCount)
                let intervals = dateCol.timeIntervals
                let nulls = dateCol.nullBitmap
                indices.sort { i, j in
                    let ni = nulls[i]
                    let nj = nulls[j]
                    if ni != nj { return !ni }
                    if ni { return i < j }
                    let vi = intervals[i]
                    let vj = intervals[j]
                    if vi != vj { return ascending ? vi < vj : vi > vj }
                    return i < j
                }
                return try rows(at: indices)
            } else if let sCol = col.storage as? StringColumn {
                var indices = Array(0..<rowCount)
                let nulls = sCol.nullBitmap
                indices.sort { i, j in
                    let ni = nulls[i]
                    let nj = nulls[j]
                    if ni != nj { return !ni }
                    if ni { return i < j }
                    let si = sCol.string(at: i)
                    let sj = sCol.string(at: j)
                    if si != sj { return ascending ? si < sj : si > sj }
                    return i < j
                }
                return try rows(at: indices)
            }
        }

        let sortColumns = try descriptors.map { descriptor in
            (descriptor.order, try values(in: descriptor.column))
        }
        let indices = (0..<rowCount).sorted { left, right in
            for (order, values) in sortColumns {
                let lhs = values[left]
                let rhs = values[right]
                guard lhs != rhs else { continue }
                if lhs == .missing { return false }
                if rhs == .missing { return true }
                switch order {
                case .ascending:
                    return isOrderedBefore(lhs, rhs)
                case .descending:
                    return isOrderedBefore(rhs, lhs)
                }
            }
            return left < right
        }
        return try rows(at: indices)
    }

    /// Stable $O(N)$ LSD radix sort over IEEE-754 `Double` values with `null`s placed last.
    private static func radixSortFloat64Indices(
        values: ContiguousArray<Double>,
        nullBitmap: BitSet,
        ascending: Bool
    ) -> [Int] {
        let n = values.count
        let hasNulls = !nullBitmap.allFalse
        var hist = [Int](repeating: 0, count: 8 * 256)

        var keys = ContiguousArray<UInt64>(unsafeUninitializedCapacity: n) { keyBuf, initCount in
            values.withUnsafeBufferPointer { valBuf in
                hist.withUnsafeMutableBufferPointer { histBuf in
                    guard let valBase = valBuf.baseAddress,
                          let keyBase = keyBuf.baseAddress,
                          let hBase = histBuf.baseAddress else { return }
                    if !hasNulls {
                        for i in 0..<n {
                            let v = valBase[i]
                            let bits = v == 0.0 ? 0 : v.bitPattern
                            let signMask = UInt64(Int64(bitPattern: bits) >> 63)
                            let ordered = bits ^ (signMask | 0x8000_0000_0000_0000)
                            let k = ascending ? ordered : ~ordered
                            keyBase[i] = k
                            hBase[Int(k & 0xFF)] &+= 1
                            hBase[256 + Int((k &>> 8) & 0xFF)] &+= 1
                            hBase[512 + Int((k &>> 16) & 0xFF)] &+= 1
                            hBase[768 + Int((k &>> 24) & 0xFF)] &+= 1
                            hBase[1024 + Int((k &>> 32) & 0xFF)] &+= 1
                            hBase[1280 + Int((k &>> 40) & 0xFF)] &+= 1
                            hBase[1536 + Int((k &>> 48) & 0xFF)] &+= 1
                            hBase[1792 + Int((k &>> 56) & 0xFF)] &+= 1
                        }
                    } else {
                        for i in 0..<n {
                            let k: UInt64
                            if nullBitmap[i] {
                                k = UInt64.max
                            } else {
                                let v = valBase[i]
                                let bits = v == 0.0 ? 0 : v.bitPattern
                                let signMask = UInt64(Int64(bitPattern: bits) >> 63)
                                let ordered = bits ^ (signMask | 0x8000_0000_0000_0000)
                                k = ascending ? ordered : ~ordered
                            }
                            keyBase[i] = k
                            hBase[Int(k & 0xFF)] &+= 1
                            hBase[256 + Int((k &>> 8) & 0xFF)] &+= 1
                            hBase[512 + Int((k &>> 16) & 0xFF)] &+= 1
                            hBase[768 + Int((k &>> 24) & 0xFF)] &+= 1
                            hBase[1024 + Int((k &>> 32) & 0xFF)] &+= 1
                            hBase[1280 + Int((k &>> 40) & 0xFF)] &+= 1
                            hBase[1536 + Int((k &>> 48) & 0xFF)] &+= 1
                            hBase[1792 + Int((k &>> 56) & 0xFF)] &+= 1
                        }
                    }
                }
            }
            initCount = n
        }

        return executeRadixPasses(keys: &keys, hist: &hist, count: n)
    }

    /// Stable $O(N)$ LSD radix sort over `Int64` values with `null`s placed last.
    private static func radixSortInt64Indices(
        values: ContiguousArray<Int64>,
        nullBitmap: BitSet,
        ascending: Bool
    ) -> [Int] {
        let n = values.count
        let hasNulls = !nullBitmap.allFalse
        var hist = [Int](repeating: 0, count: 8 * 256)

        var keys = ContiguousArray<UInt64>(unsafeUninitializedCapacity: n) { keyBuf, initCount in
            values.withUnsafeBufferPointer { valBuf in
                hist.withUnsafeMutableBufferPointer { histBuf in
                    guard let valBase = valBuf.baseAddress,
                          let keyBase = keyBuf.baseAddress,
                          let hBase = histBuf.baseAddress else { return }
                    for i in 0..<n {
                        let k: UInt64
                        if hasNulls && nullBitmap[i] {
                            k = UInt64.max
                        } else {
                            let u = UInt64(bitPattern: valBase[i]) ^ 0x8000_0000_0000_0000
                            let raw = ascending ? u : ~u
                            k = raw == UInt64.max ? (UInt64.max - 1) : raw
                        }
                        keyBase[i] = k
                        hBase[Int(k & 0xFF)] &+= 1
                        hBase[256 + Int((k &>> 8) & 0xFF)] &+= 1
                        hBase[512 + Int((k &>> 16) & 0xFF)] &+= 1
                        hBase[768 + Int((k &>> 24) & 0xFF)] &+= 1
                        hBase[1024 + Int((k &>> 32) & 0xFF)] &+= 1
                        hBase[1280 + Int((k &>> 40) & 0xFF)] &+= 1
                        hBase[1536 + Int((k &>> 48) & 0xFF)] &+= 1
                        hBase[1792 + Int((k &>> 56) & 0xFF)] &+= 1
                    }
                }
            }
            initCount = n
        }

        return executeRadixPasses(keys: &keys, hist: &hist, count: n)
    }

    private static func executeRadixPasses(
        keys: inout ContiguousArray<UInt64>,
        hist: inout [Int],
        count n: Int
    ) -> [Int] {
        var activePasses: [Int] = []
        activePasses.reserveCapacity(8)
        for p in 0..<8 {
            let base = p * 256
            var trivial = false
            for b in 0..<256 where hist[base + b] == n {
                trivial = true
                break
            }
            if !trivial {
                activePasses.append(p)
            }
        }

        var indices = [Int](unsafeUninitializedCapacity: n) { buf, initCount in
            guard let base = buf.baseAddress else { return }
            for i in 0..<n { base[i] = i }
            initCount = n
        }
        guard !activePasses.isEmpty else { return indices }

        var tmpKeys = ContiguousArray<UInt64>(unsafeUninitializedCapacity: n) { _, initCount in
            initCount = n
        }
        var tmpIndices = [Int](unsafeUninitializedCapacity: n) { _, initCount in
            initCount = n
        }

        var offsets = [Int](repeating: 0, count: 256)
        let lastPassIndex = activePasses.count - 1

        for (stepIdx, p) in activePasses.enumerated() {
            let shift = UInt64(p * 8)
            let hBase = p * 256
            var sum = 0
            for b in 0..<256 {
                offsets[b] = sum
                sum &+= hist[hBase + b]
            }

            let isLast = stepIdx == lastPassIndex
            keys.withUnsafeBufferPointer { srcKeyBuf in
                indices.withUnsafeBufferPointer { srcIdxBuf in
                    tmpKeys.withUnsafeMutableBufferPointer { dstKeyBuf in
                        tmpIndices.withUnsafeMutableBufferPointer { dstIdxBuf in
                            offsets.withUnsafeMutableBufferPointer { offBuf in
                                guard let sk = srcKeyBuf.baseAddress,
                                      let si = srcIdxBuf.baseAddress,
                                      let dk = dstKeyBuf.baseAddress,
                                      let di = dstIdxBuf.baseAddress,
                                      let off = offBuf.baseAddress else { return }
                                if isLast {
                                    for i in 0..<n {
                                        let bucket = Int((sk[i] &>> shift) & 0xFF)
                                        let pos = off[bucket]
                                        off[bucket] = pos &+ 1
                                        di[pos] = si[i]
                                    }
                                } else {
                                    for i in 0..<n {
                                        let k = sk[i]
                                        let bucket = Int((k &>> shift) & 0xFF)
                                        let pos = off[bucket]
                                        off[bucket] = pos &+ 1
                                        dk[pos] = k
                                        di[pos] = si[i]
                                    }
                                }
                            }
                        }
                    }
                }
            }

            swap(&indices, &tmpIndices)
            if !isLast {
                swap(&keys, &tmpKeys)
            }
        }

        return indices
    }
}

func isOrderedBefore(_ lhs: DataValue, _ rhs: DataValue) -> Bool {
    switch (lhs, rhs) {
    case (.number(let left), .number(let right)):
        left < right
    case (.string(let left), .string(let right)):
        left < right
    case (.boolean(let left), .boolean(let right)):
        !left && right
    case (.date(let left), .date(let right)):
        left < right
    case (.missing, _), (_, .missing):
        false
    case _:
        valueOrder(lhs) < valueOrder(rhs)
    }
}

private func valueOrder(_ value: DataValue) -> Int {
    switch value {
    case .number: 0
    case .boolean: 1
    case .date: 2
    case .string: 3
    case .missing: 4
    }
}
