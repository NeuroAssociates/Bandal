import Foundation

extension DataFrame {
    /// Returns a new frame sorted by one or more columns.
    ///
    /// Sort keys are applied in order, ties preserve their input order, and
    /// missing values remain last in both ascending and descending order.
    public func sorted(by descriptors: [SortDescriptor]) throws -> DataFrame {
        guard !descriptors.isEmpty else { return self }

        if descriptors.count == 1 {
            let desc = descriptors[0]
            guard let col = columns.first(where: { $0.name == desc.column }) else {
                throw DataFrameError.columnNotFound(desc.column)
            }
            let ascending = desc.order == .ascending
            var indices = Array(0..<rowCount)

            if let fCol = col.storage as? Float64Column {
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
            } else if let catCol = col.storage as? CategoricalColumn {
                let dict = catCol.dictionary
                let sortedDictIndices = (0..<dict.count).sorted { dict[$0] < dict[$1] }
                var rank = [Int](repeating: 0, count: dict.count)
                for (r, dIdx) in sortedDictIndices.enumerated() {
                    rank[dIdx] = r
                }
                let codes = catCol.codes
                let nulls = catCol.nullBitmap
                indices.sort { i, j in
                    let ni = nulls[i]
                    let nj = nulls[j]
                    if ni != nj { return !ni }
                    if ni { return i < j }
                    let ri = rank[Int(codes[i])]
                    let rj = rank[Int(codes[j])]
                    if ri != rj { return ascending ? ri < rj : ri > rj }
                    return i < j
                }
                return try rows(at: indices)
            } else if let dateCol = col.storage as? DateColumn {
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
