extension DataFrame {
    /// Returns a numeric column as Swift values.
    public func column(_ name: String) throws -> [Double] {
        let buffer = try numericBuffer(in: name)
        guard !buffer.hasMissing else {
            throw DataFrameError.nonNumericColumn(name)
        }
        return buffer.values
    }

    /// Returns the values in a numeric, text, or mixed column.
    public func values(in name: String) throws -> [DataValue] {
        guard let column = columns.first(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        return column.values
    }

    /// Returns a Boolean mask indicating which values in a column are missing.
    public func isMissing(in name: String) throws -> [Bool] {
        guard let column = columns.first(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        if let buffer = column.numericBuffer {
            return buffer.isMissing
        }
        let bm = column.storage.nullBitmap
        let n = column.storage.count
        if bm.allFalse {
            return Array(repeating: false, count: n)
        }
        return (0..<n).map { bm[$0] }
    }

    /// Returns the number of missing values in each column.
    public func missingCounts() -> [String: Int] {
        Dictionary(uniqueKeysWithValues: columns.map { column in
            (column.name, column.storage.nullCount)
        })
    }

    /// Returns a copy with missing values in the selected columns replaced.
    ///
    /// When `columns` is omitted, every column is filled. Pass `.missing` to
    /// leave values unchanged. The source frame is never modified.
    public func fillMissing(
        in selectedColumns: [String]? = nil,
        with replacement: DataValue
    ) throws -> DataFrame {
        let names = try validatedColumnNames(selectedColumns)
        if case .number(let number) = replacement, !number.isFinite {
            for col in columns where names.contains(col.name) {
                for row in 0..<col.storage.count where col.storage.isNull(at: row) {
                    throw DataFrameError.nonFiniteValue(column: col.name, row: row)
                }
            }
        }
        guard replacement != .missing else { return self }
        let selected = Set(names)
        let updated: [DataColumn] = columns.map { column in
            guard selected.contains(column.name), column.storage.hasNulls else {
                return column
            }
            if let floatCol = column.storage as? Float64Column,
               case .number(let number) = replacement
            {
                var newVals = floatCol.values
                let bm = floatCol.nullBitmap
                for i in 0..<newVals.count where bm[i] {
                    newVals[i] = number
                }
                let filled = Float64Column(
                    name: column.name,
                    values: newVals,
                    nullBitmap: BitSet(count: newVals.count)
                )
                return DataColumn(storage: filled)
            }
            if let strCol = column.storage as? StringColumn,
               case .string(let text) = replacement
            {
                var newStrings = strCol.stringData
                let bm = strCol.nullBitmap
                for i in 0..<newStrings.count where bm[i] {
                    newStrings[i] = text
                }
                let filled = StringColumn(
                    name: column.name,
                    stringData: newStrings,
                    nullBitmap: BitSet(count: newStrings.count)
                )
                return DataColumn(storage: filled)
            }
            let filledValues = column.values.map { $0 == .missing ? replacement : $0 }
            return DataColumn(name: column.name, values: filledValues)
        }
        return try DataFrame(dataColumns: updated)
    }

    /// Returns a copy with rows containing missing values removed.
    ///
    /// When `in` is omitted, a row is removed if any column is missing.
    public func dropMissing(in selectedColumns: [String]? = nil) throws -> DataFrame {
        let names = try validatedColumnNames(selectedColumns)
        let selected = Set(names)
        let selectedCols = columns.filter { selected.contains($0.name) }
        var combinedNulls = BitSet(count: rowCount)
        for col in selectedCols where col.storage.hasNulls {
            combinedNulls = combinedNulls.bitwiseOr(col.storage.nullBitmap)
        }
        if combinedNulls.allFalse {
            return self
        }
        return try rows(matching: combinedNulls.inverted())
    }

    private func validatedColumnNames(_ requestedNames: [String]?) throws -> [String] {
        guard let requestedNames else { return columnNames }
        guard !requestedNames.isEmpty else { throw DataFrameError.emptyColumns }
        for name in requestedNames {
            guard columnNames.contains(name) else {
                throw DataFrameError.columnNotFound(name)
            }
        }
        return requestedNames
    }
}
