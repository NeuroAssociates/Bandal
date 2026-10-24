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
        return column.values.map { $0 == .missing }
    }

    /// Returns the number of missing values in each column.
    public func missingCounts() -> [String: Int] {
        Dictionary(uniqueKeysWithValues: columns.map { column in
            if let buffer = column.numericBuffer {
                return (column.name, buffer.count - buffer.nonMissingCount)
            }
            return (column.name, column.values.filter { $0 == .missing }.count)
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
        let selected = Set(names)
        let updated = columns.map { column in
            (
                name: column.name,
                values: selected.contains(column.name)
                    ? column.values.map { $0 == .missing ? replacement : $0 }
                    : column.values
            )
        }
        return try DataFrame(columns: updated)
    }

    /// Returns a copy with rows containing missing values removed.
    ///
    /// When `in` is omitted, a row is removed if any column is missing.
    public func dropMissing(in selectedColumns: [String]? = nil) throws -> DataFrame {
        let names = try validatedColumnNames(selectedColumns)
        let selected = Set(names)
        let selectedValues = columns.filter { selected.contains($0.name) }.map(\.values)
        let indices = (0..<rowCount).filter { row in
            selectedValues.allSatisfy { $0[row] != .missing }
        }
        return try rows(at: indices)
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
