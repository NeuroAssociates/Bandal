import Foundation

extension DataFrame {
    /// Converts selected identifier columns and remaining columns into
    /// variable/value rows.
    ///
    /// Output is column-major: values for each source column retain their input
    /// row order, and source columns retain their original order.
    public func melt(
        idColumns: [String],
        variableName: String = "variable",
        valueName: String = "value"
    ) throws -> DataFrame {
        guard !variableName.isEmpty, !valueName.isEmpty else {
            throw DataFrameError.emptyColumnName
        }

        var outputNames = Set<String>()
        for name in idColumns + [variableName, valueName] {
            guard outputNames.insert(name).inserted else {
                throw DataFrameError.duplicateColumnName(name)
            }
        }
        let idColumnValues = try idColumns.map { try values(in: $0) }
        let idNames = Set(idColumns)
        let valueColumns = columns.filter { !idNames.contains($0.name) }
        var outputColumns = idColumns.map { name in
            (name: name, values: [DataValue]())
        }
        outputColumns.append((name: variableName, values: []))
        outputColumns.append((name: valueName, values: []))

        for sourceColumn in valueColumns {
            for row in 0..<rowCount {
                for index in idColumns.indices {
                    outputColumns[index].values.append(idColumnValues[index][row])
                }
                outputColumns[idColumns.count].values.append(.string(sourceColumn.name))
                outputColumns[idColumns.count + 1].values.append(sourceColumn.values[row])
            }
        }
        return try DataFrame(columns: outputColumns)
    }

    /// Aggregates values into a two-dimensional table.
    ///
    /// Rows with missing index or pivot keys are omitted, matching `groupBy`.
    /// Missing values in `values` are ignored by the selected aggregation.
    /// Pivot columns are sorted by their typed `DataValue` order and named with
    /// the value column, reflected pivot key, and aggregation operation.
    public func pivotTable(
        index: [String],
        columns pivotColumn: String,
        values valueColumn: String,
        using operation: AggregateFunction
    ) throws -> DataFrame {
        guard !index.isEmpty else { throw DataFrameError.emptyColumns }
        var keyNames = Set<String>()
        for name in index + [pivotColumn, valueColumn] {
            guard keyNames.insert(name).inserted else {
                throw DataFrameError.duplicateColumnName(name)
            }
            _ = try values(in: name)
        }

        let pivotValues = Set(try values(in: pivotColumn).filter { $0 != .missing })
            .sorted(by: isOrderedBefore)
        let groupingKeys = index + [pivotColumn]
        let grouped = try aggregate(
            keys: groupingKeys,
            aggregations: [Aggregation(valueColumn, using: operation)]
        )
        let groupedKeyColumns = try groupingKeys.map { try grouped.values(in: $0) }
        let groupedValues = try grouped.values(in: "\(valueColumn)_\(operation.name)")
        var valuesByKey: [[DataValue]: DataValue] = [:]
        for row in 0..<grouped.rowCount {
            valuesByKey[groupedKeyColumns.map { $0[row] }] = groupedValues[row]
        }

        let indexKeys = Set((0..<grouped.rowCount).map { row in
            groupedKeyColumns.prefix(index.count).map { $0[row] }
        }).sorted(by: dataValueKeysAreOrderedBefore)
        var outputColumns = index.enumerated().map { indexPosition, name in
            (
                name: name,
                values: indexKeys.map { $0[indexPosition] }
            )
        }

        for pivotValue in pivotValues {
            let name = "\(valueColumn)_\(pivotKeyName(pivotValue))_\(operation.name)"
            let resultValues = indexKeys.map { indexKey in
                valuesByKey[indexKey + [pivotValue]] ?? .missing
            }
            outputColumns.append((name: name, values: resultValues))
        }
        return try DataFrame(columns: outputColumns)
    }

    /// Splits delimited string entries in `column` into separate rows,
    /// duplicating the other columns in each expanded row (similar to `df.explode()`).
    ///
    /// Whitespace around split items is trimmed. Non-string or `.missing` cells
    /// remain a single row unchanged.
    public func exploded(
        _ column: String,
        separator: String = ","
    ) throws -> DataFrame {
        guard let targetIndex = columns.firstIndex(where: { $0.name == column }) else {
            throw DataFrameError.columnNotFound(column)
        }
        guard !separator.isEmpty else {
            return self
        }
        var outputColumns = columns.map { (name: $0.name, values: [DataValue]()) }
        for row in 0..<rowCount {
            let cell = columns[targetIndex].values[row]
            let expandedValues: [DataValue]
            if case .string(let text) = cell {
                let parts = text
                    .components(separatedBy: separator)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                expandedValues = parts.isEmpty
                    ? [.string(text)]
                    : parts.map { $0.isEmpty ? .missing : .string($0) }
            } else {
                expandedValues = [cell]
            }

            for item in expandedValues {
                for colIndex in columns.indices {
                    if colIndex == targetIndex {
                        outputColumns[colIndex].values.append(item)
                    } else {
                        outputColumns[colIndex].values.append(columns[colIndex].values[row])
                    }
                }
            }
        }
        return try DataFrame(columns: outputColumns)
    }

    /// Transposes rows and columns (similar to `df.T`).
    ///
    /// The first column (`indexColumnName`) stores the original column names,
    /// and each original row `i` becomes a column named `"\(rowPrefix)\(i)"`.
    public func transposed(
        indexColumnName: String = "column",
        rowPrefix: String = "row_"
    ) throws -> DataFrame {
        guard !indexColumnName.isEmpty else {
            throw DataFrameError.emptyColumnName
        }
        var output: [(name: String, values: [DataValue])] = [
            (name: indexColumnName, values: columnNames.map(DataValue.string)),
        ]
        for row in 0..<rowCount {
            let colName = "\(rowPrefix)\(row)"
            let colValues = columns.map { $0.values[row] }
            output.append((name: colName, values: colValues))
        }
        return try DataFrame(columns: output)
    }
}

private func pivotKeyName(_ value: DataValue) -> String {
    switch value {
    case .number(let number): "number_\(number)"
    case .string(let string): "string_\(string)"
    case .boolean(let boolean): "boolean_\(boolean)"
    case .date(let date): "date_\(formatDateAsISO8601(date))"
    case .missing: "missing"
    }
}

private func dataValueKeysAreOrderedBefore(
    _ lhs: [DataValue],
    _ rhs: [DataValue]
) -> Bool {
    for (left, right) in zip(lhs, rhs) where left != right {
        return isOrderedBefore(left, right)
    }
    return false
}
