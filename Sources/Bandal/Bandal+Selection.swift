import Foundation

/// A composable, type-safe row predicate used with ``DataFrame/query(_:)-5r82k``.
public struct RowPredicate: Sendable {
    let evaluate: @Sendable ([String: DataValue]) throws -> Bool
    let fastMask: (@Sendable (DataFrame) throws -> BitSet?)?

    /// Creates a predicate from a closure evaluated against each row.
    ///
    /// - Parameter evaluate: A closure receiving the row as a dictionary of column name to value, returning `true` to keep the row. Errors it throws propagate to the caller.
    public init(_ evaluate: @escaping @Sendable ([String: DataValue]) throws -> Bool) {
        self.evaluate = evaluate
        self.fastMask = nil
    }

    init(
        evaluate: @escaping @Sendable ([String: DataValue]) throws -> Bool,
        fastMask: (@Sendable (DataFrame) throws -> BitSet?)?
    ) {
        self.evaluate = evaluate
        self.fastMask = fastMask
    }

    private static func requireValue(in row: [String: DataValue], column: String) throws -> DataValue {
        guard let value = row[column] else {
            throw DataFrameError.columnNotFound(column)
        }
        return value
    }

    private static func requireColumn(in df: DataFrame, name: String) throws -> DataColumn {
        guard let col = df.columns.first(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        return col
    }

    private static func numericMask(
        buffer: NumericBuffer,
        predicate: (Double) -> Bool
    ) -> BitSet {
        let n = buffer.count
        if n == 0 { return BitSet() }
        let wordCount = (n + 63) >> 6
        let hasNulls = buffer.hasMissing
        let nullWords = buffer.floatColumn.nullBitmap.words
        var anyTrue = false
        let words = ContiguousArray<UInt64>(unsafeUninitializedCapacity: wordCount) { outBuf, initCount in
            buffer.floatColumn.values.withUnsafeBufferPointer { valBuf in
                guard let valBase = valBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                let fullWords = n >> 6
                if !hasNulls {
                    for w in 0..<fullWords {
                        let baseIdx = w &<< 6
                        var word: UInt64 = 0
                        for bit in 0..<64 {
                            if predicate(valBase[baseIdx + bit]) {
                                word |= (1 &<< UInt64(bit))
                            }
                        }
                        if word != 0 { anyTrue = true }
                        outBase[w] = word
                    }
                    let rem = n & 63
                    if rem > 0 {
                        let baseIdx = fullWords &<< 6
                        var word: UInt64 = 0
                        for bit in 0..<rem {
                            if predicate(valBase[baseIdx + bit]) {
                                word |= (1 &<< UInt64(bit))
                            }
                        }
                        if word != 0 { anyTrue = true }
                        outBase[fullWords] = word
                    }
                } else {
                    for w in 0..<fullWords {
                        let baseIdx = w &<< 6
                        let nWord = nullWords[w]
                        var word: UInt64 = 0
                        for bit in 0..<64 {
                            let bitMask = (1 as UInt64) &<< UInt64(bit)
                            if (nWord & bitMask) == 0 && predicate(valBase[baseIdx + bit]) {
                                word |= bitMask
                            }
                        }
                        if word != 0 { anyTrue = true }
                        outBase[w] = word
                    }
                    let rem = n & 63
                    if rem > 0 {
                        let baseIdx = fullWords &<< 6
                        let nWord = nullWords[fullWords]
                        var word: UInt64 = 0
                        for bit in 0..<rem {
                            let bitMask = (1 as UInt64) &<< UInt64(bit)
                            if (nWord & bitMask) == 0 && predicate(valBase[baseIdx + bit]) {
                                word |= bitMask
                            }
                        }
                        if word != 0 { anyTrue = true }
                        outBase[fullWords] = word
                    }
                }
            }
            initCount = wordCount
        }
        return BitSet(words: words, count: n, mayHaveTrueBits: anyTrue)
    }

    /// Matches rows where `column` equals `value`.
    public static func column(_ name: String, equals value: DataValue) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                try requireValue(in: row, column: name) == value
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                if value == .missing {
                    return col.storage.nullBitmap
                }
                if let buf = col.numericBuffer, case .number(let target) = value {
                    return numericMask(buffer: buf) { $0 == target }
                }
                var mask = BitSet(count: df.rowCount)
                if let strCol = col.storage as? StringColumn, case .string(let target) = value {
                    let vals = strCol.stringData
                    let nulls = strCol.nullBitmap
                    for i in 0..<vals.count where !nulls[i] && vals[i] == target {
                        mask[i] = true
                    }
                    return mask
                }
                if let catCol = col.storage as? CategoricalColumn, case .string(let target) = value {
                    guard let code = catCol.dictionary.firstIndex(of: target).map(UInt32.init) else {
                        return mask
                    }
                    let codes = catCol.codes
                    let nulls = catCol.nullBitmap
                    for i in 0..<codes.count where !nulls[i] && codes[i] == code {
                        mask[i] = true
                    }
                    return mask
                }
                if let boolCol = col.storage as? BooleanColumn, case .boolean(let target) = value {
                    let bits = boolCol.bitmap
                    let nulls = boolCol.nullBitmap
                    for i in 0..<boolCol.count where !nulls[i] && bits[i] == target {
                        mask[i] = true
                    }
                    return mask
                }
                for i in 0..<df.rowCount where col.storage.value(at: i) == value {
                    mask[i] = true
                }
                return mask
            }
        )
    }

    /// Matches rows where `column` does not equal `value`.
    public static func column(_ name: String, notEquals value: DataValue) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                try requireValue(in: row, column: name) != value
            },
            fastMask: { df in
                if let eqMask = try RowPredicate.column(name, equals: value).fastMask?(df) {
                    return eqMask.inverted()
                }
                return nil
            }
        )
    }

    /// Matches rows where numeric `column` is strictly greater than `threshold`.
    public static func column(_ name: String, greaterThan threshold: Double) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                guard case .number(let n) = try requireValue(in: row, column: name) else { return false }
                return n > threshold
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                if let buf = col.numericBuffer {
                    return numericMask(buffer: buf) { $0 > threshold }
                }
                var mask = BitSet(count: df.rowCount)
                for i in 0..<df.rowCount {
                    if case .number(let n) = col.storage.value(at: i), n > threshold {
                        mask[i] = true
                    }
                }
                return mask
            }
        )
    }

    /// Matches rows where numeric `column` is greater than or equal to `threshold`.
    public static func column(_ name: String, greaterThanOrEqual threshold: Double) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                guard case .number(let n) = try requireValue(in: row, column: name) else { return false }
                return n >= threshold
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                if let buf = col.numericBuffer {
                    return numericMask(buffer: buf) { $0 >= threshold }
                }
                var mask = BitSet(count: df.rowCount)
                for i in 0..<df.rowCount {
                    if case .number(let n) = col.storage.value(at: i), n >= threshold {
                        mask[i] = true
                    }
                }
                return mask
            }
        )
    }

    /// Matches rows where numeric `column` is strictly less than `threshold`.
    public static func column(_ name: String, lessThan threshold: Double) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                guard case .number(let n) = try requireValue(in: row, column: name) else { return false }
                return n < threshold
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                if let buf = col.numericBuffer {
                    return numericMask(buffer: buf) { $0 < threshold }
                }
                var mask = BitSet(count: df.rowCount)
                for i in 0..<df.rowCount {
                    if case .number(let n) = col.storage.value(at: i), n < threshold {
                        mask[i] = true
                    }
                }
                return mask
            }
        )
    }

    /// Matches rows where numeric `column` is less than or equal to `threshold`.
    public static func column(_ name: String, lessThanOrEqual threshold: Double) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                guard case .number(let n) = try requireValue(in: row, column: name) else { return false }
                return n <= threshold
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                if let buf = col.numericBuffer {
                    return numericMask(buffer: buf) { $0 <= threshold }
                }
                var mask = BitSet(count: df.rowCount)
                for i in 0..<df.rowCount {
                    if case .number(let n) = col.storage.value(at: i), n <= threshold {
                        mask[i] = true
                    }
                }
                return mask
            }
        )
    }

    /// Matches rows where numeric `column` lies in `lower...upper` (or `(lower, upper)` when `inclusive` is `false`).
    public static func column(
        _ name: String,
        between lower: Double,
        and upper: Double,
        inclusive: Bool = true
    ) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                guard case .number(let n) = try requireValue(in: row, column: name) else { return false }
                return inclusive ? (n >= lower && n <= upper) : (n > lower && n < upper)
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                if let buf = col.numericBuffer {
                    if inclusive {
                        return numericMask(buffer: buf) { $0 >= lower && $0 <= upper }
                    } else {
                        return numericMask(buffer: buf) { $0 > lower && $0 < upper }
                    }
                }
                var mask = BitSet(count: df.rowCount)
                for i in 0..<df.rowCount {
                    if case .number(let n) = col.storage.value(at: i) {
                        if inclusive ? (n >= lower && n <= upper) : (n > lower && n < upper) {
                            mask[i] = true
                        }
                    }
                }
                return mask
            }
        )
    }

    /// Matches rows where `column` belongs to `candidates`.
    public static func column(_ name: String, in candidates: Set<DataValue>) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                candidates.contains(try requireValue(in: row, column: name))
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                var mask = BitSet(count: df.rowCount)
                for i in 0..<df.rowCount where candidates.contains(col.storage.value(at: i)) {
                    mask[i] = true
                }
                return mask
            }
        )
    }

    /// Matches rows where string `column` contains `substring`.
    public static func column(
        _ name: String,
        contains substring: String,
        caseSensitive: Bool = true
    ) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                guard case .string(let s) = try requireValue(in: row, column: name) else { return false }
                if caseSensitive {
                    return s.contains(substring)
                }
                return s.range(of: substring, options: .caseInsensitive) != nil
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                var mask = BitSet(count: df.rowCount)
                for i in 0..<df.rowCount {
                    if case .string(let s) = col.storage.value(at: i) {
                        let matches = caseSensitive
                            ? s.contains(substring)
                            : (s.range(of: substring, options: .caseInsensitive) != nil)
                        if matches {
                            mask[i] = true
                        }
                    }
                }
                return mask
            }
        )
    }

    /// Matches rows where `column` is `.missing`.
    public static func isMissing(_ name: String) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                try requireValue(in: row, column: name) == .missing
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                return col.storage.nullBitmap
            }
        )
    }

    /// Matches rows where `column` is not `.missing`.
    public static func notMissing(_ name: String) -> RowPredicate {
        RowPredicate(
            evaluate: { row in
                try requireValue(in: row, column: name) != .missing
            },
            fastMask: { df in
                let col = try requireColumn(in: df, name: name)
                return col.storage.nullBitmap.inverted()
            }
        )
    }

    /// Logical conjunction (`AND`) of two predicates.
    public func and(_ other: RowPredicate) -> RowPredicate {
        let lhs = evaluate
        let rhs = other.evaluate
        let lhsFast = fastMask
        let rhsFast = other.fastMask
        return RowPredicate(
            evaluate: { row in
                try lhs(row) && rhs(row)
            },
            fastMask: { df in
                guard let m1 = try lhsFast?(df), let m2 = try rhsFast?(df) else { return nil }
                return m1.bitwiseAnd(m2)
            }
        )
    }

    /// Logical disjunction (`OR`) of two predicates.
    public func or(_ other: RowPredicate) -> RowPredicate {
        let lhs = evaluate
        let rhs = other.evaluate
        let lhsFast = fastMask
        let rhsFast = other.fastMask
        return RowPredicate(
            evaluate: { row in
                try lhs(row) || rhs(row)
            },
            fastMask: { df in
                guard let m1 = try lhsFast?(df), let m2 = try rhsFast?(df) else { return nil }
                return m1.bitwiseOr(m2)
            }
        )
    }

    /// Logical negation (`NOT`) of a predicate.
    public var negated: RowPredicate {
        let inner = evaluate
        let innerFast = fastMask
        return RowPredicate(
            evaluate: { row in
                try !inner(row)
            },
            fastMask: { df in
                guard let m = try innerFast?(df) else { return nil }
                return m.inverted()
            }
        )
    }

    /// Returns a predicate that matches rows satisfying both operands.
    public static func && (lhs: RowPredicate, rhs: RowPredicate) -> RowPredicate {
        lhs.and(rhs)
    }

    /// Returns a predicate that matches rows satisfying either operand.
    public static func || (lhs: RowPredicate, rhs: RowPredicate) -> RowPredicate {
        lhs.or(rhs)
    }

    /// Returns a predicate that matches rows not satisfying the operand.
    public static prefix func ! (predicate: RowPredicate) -> RowPredicate {
        predicate.negated
    }
}

/// Date components extractable from `.date` columns in UTC.
public enum DateComponent: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case year
    case month
    case day
    case weekday
    case hour
    case minute
    case second

    /// The lowercase name of the component.
    public var description: String { rawValue }
}

extension DataFrame {
    /// Returns a new data frame containing the requested columns in the given order.
    public func select(_ names: String...) throws -> DataFrame {
        try select(columns: names)
    }

    /// Returns a new data frame containing the requested columns in the given order.
    public func select(columns names: [String]) throws -> DataFrame {
        guard !names.isEmpty else {
            throw DataFrameError.emptyColumns
        }
        let selected = try names.map { name -> DataColumn in
            guard let col = columns.first(where: { $0.name == name }) else {
                throw DataFrameError.columnNotFound(name)
            }
            return col
        }
        return try DataFrame(dataColumns: selected)
    }

    /// Returns the row at `index`, or `nil` when `index` is out of bounds.
    public func row(at index: Int) -> DataRow? {
        guard (0..<rowCount).contains(index) else { return nil }
        var map: [String: DataValue] = [:]
        map.reserveCapacity(columns.count)
        for column in columns {
            map[column.name] = column.storage.value(at: index)
        }
        return DataRow(index: index, columnOrder: columnNames, values: map)
    }

    /// All rows of the frame as ``DataRow`` values.
    public var rows: [DataRow] {
        (0..<rowCount).compactMap { row(at: $0) }
    }

    /// Accesses a row by zero-based index (`df[row: 0]`).
    public subscript(row index: Int) -> DataRow? {
        row(at: index)
    }

    /// Accesses a contiguous slice of rows (`df[rows: 0..<5]`).
    public subscript(rows range: Range<Int>) -> DataFrame? {
        let lower = Swift.max(0, Swift.min(range.lowerBound, rowCount))
        let upper = Swift.max(lower, Swift.min(range.upperBound, rowCount))
        return try? rows(in: lower..<upper)
    }

    /// Accesses a single cell by column name and row index (`df["price", 0]`).
    public subscript(column: String, row: Int) -> DataValue? {
        guard (0..<rowCount).contains(row),
              let col = columns.first(where: { $0.name == column })
        else {
            return nil
        }
        return col.storage.value(at: row)
    }

    /// Exports the frame as a dictionary mapping each column name to its `[DataValue]` array.
    public func toDictionary() -> [String: [DataValue]] {
        Dictionary(uniqueKeysWithValues: columns.map { ($0.name, $0.values) })
    }

    /// Returns the first `count` rows.
    public func head(_ count: Int = 5) throws -> DataFrame {
        try rows(in: 0..<min(max(count, 0), rowCount))
    }

    /// Returns the last `count` rows.
    public func tail(_ count: Int = 5) throws -> DataFrame {
        let size = min(max(count, 0), rowCount)
        return try rows(in: (rowCount - size)..<rowCount)
    }

    /// Returns a new frame containing rows for which the predicate is true.
    public func filterRows(
        _ predicate: ([String: DataValue]) throws -> Bool
    ) throws -> DataFrame {
        var mask = BitSet(count: rowCount)
        var row: [String: DataValue] = [:]
        row.reserveCapacity(columns.count)
        for rowIndex in 0..<rowCount {
            row.removeAll(keepingCapacity: true)
            for column in columns {
                row[column.name] = column.storage.value(at: rowIndex)
            }
            if try predicate(row) {
                mask[rowIndex] = true
            }
        }
        return try rows(matching: mask)
    }

    /// Filters rows using a composable ``RowPredicate``.
    public func query(_ predicate: RowPredicate) throws -> DataFrame {
        if let mask = try predicate.fastMask?(self) {
            return try rows(matching: mask)
        }
        return try filterRows(predicate.evaluate)
    }

    /// Filters rows using a concise query expression such as `"score >= 80 && active == true"`.
    ///
    /// Supported operators: `==`, `!=`, `>=`, `<=`, `>`, `<`, combined with
    /// `&&` / `and` or `||` / `or`. Right-hand literals may be numbers,
    /// `true`/`false`, `nil`/`null`/`NaN` (for `.missing`), or quoted strings.
    public func query(_ expression: String) throws -> DataFrame {
        let predicate = try Self.parseQueryExpression(expression, availableColumns: Set(columnNames))
        return try query(predicate)
    }

    /// Internal helper to add or replace a column directly from an `AnyColumn` storage.
    func withColumn(storage: any AnyColumn) throws -> DataFrame {
        guard !storage.name.isEmpty else {
            throw DataFrameError.emptyColumnName
        }
        guard storage.count == rowCount else {
            throw DataFrameError.unequalColumnLengths(
                column: storage.name,
                expected: rowCount,
                actual: storage.count
            )
        }
        let newCol = DataColumn(storage: storage)
        var updated = columns
        if let index = updated.firstIndex(where: { $0.name == storage.name }) {
            updated[index] = newCol
        } else {
            updated.append(newCol)
        }
        return try DataFrame(dataColumns: updated)
    }

    /// Returns a copy with a new or replaced column.
    public func withColumn(name: String, values: [DataValue]) throws -> DataFrame {
        guard !name.isEmpty else {
            throw DataFrameError.emptyColumnName
        }
        guard values.count == rowCount else {
            throw DataFrameError.unequalColumnLengths(
                column: name,
                expected: rowCount,
                actual: values.count
            )
        }
        for (row, value) in values.enumerated() {
            if case .number(let number) = value, !number.isFinite {
                throw DataFrameError.nonFiniteValue(column: name, row: row)
            }
        }
        let newCol = DataColumn(name: name, values: values)
        var updated = columns
        if let index = updated.firstIndex(where: { $0.name == name }) {
            updated[index] = newCol
        } else {
            updated.append(newCol)
        }
        return try DataFrame(dataColumns: updated)
    }

    /// Returns a copy with a new or replaced column computed row-by-row (similar to `df.assign()`).
    public func withColumn(
        name: String,
        computing transform: (DataRow) throws -> DataValue
    ) throws -> DataFrame {
        let computed = try (0..<rowCount).map { rowIndex -> DataValue in
            guard let row = row(at: rowIndex) else { return .missing }
            return try transform(row)
        }
        return try withColumn(name: name, values: computed)
    }

    /// Returns a copy with the specified column names changed.
    ///
    /// Unspecified columns keep their names. Unknown source names and resulting
    /// duplicate or empty names are rejected.
    public func renamedColumns(_ names: [String: String]) throws -> DataFrame {
        for name in names.keys where !columnNames.contains(name) {
            throw DataFrameError.columnNotFound(name)
        }
        let updated = columns.map { column in
            if let newName = names[column.name] {
                return column.renamed(to: newName)
            }
            return column
        }
        return try DataFrame(dataColumns: updated)
    }

    /// Converts a column to the requested concrete type without mutating the frame.
    ///
    /// Missing values remain missing. String-to-number conversion accepts the
    /// same comma and dollar-sign conventions as CSV numeric inference. Dates
    /// use `dateFormat` when supplied, or ISO 8601 otherwise.
    public func castColumn(
        _ name: String,
        to type: DataType,
        dateFormat: String? = nil
    ) throws -> DataFrame {
        guard type == .number || type == .string || type == .boolean || type == .date else {
            throw DataFrameError.invalidCSV("cannot cast a column to \(dataTypeName(type))")
        }
        let source = try values(in: name)
        let converted = try source.enumerated().map { row, value in
            try convert(value, to: type, dateFormat: dateFormat)
                .mapError { _ in
                    DataFrameError.invalidConversion(column: name, row: row, type: type)
                }.get()
        }
        return try withColumn(name: name, values: converted)
    }

    /// Extracts a calendar component (`.year`, `.month`, `.day`, `.weekday`,
    /// `.hour`, `.minute`, `.second`) in UTC from a `.date` column into a
    /// numeric column.
    public func dateComponent(
        _ component: DateComponent,
        from column: String,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        let source = try values(in: column)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let targetName = outputColumn ?? "\(column)_\(component.rawValue)"

        let extracted = try source.enumerated().map { row, value -> DataValue in
            switch value {
            case .missing:
                return .missing
            case .date(let date):
                let result: Int
                switch component {
                case .year: result = calendar.component(.year, from: date)
                case .month: result = calendar.component(.month, from: date)
                case .day: result = calendar.component(.day, from: date)
                case .weekday: result = calendar.component(.weekday, from: date)
                case .hour: result = calendar.component(.hour, from: date)
                case .minute: result = calendar.component(.minute, from: date)
                case .second: result = calendar.component(.second, from: date)
                }
                return .number(Double(result))
            default:
                throw DataFrameError.invalidConversion(column: column, row: row, type: .date)
            }
        }
        return try withColumn(name: targetName, values: extracted)
    }

    // MARK: String column operations

    /// Converts string values in `column` to lowercase. Non-string values (other than `.missing`) throw `.invalidConversion`.
    public func lowercased(_ column: String, as outputColumn: String? = nil) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn) { .string($0.lowercased()) }
    }

    /// Converts string values in `column` to uppercase.
    public func uppercased(_ column: String, as outputColumn: String? = nil) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn) { .string($0.uppercased()) }
    }

    /// Trims leading and trailing whitespace and newlines from string values in `column`.
    public func trimmed(_ column: String, as outputColumn: String? = nil) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn) {
            .string($0.trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }

    /// Computes character counts of string values in `column` as a numeric column.
    public func stringLength(_ column: String, as outputColumn: String? = nil) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn ?? "\(column)_length") {
            .number(Double($0.count))
        }
    }

    /// Replaces occurrences of `target` with `replacement` in string values of `column`.
    public func replacingSubstring(
        in column: String,
        target: String,
        with replacement: String,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn) {
            .string($0.replacingOccurrences(of: target, with: replacement))
        }
    }

    /// Produces a Boolean column indicating whether each string in `column` contains `substring`.
    public func containsSubstring(
        _ column: String,
        _ substring: String,
        caseSensitive: Bool = true,
        as outputColumn: String? = nil
    ) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn ?? "\(column)_contains") { text in
            if caseSensitive {
                return .boolean(text.contains(substring))
            }
            return .boolean(text.range(of: substring, options: .caseInsensitive) != nil)
        }
    }

    private func transformStrings(
        in column: String,
        as outputColumn: String?,
        _ transform: (String) -> DataValue
    ) throws -> DataFrame {
        let source = try values(in: column)
        let transformed = try source.enumerated().map { row, value -> DataValue in
            switch value {
            case .missing:
                return .missing
            case .string(let text):
                return transform(text)
            default:
                throw DataFrameError.invalidConversion(column: column, row: row, type: .string)
            }
        }
        return try withColumn(name: outputColumn ?? column, values: transformed)
    }

    /// Returns a row-aligned mask that marks duplicate rows.
    ///
    /// If `in` is omitted, all columns are compared. Missing values compare as
    /// equal. `keep` controls which occurrence is not marked as a duplicate.
    public func duplicatedRows(
        in selectedColumns: [String]? = nil,
        keep: DuplicateKeep = .first
    ) throws -> [Bool] {
        let names = try validatedColumnNames(selectedColumns)
        let selectedCols = try names.map { name -> any AnyColumn in
            guard let col = columns.first(where: { $0.name == name }) else {
                throw DataFrameError.columnNotFound(name)
            }
            return col.storage
        }
        let rowKeys = (0..<rowCount).map { row in
            selectedCols.map { $0.value(at: row) }
        }
        var duplicateMask = Array(repeating: false, count: rowCount)

        switch keep {
        case .first:
            var seen = Set<[DataValue]>()
            for row in rowKeys.indices {
                duplicateMask[row] = !seen.insert(rowKeys[row]).inserted
            }
        case .last:
            var seen = Set<[DataValue]>()
            for row in rowKeys.indices.reversed() {
                duplicateMask[row] = !seen.insert(rowKeys[row]).inserted
            }
        case .none:
            var counts: [[DataValue]: Int] = [:]
            for key in rowKeys {
                counts[key, default: 0] += 1
            }
            for row in rowKeys.indices {
                duplicateMask[row] = counts[rowKeys[row], default: 0] > 1
            }
        }
        return duplicateMask
    }

    /// Returns a copy with duplicate rows removed.
    ///
    /// Rows are kept in their original order. If `in` is omitted, all columns
    /// determine whether rows are duplicates.
    public func dropDuplicates(
        in selectedColumns: [String]? = nil,
        keep: DuplicateKeep = .first
    ) throws -> DataFrame {
        let duplicateMask = try duplicatedRows(in: selectedColumns, keep: keep)
        var keepBits = BitSet(count: rowCount)
        for i in 0..<rowCount where !duplicateMask[i] {
            keepBits[i] = true
        }
        return try rows(matching: keepBits)
    }

    /// Concatenates frames vertically, requiring identical column names and order.
    ///
    /// Values and rows retain their original order. An empty frame list is
    /// invalid because a `DataFrame` must have at least one column.
    public static func concatenatingRows(_ frames: [DataFrame]) throws -> DataFrame {
        guard let first = frames.first else {
            throw DataFrameError.emptyColumns
        }
        var combined = first.columns.map { (name: $0.name, values: $0.values) }
        for frame in frames.dropFirst() {
            guard frame.columnNames == first.columnNames else {
                throw DataFrameError.incompatibleSchema(
                    expected: first.columnNames,
                    actual: frame.columnNames
                )
            }
            for index in combined.indices {
                combined[index].values.append(contentsOf: frame.columns[index].values)
            }
        }
        return try DataFrame(columns: combined)
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

    private func convert(
        _ value: DataValue,
        to type: DataType,
        dateFormat: String?
    ) -> Result<DataValue, DataFrameError> {
        guard value != .missing else { return .success(.missing) }
        switch (value, type) {
        case (.number, .number), (.string, .string), (.boolean, .boolean), (.date, .date):
            return .success(value)
        case (.number(let number), .string):
            return .success(.string(String(number)))
        case (.boolean(let boolean), .string):
            return .success(.string(String(boolean)))
        case (.date(let date), .string):
            return .success(.string(formatDateAsISO8601(date)))
        case (.string(let string), .number):
            let normalized = string
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: ",", with: "")
                .replacingOccurrences(of: "$", with: "")
            guard let number = Double(normalized), number.isFinite else {
                return .failure(.invalidCSV("invalid numeric value"))
            }
            return .success(.number(number))
        case (.string(let string), .boolean):
            return switch string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "true": .success(.boolean(true))
            case "false": .success(.boolean(false))
            default: .failure(.invalidCSV("invalid Boolean value"))
            }
        case (.string(let string), .date):
            let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
            let date = dateFormat.flatMap { parseDate(trimmed, format: $0) }
                ?? (dateFormat == nil ? ISO8601DateFormatter().date(from: trimmed) : nil)
            return date.map { .success(.date($0)) }
                ?? .failure(.invalidCSV("invalid date value"))
        default:
            return .failure(.invalidCSV("incompatible value type"))
        }
    }

    // MARK: Query expression parser

    private static func parseQueryExpression(
        _ expression: String,
        availableColumns: Set<String>
    ) throws -> RowPredicate {
        let trimmed = expression.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw DataFrameError.invalidCSV("query expression cannot be empty")
        }

        let orClauses = trimmed.components(separatedBy: "||")
        if orClauses.count > 1 {
            let subPredicates = try orClauses.map {
                try parseQueryExpression($0, availableColumns: availableColumns)
            }
            return subPredicates.dropFirst().reduce(subPredicates[0]) { $0.or($1) }
        }

        let andClauses = trimmed.components(separatedBy: "&&")
        if andClauses.count > 1 {
            let subPredicates = try andClauses.map {
                try parseQueryExpression($0, availableColumns: availableColumns)
            }
            return subPredicates.dropFirst().reduce(subPredicates[0]) { $0.and($1) }
        }

        let operators = [">=", "<=", "==", "!=", ">", "<"]
        for op in operators {
            if let range = trimmed.range(of: op) {
                let colName = trimmed[..<range.lowerBound].trimmingCharacters(in: .whitespaces)
                let rhsText = trimmed[range.upperBound...].trimmingCharacters(in: .whitespaces)
                guard !colName.isEmpty, !rhsText.isEmpty else {
                    throw DataFrameError.invalidCSV("invalid query clause '\(trimmed)'")
                }
                guard availableColumns.contains(colName) else {
                    throw DataFrameError.columnNotFound(colName)
                }
                let literal = parseQueryLiteral(rhsText)
                switch op {
                case "==":
                    return .column(colName, equals: literal)
                case "!=":
                    return .column(colName, notEquals: literal)
                case ">":
                    guard case .number(let n) = literal else {
                        throw DataFrameError.invalidCSV("numeric literal required for '>' in '\(trimmed)'")
                    }
                    return .column(colName, greaterThan: n)
                case ">=":
                    guard case .number(let n) = literal else {
                        throw DataFrameError.invalidCSV("numeric literal required for '>=' in '\(trimmed)'")
                    }
                    return .column(colName, greaterThanOrEqual: n)
                case "<":
                    guard case .number(let n) = literal else {
                        throw DataFrameError.invalidCSV("numeric literal required for '<' in '\(trimmed)'")
                    }
                    return .column(colName, lessThan: n)
                case "<=":
                    guard case .number(let n) = literal else {
                        throw DataFrameError.invalidCSV("numeric literal required for '<=' in '\(trimmed)'")
                    }
                    return .column(colName, lessThanOrEqual: n)
                default:
                    break
                }
            }
        }
        throw DataFrameError.invalidCSV("unsupported query expression '\(trimmed)'")
    }

    private static func parseQueryLiteral(_ raw: String) -> DataValue {
        if (raw.hasPrefix("\"") && raw.hasSuffix("\"")) || (raw.hasPrefix("'") && raw.hasSuffix("'")),
           raw.count >= 2
        {
            return .string(String(raw.dropFirst().dropLast()))
        }
        switch raw.lowercased() {
        case "nil", "null", "nan":
            return .missing
        case "true":
            return .boolean(true)
        case "false":
            return .boolean(false)
        default:
            if let number = Double(raw), number.isFinite {
                return .number(number)
            }
            return .string(raw)
        }
    }
}

func dataTypeName(_ type: DataType) -> String {
    switch type {
    case .number: "number"
    case .string: "string"
    case .boolean: "boolean"
    case .date: "date"
    case .mixed: "mixed"
    case .unknown: "unknown"
    }
}
