import Foundation

/// A composable, type-safe row predicate used with ``DataFrame/query(_:)-(RowPredicate)``.
public struct RowPredicate: Sendable {
    /// Row-by-row evaluation closure used when a vectorized fast mask is unavailable.
    let evaluate: @Sendable ([String: DataValue]) throws -> Bool
    /// Optional vectorized columnar evaluation closure returning a ``BitSet`` mask directly.
    let fastMask: (@Sendable (DataFrame) throws -> BitSet?)?

    /// Creates a predicate from a closure evaluated against each row.
    ///
    /// - Parameter evaluate: A closure receiving the row as a dictionary of column name to value, returning `true` to keep the row. Errors it throws propagate to the caller.
    public init(_ evaluate: @escaping @Sendable ([String: DataValue]) throws -> Bool) {
        self.evaluate = evaluate
        self.fastMask = nil
    }

    /// Creates a predicate with both a row-wise fallback closure and a columnar `fastMask` evaluator.
    ///
    /// - Parameters:
    ///   - evaluate: Row-wise evaluation closure.
    ///   - fastMask: Optional vectorized ``BitSet`` evaluator.
    init(
        evaluate: @escaping @Sendable ([String: DataValue]) throws -> Bool,
        fastMask: (@Sendable (DataFrame) throws -> BitSet?)?
    ) {
        self.evaluate = evaluate
        self.fastMask = fastMask
    }

    /// Looks up `column` in a row dictionary or throws ``DataFrameError/columnNotFound(_:)``.
    private static func requireValue(in row: [String: DataValue], column: String) throws -> DataValue {
        guard let value = row[column] else {
            throw DataFrameError.columnNotFound(column)
        }
        return value
    }

    /// Looks up `name` in `df` or throws ``DataFrameError/columnNotFound(_:)``.
    private static func requireColumn(in df: DataFrame, name: String) throws -> DataColumn {
        guard let col = df.columns.first(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        return col
    }

    /// Evaluates `predicate` across a ``NumericBuffer`` and packs results into 64-bit ``BitSet`` words.
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
    ///
    /// - Parameters:
    ///   - name: Column name to inspect.
    ///   - value: Target ``DataValue`` to match.
    /// - Returns: A ``RowPredicate`` matching rows where `column == value`.
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
    ///
    /// - Parameters:
    ///   - name: Column name to inspect.
    ///   - value: ``DataValue`` to exclude.
    /// - Returns: A ``RowPredicate`` matching rows where `column != value`.
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
    ///
    /// - Parameters:
    ///   - name: Numeric column name.
    ///   - threshold: Exclusive lower bound.
    /// - Returns: A ``RowPredicate`` matching rows where `column > threshold`.
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
    ///
    /// - Parameters:
    ///   - name: Numeric column name.
    ///   - threshold: Inclusive lower bound.
    /// - Returns: A ``RowPredicate`` matching rows where `column >= threshold`.
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
    ///
    /// - Parameters:
    ///   - name: Numeric column name.
    ///   - threshold: Exclusive upper bound.
    /// - Returns: A ``RowPredicate`` matching rows where `column < threshold`.
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
    ///
    /// - Parameters:
    ///   - name: Numeric column name.
    ///   - threshold: Inclusive upper bound.
    /// - Returns: A ``RowPredicate`` matching rows where `column <= threshold`.
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
    ///
    /// - Parameters:
    ///   - name: Numeric column name.
    ///   - lower: Lower bound.
    ///   - upper: Upper bound.
    ///   - inclusive: Whether endpoints are included. Defaults to `true`.
    /// - Returns: A ``RowPredicate`` matching rows within the specified interval.
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
    ///
    /// - Parameters:
    ///   - name: Column name to inspect.
    ///   - candidates: Set of allowed ``DataValue`` elements.
    /// - Returns: A ``RowPredicate`` matching rows whose cell value is in `candidates`.
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
    ///
    /// - Parameters:
    ///   - name: String column name.
    ///   - substring: Substring to search for.
    ///   - caseSensitive: Whether the search is case-sensitive. Defaults to `true`.
    /// - Returns: A ``RowPredicate`` matching rows whose string contains `substring`.
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
    ///
    /// - Parameter name: Column name to check.
    /// - Returns: A ``RowPredicate`` matching rows where `column` is `.missing`.
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
    ///
    /// - Parameter name: Column name to check.
    /// - Returns: A ``RowPredicate`` matching rows where `column` is not `.missing`.
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
    ///
    /// - Parameter other: The second predicate.
    /// - Returns: A combined ``RowPredicate`` that requires both predicates to match.
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
    ///
    /// - Parameter other: The second predicate.
    /// - Returns: A combined ``RowPredicate`` that matches when either predicate matches.
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
    ///
    /// - Parameters:
    ///   - lhs: Left-hand predicate.
    ///   - rhs: Right-hand predicate.
    /// - Returns: Conjunction of `lhs` and `rhs`.
    public static func && (lhs: RowPredicate, rhs: RowPredicate) -> RowPredicate {
        lhs.and(rhs)
    }

    /// Returns a predicate that matches rows satisfying either operand.
    ///
    /// - Parameters:
    ///   - lhs: Left-hand predicate.
    ///   - rhs: Right-hand predicate.
    /// - Returns: Disjunction of `lhs` and `rhs`.
    public static func || (lhs: RowPredicate, rhs: RowPredicate) -> RowPredicate {
        lhs.or(rhs)
    }

    /// Returns a predicate that matches rows not satisfying the operand.
    ///
    /// - Parameter predicate: Predicate to negate.
    /// - Returns: Negated ``RowPredicate``.
    public static prefix func ! (predicate: RowPredicate) -> RowPredicate {
        predicate.negated
    }
}

/// Date components extractable from `.date` columns in UTC.
public enum DateComponent: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    /// Four-digit calendar year in UTC.
    case year
    /// Calendar month (`1...12`) in UTC.
    case month
    /// Day of the month (`1...31`) in UTC.
    case day
    /// Gregorian weekday (`1...7`, Sunday = 1) in UTC.
    case weekday
    /// Hour of the day (`0...23`) in UTC.
    case hour
    /// Minute of the hour (`0...59`) in UTC.
    case minute
    /// Second of the minute (`0...59`) in UTC.
    case second

    /// The lowercase name of the component.
    public var description: String { rawValue }
}

extension DataFrame {
    /// Returns a new data frame containing the requested columns in the given order.
    ///
    /// - Parameter names: Variadic list of column names to keep.
    /// - Returns: A projected ``DataFrame``.
    /// - Throws: ``DataFrameError`` if `names` is empty or any column does not exist.
    public func select(_ names: String...) throws -> DataFrame {
        try select(columns: names)
    }

    /// Returns a new data frame containing the requested columns in the given order.
    ///
    /// - Parameter names: Array of column names to keep.
    /// - Returns: A projected ``DataFrame``.
    /// - Throws: ``DataFrameError`` if `names` is empty or any column does not exist.
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
    ///
    /// - Parameter index: Zero-based row index.
    /// - Returns: The ``DataRow`` at `index`, or `nil` if out of range.
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
    ///
    /// - Parameter index: Zero-based row index.
    /// - Returns: The ``DataRow`` at `index`, or `nil` if out of range.
    public subscript(row index: Int) -> DataRow? {
        row(at: index)
    }

    /// Accesses a contiguous slice of rows (`df[rows: 0..<5]`).
    ///
    /// - Parameter range: Half-open row index range, clamped to `0..<rowCount`.
    /// - Returns: A sliced ``DataFrame``.
    public subscript(rows range: Range<Int>) -> DataFrame? {
        let lower = Swift.max(0, Swift.min(range.lowerBound, rowCount))
        let upper = Swift.max(lower, Swift.min(range.upperBound, rowCount))
        return try? rows(in: lower..<upper)
    }

    /// Accesses a single cell by column name and row index (`df["price", 0]`).
    ///
    /// - Parameters:
    ///   - column: Column name.
    ///   - row: Zero-based row index.
    /// - Returns: The cell's ``DataValue``, or `nil` if `column` or `row` is invalid.
    public subscript(column: String, row: Int) -> DataValue? {
        guard (0..<rowCount).contains(row),
              let col = columns.first(where: { $0.name == column })
        else {
            return nil
        }
        return col.storage.value(at: row)
    }

    /// Exports the frame as a dictionary mapping each column name to its `[DataValue]` array.
    ///
    /// - Returns: Dictionary mapping column names to `[DataValue]` arrays.
    public func toDictionary() -> [String: [DataValue]] {
        Dictionary(uniqueKeysWithValues: columns.map { ($0.name, $0.values) })
    }

    /// Returns the first `count` rows.
    ///
    /// - Parameter count: Maximum number of leading rows to return. Defaults to `5`.
    /// - Returns: A ``DataFrame`` containing up to `count` leading rows.
    /// - Throws: ``DataFrameError`` if slicing fails.
    public func head(_ count: Int = 5) throws -> DataFrame {
        try rows(in: 0..<min(max(count, 0), rowCount))
    }

    /// Returns the last `count` rows.
    ///
    /// - Parameter count: Maximum number of trailing rows to return. Defaults to `5`.
    /// - Returns: A ``DataFrame`` containing up to `count` trailing rows.
    /// - Throws: ``DataFrameError`` if slicing fails.
    public func tail(_ count: Int = 5) throws -> DataFrame {
        let size = min(max(count, 0), rowCount)
        return try rows(in: (rowCount - size)..<rowCount)
    }

    /// Returns a new frame containing rows for which the predicate is true.
    ///
    /// - Parameter predicate: Closure evaluated for each row dictionary; returns `true` to keep the row.
    /// - Returns: A filtered ``DataFrame``.
    /// - Throws: Any error thrown by `predicate`.
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
    ///
    /// - Parameter predicate: The ``RowPredicate`` to evaluate.
    /// - Returns: A filtered ``DataFrame``.
    /// - Throws: ``DataFrameError`` if a referenced column does not exist.
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
    ///
    /// - Parameter expression: Query string to parse and evaluate.
    /// - Returns: A filtered ``DataFrame``.
    /// - Throws: ``DataFrameError`` if the expression syntax is invalid or references an unknown column.
    public func query(_ expression: String) throws -> DataFrame {
        let predicate = try Self.parseQueryExpression(expression, availableColumns: Set(columnNames))
        return try query(predicate)
    }

    /// Internal helper to add or replace a column directly from an `AnyColumn` storage.
    ///
    /// - Parameter storage: Typed column storage buffer.
    /// - Returns: A new ``DataFrame`` with the added or replaced column.
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
    ///
    /// - Parameters:
    ///   - name: Name of the column to insert or replace.
    ///   - values: Cell values aligned with the frame's rows.
    /// - Returns: A new ``DataFrame`` containing the updated column.
    /// - Throws: ``DataFrameError`` if `name` is empty, `values.count != rowCount`, or any numeric value is non-finite.
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
    ///
    /// - Parameters:
    ///   - name: Name of the column to insert or replace.
    ///   - transform: Closure receiving each ``DataRow`` and returning its computed ``DataValue``.
    /// - Returns: A new ``DataFrame`` containing the computed column.
    /// - Throws: Any error thrown by `transform` or ``withColumn(name:values:)``.
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
    ///
    /// - Parameter names: Dictionary mapping existing column names to replacement names.
    /// - Returns: A new ``DataFrame`` with renamed columns.
    /// - Throws: ``DataFrameError`` if a source name does not exist or a target name is empty or duplicated.
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
    ///
    /// - Parameters:
    ///   - name: Name of the column to convert.
    ///   - type: Target concrete ``DataType`` (`.number`, `.string`, `.boolean`, or `.date`).
    ///   - dateFormat: Optional `DateFormatter` pattern used when parsing `.date` values.
    /// - Returns: A new ``DataFrame`` with the converted column.
    /// - Throws: ``DataFrameError`` if `name` is not found or any cell cannot be converted.
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
    ///
    /// - Parameters:
    ///   - component: The ``DateComponent`` to extract.
    ///   - column: Source `.date` column name.
    ///   - outputColumn: Optional name for the resulting column; defaults to `"\(column)_\(component.rawValue)"`.
    /// - Returns: A new ``DataFrame`` with the extracted numeric component column.
    /// - Throws: ``DataFrameError`` if `column` does not exist or contains non-date values.
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
    ///
    /// - Parameters:
    ///   - column: Source string column name.
    ///   - outputColumn: Optional output column name; replaces `column` when `nil`.
    /// - Returns: A new ``DataFrame`` with lowercased strings.
    /// - Throws: ``DataFrameError`` if `column` is missing or contains non-string values.
    public func lowercased(_ column: String, as outputColumn: String? = nil) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn) { .string($0.lowercased()) }
    }

    /// Converts string values in `column` to uppercase.
    ///
    /// - Parameters:
    ///   - column: Source string column name.
    ///   - outputColumn: Optional output column name; replaces `column` when `nil`.
    /// - Returns: A new ``DataFrame`` with uppercased strings.
    /// - Throws: ``DataFrameError`` if `column` is missing or contains non-string values.
    public func uppercased(_ column: String, as outputColumn: String? = nil) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn) { .string($0.uppercased()) }
    }

    /// Trims leading and trailing whitespace and newlines from string values in `column`.
    ///
    /// - Parameters:
    ///   - column: Source string column name.
    ///   - outputColumn: Optional output column name; replaces `column` when `nil`.
    /// - Returns: A new ``DataFrame`` with trimmed strings.
    /// - Throws: ``DataFrameError`` if `column` is missing or contains non-string values.
    public func trimmed(_ column: String, as outputColumn: String? = nil) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn) {
            .string($0.trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }

    /// Computes character counts of string values in `column` as a numeric column.
    ///
    /// - Parameters:
    ///   - column: Source string column name.
    ///   - outputColumn: Optional output column name; defaults to `"\(column)_length"`.
    /// - Returns: A new ``DataFrame`` with the character-length column.
    /// - Throws: ``DataFrameError`` if `column` is missing or contains non-string values.
    public func stringLength(_ column: String, as outputColumn: String? = nil) throws -> DataFrame {
        try transformStrings(in: column, as: outputColumn ?? "\(column)_length") {
            .number(Double($0.count))
        }
    }

    /// Replaces occurrences of `target` with `replacement` in string values of `column`.
    ///
    /// - Parameters:
    ///   - column: Source string column name.
    ///   - target: Substring to find.
    ///   - replacement: Replacement string.
    ///   - outputColumn: Optional output column name; replaces `column` when `nil`.
    /// - Returns: A new ``DataFrame`` with updated string values.
    /// - Throws: ``DataFrameError`` if `column` is missing or contains non-string values.
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
    ///
    /// - Parameters:
    ///   - column: Source string column name.
    ///   - substring: Substring to search for.
    ///   - caseSensitive: Whether matching is case-sensitive. Defaults to `true`.
    ///   - outputColumn: Optional output column name; defaults to `"\(column)_contains"`.
    /// - Returns: A new ``DataFrame`` with the Boolean match column.
    /// - Throws: ``DataFrameError`` if `column` is missing or contains non-string values.
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

    /// Applies `transform` to every non-missing `.string` cell in `column` and writes the result to `outputColumn ?? column`.
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
    ///
    /// - Parameters:
    ///   - selectedColumns: Optional subset of column names used to identify duplicates; `nil` uses all columns.
    ///   - keep: Which duplicate occurrence to mark as `false` (non-duplicate). Defaults to `.first`.
    /// - Returns: A `[Bool]` array of length `rowCount` where `true` marks a duplicate row.
    /// - Throws: ``DataFrameError`` if `selectedColumns` is empty or contains an unknown column name.
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
    ///
    /// - Parameters:
    ///   - selectedColumns: Optional subset of column names used to identify duplicates; `nil` uses all columns.
    ///   - keep: Which occurrence of a duplicated row to retain. Defaults to `.first`.
    /// - Returns: A deduplicated ``DataFrame``.
    /// - Throws: ``DataFrameError`` if `selectedColumns` is empty or contains an unknown column name.
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
    ///
    /// - Parameter frames: Non-empty array of frames with identical schemas.
    /// - Returns: A vertically concatenated ``DataFrame``.
    /// - Throws: ``DataFrameError/emptyColumns`` if `frames` is empty, or ``DataFrameError/incompatibleSchema(expected:actual:)`` if schemas differ.
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

    /// Validates that `requestedNames` (when non-nil) is non-empty and every column exists in the frame.
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

    /// Converts a single ``DataValue`` to `type`, using `dateFormat` when parsing dates.
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

    /// Parses a string query expression into a composable ``RowPredicate``.
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

    /// Parses the right-hand literal token of a query clause into a ``DataValue``.
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

/// Returns the canonical lowercase string name for `type`.
///
/// - Parameter type: The ``DataType`` to describe.
/// - Returns: Lowercase string name of `type`.
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
