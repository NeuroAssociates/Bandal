import Foundation

/// Formats a `Date` as an ISO 8601 UTC timestamp string.
///
/// - Parameter date: The date to format.
/// - Returns: An ISO 8601 string representation of `date`.
func formatDateAsISO8601(_ date: Date) -> String {
    ISO8601DateFormatter().string(from: date)
}

/// Parses a date string strictly using the given `DateFormatter` format pattern in UTC.
///
/// - Parameters:
///   - value: The raw date string to parse.
///   - format: A `DateFormatter` pattern such as `"yyyy-MM-dd"`.
/// - Returns: The parsed `Date` if `value` round-trips through `format`; otherwise `nil`.
func parseDate(_ value: String, format: String) -> Date? {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.calendar = Calendar(identifier: .gregorian)
    formatter.dateFormat = format
    formatter.isLenient = false
    guard let date = formatter.date(from: value), formatter.string(from: date) == value else {
        return nil
    }
    return date
}

/// Values supported by a ``DataFrame``.
public enum DataValue: Hashable, Sendable, Codable,
    CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable,
    ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByStringLiteral, ExpressibleByBooleanLiteral, ExpressibleByNilLiteral
{
    /// A finite 64-bit floating-point number.
    case number(Double)
    /// A UTF-8 text string.
    case string(String)
    /// A Boolean truth value (`true` or `false`).
    case boolean(Bool)
    /// A calendar timestamp (`Date`).
    case date(Date)
    /// A missing (`null` / `nil`) cell value.
    case missing

    /// Creates a `.number` value from a floating-point literal.
    ///
    /// - Parameter value: The floating-point literal value.
    public init(floatLiteral value: Double) {
        self = .number(value)
    }

    /// Creates a `.number` value from an integer literal, converted to `Double`.
    ///
    /// - Parameter value: The integer literal value.
    public init(integerLiteral value: Int) {
        self = .number(Double(value))
    }

    /// Creates a `.string` value from a string literal.
    ///
    /// - Parameter value: The string literal value.
    public init(stringLiteral value: String) {
        self = .string(value)
    }

    /// Creates a `.boolean` value from a Boolean literal.
    ///
    /// - Parameter value: The Boolean literal value.
    public init(booleanLiteral value: Bool) {
        self = .boolean(value)
    }

    /// Creates a `.missing` value from a `nil` literal.
    ///
    /// - Parameter nilLiteral: The `nil` literal tuple.
    public init(nilLiteral: ()) {
        self = .missing
    }

    /// Returns the underlying `Double` when the cell is `.number`, or `nil` otherwise.
    public var numberValue: Double? {
        if case .number(let value) = self { return value }
        return nil
    }

    /// Returns the underlying `String` when the cell is `.string`, or `nil` otherwise.
    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    /// Returns the underlying `Bool` when the cell is `.boolean`, or `nil` otherwise.
    public var boolValue: Bool? {
        if case .boolean(let value) = self { return value }
        return nil
    }

    /// Returns the underlying `Date` when the cell is `.date`, or `nil` otherwise.
    public var dateValue: Date? {
        if case .date(let value) = self { return value }
        return nil
    }

    /// `true` when the cell is `.missing`.
    public var isMissing: Bool {
        self == .missing
    }

    /// A text representation of the cell; dates use ISO 8601 and missing values render as `"nil"`.
    public var description: String {
        switch self {
        case .number(let value):
            String(value)
        case .string(let value):
            value
        case .boolean(let value):
            String(value)
        case .date(let value):
            formatDateAsISO8601(value)
        case .missing:
            "nil"
        }
    }

    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    ///
    /// - Parameter target: The stream to write to.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// The inferred or explicitly requested type of a data-frame column.
public enum DataType: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible, CustomDebugStringConvertible {
    /// Numeric values stored as 64-bit floating-point or integer numbers.
    case number
    /// Text values stored as UTF-8 strings or dictionary-encoded categories.
    case string
    /// Boolean values (`true` or `false`).
    case boolean
    /// Calendar timestamps (`Date`).
    case date
    /// Multiple distinct value types within the same column.
    case mixed
    /// No non-missing values are present to infer a concrete type.
    case unknown

    /// The raw string value of the type, such as `"number"`.
    public var description: String { rawValue }
    /// A debugging representation identical to ``description``.
    public var debugDescription: String { rawValue }
}

/// Errors raised while creating or querying a ``DataFrame``.
public enum DataFrameError: Error, Equatable, Sendable, LocalizedError {
    /// A ``DataFrame`` was initialized or selected with zero columns.
    case emptyColumns
    /// A column name is the empty string (`""`).
    case emptyColumnName
    /// Two or more columns share the same name.
    case duplicateColumnName(String)
    /// A column's element count does not match the expected row count of the frame.
    case unequalColumnLengths(column: String, expected: Int, actual: Int)
    /// A referenced column name does not exist in the frame.
    case columnNotFound(String)
    /// A numeric cell contains `NaN` or infinity (`±∞`).
    case nonFiniteValue(column: String, row: Int)
    /// A numeric operation was requested on a column that contains non-numeric values.
    case nonNumericColumn(String)
    /// CSV or JSON input could not be parsed due to structural or formatting errors.
    case invalidCSV(String)
    /// An operation would create a column whose name already exists in the frame.
    case columnNameConflict(String)
    /// Two frames have incompatible column schemas for concatenation or comparison.
    case incompatibleSchema(expected: [String], actual: [String])
    /// A cell value could not be converted to the requested ``DataType``.
    case invalidConversion(column: String, row: Int, type: DataType)
    /// A rolling, lag, bin, or quantile parameter is outside the valid positive range.
    case invalidWindowSize(Int)

    /// A human-readable message describing the error.
    public var errorDescription: String? {
        switch self {
        case .emptyColumns:
            "A data frame must contain at least one column."
        case .emptyColumnName:
            "Column names cannot be empty."
        case .duplicateColumnName(let name):
            "Column name '\(name)' is duplicated."
        case .unequalColumnLengths(let column, let expected, let actual):
            "Column '\(column)' has \(actual) values; expected \(expected)."
        case .columnNotFound(let name):
            "Column '\(name)' does not exist."
        case .nonFiniteValue(let column, let row):
            "Column '\(column)' contains a non-finite value at row \(row)."
        case .nonNumericColumn(let name):
            "Column '\(name)' contains non-numeric or missing values."
        case .invalidCSV(let reason):
            "Could not read CSV: \(reason)"
        case .columnNameConflict(let name):
            "The joined data frames both contain column '\(name)'."
        case .incompatibleSchema(let expected, let actual):
            "Data frame schema \(actual) does not match expected schema \(expected)."
        case .invalidConversion(let column, let row, let type):
            "Cannot convert value in column '\(column)' at row \(row) to \(type)."
        case .invalidWindowSize(let size):
            "Window size must be positive; received \(size)."
        }
    }
}

/// Summary statistics for one numeric column.
///
/// `count` is the number of numeric, non-missing values. Empty numeric columns
/// have a count of zero and `nil` for the remaining statistics. Standard
/// deviation is the sample standard deviation and is `nil` for fewer than two
/// numeric values.
public struct ColumnSummary: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// The number of numeric, non-missing values.
    public let count: Int
    /// The arithmetic mean, or `nil` when there are no numeric values.
    public let mean: Double?
    /// The sample standard deviation, or `nil` for fewer than two numeric values.
    public let standardDeviation: Double?
    /// The median, or `nil` when there are no numeric values.
    public let median: Double?
    /// The smallest value, or `nil` when there are no numeric values.
    public let min: Double?
    /// The largest value, or `nil` when there are no numeric values.
    public let max: Double?

    /// Creates a column summary with precomputed statistics.
    ///
    /// - Parameters:
    ///   - count: Number of non-missing numeric values.
    ///   - mean: Arithmetic mean, or `nil` if empty.
    ///   - standardDeviation: Sample standard deviation, or `nil` if fewer than 2 values.
    ///   - median: Median value, or `nil` if empty.
    ///   - min: Minimum value, or `nil` if empty.
    ///   - max: Maximum value, or `nil` if empty.
    init(
        count: Int,
        mean: Double?,
        standardDeviation: Double?,
        median: Double?,
        min: Double?,
        max: Double?
    ) {
        self.count = count
        self.mean = mean
        self.standardDeviation = standardDeviation
        self.median = median
        self.min = min
        self.max = max
    }
}

/// The reduction to perform on a grouped numeric column.
///
/// `count` counts non-missing values of any type. Every other function
/// requires numeric values and ignores missing ones. `variance` and
/// `standardDeviation` use the sample formula (`n - 1`) and are missing for
/// fewer than two numeric values. `first` and `last` return the first or last
/// non-missing value in row order and work for any value type.
public enum AggregateFunction: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    /// Counts non-missing values of any type.
    case count
    /// Computes the arithmetic sum of non-missing numeric values.
    case sum
    /// Computes the arithmetic mean of non-missing numeric values.
    case mean
    /// Finds the smallest non-missing numeric value.
    case min
    /// Finds the largest non-missing numeric value.
    case max
    /// Computes the 50th percentile (median) of non-missing numeric values.
    case median
    /// Computes the unbiased sample variance (`ddof = 1`) of non-missing numeric values.
    case variance
    /// Computes the unbiased sample standard deviation (`ddof = 1`) of non-missing numeric values.
    case standardDeviation
    /// Returns the first non-missing value in row order.
    case first
    /// Returns the last non-missing value in row order.
    case last

    /// The raw string value of the function, such as `"sum"`.
    public var description: String { rawValue }
}

/// A column and reduction to compute as part of a grouped aggregation.
public struct Aggregation: Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible {
    /// The name of the column to reduce.
    public let column: String
    /// The reduction to apply to the column.
    public let function: AggregateFunction
    /// The name of the result column, or `nil` to use a default name.
    public let alias: String?

    /// Creates an aggregation.
    ///
    /// - Parameters:
    ///   - column: The name of the column to reduce.
    ///   - function: The reduction to apply.
    ///   - alias: An optional name for the result column.
    public init(_ column: String, using function: AggregateFunction, as alias: String? = nil) {
        self.column = column
        self.function = function
        self.alias = alias
    }
}

/// The join strategy used to combine rows with matching key values.
public enum JoinKind: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    /// Keeps all rows from the left frame and fills unmatched right columns with `.missing`.
    case left
    /// Keeps only rows whose join keys exist in both frames.
    case inner
    /// Keeps all rows from the right frame and fills unmatched left columns with `.missing`.
    case right
    /// Keeps all rows from both frames, filling unmatched columns on either side with `.missing`.
    case full

    /// The raw string value of the join kind, such as `"inner"`.
    public var description: String { rawValue }
}

/// Which row to retain when removing duplicate rows.
public enum DuplicateKeep: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    /// Retains the first occurrence of each duplicate key tuple.
    case first
    /// Retains the last occurrence of each duplicate key tuple.
    case last
    /// Drops all rows that share a duplicate key tuple.
    case none

    /// The raw string value of the option, such as `"first"`.
    public var description: String { rawValue }
}

/// Configuration for loading a CSV file into a `DataFrame`.
public struct CSVReadOptions: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    /// The character that separates fields.
    public var delimiter: Character
    /// The maximum number of data rows to read, or `nil` to read all rows.
    public var maximumRows: Int?
    /// Whether the first record contains column names.
    public var hasHeader: Bool
    /// Explicit column names that override or replace the header, or `nil`.
    public var columnNames: [String]?
    /// The columns to keep, or `nil` to keep all columns.
    public var selectedColumns: [String]?
    /// Field values that are read as missing.
    public var missingTokens: Set<String>
    /// Whether fields that look numeric are parsed as numbers.
    public var infersNumbers: Bool
    /// Explicit types keyed by column name, overriding inference.
    public var columnTypes: [String: DataType]
    /// The date format used to parse `.date` columns, or `nil`.
    public var dateFormat: String?

    /// Creates CSV read options.
    ///
    /// - Parameters:
    ///   - delimiter: The field separator.
    ///   - maximumRows: The maximum number of data rows to read, or `nil` for all.
    ///   - hasHeader: Whether the first record contains column names.
    ///   - columnNames: Explicit column names, or `nil`.
    ///   - selectedColumns: The columns to keep, or `nil` for all.
    ///   - missingTokens: Field values treated as missing.
    ///   - infersNumbers: Whether numeric-looking fields are parsed as numbers.
    ///   - columnTypes: Explicit types keyed by column name.
    ///   - dateFormat: The date format for `.date` columns, or `nil`.
    public init(
        delimiter: Character = ",",
        maximumRows: Int? = nil,
        hasHeader: Bool = true,
        columnNames: [String]? = nil,
        selectedColumns: [String]? = nil,
        missingTokens: Set<String> = [""],
        infersNumbers: Bool = true,
        columnTypes: [String: DataType] = [:],
        dateFormat: String? = nil
    ) {
        self.delimiter = delimiter
        self.maximumRows = maximumRows
        self.hasHeader = hasHeader
        self.columnNames = columnNames
        self.selectedColumns = selectedColumns
        self.missingTokens = missingTokens
        self.infersNumbers = infersNumbers
        self.columnTypes = columnTypes
        self.dateFormat = dateFormat
    }
}

/// The direction used to order values in a data frame.
public enum SortOrder: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    /// Orders values from smallest to largest (missing values last).
    case ascending
    /// Orders values from largest to smallest (missing values last).
    case descending

    /// The raw string value of the order, such as `"ascending"`.
    public var description: String { rawValue }
}

/// A column and direction used when sorting a data frame.
public struct SortDescriptor: Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible {
    /// The name of the column to sort by.
    public let column: String
    /// The direction of the sort.
    public let order: SortOrder

    /// Creates a sort descriptor.
    ///
    /// - Parameters:
    ///   - column: The name of the column to sort by.
    ///   - order: The sort direction. The default is `.ascending`.
    public init(_ column: String, order: SortOrder = .ascending) {
        self.column = column
        self.order = order
    }
}

/// One row of a ``DataFrame``, preserving column order and providing subscript
/// access by column name or position.
public struct DataRow: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Zero-based row index in the source `DataFrame`.
    public let index: Int
    /// Column names in the frame's deterministic order.
    public let columnOrder: [String]
    /// Mapping of column name to cell value.
    public let values: [String: DataValue]

    /// Creates a row.
    ///
    /// - Parameters:
    ///   - index: The zero-based row index in the source frame.
    ///   - columnOrder: Column names in display order.
    ///   - values: Cell values keyed by column name.
    public init(index: Int, columnOrder: [String], values: [String: DataValue]) {
        self.index = index
        self.columnOrder = columnOrder
        self.values = values
    }

    /// Accesses a value by column name.
    ///
    /// - Parameter column: The column name to look up.
    /// - Returns: The cell's ``DataValue``, or `nil` if `column` is not in the row.
    public subscript(_ column: String) -> DataValue? {
        values[column]
    }

    /// Accesses a value by zero-based column position.
    ///
    /// - Parameter position: A zero-based index into ``columnOrder``.
    /// - Returns: The cell's ``DataValue``, or `.missing` if not present.
    public subscript(position: Int) -> DataValue {
        values[columnOrder[position]] ?? .missing
    }

    /// A pandas-style text rendering with one line per column and a row summary footer.
    public var description: String {
        let width = columnOrder.map(\.count).max() ?? 0
        var lines = columnOrder.map { name in
            let padded = name.padding(toLength: width, withPad: " ", startingAt: 0)
            return "\(padded)    \(values[name] ?? .missing)"
        }
        lines.append("Row: \(index), Columns: \(columnOrder.count)")
        return lines.joined(separator: "\n")
    }

    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    ///
    /// - Parameter target: The stream to write to.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// Internal wrapper pairing a typed ``AnyColumn`` storage buffer with an optional pre-extracted ``NumericBuffer``.
struct DataColumn: Equatable, Hashable, Sendable, Codable {
    /// The column's name.
    let name: String
    /// The underlying columnar storage buffer.
    let storage: any AnyColumn
    /// Cached contiguous numeric buffer when the column holds numbers; `nil` for non-numeric columns.
    let numericBuffer: NumericBuffer?
    /// Optional cached `[DataValue]` array when constructed directly from boxed values.
    private let _cachedValues: [DataValue]?

    /// All cell values materialized as `[DataValue]`.
    var values: [DataValue] {
        if let cached = _cachedValues { return cached }
        return storage.toDataValues()
    }

    /// Creates a column wrapper from a typed ``AnyColumn`` storage buffer.
    ///
    /// - Parameters:
    ///   - storage: The underlying column buffer.
    ///   - numericBuffer: Optional prebuilt numeric buffer.
    ///   - cachedValues: Optional pre-materialized `[DataValue]` array.
    init(storage: any AnyColumn, numericBuffer: NumericBuffer? = nil, cachedValues: [DataValue]? = nil) {
        self.name = storage.name
        self.storage = storage
        if let numericBuffer {
            self.numericBuffer = numericBuffer
        } else if let floatCol = storage as? Float64Column {
            self.numericBuffer = NumericBuffer(floatColumn: floatCol)
        } else if let intCol = storage as? Int64Column {
            self.numericBuffer = NumericBuffer(intColumn: intCol)
        } else if let anyCol = storage as? AnyValueColumn, anyCol.validCount == 0 {
            self.numericBuffer = NumericBuffer(
                floatColumn: Float64Column(
                    name: anyCol.name,
                    values: ContiguousArray(repeating: 0.0, count: anyCol.count),
                    nullBitmap: BitSet(count: anyCol.count, repeating: true)
                )
            )
        } else {
            self.numericBuffer = nil
        }
        self._cachedValues = cachedValues
    }

    /// Creates a column by inferring the most specific columnar storage for `values`.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - values: Boxed ``DataValue`` elements.
    init(name: String, values: [DataValue]) {
        let col = AnyColumnFactory.make(name: name, values: values)
        self.init(storage: col, cachedValues: values)
    }

    /// Creates a contiguous `Double` column with no missing values.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - numbers: Non-missing `Double` values.
    init(name: String, numbers: [Double]) {
        let floatCol = Float64Column(name: name, numbers: numbers)
        self.init(
            storage: floatCol,
            numericBuffer: NumericBuffer(floatColumn: floatCol),
            cachedValues: nil
        )
    }

    /// Returns a copy of this column with `newName`.
    ///
    /// - Parameter newName: Replacement column name.
    /// - Returns: A renamed `DataColumn` sharing the underlying buffers.
    func renamed(to newName: String) -> DataColumn {
        guard newName != name else { return self }
        let newStorage = storage.renamed(to: newName)
        let newBuffer: NumericBuffer?
        if let buf = numericBuffer {
            let renamedFloatCol = Float64Column(
                name: newName,
                values: buf.floatColumn.values,
                nullBitmap: buf.floatColumn.nullBitmap
            )
            newBuffer = NumericBuffer(floatColumn: renamedFloatCol)
        } else {
            newBuffer = nil
        }
        return DataColumn(storage: newStorage, numericBuffer: newBuffer, cachedValues: _cachedValues)
    }

    /// Coding keys used for `Codable` serialization of a column.
    enum CodingKeys: String, CodingKey {
        case name
        case values
    }

    /// Decodes a column from its name and `[DataValue]` array.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedName = try container.decode(String.self, forKey: .name)
        let decodedValues = try container.decode([DataValue].self, forKey: .values)
        self.init(name: decodedName, values: decodedValues)
    }

    /// Encodes the column's name and `[DataValue]` array.
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(values, forKey: .values)
    }

    /// Returns `true` when two columns have the same name and cell values.
    static func == (lhs: DataColumn, rhs: DataColumn) -> Bool {
        lhs.name == rhs.name && lhs.values == rhs.values
    }

    /// Hashes the column name and cell values into `hasher`.
    func hash(into hasher: inout Hasher) {
        hasher.combine(name)
        hasher.combine(values)
    }
}

/// A column-oriented table with numeric, text, Boolean, date, and missing values.
///
/// Numeric reductions run through MLX (`float64`, CPU stream) for large
/// inputs when MLX is available, and native Swift `Double` otherwise. CSV fields that
/// contain a number (including quoted thousands separators and a leading dollar
/// sign) are inferred as numeric; other non-empty fields remain strings.
public struct DataFrame: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Internal ordered list of columns.
    let columns: [DataColumn]

    /// The number of rows in the data frame.
    public let rowCount: Int

    /// Column names in deterministic order.
    public let columnNames: [String]

    /// A two-element `[rows, columns]` shape, matching the common Pandas form.
    public var shape: [Int] { [rowCount, columnNames.count] }

    /// Returns the inferred type of a column, ignoring missing values.
    ///
    /// Empty or entirely missing columns have type `.unknown`; columns with
    /// more than one concrete value type have type `.mixed`.
    ///
    /// - Parameter name: The name of the column to inspect.
    /// - Returns: The inferred ``DataType`` of the column.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if `name` does not exist.
    public func dataType(of name: String) throws -> DataType {
        guard let col = columns.first(where: { $0.name == name }) else {
            throw DataFrameError.columnNotFound(name)
        }
        let type = col.storage.dataType
        if type != .unknown && type != .mixed {
            return type
        }
        let values = try values(in: name)
        var types = Set<DataType>()
        for value in values where value != .missing {
            switch value {
            case .number: types.insert(.number)
            case .string: types.insert(.string)
            case .boolean: types.insert(.boolean)
            case .date: types.insert(.date)
            case .missing: break
            }
        }
        guard !types.isEmpty else { return .unknown }
        guard types.count == 1, let type = types.first else { return .mixed }
        return type
    }

    /// Selects all columns whose inferred type matches `type`.
    ///
    /// - Parameter type: The ``DataType`` to filter columns by.
    /// - Returns: A new ``DataFrame`` containing only columns of `type`.
    /// - Throws: ``DataFrameError/emptyColumns`` if no columns match `type`.
    public func select(columnsOfType type: DataType) throws -> DataFrame {
        let selected = try columns.filter { try dataType(of: $0.name) == type }
        guard !selected.isEmpty else { throw DataFrameError.emptyColumns }
        return try DataFrame(dataColumns: selected)
    }

    /// Creates a data frame from named numeric columns.
    ///
    /// Dictionary input is sorted by column name because dictionary iteration order
    /// is not part of this API's column-order contract.
    ///
    /// - Parameter columns: Dictionary mapping column names to `[Double]` arrays.
    /// - Throws: ``DataFrameError`` if `columns` is empty, a column name is empty, lengths differ, or a value is non-finite.
    public init(columns: [String: [Double]]) throws {
        try self.init(
            columns: columns
                .map { (name: $0.key, values: $0.value) }
                .sorted { $0.name < $1.name }
        )
    }

    /// Creates a data frame from ordered named numeric columns.
    ///
    /// - Parameter columns: Ordered array of `(name, values)` numeric column tuples.
    /// - Throws: ``DataFrameError`` if `columns` is empty, a name is empty or duplicated, lengths differ, or a value is non-finite.
    public init(columns: [(name: String, values: [Double])]) throws {
        guard !columns.isEmpty else {
            throw DataFrameError.emptyColumns
        }
        var names = Set<String>()
        for column in columns {
            guard !column.name.isEmpty else {
                throw DataFrameError.emptyColumnName
            }
            guard names.insert(column.name).inserted else {
                throw DataFrameError.duplicateColumnName(column.name)
            }
        }
        let expectedCount = columns[0].values.count
        for column in columns {
            guard column.values.count == expectedCount else {
                throw DataFrameError.unequalColumnLengths(
                    column: column.name,
                    expected: expectedCount,
                    actual: column.values.count
                )
            }
            for (row, number) in column.values.enumerated() where !number.isFinite {
                throw DataFrameError.nonFiniteValue(column: column.name, row: row)
            }
        }
        self.columns = columns.map { DataColumn(name: $0.name, numbers: $0.values) }
        self.rowCount = expectedCount
        self.columnNames = columns.map(\.name)
    }

    /// Creates a data frame from ordered mixed-type columns.
    ///
    /// - Parameter columns: Ordered array of `(name, values)` mixed-type column tuples.
    /// - Throws: ``DataFrameError`` if `columns` is empty, a name is empty or duplicated, lengths differ, or a numeric value is non-finite.
    @_disfavoredOverload
    public init(columns: [(name: String, values: [DataValue])]) throws {
        guard !columns.isEmpty else {
            throw DataFrameError.emptyColumns
        }

        var names = Set<String>()
        for column in columns {
            guard !column.name.isEmpty else {
                throw DataFrameError.emptyColumnName
            }
            guard names.insert(column.name).inserted else {
                throw DataFrameError.duplicateColumnName(column.name)
            }
        }

        let expectedCount = columns[0].values.count
        for column in columns {
            guard column.values.count == expectedCount else {
                throw DataFrameError.unequalColumnLengths(
                    column: column.name,
                    expected: expectedCount,
                    actual: column.values.count
                )
            }
            for (row, value) in column.values.enumerated() {
                if case .number(let number) = value, !number.isFinite {
                    throw DataFrameError.nonFiniteValue(column: column.name, row: row)
                }
            }
        }

        self.columns = columns.map { DataColumn(name: $0.name, values: $0.values) }
        self.rowCount = expectedCount
        self.columnNames = columns.map(\.name)
    }

    /// Creates a data frame directly from validated internal `DataColumn` instances.
    ///
    /// - Parameter dataColumns: Non-empty array of equal-length columns.
    /// - Throws: ``DataFrameError`` if `dataColumns` is empty, a name is empty or duplicated, or lengths differ.
    init(dataColumns: [DataColumn]) throws {
        guard !dataColumns.isEmpty else {
            throw DataFrameError.emptyColumns
        }
        var names = Set<String>()
        for column in dataColumns {
            guard !column.name.isEmpty else {
                throw DataFrameError.emptyColumnName
            }
            guard names.insert(column.name).inserted else {
                throw DataFrameError.duplicateColumnName(column.name)
            }
        }
        let expectedCount = dataColumns[0].storage.count
        for column in dataColumns {
            guard column.storage.count == expectedCount else {
                throw DataFrameError.unequalColumnLengths(
                    column: column.name,
                    expected: expectedCount,
                    actual: column.storage.count
                )
            }
        }
        self.columns = dataColumns
        self.rowCount = expectedCount
        self.columnNames = dataColumns.map(\.name)
    }

    /// Creates a data frame from an array of typed ``AnyColumn`` storage buffers.
    ///
    /// - Parameter storageColumns: Typed column storage buffers.
    /// - Throws: ``DataFrameError`` if validation fails.
    init(storageColumns: [any AnyColumn]) throws {
        try self.init(dataColumns: storageColumns.map { DataColumn(storage: $0) })
    }

    /// Returns a new frame containing the rows within `range`.
    ///
    /// - Parameter range: Contiguous row index range within `0..<rowCount`.
    /// - Returns: A sliced ``DataFrame``.
    func rows(in range: Range<Int>) throws -> DataFrame {
        try DataFrame(
            storageColumns: columns.map { $0.storage.slice(range: range) }
        )
    }

    /// Returns a new frame containing the rows at `indices` in the specified order.
    ///
    /// - Parameter indices: Row indices within `0..<rowCount`.
    /// - Returns: A gathered ``DataFrame``.
    func rows(at indices: [Int]) throws -> DataFrame {
        if columns.count > 1 && indices.count >= 65_536 {
            let colCount = columns.count
            let cols = columns
            let boxed = DisjointSliceBuffer(capacity: colCount)
            defer { boxed.release(count: colCount) }
            DispatchQueue.concurrentPerform(iterations: colCount) { idx in
                boxed.initialize(at: idx, to: cols[idx].storage.slice(at: indices))
            }
            let sliced = (0..<colCount).map { boxed.pointer[$0] }
            return try DataFrame(storageColumns: sliced)
        }
        return try DataFrame(
            storageColumns: columns.map { $0.storage.slice(at: indices) }
        )
    }

    /// Returns a new frame containing the rows whose bit in `mask` is `true`.
    ///
    /// - Parameter mask: A ``BitSet`` of length `rowCount`.
    /// - Returns: A filtered ``DataFrame``.
    func rows(matching mask: BitSet) throws -> DataFrame {
        if columns.count > 1 && rowCount >= 65_536 {
            let colCount = columns.count
            let cols = columns
            let boxed = DisjointSliceBuffer(capacity: colCount)
            defer { boxed.release(count: colCount) }
            DispatchQueue.concurrentPerform(iterations: colCount) { idx in
                boxed.initialize(at: idx, to: cols[idx].storage.slice(mask: mask))
            }
            let sliced = (0..<colCount).map { boxed.pointer[$0] }
            return try DataFrame(storageColumns: sliced)
        }
        return try DataFrame(
            storageColumns: columns.map { $0.storage.slice(mask: mask) }
        )
    }
}

extension ColumnSummary {
    /// A one-line summary of the statistics; unavailable values render as `NaN`.
    public var description: String {
        func f(_ v: Double?) -> String { v.map { String(format: "%.6g", $0) } ?? "NaN" }
        return "count \(count), mean \(f(mean)), std \(f(standardDeviation)), "
            + "median \(f(median)), min \(f(min)), max \(f(max))"
    }

    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    ///
    /// - Parameter target: The stream to write to.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

extension Aggregation {
    /// A text form such as `sum(price) as total`, omitting the alias when absent.
    public var description: String {
        if let alias {
            return "\(function.name)(\(column)) as \(alias)"
        }
        return "\(function.name)(\(column))"
    }
    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }
}

extension SortDescriptor {
    /// A text form such as `price descending`.
    public var description: String { "\(column) \(order == .ascending ? "ascending" : "descending")" }
    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }
}

extension CSVReadOptions {
    /// A one-line summary of the main options.
    public var description: String {
        "CSVReadOptions(delimiter: \"\(delimiter)\", hasHeader: \(hasHeader), "
            + "maximumRows: \(maximumRows.map(String.init) ?? "all"), "
            + "infersNumbers: \(infersNumbers), missingTokens: \(missingTokens.sorted()))"
    }

    /// A debugging representation identical to ``description``.
    public var debugDescription: String { description }
}

/// Output slots written by `DispatchQueue.concurrentPerform`; each index is written by exactly one iteration.
private struct DisjointSliceBuffer: @unchecked Sendable {
    /// Raw buffer pointer holding sliced columns.
    let pointer: UnsafeMutablePointer<any AnyColumn>

    /// Allocates uninitialized storage for `capacity` columns.
    ///
    /// - Parameter capacity: Number of column slots to allocate.
    init(capacity: Int) {
        pointer = .allocate(capacity: capacity)
    }

    /// Initializes the slot at `index` with `value`.
    ///
    /// - Parameters:
    ///   - index: Target slot index.
    ///   - value: Sliced column to store.
    func initialize(at index: Int, to value: any AnyColumn) {
        (pointer + index).initialize(to: value)
    }

    /// Deinitializes `count` slots and deallocates the underlying pointer.
    ///
    /// - Parameter count: Number of initialized slots.
    func release(count: Int) {
        pointer.deinitialize(count: count)
        pointer.deallocate()
    }
}
