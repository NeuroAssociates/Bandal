import Foundation

/// Incremental builder that accumulates CSV fields into either a contiguous ``Float64Column`` or a boxed `[DataValue]` array.
private struct CSVColumnBuilder {
    /// Column name.
    let name: String
    /// Explicitly requested column type from ``CSVReadOptions/columnTypes``, or `nil` for automatic inference.
    let explicitType: DataType?
    /// Whether numeric-looking strings should be parsed as numbers.
    let infersNumbers: Bool
    /// Whether the builder is currently accumulating into the unboxed `numbers` fast path.
    private(set) var isNumeric: Bool
    /// Unboxed numeric storage used while all values are numbers or missing.
    private var numbers: ContiguousArray<Double> = []
    /// Indices of missing values within `numbers`.
    private var nullIndices: [Int] = []
    /// Fallback boxed storage used when non-numeric values are encountered.
    private var boxed: [DataValue] = []

    /// Creates a CSV column builder.
    ///
    /// - Parameters:
    ///   - name: Column name.
    ///   - explicitType: Optional explicit target ``DataType``.
    ///   - infersNumbers: Whether numbers are inferred automatically.
    ///   - estimatedRows: Estimated row count for capacity reservation.
    init(name: String, explicitType: DataType?, infersNumbers: Bool, estimatedRows: Int) {
        self.name = name
        self.explicitType = explicitType
        self.infersNumbers = infersNumbers
        if explicitType == .number || (explicitType == nil && infersNumbers) {
            self.isNumeric = true
            self.numbers.reserveCapacity(estimatedRows)
        } else {
            self.isNumeric = false
            self.boxed.reserveCapacity(estimatedRows)
        }
    }

    /// Appends a parsed `Double` value.
    ///
    /// - Parameter number: Finite floating-point value.
    mutating func appendNumber(_ number: Double) {
        if isNumeric {
            numbers.append(number)
        } else {
            boxed.append(.number(number))
        }
    }

    /// Appends a missing (`null`) value.
    mutating func appendMissing() {
        if isNumeric {
            nullIndices.append(numbers.count)
            numbers.append(0.0)
        } else {
            boxed.append(.missing)
        }
    }

    /// Appends an arbitrary ``DataValue``, demoting to boxed storage if necessary.
    ///
    /// - Parameter value: Parsed cell value.
    mutating func appendValue(_ value: DataValue) {
        if isNumeric {
            switch value {
            case .number(let number):
                numbers.append(number)
                return
            case .missing:
                nullIndices.append(numbers.count)
                numbers.append(0.0)
                return
            default:
                demoteToBoxed()
            }
        }
        boxed.append(value)
    }

    /// Converts previously accumulated numeric values into boxed `[DataValue]` storage when a non-numeric value arrives.
    private mutating func demoteToBoxed() {
        guard isNumeric else { return }
        isNumeric = false
        var converted = [DataValue]()
        converted.reserveCapacity(numbers.count + 16)
        let nullSet = Set(nullIndices)
        for i in 0..<numbers.count {
            if nullSet.contains(i) {
                converted.append(.missing)
            } else {
                converted.append(.number(numbers[i]))
            }
        }
        numbers.removeAll(keepingCapacity: false)
        nullIndices.removeAll(keepingCapacity: false)
        boxed = converted
    }

    /// Finalizes the builder and produces a typed `DataColumn`.
    ///
    /// - Returns: A `DataColumn` backed by ``Float64Column`` or the inferred column type.
    func finish() -> DataColumn {
        if isNumeric {
            let n = numbers.count
            if nullIndices.count == n && explicitType == nil {
                return DataColumn(
                    storage: AnyValueColumn(
                        name: name,
                        dataValues: ContiguousArray(repeating: .missing, count: n),
                        nullBitmap: BitSet(count: n, repeating: true)
                    )
                )
            }
            var bm = BitSet(count: n)
            for idx in nullIndices {
                bm[idx] = true
            }
            let floatCol = Float64Column(name: name, values: numbers, nullBitmap: bm)
            return DataColumn(storage: floatCol)
        }
        return DataColumn(name: name, values: boxed)
    }
}

extension DataFrame {
    /// Loads a CSV file, using the first record as its column names.
    ///
    /// - Parameters:
    ///   - url: File URL of the UTF-8 CSV file.
    ///   - delimiter: Single ASCII character used as the field separator. Defaults to `","`.
    ///   - maximumRows: Optional maximum number of data rows to read.
    /// - Throws: ``DataFrameError`` if the file cannot be read or the CSV structure is invalid.
    public init(
        contentsOfCSV url: URL,
        delimiter: Character = ",",
        maximumRows: Int? = nil
    ) throws {
        try self.init(
            contentsOfCSV: url,
            options: CSVReadOptions(delimiter: delimiter, maximumRows: maximumRows)
        )
    }

    /// Loads a CSV file using explicit parsing and type-inference options.
    ///
    /// Records are read incrementally, so `maximumRows` stops file processing
    /// after the requested number of data records.
    ///
    /// - Parameters:
    ///   - url: File URL of the UTF-8 CSV file.
    ///   - options: Parsing and schema options.
    /// - Throws: ``DataFrameError`` if the file cannot be read or parsing fails.
    public init(contentsOfCSV url: URL, options: CSVReadOptions) throws {
        try self.init(dataColumns: Self.readCSVColumns(url, options: options))
    }

    /// Writes the frame as UTF-8 CSV. Missing values are written as empty fields.
    ///
    /// - Parameters:
    ///   - url: Destination file URL.
    ///   - delimiter: Single ASCII character used as the field separator. Defaults to `","`.
    ///   - includeHeader: Whether to write the column names as the first row. Defaults to `true`.
    /// - Throws: ``DataFrameError/invalidCSV(_:)`` if the delimiter is invalid or writing fails.
    public func writeCSV(
        to url: URL,
        delimiter: Character = ",",
        includeHeader: Bool = true
    ) throws {
        let delimiterByte = try Self.csvDelimiterByte(delimiter)
        let file: FileHandle
        do {
            try Data().write(to: url, options: .atomic)
            file = try FileHandle(forWritingTo: url)
        } catch {
            throw DataFrameError.invalidCSV(error.localizedDescription)
        }

        func encodedRecord(_ fields: [String]) -> Data {
            var data = Data()
            for (index, field) in fields.enumerated() {
                if index > 0 {
                    data.append(delimiterByte)
                }
                let needsQuotes = field.utf8.contains { byte in
                    byte == delimiterByte || byte == 0x22 || byte == 0x0A || byte == 0x0D
                }
                if needsQuotes {
                    data.append(0x22)
                    for byte in field.utf8 {
                        data.append(byte)
                        if byte == 0x22 {
                            data.append(0x22)
                        }
                    }
                    data.append(0x22)
                } else {
                    data.append(contentsOf: field.utf8)
                }
            }
            data.append(0x0A)
            return data
        }

        do {
            if includeHeader {
                try file.write(contentsOf: encodedRecord(columnNames))
            }
            for row in 0..<rowCount {
                try file.write(contentsOf: encodedRecord(columns.map { column in
                    switch column.storage.value(at: row) {
                    case .number(let number): String(number)
                    case .string(let string): string
                    case .boolean(let value): String(value)
                    case .date(let value): formatDateAsISO8601(value)
                    case .missing: ""
                    }
                }))
            }
            try file.close()
        } catch {
            try? file.close()
            throw DataFrameError.invalidCSV(error.localizedDescription)
        }
    }

    // MARK: JSON I/O

    /// Creates a `DataFrame` from UTF-8 JSON data.
    ///
    /// Accepts either:
    /// - an array of JSON objects (`[{"a": 1, "b": "x"}, ...]`, records orientation), or
    /// - a JSON object mapping column names to arrays (`{"a": [1, 2], "b": ["x", "y"]}`).
    ///
    /// JSON `null` values become `.missing`. When `dateFormat` is supplied,
    /// string values matching the format are parsed as `.date`.
    ///
    /// - Parameters:
    ///   - data: UTF-8 encoded JSON payload.
    ///   - dateFormat: Optional `DateFormatter` format string for parsing `.date` values.
    /// - Throws: ``DataFrameError`` if the JSON is malformed, empty, or contains non-finite numbers.
    public init(jsonData data: Data, dateFormat: String? = nil) throws {
        let topLevel: Any
        do {
            topLevel = try JSONSerialization.jsonObject(with: data, options: [])
        } catch {
            throw DataFrameError.invalidCSV("invalid JSON: \(error.localizedDescription)")
        }

        if let records = topLevel as? [[String: Any]] {
            guard !records.isEmpty else {
                throw DataFrameError.emptyColumns
            }
            var orderedKeys: [String] = []
            var seen = Set<String>()
            for record in records {
                for key in record.keys.sorted() where seen.insert(key).inserted {
                    orderedKeys.append(key)
                }
            }
            guard !orderedKeys.isEmpty else {
                throw DataFrameError.emptyColumns
            }
            let parsedColumns: [(name: String, values: [DataValue])] = try orderedKeys.map { key in
                let colValues = try records.enumerated().map { row, record in
                    try Self.dataValueFromJSON(record[key], column: key, row: row, dateFormat: dateFormat)
                }
                return (name: key, values: colValues)
            }
            try self.init(columns: parsedColumns)
        } else if let columnDict = topLevel as? [String: [Any]] {
            guard !columnDict.isEmpty else {
                throw DataFrameError.emptyColumns
            }
            let sortedKeys = columnDict.keys.sorted()
            let parsedColumns: [(name: String, values: [DataValue])] = try sortedKeys.map { key in
                let array = columnDict[key] ?? []
                let colValues = try array.enumerated().map { row, item in
                    try Self.dataValueFromJSON(item, column: key, row: row, dateFormat: dateFormat)
                }
                return (name: key, values: colValues)
            }
            try self.init(columns: parsedColumns)
        } else {
            throw DataFrameError.invalidCSV("JSON root must be an array of objects or a dictionary of column arrays")
        }
    }

    /// Loads a `DataFrame` from a JSON file on disk.
    ///
    /// - Parameters:
    ///   - url: File URL of the JSON file.
    ///   - dateFormat: Optional `DateFormatter` format string for parsing `.date` values.
    /// - Throws: ``DataFrameError`` if reading or decoding fails.
    public init(contentsOfJSON url: URL, dateFormat: String? = nil) throws {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw DataFrameError.invalidCSV("could not read JSON file: \(error.localizedDescription)")
        }
        try self.init(jsonData: data, dateFormat: dateFormat)
    }

    /// Serializes the frame as records-oriented UTF-8 JSON (`[{"col": value, ...}, ...]`).
    ///
    /// `.missing` values are encoded as JSON `null`, and `.date` values are
    /// encoded as ISO-8601 strings.
    ///
    /// - Parameter prettyPrinted: Whether to format the JSON output with indentation and newlines.
    /// - Returns: UTF-8 encoded JSON `Data`.
    /// - Throws: ``DataFrameError/invalidCSV(_:)`` if serialization fails.
    public func jsonData(prettyPrinted: Bool = false) throws -> Data {
        let records: [[String: Any]] = (0..<rowCount).map { row in
            var object: [String: Any] = [:]
            object.reserveCapacity(columns.count)
            for column in columns {
                switch column.storage.value(at: row) {
                case .number(let number):
                    object[column.name] = number
                case .string(let string):
                    object[column.name] = string
                case .boolean(let flag):
                    object[column.name] = flag
                case .date(let date):
                    object[column.name] = formatDateAsISO8601(date)
                case .missing:
                    object[column.name] = NSNull()
                }
            }
            return object
        }
        var options: JSONSerialization.WritingOptions = [.sortedKeys]
        if prettyPrinted {
            options.insert(.prettyPrinted)
        }
        do {
            return try JSONSerialization.data(withJSONObject: records, options: options)
        } catch {
            throw DataFrameError.invalidCSV("could not encode JSON: \(error.localizedDescription)")
        }
    }

    /// Writes the frame as records-oriented UTF-8 JSON to `url`.
    ///
    /// - Parameters:
    ///   - url: Destination file URL.
    ///   - prettyPrinted: Whether to format the JSON output with indentation. Defaults to `true`.
    /// - Throws: ``DataFrameError/invalidCSV(_:)`` if encoding or writing fails.
    public func writeJSON(to url: URL, prettyPrinted: Bool = true) throws {
        let data = try jsonData(prettyPrinted: prettyPrinted)
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw DataFrameError.invalidCSV("could not write JSON file: \(error.localizedDescription)")
        }
    }

    /// Converts a decoded `JSONSerialization` value into a ``DataValue``.
    ///
    /// - Parameters:
    ///   - raw: Raw value from `JSONSerialization`.
    ///   - column: Column name for error reporting.
    ///   - row: Zero-based row index for error reporting.
    ///   - dateFormat: Optional date format string.
    /// - Returns: The converted ``DataValue``.
    private static func dataValueFromJSON(
        _ raw: Any?,
        column: String,
        row: Int,
        dateFormat: String?
    ) throws -> DataValue {
        guard let raw, !(raw is NSNull) else {
            return .missing
        }
        if let number = raw as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return .boolean(number.boolValue)
            }
            let doubleValue = number.doubleValue
            guard doubleValue.isFinite else {
                throw DataFrameError.nonFiniteValue(column: column, row: row)
            }
            return .number(doubleValue)
        }
        if let text = raw as? String {
            if let dateFormat, let date = parseDate(text, format: dateFormat) {
                return .date(date)
            }
            return .string(text)
        }
        return .string(String(describing: raw))
    }

    /// Incrementally streams and parses a CSV file into an array of `DataColumn` instances.
    ///
    /// - Parameters:
    ///   - url: Source CSV file URL.
    ///   - options: CSV reading configuration.
    /// - Returns: Parsed columns in schema order.
    private static func readCSVColumns(
        _ url: URL,
        options: CSVReadOptions
    ) throws -> [DataColumn] {
        if let maximumRows = options.maximumRows, maximumRows < 0 {
            throw DataFrameError.invalidCSV("maximumRows cannot be negative")
        }
        let delimiterByte = try csvDelimiterByte(options.delimiter)
        let file: FileHandle
        do {
            file = try FileHandle(forReadingFrom: url)
        } catch {
            throw DataFrameError.invalidCSV(error.localizedDescription)
        }
        defer { try? file.close() }
        let fileSize: UInt64 = (try? file.seekToEnd()) ?? 0
        try? file.seek(toOffset: 0)

        var names: [String]?
        var selectedNames: [String] = []
        var selectedIndices: [Int] = []
        var builders: [CSVColumnBuilder] = []
        var dataRowCount = 0
        var recordNumber = 0
        var recordBytes: [UInt8] = []
        var inQuotes = false
        var quoteMayClose = false
        var fieldIsEmpty = true
        var skipLineFeed = false
        var finished = false

        func processRecord() throws {
            if recordNumber == 0 && recordBytes.starts(with: [0xEF, 0xBB, 0xBF]) {
                recordBytes.removeFirst(3)
            }
            guard let decoded = String(bytes: recordBytes, encoding: .utf8) else {
                throw DataFrameError.invalidCSV("input is not valid UTF-8")
            }
            let hasQuotes = recordBytes.contains(0x22)
            recordBytes = []
            let record: [Substring]
            if !hasQuotes {
                let delimChar = Character(UnicodeScalar(delimiterByte))
                record = decoded.split(separator: delimChar, omittingEmptySubsequences: false)
            } else {
                let parsedRecords = try parseCSVRecord(decoded, delimiterByte: delimiterByte)
                guard let first = parsedRecords.first else { return }
                record = first.map { Substring($0) }
            }
            recordNumber += 1

            if names == nil {
                let fieldCount = record.count
                let resolvedNames: [String]
                if let configuredNames = options.columnNames {
                    guard configuredNames.count == fieldCount else {
                        throw DataFrameError.invalidCSV(
                            "received \(configuredNames.count) column names; expected \(fieldCount)"
                        )
                    }
                    resolvedNames = configuredNames
                } else if options.hasHeader {
                    resolvedNames = record.map(String.init)
                } else {
                    resolvedNames = (0..<fieldCount).map { "column_\($0)" }
                }
                guard !resolvedNames.isEmpty else { throw DataFrameError.emptyColumns }
                var uniqueNames = Set<String>()
                for name in resolvedNames {
                    guard !name.isEmpty else { throw DataFrameError.emptyColumnName }
                    guard uniqueNames.insert(name).inserted else {
                        throw DataFrameError.duplicateColumnName(name)
                    }
                }
                for (name, type) in options.columnTypes {
                    guard resolvedNames.contains(name) else {
                        throw DataFrameError.columnNotFound(name)
                    }
                    guard type == .number || type == .string || type == .boolean || type == .date
                    else {
                        throw DataFrameError.invalidCSV(
                            "\(dataTypeName(type)) is not a concrete CSV column type"
                        )
                    }
                }

                selectedNames = options.selectedColumns ?? resolvedNames
                guard !selectedNames.isEmpty else { throw DataFrameError.emptyColumns }
                for name in selectedNames {
                    guard let index = resolvedNames.firstIndex(of: name) else {
                        throw DataFrameError.columnNotFound(name)
                    }
                    selectedIndices.append(index)
                }
                names = resolvedNames
                let estimatedRows = options.maximumRows ?? (fileSize > 0 ? max(100, Int(fileSize / 40)) : 100)
                builders = selectedNames.map { colName in
                    CSVColumnBuilder(
                        name: colName,
                        explicitType: options.columnTypes[colName],
                        infersNumbers: options.infersNumbers,
                        estimatedRows: estimatedRows
                    )
                }

                if !options.hasHeader {
                    try appendDataRecord(record)
                }
                if options.hasHeader, options.maximumRows == 0 {
                    finished = true
                }
                return
            }

            try appendDataRecord(record)
        }

        func appendDataRecord(_ record: [Substring]) throws {
            guard let names else { return }
            guard record.count == names.count else {
                throw DataFrameError.invalidCSV(
                    "record \(recordNumber) has \(record.count) fields; expected \(names.count)"
                )
            }
            let canAppendRow = options.maximumRows.map { dataRowCount < $0 } ?? true
            if canAppendRow {
                for (outputIndex, sourceIndex) in selectedIndices.enumerated() {
                    let rawSub = record[sourceIndex]
                    if builders[outputIndex].isNumeric, builders[outputIndex].explicitType == nil {
                        var sub = rawSub
                        while let first = sub.first, first.isWhitespace { sub.removeFirst() }
                        while let last = sub.last, last.isWhitespace { sub.removeLast() }
                        if sub.isEmpty, options.missingTokens.contains("") {
                            builders[outputIndex].appendMissing()
                            continue
                        }
                        if let number = Double(sub), number.isFinite {
                            builders[outputIndex].appendNumber(number)
                            continue
                        }
                    }
                    let inferred = try Self.inferValue(
                        rawSub,
                        missingTokens: options.missingTokens,
                        infersNumbers: options.infersNumbers,
                        columnType: builders[outputIndex].explicitType,
                        dateFormat: options.dateFormat
                    )
                    builders[outputIndex].appendValue(inferred)
                }
                dataRowCount += 1
            }
            if let maximumRows = options.maximumRows, dataRowCount >= maximumRows {
                finished = true
            }
        }

        do {
            while !finished, let chunk = try file.read(upToCount: 64 * 1024), !chunk.isEmpty {
                for byte in chunk {
                    if skipLineFeed {
                        skipLineFeed = false
                        if byte == 0x0A { continue }
                    }

                    if inQuotes {
                        if quoteMayClose {
                            quoteMayClose = false
                            if byte == 0x22 {
                                recordBytes.append(byte)
                                fieldIsEmpty = false
                                continue
                            }
                            inQuotes = false
                        } else {
                            recordBytes.append(byte)
                            if byte == 0x22 {
                                quoteMayClose = true
                            } else {
                                fieldIsEmpty = false
                            }
                            continue
                        }
                    }

                    switch byte {
                    case delimiterByte:
                        recordBytes.append(byte)
                        fieldIsEmpty = true
                    case 0x0A, 0x0D:
                        try processRecord()
                        fieldIsEmpty = true
                        skipLineFeed = byte == 0x0D
                    case 0x22 where fieldIsEmpty:
                        recordBytes.append(byte)
                        inQuotes = true
                        fieldIsEmpty = false
                    default:
                        recordBytes.append(byte)
                        fieldIsEmpty = false
                    }
                    if finished { break }
                }
            }

            if !finished {
                if quoteMayClose {
                    inQuotes = false
                    quoteMayClose = false
                }
                guard !inQuotes else {
                    throw DataFrameError.invalidCSV("unterminated quoted field")
                }
                if !recordBytes.isEmpty {
                    try processRecord()
                }
            }
        } catch let error as DataFrameError {
            throw error
        } catch {
            throw DataFrameError.invalidCSV(error.localizedDescription)
        }

        guard names != nil else { throw DataFrameError.emptyColumns }
        return builders.map { $0.finish() }
    }

    /// Infers or parses a single CSV field into a ``DataValue`` according to the column's schema options.
    ///
    /// - Parameters:
    ///   - rawValue: Raw field substring.
    ///   - missingTokens: Tokens treated as `.missing`.
    ///   - infersNumbers: Whether numeric-looking strings should be parsed as numbers.
    ///   - columnType: Optional explicit ``DataType``.
    ///   - dateFormat: Optional `DateFormatter` format pattern for `.date` columns.
    /// - Returns: The parsed ``DataValue``.
    private static func inferValue(
        _ rawValue: Substring,
        missingTokens: Set<String>,
        infersNumbers: Bool,
        columnType: DataType?,
        dateFormat: String?
    ) throws -> DataValue {
        var sub = rawValue
        while let first = sub.first, first.isWhitespace {
            sub.removeFirst()
        }
        while let last = sub.last, last.isWhitespace {
            sub.removeLast()
        }
        if sub.isEmpty {
            if missingTokens.contains("") { return .missing }
            if let columnType, columnType == .string { return .string("") }
        }
        if let columnType {
            switch columnType {
            case .number:
                if let number = Double(sub), number.isFinite {
                    return .number(number)
                }
                let strVal = String(sub)
                guard !missingTokens.contains(strVal) else { return .missing }
                let normalized = strVal
                    .replacingOccurrences(of: ",", with: "")
                    .replacingOccurrences(of: "$", with: "")
                guard let number = Double(normalized), number.isFinite else {
                    throw DataFrameError.invalidCSV("cannot parse '\(strVal)' as number")
                }
                return .number(number)
            case .string:
                let strVal = String(sub)
                guard !missingTokens.contains(strVal) else { return .missing }
                return .string(strVal)
            case .boolean:
                let strVal = String(sub)
                guard !missingTokens.contains(strVal) else { return .missing }
                switch strVal.lowercased() {
                case "true": return .boolean(true)
                case "false": return .boolean(false)
                default: throw DataFrameError.invalidCSV("cannot parse '\(strVal)' as Boolean")
                }
            case .date:
                let strVal = String(sub)
                guard !missingTokens.contains(strVal) else { return .missing }
                let date = dateFormat.flatMap { parseDate(strVal, format: $0) }
                    ?? (dateFormat == nil ? ISO8601DateFormatter().date(from: strVal) : nil)
                guard let date else {
                    throw DataFrameError.invalidCSV("cannot parse '\(strVal)' as date")
                }
                return .date(date)
            case .mixed, .unknown:
                throw DataFrameError.invalidCSV(
                    "\(dataTypeName(columnType)) is not a concrete CSV column type"
                )
            }
        }
        if infersNumbers, let number = Double(sub), number.isFinite {
            return .number(number)
        }
        let strVal = String(sub)
        guard !missingTokens.contains(strVal) else { return .missing }
        guard infersNumbers else { return .string(strVal) }
        let normalized = strVal
            .replacingOccurrences(of: ",", with: "")
            .replacingOccurrences(of: "$", with: "")
        if let number = Double(normalized), number.isFinite {
            return .number(number)
        }
        return .string(strVal)
    }

    /// Validates that `delimiter` is a single ASCII byte and returns its `UInt8` code.
    ///
    /// - Parameter delimiter: Delimiter character.
    /// - Returns: Single ASCII byte representing `delimiter`.
    private static func csvDelimiterByte(_ delimiter: Character) throws -> UInt8 {
        let bytes = Array(String(delimiter).utf8)
        guard bytes.count == 1, let byte = bytes.first else {
            throw DataFrameError.invalidCSV("delimiter must be a single ASCII character")
        }
        return byte
    }

    /// Parses a quoted RFC 4180 CSV record line into fields.
    ///
    /// - Parameters:
    ///   - contents: Raw CSV record text.
    ///   - delimiterByte: ASCII delimiter byte.
    /// - Returns: Array of parsed records, each containing field strings.
    private static func parseCSVRecord(
        _ contents: String,
        delimiterByte: UInt8
    ) throws -> [[String]] {
        var bytes = Array(contents.utf8)
        if bytes.starts(with: [0xEF, 0xBB, 0xBF]) {
            bytes.removeFirst(3)
        }
        var records: [[String]] = []
        var record: [String] = []
        var field: [UInt8] = []
        var inQuotes = false
        var index = 0

        while index < bytes.count {
            let byte = bytes[index]
            if inQuotes {
                if byte == 0x22 {
                    if index + 1 < bytes.count, bytes[index + 1] == 0x22 {
                        field.append(0x22)
                        index += 1
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(byte)
                }
            } else {
                switch byte {
                case 0x22 where field.isEmpty:
                    inQuotes = true
                case delimiterByte:
                    record.append(String(decoding: field, as: UTF8.self))
                    field = []
                case 0x22:
                    throw DataFrameError.invalidCSV("unexpected quote in an unquoted field")
                default:
                    field.append(byte)
                }
            }
            index += 1
        }

        guard !inQuotes else {
            throw DataFrameError.invalidCSV("unterminated quoted field")
        }
        if !field.isEmpty || !record.isEmpty {
            record.append(String(decoding: field, as: UTF8.self))
            records.append(record)
        }
        return records
    }
}
