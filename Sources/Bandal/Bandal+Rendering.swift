import Foundation

/// Options controlling how ``DataFrame/render(options:)`` formats tabular output.
public struct RenderOptions: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible {
    /// Maximum number of rows to display before inserting an ellipsis (`…`).
    public var maxRows: Int
    /// Maximum number of columns to display before inserting an ellipsis (`…`).
    public var maxColumns: Int
    /// Optional maximum width per cell; longer cells are truncated with `…`.
    public var maxColumnWidth: Int?

    public init(
        maxRows: Int = 10,
        maxColumns: Int = 8,
        maxColumnWidth: Int? = nil
    ) {
        self.maxRows = maxRows
        self.maxColumns = maxColumns
        self.maxColumnWidth = maxColumnWidth
    }

    public var description: String {
        "RenderOptions(maxRows: \(maxRows), maxColumns: \(maxColumns), maxColumnWidth: \(maxColumnWidth.map(String.init) ?? "none"))"
    }

    public var debugDescription: String { description }
}

extension DataFrame {
    /// A compact, tabular preview of the frame, similar to a Pandas DataFrame repr.
    ///
    /// At most ten data rows and eight data columns are shown. Larger frames
    /// include ellipses to indicate omitted rows or columns.
    public var description: String {
        render()
    }

    public var debugDescription: String {
        description
    }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }

    /// Renders the frame with configurable `RenderOptions`.
    public func render(options: RenderOptions) -> String {
        let visibleRows = Self.visibleIndices(count: rowCount, limit: options.maxRows)
        let visibleColumns = Self.visibleIndices(count: columns.count, limit: options.maxColumns)
        let labels = visibleRows.map { $0.map { String($0) } ?? "…" }
        let headers = visibleColumns.map { $0.map { Self.truncate(columns[$0].name, limit: options.maxColumnWidth) } ?? "…" }
        let values = visibleRows.map { row in
            visibleColumns.map { columnIndex in
                guard let row, let columnIndex else { return "…" }
                let formatted = Self.displayValue(columns[columnIndex].values[row])
                return Self.truncate(formatted, limit: options.maxColumnWidth)
            }
        }
        let table = Self.renderTable(rowLabels: labels, columnNames: headers, values: values)
        return "\(table)\n\n[\(rowCount) rows x \(columnNames.count) columns]"
    }

    /// Renders the frame with configurable row and column limits.
    ///
    /// A limit of zero displays no values along that axis. Negative limits are
    /// treated as zero.
    public func render(maxRows: Int = 10, maxColumns: Int = 8) -> String {
        render(options: RenderOptions(maxRows: maxRows, maxColumns: maxColumns))
    }

    /// Renders the frame as a GitHub-flavored Markdown table.
    public func toMarkdown(maxRows: Int? = nil, maxColumns: Int? = nil) -> String {
        let rowLimit = maxRows ?? rowCount
        let colLimit = maxColumns ?? columns.count
        let visibleRows = Self.visibleIndices(count: rowCount, limit: rowLimit)
        let visibleColumns = Self.visibleIndices(count: columns.count, limit: colLimit)
        let headers = visibleColumns.map { $0.map { Self.escapeMarkdown(columns[$0].name) } ?? "…" }
        var lines: [String] = []
        lines.append("| " + headers.joined(separator: " | ") + " |")
        lines.append("| " + headers.map { _ in "---" }.joined(separator: " | ") + " |")
        for row in visibleRows {
            let cells = visibleColumns.map { columnIndex -> String in
                guard let row, let columnIndex else { return "…" }
                return Self.escapeMarkdown(Self.displayValue(columns[columnIndex].values[row]))
            }
            lines.append("| " + cells.joined(separator: " | ") + " |")
        }
        return lines.joined(separator: "\n")
    }

    /// Returns per-column non-missing counts, inferred types, and estimated memory usage.
    ///
    /// The estimates count numeric payloads as eight bytes and strings by their
    /// UTF-8 byte length; they exclude Swift collection and object overhead.
    public func info() -> String {
        let nameWidth = max("Column".count, columns.map(\.name.count).max() ?? 0)
        let countHeader = "Non-Null Count"
        let countWidth = max(countHeader.count, String(rowCount).count + 10)
        let typeWidth = max("Dtype".count, columns.map { Self.dataType(of: $0).count }.max() ?? 0)
        var lines = [
            "DataFrame",
            rowCount == 0
                ? "RangeIndex: 0 entries"
                : "RangeIndex: \(rowCount) entries, 0 to \(rowCount - 1)",
            "Data columns (total \(columns.count) columns):",
            " #  \(Self.padded("Column", to: nameWidth))  "
                + "\(Self.padded(countHeader, to: countWidth))  "
                + Self.padded("Dtype", to: typeWidth),
        ]
        var memoryUsage = 0
        var typeCounts: [String: Int] = [:]
        for (index, column) in columns.enumerated() {
            let nonMissingCount = column.values.filter { $0 != .missing }.count
            let type = Self.dataType(of: column)
            typeCounts[type, default: 0] += 1
            memoryUsage += column.values.reduce(into: 0) { total, value in
                switch value {
                case .number:
                    total += 8
                case .string(let string):
                    total += string.utf8.count
                case .boolean:
                    total += 1
                case .date:
                    total += 8
                case .missing:
                    break
                }
            }
            let indexText = String(index)
            lines.append(
                "\(Self.padded(indexText, to: 2, alignment: .right))  "
                    + "\(Self.padded(column.name, to: nameWidth))  "
                    + "\(Self.padded("\(nonMissingCount) non-null", to: countWidth))  "
                    + Self.padded(type, to: typeWidth)
            )
        }
        let typeSummary = typeCounts.keys.sorted().map { "\($0)(\(typeCounts[$0, default: 0]))" }
            .joined(separator: ", ")
        lines.append("---")
        lines.append("dtypes: \(typeSummary)")
        lines.append("memory usage: ~\(memoryUsage) bytes (estimated payload)")
        return lines.joined(separator: "\n")
    }

    /// Returns a Pandas-style descriptive-statistics table.
    ///
    /// Numeric columns include count, mean, sample standard deviation, minimum,
    /// quartiles (or custom `percentiles`), and maximum. Text columns include
    /// count, unique, most frequent value, and its frequency. Missing values are
    /// excluded from statistics.
    ///
    /// This method delegates to `describeStructured(percentiles:include:).describe()`
    /// so that callers can also inspect the result as a Swift object before rendering text.
    public func describe(
        percentiles: [Double] = [0.25, 0.5, 0.75],
        include: Set<DataType>? = nil
    ) -> String {
        describeStructured(percentiles: percentiles, include: include).describe()
    }

    private static func truncate(_ value: String, limit: Int?) -> String {
        guard let limit, limit > 0, value.count > limit else { return value }
        if limit == 1 { return "…" }
        return String(value.prefix(limit - 1)) + "…"
    }

    private static func escapeMarkdown(_ value: String) -> String {
        value
            .replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "\n", with: " ")
    }

    private static func visibleIndices(count: Int, limit: Int) -> [Int?] {
        let limit = max(limit, 0)
        guard count > limit else { return Array((0..<count).map(Optional.some)) }
        guard limit > 0 else { return [] }
        guard limit >= 3 else {
            return Array((0..<limit).map(Optional.some)) + [nil]
        }
        let leadingCount = (limit + 1) / 2
        let trailingCount = limit / 2
        return Array((0..<leadingCount).map(Optional.some))
            + [nil]
            + Array(((count - trailingCount)..<count).map(Optional.some))
    }

    private static func dataType(of column: DataColumn) -> String {
        let types = Set(column.values.compactMap { value -> DataType? in
            switch value {
            case .number: .number
            case .string: .string
            case .boolean: .boolean
            case .date: .date
            case .missing: nil
            }
        })
        guard !types.isEmpty else { return "unknown" }
        guard types.count == 1, let type = types.first else { return "mixed" }
        return switch type {
        case .number: "float64"
        case .string: "object"
        case .boolean: "bool"
        case .date: "datetime"
        case .mixed: "mixed"
        case .unknown: "unknown"
        }
    }

    static func displayValue(_ value: DataValue) -> String {
        switch value {
        case .number(let number):
            String(format: "%.6g", number)
        case .string(let string):
            string
        case .boolean(let value):
            String(value)
        case .date(let value):
            formatDateAsISO8601(value)
        case .missing:
            "NaN"
        }
    }

    static func renderTable(
        rowLabels: [String],
        columnNames: [String],
        values: [[String]]
    ) -> String {
        let rowLabelWidth = max(0, rowLabels.map(\.count).max() ?? 0)
        let columnWidths = columnNames.indices.map { columnIndex in
            max(
                columnNames[columnIndex].count,
                values.compactMap { $0.indices.contains(columnIndex) ? $0[columnIndex].count : nil }
                    .max() ?? 0
            )
        }
        let header = Self.padded("", to: rowLabelWidth)
            + (columnNames.isEmpty ? "" : "  ")
            + zip(columnNames, columnWidths)
                .map { Self.padded($0.0, to: $0.1) }
                .joined(separator: "  ")
        let body = rowLabels.indices.map { rowIndex in
            let label = Self.padded(rowLabels[rowIndex], to: rowLabelWidth)
            let cells = columnNames.indices.map { columnIndex in
                let value = values.indices.contains(rowIndex)
                    && values[rowIndex].indices.contains(columnIndex)
                    ? values[rowIndex][columnIndex]
                    : ""
                return Self.padded(value, to: columnWidths[columnIndex])
            }
            return label + (cells.isEmpty ? "" : "  ") + cells.joined(separator: "  ")
        }
        return ([header] + body).joined(separator: "\n")
    }

    private enum TextAlignment {
        case left
        case right
    }

    private static func padded(
        _ value: String,
        to width: Int,
        alignment: TextAlignment = .left
    ) -> String {
        let padding = String(repeating: " ", count: max(width - value.count, 0))
        return switch alignment {
        case .left: value + padding
        case .right: padding + value
        }
    }
}

// MARK: - DataFrameDescription text rendering

extension DataFrameDescription {
    /// Renders a pandas-style descriptive-statistics table as a string.
    ///
    /// The output format matches `DataFrame.describe()` so existing callers
    /// see identical text while gaining programmatic access to the result.
    public func describe() -> String {
        let statistics = ["count", "mean", "std", "min"]
            + percentileLabels
            + ["max", "unique", "top", "freq"]

        let rows = statistics.map { statistic in
            columnOrder.map { name in
                Self.describeValue(statistic, for: name, in: self)
            }
        }

        return DataFrame.renderTable(
            rowLabels: statistics,
            columnNames: columnOrder,
            values: rows
        )
    }

    private static func describeValue(
        _ statistic: String,
        for columnName: String,
        in description: DataFrameDescription
    ) -> String {
        if let stats = description.numeric[columnName] {
            switch statistic {
            case "count": return String(stats.count)
            case "mean": return stats.mean.map { String(format: "%.6g", $0) } ?? "NaN"
            case "std": return stats.standardDeviation.map { String(format: "%.6g", $0) } ?? "NaN"
            case "min": return stats.min.map { String(format: "%.6g", $0) } ?? "NaN"
            case "25%": return (stats.percentiles["25%"] ?? stats.q1).map { String(format: "%.6g", $0) } ?? "NaN"
            case "50%": return (stats.percentiles["50%"] ?? stats.median).map { String(format: "%.6g", $0) } ?? "NaN"
            case "75%": return (stats.percentiles["75%"] ?? stats.q3).map { String(format: "%.6g", $0) } ?? "NaN"
            case "max": return stats.max.map { String(format: "%.6g", $0) } ?? "NaN"
            default:
                if let p = stats.percentiles[statistic] {
                    return String(format: "%.6g", p)
                }
                return "NaN"
            }
        } else if let summary = description.nonNumeric[columnName] {
            switch statistic {
            case "count": return String(summary.count)
            case "unique": return String(summary.unique)
            case "top": return summary.top.map { DataFrame.displayValue($0) } ?? "NaN"
            case "freq": return String(summary.freq)
            default: return "NaN"
            }
        }
        return "NaN"
    }
}
