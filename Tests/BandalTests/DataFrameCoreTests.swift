import Foundation
import Testing
@testable import Bandal

@Test func createsAndSelectsColumns() throws {
    let data = try DataFrame(columns: [
        "temperature": [18.0, 21.5, 19.0],
        "humidity": [0.45, 0.50, 0.48],
    ])

    #expect(data.rowCount == 3)
    #expect(data.columnNames == ["humidity", "temperature"])
    #expect(try data.column("temperature") == [18.0, 21.5, 19.0])

    let temperatures = try data.select("temperature")
    #expect(temperatures.rowCount == 3)
    #expect(temperatures.columnNames == ["temperature"])
    #expect(try temperatures.column("temperature") == [18.0, 21.5, 19.0])
    #expect(data.columnNames.count == 2)
}

@Test func detectsAndDropsDuplicateRowsWithSelectableKeepPolicy() throws {
    let data = try DataFrame(columns: [
        (name: "group", values: [
            .string("A"), .string("A"), .string("A"), .string("B"), .string("B"),
        ]),
        (name: "value", values: [
            .number(1), .number(1), .number(2), .missing, .missing,
        ]),
    ])

    #expect(try data.duplicatedRows() == [false, true, false, false, true])
    #expect(try data.duplicatedRows(in: ["group"]) == [false, true, true, false, true])
    #expect(try data.duplicatedRows(in: ["group"], keep: .last)
        == [true, true, false, true, false])
    #expect(try data.duplicatedRows(in: ["group"], keep: .none)
        == [true, true, true, true, true])

    let firstKept = try data.dropDuplicates(in: ["group"])
    #expect(firstKept.rowCount == 2)
    #expect(try firstKept.values(in: "value") == [.number(1), .missing])

    let lastKept = try data.dropDuplicates(in: ["group"], keep: .last)
    #expect(try lastKept.values(in: "value") == [.number(2), .missing])

    let noDuplicatesKept = try data.dropDuplicates(in: ["group"], keep: .none)
    #expect(noDuplicatesKept.rowCount == 0)
    #expect(data.rowCount == 5)
}

@Test func renamesColumnsWithoutChangingOrderOrValues() throws {
    let data = try DataFrame(columns: [
        (name: "oldA", values: [.number(1), .number(2)]),
        (name: "oldB", values: [.string("x"), .missing]),
    ])

    let renamed = try data.renamedColumns(["oldA": "newA", "oldB": "newB"])
    #expect(renamed.columnNames == ["newA", "newB"])
    #expect(try renamed.values(in: "newA") == [.number(1), .number(2)])
    #expect(try renamed.values(in: "newB") == [.string("x"), .missing])
    #expect(data.columnNames == ["oldA", "oldB"])

    expectDataFrameError(.columnNotFound("unknown")) {
        _ = try data.renamedColumns(["unknown": "new"])
    }
    expectDataFrameError(.duplicateColumnName("oldB")) {
        _ = try data.renamedColumns(["oldA": "oldB"])
    }
    expectDataFrameError(.emptyColumnName) {
        _ = try data.renamedColumns(["oldA": ""])
    }
}

@Test func concatenatesRowsWithMatchingOrderedSchemas() throws {
    let first = try DataFrame(columns: [
        (name: "id", values: [.number(1), .number(2)]),
        (name: "label", values: [.string("a"), .missing]),
    ])
    let second = try DataFrame(columns: [
        (name: "id", values: [.number(3)]),
        (name: "label", values: [.string("c")]),
    ])

    let combined = try DataFrame.concatenatingRows([first, second])
    #expect(combined.shape == [3, 2])
    #expect(combined.columnNames == ["id", "label"])
    #expect(try combined.values(in: "id") == [.number(1), .number(2), .number(3)])
    #expect(try combined.values(in: "label") == [
        .string("a"), .missing, .string("c"),
    ])

    expectDataFrameError(
        .incompatibleSchema(expected: ["id", "label"], actual: ["label", "id"])
    ) {
        _ = try DataFrame.concatenatingRows([
            first,
            try DataFrame(columns: [
                (name: "label", values: [.string("x")]),
                (name: "id", values: [.number(4)]),
            ]),
        ])
    }
    expectDataFrameError(.emptyColumns) {
        _ = try DataFrame.concatenatingRows([])
    }
}

@Test func supportsBooleanAndDateTypesAndTypedColumnSelection() throws {
    let firstDate = try #require(
        ISO8601DateFormatter().date(from: "2024-01-02T03:04:05Z")
    )
    let secondDate = try #require(
        ISO8601DateFormatter().date(from: "2024-01-03T03:04:05Z")
    )
    let data = try DataFrame(columns: [
        (name: "active", values: [.boolean(true), .boolean(false), .missing]),
        (name: "created", values: [.date(secondDate), .date(firstDate), .missing]),
        (name: "mixed", values: [.number(1), .string("one"), .missing]),
        (name: "empty", values: [.missing, .missing, .missing]),
    ])

    #expect(try data.dataType(of: "active") == .boolean)
    #expect(try data.dataType(of: "created") == .date)
    #expect(try data.dataType(of: "mixed") == .mixed)
    #expect(try data.dataType(of: "empty") == .unknown)

    let typedSelection = try data.select(columnsOfType: .date)
    #expect(typedSelection.columnNames == ["created"])

    let sorted = try data.sorted(by: [SortDescriptor("created")])
    #expect(try sorted.values(in: "created") == [
        .date(firstDate), .date(secondDate), .missing,
    ])
    #expect(data.info().contains("bool(1)"))
    #expect(data.info().contains("datetime(1)"))
    #expect(data.info().contains("mixed(1)"))
    #expect(data.summary().isEmpty)
}

@Test func castsColumnsExplicitlyAndReportsInvalidRows() throws {
    let data = try DataFrame(columns: [
        (name: "amount", values: [.string("$1,200"), .missing, .string("3.5")]),
        (name: "enabled", values: [.string("TRUE"), .string("false"), .missing]),
        (name: "date", values: [.string("2024/02/01"), .string("2024/02/02"), .missing]),
    ])

    let casted = try data
        .castColumn("amount", to: .number)
        .castColumn("enabled", to: .boolean)
        .castColumn("date", to: .date, dateFormat: "yyyy/MM/dd")
    #expect(try casted.values(in: "amount") == [.number(1200), .missing, .number(3.5)])
    #expect(try casted.values(in: "enabled") == [
        .boolean(true), .boolean(false), .missing,
    ])
    #expect(try casted.dataType(of: "date") == .date)
    #expect(try data.dataType(of: "date") == .string)

    let invalid = try DataFrame(columns: [
        (name: "flag", values: [.string("true"), .string("not-a-boolean")]),
    ])
    expectDataFrameError(.invalidConversion(column: "flag", row: 1, type: .boolean)) {
        _ = try invalid.castColumn("flag", to: .boolean)
    }
}

@Test func sortsByMultipleColumnsAndKeepsMissingValuesLast() throws {
    let data = try DataFrame(columns: [
        (name: "score", values: [.number(85), .number(92), .number(92), .missing]),
        (name: "attendance", values: [.number(95), .number(100), .number(90), .number(80)]),
        (name: "name", values: [.string("Min"), .string("Ji"), .string("Ha"), .string("Do")]),
    ])

    let sorted = try data.sorted(by: [
        SortDescriptor("score", order: .descending),
        SortDescriptor("attendance"),
    ])

    #expect(try sorted.values(in: "name") == [
        .string("Ha"), .string("Ji"), .string("Min"), .string("Do"),
    ])
}

@Test func preservesOrderedInputColumns() throws {
    let data = try DataFrame(columns: [
        (name: "z", values: [1.0]),
        (name: "a", values: [2.0]),
    ])

    #expect(data.columnNames == ["z", "a"])
}

@Test func reportsInvalidColumns() {
    expectDataFrameError(.emptyColumns) {
        _ = try DataFrame(columns: [] as [(name: String, values: [Double])])
    }
    expectDataFrameError(.emptyColumnName) {
        _ = try DataFrame(columns: [(name: "", values: [1.0])])
    }
    expectDataFrameError(.duplicateColumnName("value")) {
        _ = try DataFrame(columns: [
            (name: "value", values: [1.0]),
            (name: "value", values: [2.0]),
        ])
    }
    expectDataFrameError(
        .unequalColumnLengths(column: "second", expected: 1, actual: 2)
    ) {
        _ = try DataFrame(columns: [
            (name: "first", values: [1.0]),
            (name: "second", values: [2.0, 3.0]),
        ])
    }
}

@Test func reportsUnknownColumns() throws {
    let data = try DataFrame(columns: ["known": [1.0]])

    expectDataFrameError(.columnNotFound("missing")) {
        _ = try data.column("missing")
    }
    expectDataFrameError(.columnNotFound("missing")) {
        _ = try data.select("missing")
    }
    expectDataFrameError(.emptyColumns) {
        _ = try data.select()
    }
}

@Test func rejectsNonFiniteValues() {
    expectDataFrameError(.nonFiniteValue(column: "value", row: 1)) {
        _ = try DataFrame(columns: ["value": [1.0, .nan]])
    }
}

@Test func supportsZeroRowColumns() throws {
    let data = try DataFrame(columns: ["empty": []])

    #expect(data.rowCount == 0)
    #expect(try data.column("empty").isEmpty)

    let summary = try #require(data.summary()["empty"])
    #expect(summary.count == 0)
    #expect(summary.mean == nil)
    #expect(summary.min == nil)
    #expect(summary.max == nil)
}
