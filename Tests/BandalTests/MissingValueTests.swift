import Foundation
import Testing
@testable import Bandal

@Test func inspectsFillsAndDropsMissingValuesWithoutMutatingSource() throws {
    let data = try DataFrame(columns: [
        (name: "value", values: [.number(10), .missing, .number(30)]),
        (name: "label", values: [.string("a"), .string("b"), .missing]),
    ])

    #expect(try data.isMissing(in: "value") == [false, true, false])
    #expect(data.missingCounts() == ["value": 1, "label": 1])

    let filled = try data.fillMissing(in: ["value"], with: .number(0))
    #expect(try filled.values(in: "value") == [.number(10), .number(0), .number(30)])
    #expect(try filled.values(in: "label") == [.string("a"), .string("b"), .missing])
    #expect(try data.values(in: "value") == [.number(10), .missing, .number(30)])

    let completeValues = try data.dropMissing(in: ["value"])
    #expect(completeValues.rowCount == 2)
    #expect(try completeValues.values(in: "label") == [.string("a"), .missing])

    let completeRows = try data.dropMissing()
    #expect(completeRows.rowCount == 1)
    #expect(try completeRows.values(in: "value") == [.number(10)])
}

@Test func reportsMissingValueColumnErrors() throws {
    let data = try DataFrame(columns: ["value": [1.0]])

    expectDataFrameError(.columnNotFound("unknown")) {
        _ = try data.isMissing(in: "unknown")
    }
    expectDataFrameError(.columnNotFound("unknown")) {
        _ = try data.fillMissing(in: ["unknown"], with: .number(0))
    }
    expectDataFrameError(.emptyColumns) {
        _ = try data.dropMissing(in: [])
    }
    let dataWithMissing = try DataFrame(columns: [
        (name: "value", values: [.number(1), .missing]),
    ])
    expectDataFrameError(.nonFiniteValue(column: "value", row: 1)) {
        _ = try dataWithMissing.fillMissing(in: ["value"], with: .number(.infinity))
    }
}
