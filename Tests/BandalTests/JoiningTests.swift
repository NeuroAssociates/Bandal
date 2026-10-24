import Foundation
import Testing
@testable import Bandal

@Test func supportsInnerRightAndFullJoinsWithDuplicateAndMissingKeys() throws {
    let left = try DataFrame(columns: [
        (name: "Key", values: [
            .string("A"), .string("A"), .string("B"), .missing,
        ]),
        (name: "Left", values: [
            .number(1), .number(2), .number(3), .number(4),
        ]),
    ])
    let right = try DataFrame(columns: [
        (name: "Key", values: [
            .string("A"), .string("A"), .string("C"), .missing,
        ]),
        (name: "Right", values: [
            .string("first"), .string("second"), .string("third"), .string("missing"),
        ]),
    ])

    let inner = try left.join(right, on: "Key", kind: .inner)
    #expect(inner.rowCount == 4)
    #expect(try inner.values(in: "Left") == [
        .number(1), .number(1), .number(2), .number(2),
    ])
    #expect(try inner.values(in: "Right") == [
        .string("first"), .string("second"), .string("first"), .string("second"),
    ])

    let rightJoined = try left.join(right, on: "Key", kind: .right)
    #expect(rightJoined.rowCount == 6)
    #expect(try rightJoined.values(in: "Key") == [
        .string("A"), .string("A"), .string("A"), .string("A"),
        .string("C"), .missing,
    ])
    #expect(try rightJoined.values(in: "Left").suffix(2) == [.missing, .missing])

    let full = try left.join(right, on: "Key", kind: .full)
    #expect(full.rowCount == 8)
    #expect(try full.values(in: "Key") == [
        .string("A"), .string("A"), .string("A"), .string("A"),
        .string("B"), .missing, .string("C"), .missing,
    ])
    #expect(try full.values(in: "Right").suffix(4) == [
        .missing, .missing, .string("third"), .string("missing"),
    ])
}

@Test func handlesEmptyJoinsAndRejectsConflictingColumns() throws {
    let emptyLeft = try DataFrame(columns: [
        (name: "Key", values: [DataValue]()),
        (name: "Left", values: [DataValue]()),
    ])
    let emptyRight = try DataFrame(columns: [
        (name: "Key", values: [DataValue]()),
        (name: "Right", values: [DataValue]()),
    ])

    for kind in [JoinKind.left, .inner, .right, .full] {
        let joined = try emptyLeft.join(emptyRight, on: "Key", kind: kind)
        #expect(joined.shape == [0, 3])
    }

    let conflictingRight = try DataFrame(columns: [
        (name: "Key", values: [.string("A")]),
        (name: "Left", values: [.string("conflict")]),
    ])
    expectDataFrameError(.columnNameConflict("Left")) {
        _ = try emptyLeft.join(conflictingRight, on: "Key", kind: .full)
    }
}
