import Foundation
import Testing
@testable import Bandal

@Test func meltsAndPivotsFramesWithDeterministicOrdering() throws {
    let data = try DataFrame(columns: [
        (name: "region", values: [.string("East"), .string("West"), .missing]),
        (name: "sales", values: [.number(10), .number(30), .number(99)]),
        (name: "units", values: [.number(1), .missing, .number(4)]),
    ])

    let melted = try data.melt(idColumns: ["region"])
    #expect(melted.columnNames == ["region", "variable", "value"])
    #expect(try melted.values(in: "region") == [
        .string("East"), .string("West"), .missing,
        .string("East"), .string("West"), .missing,
    ])
    #expect(try melted.values(in: "variable") == [
        .string("sales"), .string("sales"), .string("sales"),
        .string("units"), .string("units"), .string("units"),
    ])
    #expect(try melted.values(in: "value") == [
        .number(10), .number(30), .number(99),
        .number(1), .missing, .number(4),
    ])

    let pivoted = try melted.pivotTable(
        index: ["region"],
        columns: "variable",
        values: "value",
        using: .sum
    )
    #expect(pivoted.columnNames == [
        "region", "value_string_sales_sum", "value_string_units_sum",
    ])
    #expect(try pivoted.values(in: "region") == [.string("East"), .string("West")])
    #expect(try pivoted.values(in: "value_string_sales_sum") == [
        .number(10), .number(30),
    ])
    #expect(try pivoted.values(in: "value_string_units_sum") == [
        .number(1), .missing,
    ])

    let typedPivot = try DataFrame(columns: [
        (name: "index", values: [.string("row"), .string("row")]),
        (name: "category", values: [.number(1), .string("1.0")]),
        (name: "amount", values: [.number(2), .number(3)]),
    ]).pivotTable(index: ["index"], columns: "category", values: "amount", using: .sum)
    #expect(typedPivot.columnNames == [
        "index", "amount_number_1.0_sum", "amount_string_1.0_sum",
    ])

    expectDataFrameError(.duplicateColumnName("region")) {
        _ = try data.melt(idColumns: ["region"], variableName: "region")
    }
    expectDataFrameError(.columnNotFound("unknown")) {
        _ = try data.melt(idColumns: ["unknown"])
    }
}
