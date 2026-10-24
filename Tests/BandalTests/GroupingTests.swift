import Foundation
import Testing
@testable import Bandal

@Test func filtersAddsColumnsGroupsAndJoins() throws {
    let sales = try DataFrame(columns: [
        (name: "Country", values: [.string("A"), .string("B"), .string("C")]),
        (name: "Sales", values: [.number(10), .number(30), .number(20)]),
    ])
    let filtered = try sales.filterRows { row in
        guard case .number(let amount) = row["Sales"] else { return false }
        return amount >= 20
    }
    #expect(filtered.rowCount == 2)

    let withGrowth = try sales.withColumn(
        name: "Growth",
        values: [.number(1), .number(3), .number(2)]
    )
    let grouped = try withGrowth
        .withColumn(
            name: "Region",
            values: [.string("West"), .string("East"), .string("West")]
        )
        .groupBy("Region")
        .aggregate("Sales", using: .sum)
    #expect(try grouped.values(in: "Region") == [.string("East"), .string("West")])
    #expect(try grouped.column("Sales_sum") == [30, 30])

    let lookup = try DataFrame(columns: [
        (name: "Country", values: [.string("A"), .string("B")]),
        (name: "Region", values: [.string("West"), .string("East")]),
    ])
    let joined = try sales.leftJoin(lookup, on: "Country")
    #expect(joined.rowCount == 3)
    #expect(try joined.values(in: "Region") == [.string("West"), .string("East"), .missing])
}

@Test func filterRowsPreservesPreviouslyCapturedRowValues() throws {
    let data = try DataFrame(columns: [
        (name: "id", values: [.number(1), .number(2)]),
        (name: "label", values: [.string("first"), .string("second")]),
    ])
    var capturedRows: [[String: DataValue]] = []

    _ = try data.filterRows { row in
        capturedRows.append(row)
        return true
    }

    #expect(capturedRows == [
        ["id": .number(1), "label": .string("first")],
        ["id": .number(2), "label": .string("second")],
    ])
}

@Test func groupsByMultipleColumnsAndAggregatesSeveralOperations() throws {
    let data = try DataFrame(columns: [
        (name: "Region", values: [
            .string("East"), .string("East"), .string("West"),
            .string("East"), .missing,
        ]),
        (name: "Product", values: [
            .string("A"), .string("A"), .string("A"),
            .string("B"), .string("A"),
        ]),
        (name: "Sales", values: [
            .number(10), .missing, .number(30), .missing, .number(999),
        ]),
    ])

    let grouped = try data
        .groupBy("Region", "Product")
        .aggregate([
            Aggregation("Sales", using: .sum),
            Aggregation("Sales", using: .count),
            Aggregation("Sales", using: .mean),
            Aggregation("Sales", using: .min),
            Aggregation("Sales", using: .max),
        ])

    #expect(try grouped.values(in: "Region") == [
        .string("East"), .string("East"), .string("West"),
    ])
    #expect(try grouped.values(in: "Product") == [
        .string("A"), .string("B"), .string("A"),
    ])
    #expect(grouped.columnNames == [
        "Region", "Product", "Sales_sum", "Sales_count",
        "Sales_mean", "Sales_min", "Sales_max",
    ])
    let sumValues = try grouped.values(in: "Sales_sum")
    let countValues = try grouped.values(in: "Sales_count")
    #expect(sumValues == [
        .number(10), .missing, .number(30),
    ])
    #expect(countValues == [
        .number(1), .number(0), .number(1),
    ])
    #expect(try grouped.values(in: "Sales_mean") == [
        .number(10), .missing, .number(30),
    ])
    #expect(try grouped.values(in: "Sales_min") == [
        .number(10), .missing, .number(30),
    ])
    #expect(try grouped.values(in: "Sales_max") == [
        .number(10), .missing, .number(30),
    ])
    let largeGroups = try grouped.filterRows { row in
        guard case .number(let sum) = row["Sales_sum"] else { return false }
        return sum > 10
    }
    #expect(try largeGroups.values(in: "Region") == [.string("West")])

    let groupedFromArray = try data
        .groupBy(["Region", "Product"])
        .aggregate("Sales", using: .count)
    #expect(try groupedFromArray.values(in: "Sales_count") == countValues)
}

@Test func transformsGroupAggregationsBackToSourceRows() throws {
    let data = try DataFrame(columns: [
        (name: "group", values: [
            .string("A"), .string("A"), .string("B"), .missing,
        ]),
        (name: "value", values: [
            .number(1), .missing, .missing, .number(9),
        ]),
    ])

    let transformed = try data.groupBy("group").transform("value", using: .sum)
    #expect(try transformed.values(in: "value_sum") == [
        .number(1), .number(1), .missing, .missing,
    ])
    #expect(data.columnNames == ["group", "value"])

    let counted = try data.groupBy("group").transform("value", using: .count)
    #expect(try counted.values(in: "value_count") == [
        .number(1), .number(1), .number(0), .missing,
    ])

    let conflict = try DataFrame(columns: [
        (name: "group", values: [.string("A")]),
        (name: "value", values: [.number(1)]),
        (name: "value_sum", values: [.number(0)]),
    ])
    expectDataFrameError(.columnNameConflict("value_sum")) {
        _ = try conflict.groupBy("group").transform("value", using: .sum)
    }
}
