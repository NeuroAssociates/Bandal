import Foundation
import Testing
@testable import Bandal

@Test func computesSummaryStatistics() throws {
    let data = try DataFrame(columns: [
        "temperature": [18.0, 21.5, 19.0],
    ])

    let summary = try #require(data.summary()["temperature"])
    #expect(summary.count == 3)
    #expect(abs(try #require(summary.mean) - (58.5 / 3.0)) < 1e-10)
    #expect(abs(try #require(summary.standardDeviation) - sqrt(3.25)) < 1e-10)
    #expect(summary.median == 19.0)
    #expect(summary.min == 18.0)
    #expect(summary.max == 21.5)
}

@Test func computesSummaryForEvenCountAndMissingValues() throws {
    let data = try DataFrame(columns: [
        (name: "value", values: [.number(9), .missing, .number(1), .number(5)]),
    ])

    let summary = try #require(data.summary()["value"])
    #expect(summary.count == 3)
    #expect(summary.mean == 5.0)
    #expect(summary.standardDeviation == 4.0)
    #expect(summary.median == 5.0)
    #expect(summary.min == 1.0)
    #expect(summary.max == 9.0)
}

@Test func summaryRetainsDoublePrecisionBeyondFloat32Range() throws {
    let data = try DataFrame(columns: [
        "value": [1_000_000_000_000, 1_000_000_000_002],
    ])

    let summary = try #require(data.summary()["value"])
    #expect(summary.mean == 1_000_000_000_001)
    #expect(summary.median == 1_000_000_000_001)
    #expect(summary.standardDeviation == sqrt(2))
    #expect(summary.min == 1_000_000_000_000)
    #expect(summary.max == 1_000_000_000_002)
}

@Test func computesRollingAndExpandingAggregationsWithMissingValues() throws {
    let data = try DataFrame(columns: [
        (name: "value", values: [
            .number(1), .missing, .number(3), .number(5),
        ]),
    ])

    let rolling = try data.rolling("value", window: 2, using: .mean)
    #expect(try rolling.values(in: "value_mean_rolling_2") == [
        .missing, .number(1), .number(3), .number(4),
    ])

    let rollingCount = try data.rolling("value", window: 2, using: .count)
    #expect(try rollingCount.values(in: "value_count_rolling_2") == [
        .missing, .number(1), .number(1), .number(2),
    ])

    let expanding = try data.expanding("value", using: .sum)
    #expect(try expanding.values(in: "value_sum_expanding") == [
        .number(1), .number(1), .number(4), .number(9),
    ])
    #expect(data.columnNames == ["value"])

    let allMissing = try DataFrame(columns: [
        (name: "value", values: [.missing, .missing, .number(2)]),
    ])
    let rollingSum = try allMissing.rolling("value", window: 2, using: .sum)
    #expect(try rollingSum.values(in: "value_sum_rolling_2") == [
        .missing, .missing, .number(2),
    ])
    let expandingCount = try allMissing.expanding("value", using: .count)
    #expect(try expandingCount.values(in: "value_count_expanding") == [
        .number(0), .number(0), .number(1),
    ])
}

@Test func rejectsInvalidRollingWindowAndNonNumericInput() throws {
    let data = try DataFrame(columns: [
        (name: "value", values: [.number(1), .number(2)]),
    ])
    expectDataFrameError(.invalidWindowSize(0)) {
        _ = try data.rolling("value", window: 0, using: .sum)
    }
    expectDataFrameError(.invalidWindowSize(-1)) {
        _ = try data.rolling("value", window: -1, using: .sum)
    }

    let nonNumeric = try DataFrame(columns: [
        (name: "value", values: [.number(1), .string("two")]),
    ])
    expectDataFrameError(.nonNumericColumn("value")) {
        _ = try nonNumeric.expanding("value", using: .mean)
    }
}

// MARK: - Structured statistics tests

@Test func returnsStructuredStatisticsForNumericColumns() throws {
    let data = try DataFrame(columns: [
        (name: "temperature", values: [.number(10), .number(20), .number(30), .missing]),
        (name: "city", values: [.string("Seoul"), .string("Busan"), .string("Seoul"), .missing]),
    ])

    let stats = data.statistics()
    #expect(stats.rowCount == 4)
    #expect(stats.columnCount == 2)
    #expect(stats.columns.keys.contains("temperature"))
    #expect(!stats.columns.keys.contains("city"))

    let temperature = try #require(stats.columns["temperature"])
    #expect(temperature.count == 3)
    #expect(abs(try #require(temperature.mean) - 20.0) < 1e-10)
    #expect(abs(try #require(temperature.standardDeviation) - 10.0) < 1e-10)
    #expect(temperature.median == 20.0)
    #expect(temperature.min == 10.0)
    #expect(temperature.max == 30.0)
    #expect(abs(try #require(temperature.q1) - 15.0) < 1e-10)
    #expect(abs(try #require(temperature.q3) - 25.0) < 1e-10)
}

@Test func structuredStatisticsHandlesEmptyAndAllMissingColumns() throws {
    let data = try DataFrame(columns: [
        (name: "empty", values: [.missing, .missing, .missing]),
        (name: "value", values: [.number(1), .number(2), .number(3)]),
    ])

    let stats = data.statistics()
    #expect(stats.columns.keys.contains("value"))
    #expect(!stats.columns.keys.contains("empty"))

    let empty = stats.columns["empty"]
    #expect(empty == nil)

    let value = try #require(stats.columns["value"])
    #expect(value.count == 3)
    #expect(value.min == 1.0)
    #expect(value.max == 3.0)
}

@Test func describeStructuredReturnsNumericAndNonNumericSummaries() throws {
    let data = try DataFrame(columns: [
        (name: "temperature", values: [.number(10), .number(20), .number(30), .missing]),
        (name: "city", values: [.string("Seoul"), .string("Busan"), .string("Seoul"), .missing]),
    ])

    let desc = data.describeStructured()
    #expect(desc.columnOrder == ["temperature", "city"])
    #expect(desc.rowCount == 4)
    #expect(desc.columnCount == 2)

    let temperature = try #require(desc.numeric["temperature"])
    #expect(temperature.count == 3)
    #expect(abs(try #require(temperature.mean) - 20.0) < 1e-10)

    let city = try #require(desc.nonNumeric["city"])
    #expect(city.count == 3)
    #expect(city.unique == 2)
    #expect(city.freq == 2)
    #expect(city.top == .string("Seoul"))
}

@Test func describeStructuredDescribeMatchesDescribe() throws {
    let data = try DataFrame(columns: [
        (name: "temperature", values: [.number(10), .number(20), .number(30), .missing]),
        (name: "city", values: [.string("Seoul"), .string("Busan"), .string("Seoul"), .missing]),
    ])

    let structuredOutput = data.describeStructured().describe()
    let stringOutput = data.describe()

    #expect(structuredOutput == stringOutput)
    #expect(structuredOutput.contains("count"))
    #expect(structuredOutput.contains("mean"))
    #expect(structuredOutput.contains("std"))
    #expect(structuredOutput.contains("Seoul"))
    #expect(structuredOutput.contains("freq"))
}

@Test func columnStatisticsIncludesQuartiles() throws {
    let data = try DataFrame(columns: [
        (name: "value", values: [1.0, 2.0, 3.0, 4.0, 5.0]),
    ])

    let stats = data.statistics()
    let value = try #require(stats.columns["value"])
    #expect(value.count == 5)
    #expect(value.mean == 3.0)
    #expect(value.median == 3.0)
    #expect(abs(try #require(value.q1) - 2.0) < 1e-10)
    #expect(abs(try #require(value.q3) - 4.0) < 1e-10)
}

@Test func describeStructuredHandlesEvenCountMedian() throws {
    let data = try DataFrame(columns: [
        (name: "value", values: [1.0, 2.0, 3.0, 4.0]),
    ])

    let stats = data.statistics()
    let value = try #require(stats.columns["value"])
    #expect(value.count == 4)
    #expect(value.mean == 2.5)
    #expect(value.median == 2.5)
    #expect(abs(try #require(value.q1) - 1.75) < 1e-10)
    #expect(abs(try #require(value.q3) - 3.25) < 1e-10)
}

@Test func describeStructuredTextOutputContainsExpectedStatistics() throws {
    let data = try DataFrame(columns: [
        (name: "temperature", values: [.number(10), .number(20), .number(30), .missing]),
    ])

    let desc = data.describeStructured()
    let output = desc.describe()

    #expect(output.contains("count"))
    #expect(output.contains("mean"))
    #expect(output.contains("std"))
    #expect(output.contains("min"))
    #expect(output.contains("25%"))
    #expect(output.contains("50%"))
    #expect(output.contains("75%"))
    #expect(output.contains("max"))
}
