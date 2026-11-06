import Foundation
import Testing
@testable import Bandal

private func frame() throws -> DataFrame {
    try DataFrame(columns: [
        (name: "id", values: [.number(1), .number(2), .number(3), .number(4)]),
        (name: "x", values: [.number(1), .missing, .number(3), .number(6)]),
        (name: "s", values: [.string("a"), .string("b"), .string("a"), .string("a")]),
    ])
}

@Test func seriesStatisticsAndValueCounts() throws {
    let s = try frame().series("x")
    #expect(s.reduced(using: .sum) == .number(10))
    #expect(s.reduced(using: .count) == .number(3))
    let counts = try frame().series("s").valueCounts()
    #expect(try counts.values(in: "value") == [.string("a"), .string("b")])
    #expect(try counts.values(in: "count") == [.number(3), .number(1)])
    #expect(s.description.contains("Name: x, Length: 4"))
    #expect(try frame()["missing"] == nil)
}

@Test func dropsColumnsAndSelectsExtremes() throws {
    let dropped = try frame().drop(columns: ["s"])
    #expect(dropped.columnNames == ["id", "x"])
    #expect(throws: DataFrameError.self) { try frame().drop(columns: ["zzz"]) }
    #expect(try frame().largest(1, by: "id").values(in: "id") == [.number(4)])
    #expect(try frame().smallest(1, by: "id").values(in: "id") == [.number(1)])
}

@Test func cumulativeAndShiftOperations() throws {
    let data = try frame()
    #expect(try data.cumulative("x", using: .sum).values(in: "x")
        == [.number(1), .missing, .number(4), .number(10)])
    #expect(try data.shifted("id").values(in: "id")
        == [.missing, .number(1), .number(2), .number(3)])
    #expect(try data.difference("id").values(in: "id")
        == [.missing, .number(1), .number(1), .number(1)])
    #expect(try data.clipped("id", lower: 2, upper: 3).values(in: "id")
        == [.number(2), .number(2), .number(3), .number(3)])
    #expect(try data.percentChange("id").values(in: "id")[1] == .number(1))
}

@Test func fillsForwardAndBackward() throws {
    let data = try frame()
    #expect(try data.forwardFilled().values(in: "x")[1] == .number(1))
    #expect(try data.backwardFilled().values(in: "x")[1] == .number(3))
    #expect(try data.missingMask().values(in: "x")[1] == .boolean(true))
}

@Test func correlationAndReduction() throws {
    let data = try DataFrame(columns: [
        (name: "a", values: [1.0, 2, 3]), (name: "b", values: [2.0, 4, 6]),
    ])
    let corr = try data.correlation()
    #expect(try corr.values(in: "a")[1] == .number(1))
    #expect(try data.covariance().values(in: "a")[0] == .number(1))
    #expect(try data.reduced(using: .mean).values(in: "b") == [.number(4)])
}

@Test func joinsOnDifferentlyNamedMultipleKeysWithSuffixes() throws {
    let left = try DataFrame(columns: [
        (name: "k1", values: [.number(1), .number(1), .number(2)]),
        (name: "k2", values: [.string("a"), .string("b"), .string("a")]),
        (name: "v", values: [.number(10), .number(20), .number(30)]),
    ])
    let right = try DataFrame(columns: [
        (name: "j1", values: [.number(1), .number(2)]),
        (name: "j2", values: [.string("b"), .string("a")]),
        (name: "v", values: [.number(100), .number(200)]),
    ])
    let joined = try left.join(right, leftOn: ["k1", "k2"], rightOn: ["j1", "j2"], kind: .inner)
    #expect(joined.rowCount == 2)
    #expect(joined.columnNames == ["k1", "k2", "v_x", "j1", "j2", "v_y"])
    #expect(try joined.values(in: "v_y") == [.number(100), .number(200)])
}

@Test func resultTypesHaveReadableDescriptions() throws {
    let data = try frame()
    #expect(data.infoSummary().description.contains("4 rows"))
    #expect(data.statistics().description.contains("DataFrameStatistics"))
    #expect(data.summaryTable().description.contains("mean"))
    #expect(data.missingSummary().description.contains("Total missing: 1"))
    #expect(data.uniqueSummary().description.contains("Columns: 3"))
    #expect(data.uniqueSummary()["s"] == 2)
    #expect(data.dtypes.description.contains("Columns: 3"))
    #expect(data.dtypes["id"] == .number)
    #expect(Aggregation("x", using: .sum).description == "sum(x)")
    #expect(try data.groupBy("s").description.contains("keys: s"))
}

@Test func samplesRowsReproducibly() throws {
    let frame = try DataFrame(columns: [
        (name: "n", values: (0..<50).map { DataValue.number(Double($0)) }),
    ])
    let a = try frame.sample(10, seed: 42)
    let b = try frame.sample(10, seed: 42)
    #expect(a.rowCount == 10)
    #expect(try a.values(in: "n") == b.values(in: "n"))
    let numbers = try a.values(in: "n").compactMap { value -> Double? in
        if case .number(let x) = value { x } else { nil }
    }
    #expect(numbers == numbers.sorted())
    #expect(try frame.sample(500, seed: 1).rowCount == 50)
    #expect(try frame.sample(-3).isEmpty)
}

@Test func mapsColumnValues() throws {
    let frame = try DataFrame(columns: [(name: "n", values: [.number(1), .missing, .number(3)])])
    let doubled = try frame.mapped("n") { value in
        if case .number(let x) = value { .number(x * 2) } else { value }
    }
    #expect(try doubled.values(in: "n") == [.number(2), .missing, .number(6)])
}

@Test func supportsJSONRoundTrip() throws {
    let original = try DataFrame(columns: [
        (name: "city", values: ["Seoul", "Busan"]),
        (name: "temp", values: [21.5, nil]),
        (name: "capital", values: [true, false]),
    ])
    let encoded = try original.jsonData(prettyPrinted: true)
    let decoded = try DataFrame(jsonData: encoded)
    #expect(decoded.shape == [2, 3])
    #expect(try decoded.values(in: "city") == ["Seoul", "Busan"])
    #expect(try decoded.values(in: "temp") == [21.5, nil])
    #expect(try decoded.values(in: "capital") == [true, false])

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("bandal-json-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: url) }
    try original.writeJSON(to: url)
    let fromFile = try DataFrame(contentsOfJSON: url)
    #expect(fromFile == decoded)
}

@Test func supportsRowAccessorsSubscriptsAndComputedColumns() throws {
    let df = try DataFrame(columns: [
        (name: "a", values: [10.0, 20.0, 30.0]),
        (name: "b", values: [1.0, 2.0, 3.0]),
    ])
    let firstRow = try #require(df[row: 0])
    #expect(firstRow.index == 0)
    #expect(firstRow["a"] == 10)
    #expect(firstRow.description.contains("Row: 0"))
    #expect(df.rows.count == 3)
    #expect(df["b", 2] == 3)
    #expect(df[rows: 1..<3]?.rowCount == 2)
    #expect(df.toDictionary()["a"] == [10, 20, 30])

    let withSum = try df.withColumn(name: "sum") { row in
        guard let a = row["a"]?.numberValue, let b = row["b"]?.numberValue else { return .missing }
        return .number(a + b)
    }
    #expect(try withSum.values(in: "sum") == [11, 22, 33])
}

@Test func supportsRowPredicatesAndExpressionQueries() throws {
    let df = try DataFrame(columns: [
        (name: "city", values: ["Seoul", "Busan", "Incheon", nil]),
        (name: "score", values: [95, 70, 82, 88]),
        (name: "active", values: [true, false, true, true]),
    ])
    let pred: RowPredicate =
        .column("score", between: 80, and: 100)
        && .notMissing("city")
        && !.column("city", equals: "Jeju")
    let byPred = try df.query(pred)
    #expect(try byPred.values(in: "city") == ["Seoul", "Incheon"])

    let byExpr = try df.query("score >= 80 && active == true")
    #expect(byExpr.rowCount == 3)
}

@Test func supportsDateComponentAndStringOperations() throws {
    let date = try #require(ISO8601DateFormatter().date(from: "2026-04-15T09:30:00Z"))
    let df = try DataFrame(columns: [
        (name: "ts", values: [.date(date), .missing]),
        (name: "name", values: ["  Alice Smith ", nil]),
    ])
    let transformed = try df
        .dateComponent(.year, from: "ts", as: "year")
        .dateComponent(.month, from: "ts", as: "month")
        .trimmed("name")
        .lowercased("name", as: "lower")
        .uppercased("name", as: "upper")
        .stringLength("name", as: "len")
        .replacingSubstring(in: "name", target: "Smith", with: "Kim", as: "replaced")
        .containsSubstring("name", "alice", caseSensitive: false, as: "has_alice")

    #expect(try transformed.values(in: "year") == [2026, nil])
    #expect(try transformed.values(in: "month") == [4, nil])
    #expect(try transformed.values(in: "name") == ["Alice Smith", nil])
    #expect(try transformed.values(in: "lower") == ["alice smith", nil])
    #expect(try transformed.values(in: "upper") == ["ALICE SMITH", nil])
    #expect(try transformed.values(in: "len") == [11, nil])
    #expect(try transformed.values(in: "replaced") == ["Alice Kim", nil])
    #expect(try transformed.values(in: "has_alice") == [true, nil])
}

@Test func supportsExplodedTransposedAndTimeRolling() throws {
    let tags = try DataFrame(columns: [
        (name: "id", values: [1, 2]),
        (name: "tags", values: ["a,b", "c"]),
    ])
    let exploded = try tags.exploded("tags", separator: ",")
    #expect(exploded.rowCount == 3)
    #expect(try exploded.values(in: "tags") == ["a", "b", "c"])

    let transposed = try tags.transposed(indexColumnName: "col", rowPrefix: "r_")
    #expect(transposed.columnNames == ["col", "r_0", "r_1"])
    #expect(try transposed.values(in: "col") == ["id", "tags"])

    let t0 = Date(timeIntervalSince1970: 1_000)
    let events = try DataFrame(columns: [
        (name: "ts", values: [.date(t0), .date(t0.addingTimeInterval(10)), .date(t0.addingTimeInterval(100))]),
        (name: "v", values: [10.0, 20.0, 50.0]),
    ])
    let timeRolled = try events.rolling("v", on: "ts", interval: 30, using: .sum)
    #expect(try timeRolled.values(in: "v_sum_rolling_30s") == [10.0, 30.0, 50.0])
}

@Test func supportsMarkdownExportRenderOptionsAndExtendedSeriesMath() throws {
    let df = try DataFrame(columns: [
        (name: "x", values: [-2.0, 4.0, 4.0]),
        (name: "y", values: [1.0, 2.0, 4.0]),
    ])
    let md = df.toMarkdown()
    #expect(md.contains("| x | y |"))
    let customRender = df.render(options: RenderOptions(maxRows: 2, maxColumns: 2, maxColumnWidth: 5))
    #expect(customRender.contains("[3 rows x 2 columns]"))

    let combined = try df
        .combining("x", "y", using: .add, as: "sum")
        .scaling("y", by: 10, using: .multiply, as: "y10")
        .abs("x")
        .ranked("x", ascending: true, as: "x_rank")
        .replacing(4.0, with: 8.0, in: ["x"])

    #expect(try combined.values(in: "sum") == [-1.0, 6.0, 8.0])
    #expect(try combined.values(in: "y10") == [10.0, 20.0, 40.0])
    #expect(try combined.values(in: "x") == [2.0, 8.0, 8.0])
    #expect(try combined.values(in: "x_rank") == [1.0, 2.5, 2.5])
    #expect(try df.quantile(0.5).values(in: "y") == [2.0])

    let vector = try df.toMLXVector("y")
    #expect(vector.count == 3)
    #expect(vector.description.contains("MLXVector"))
    let fromVector = try DataFrame(mlxColumns: [(name: "y", vector: vector)])
    #expect(try fromVector.values(in: "y") == [1.0, 2.0, 4.0])
}

@Test func supportsVectorizedBooleanMaskAndAdvancedAnalytics() throws {
    let df = try DataFrame(columns: [
        (name: "city", values: ["Seoul", "Busan", "Seoul", "Daegu"]),
        (name: "score", values: [100.0, 40.0, 81.0, 64.0]),
        (name: "gap", values: [.number(10), .missing, .missing, .number(40)]),
    ])
    let scoreSeries = try df.series("score")
    let citySeries = try df.series("city")

    let mask = (scoreSeries >= 80.0) && citySeries.equals("Seoul")
    #expect(mask.count == 4)
    #expect(mask.trueCount == 2)
    #expect(mask.description.contains("BooleanMask"))

    let sub = df[mask]
    #expect(sub.rowCount == 2)
    #expect(try sub.values(in: "score") == [100.0, 81.0])

    let conditional = try scoreSeries.where(scoreSeries >= 70.0, otherwise: 0.0)
    #expect(conditional.values == [.number(100.0), .number(0.0), .number(81.0), .number(0.0)])

    let interpolated = try df.interpolated("gap")
    #expect(try interpolated.values(in: "gap") == [10.0, 20.0, 30.0, 40.0])

    let transformed = try df
        .applying("score", function: .sqrt, as: "root")
        .power("root", exponent: 2.0, as: "squared")
        .standardized("score", as: "z")
        .minMaxScaled("score", as: "minmax")
        .binned("score", edges: [0, 50, 100], labels: ["low", "high"], as: "tier")
        .quantileBinned("score", quantiles: 2, labels: ["q1", "q2"], as: "qtier")
        .exponentiallyWeighted("score", alpha: 0.5, as: "ewm")

    #expect(try transformed.values(in: "root") == [10.0, 6.324555320336759, 9.0, 8.0])
    #expect(try transformed.values(in: "minmax") == [.number(1.0), .number(0.0), .number(41.0 / 60.0), .number(24.0 / 60.0)])
    #expect(try transformed.values(in: "tier") == ["high", "low", "high", "high"])
    #expect(try transformed.values(in: "qtier") == ["q2", "q1", "q2", "q1"])
    #expect(try df.indexOfMax(in: "score") == 0)
    #expect(try df.indexOfMin(in: "score") == 1)

    let sDesc = scoreSeries.describeStructured()
    #expect(sDesc.count == 4)
    #expect(sDesc.max == 100.0)
    #expect(sDesc.description.contains("Name: score"))

    let qTable = try df.quantiles([0.25, 0.5, 0.75])
    #expect(qTable[0.5, "score"] == 72.5)
    #expect(qTable.description.contains("0.5"))
}

@Test func supportsPivotedCrossTabOneHotGroupFilterAndTimeSeriesJoins() throws {
    let longFrame = try DataFrame(columns: [
        (name: "day", values: ["Mon", "Mon", "Tue", "Tue"]),
        (name: "metric", values: ["temp", "rain", "temp", "rain"]),
        (name: "val", values: [20.0, 5.0, 22.0, 0.0]),
    ])
    let wide = try longFrame.pivoted(index: "day", columns: "metric", values: "val")
    #expect(wide.columnNames == ["day", "temp", "rain"])
    #expect(try wide.values(in: "temp") == [20.0, 22.0])

    let ct = try longFrame.crossTabulated(index: "day", columns: "metric")
    #expect(try ct.values(in: "temp") == [1.0, 1.0])

    let dummies = try longFrame.oneHotEncoded(["day"], dropFirst: false)
    #expect(dummies.columnNames.contains("day_Mon"))
    #expect(dummies.columnNames.contains("day_Tue"))

    let groupFiltered = try longFrame
        .groupBy("metric")
        .filter { group in (try group.series("val").sum() ?? 0) > 10 }
    #expect(groupFiltered.rowCount == 2)
    #expect(try groupFiltered.values(in: "metric") == ["temp", "temp"])

    let aliasedAgg = try longFrame
        .groupBy("metric")
        .aggregate([Aggregation("val", using: .sum, as: "total_val")])
    #expect(aliasedAgg.columnNames == ["metric", "total_val"])

    let minPeriodsRoll = try longFrame.rolling("val", window: 3, minPeriods: 1, using: .sum)
    #expect(try minPeriodsRoll.values(in: "val_sum_rolling_3") == [20.0, 25.0, 47.0, 27.0])

    let trades = try DataFrame(columns: [
        (name: "ts", values: [10.0, 25.0, 40.0]),
        (name: "qty", values: [1.0, 2.0, 3.0]),
    ])
    let quotes = try DataFrame(columns: [
        (name: "ts", values: [8.0, 20.0, 35.0]),
        (name: "price", values: [100.0, 102.0, 105.0]),
    ])
    let asOf = try trades.joinAsOf(with: quotes, on: "ts", direction: .backward, tolerance: 10.0)
    #expect(try asOf.values(in: "price") == [100.0, 102.0, 105.0])

    let t0 = Date(timeIntervalSince1970: 0)
    let tsFrame = try DataFrame(columns: [
        (name: "ts", values: [.date(t0.addingTimeInterval(10)), .date(t0.addingTimeInterval(20)), .date(t0.addingTimeInterval(75))]),
        (name: "v", values: [1.0, 3.0, 10.0]),
    ])
    let resampled = try tsFrame.resample(
        on: "ts",
        interval: 60,
        aggregations: [Aggregation("v", using: .sum, as: "sum_v")]
    )
    #expect(resampled.rowCount == 2)
    #expect(try resampled.values(in: "sum_v") == [4.0, 10.0])

    let pipedCount = try tsFrame.pipe { $0.rowCount }
    #expect(pipedCount == 3)
}
