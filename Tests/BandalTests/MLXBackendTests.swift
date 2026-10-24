import Testing
@testable import Bandal

private func expectClose(_ lhs: Double?, _ rhs: Double?, tolerance: Double = 1e-9) {
    switch (lhs, rhs) {
    case (nil, nil):
        return
    case (.some(let a), .some(let b)):
        let scale = max(1.0, max(abs(a), abs(b)))
        #expect(abs(a - b) <= tolerance * scale, "expected \(a) ≈ \(b)")
    default:
        Issue.record("mismatch: \(String(describing: lhs)) vs \(String(describing: rhs))")
    }
}

@Test func mlxAvailabilityMatchesEnvironment() {
    if MLXBackend.requireMLXForTesting {
        #expect(BandalRuntime.isMLXAvailable, "BANDAL_REQUIRE_MLX=1 is set, so MLX must be usable")
        #expect(BandalRuntime.executionBackend.lowercased().hasPrefix("mlx"))
    }
    #expect(!BandalRuntime.executionBackend.isEmpty)
}

@Test func numericBufferCachesOnNumericColumnsAndRejectsStrings() throws {
    let df = try DataFrame(columns: [
        (name: "n", values: [.number(1), .missing, .number(3)]),
        (name: "s", values: [.string("a"), .missing, .string("b")]),
    ])
    let buffer = try df.numericBuffer(in: "n")
    #expect(buffer.count == 3)
    #expect(buffer.nonMissingCount == 2)
    #expect(buffer.numbers == [1, 3])
    #expect(buffer.isMissing == [false, true, false])
    #expect(throws: DataFrameError.nonNumericColumn("s")) {
        _ = try df.numericBuffer(in: "s")
    }
}

@Test func mlxAndNativeStatisticsAgreeAcrossSizesAndPrecisions() throws {
    // Includes values that are not exact Float32 (`1 + 1e-12`) to verify that
    // the CPU `float64` MLX path preserves Double precision.
    let values: [DataValue] = (0..<5_000).map { i in
        i.isMultiple(of: 17) ? .missing : .number(Double(i) * 0.125 + 1e-12)
    }
    let df = try DataFrame(columns: [(name: "x", values: values)])

    let native = NumericEngine.withPolicy(.native) { df.statistics()["x"]! }
    let mlx = NumericEngine.withPolicy(.mlx) { df.statistics()["x"]! }

    #expect(native.count == mlx.count)
    expectClose(native.mean, mlx.mean)
    expectClose(native.standardDeviation, mlx.standardDeviation)
    expectClose(native.min, mlx.min, tolerance: 0)
    expectClose(native.max, mlx.max, tolerance: 0)
    expectClose(native.q1, mlx.q1)
    expectClose(native.median, mlx.median)
    expectClose(native.q3, mlx.q3)
}

@Test func mlxAndNativeGroupedReductionsAgree() throws {
    let keys = ["g0", "g1", "g2", "g3"]
    let n = 5_000
    let kValues: [DataValue] = (0..<n).map { .string(keys[$0 % keys.count]) }
    let vValues: [DataValue] = (0..<n).map {
        $0.isMultiple(of: 11) ? .missing : .number(Double(($0 * 31) % 97) - 40.5)
    }
    let df = try DataFrame(columns: [
        (name: "k", values: kValues),
        (name: "v", values: vValues),
    ])
    let aggs = [
        Aggregation("v", using: .count),
        Aggregation("v", using: .sum),
        Aggregation("v", using: .mean),
        Aggregation("v", using: .min),
        Aggregation("v", using: .max),
        Aggregation("v", using: .variance),
        Aggregation("v", using: .standardDeviation),
        Aggregation("v", using: .median),
        Aggregation("v", using: .first),
        Aggregation("v", using: .last),
    ]

    let native = try NumericEngine.withPolicy(.native) { try df.groupBy("k").aggregate(aggs) }
    let mlx = try NumericEngine.withPolicy(.mlx) { try df.groupBy("k").aggregate(aggs) }

    #expect(native.columnNames == mlx.columnNames)
    #expect(try native.values(in: "k") == mlx.values(in: "k"))
    for col in native.columnNames where col != "k" {
        let a = try native.column(col)
        let b = try mlx.column(col)
        for (x, y) in zip(a, b) { expectClose(x, y) }
    }
}

@Test func mlxAndNativeRollingAndExpandingAgree() throws {
    let values: [DataValue] = (0..<5_000).map { i in
        i.isMultiple(of: 23) ? .missing : .number(Double((i * 13) % 53) * 0.5)
    }
    let df = try DataFrame(columns: [(name: "v", values: values)])

    for fn in [AggregateFunction.count, .sum, .mean, .min, .max] {
        let rNative = try NumericEngine.withPolicy(.native) { try df.rolling("v", window: 7, using: fn) }
        let rMLX = try NumericEngine.withPolicy(.mlx) { try df.rolling("v", window: 7, using: fn) }
        let col = "v_\(fn.name)_rolling_7"
        for (a, b) in zip(try rNative.values(in: col), try rMLX.values(in: col)) {
            switch (a, b) {
            case (.missing, .missing): continue
            case (.number(let x), .number(let y)): expectClose(x, y)
            default: Issue.record("mismatch in \(col): \(a) vs \(b)")
            }
        }

        let eNative = try NumericEngine.withPolicy(.native) { try df.expanding("v", using: fn) }
        let eMLX = try NumericEngine.withPolicy(.mlx) { try df.expanding("v", using: fn) }
        let ecol = "v_\(fn.name)_expanding"
        for (a, b) in zip(try eNative.values(in: ecol), try eMLX.values(in: ecol)) {
            switch (a, b) {
            case (.missing, .missing): continue
            case (.number(let x), .number(let y)): expectClose(x, y)
            default: Issue.record("mismatch in \(ecol): \(a) vs \(b)")
            }
        }
    }
}

@Test func mlxAndNativeElementwiseAndMatrixOperationsAgree() throws {
    let n = 4_500
    let aValues: [Double] = (0..<n).map { Double(($0 * 17) % 101) - 50.0 }
    let bValues: [Double] = (0..<n).map { Double(($0 * 29) % 89) + 1.0 }
    let df = try DataFrame(columns: [
        (name: "a", values: aValues),
        (name: "b", values: bValues),
    ])

    let nativeTransformed = try BandalRuntime.withExecutionPolicy(.native) {
        try df
            .combining("a", "b", using: .add, as: "sum")
            .scaling("a", by: 2.5, using: .multiply, as: "scaled")
            .cumulative("b", using: .sum)
            .clipped("a", lower: -20, upper: 20)
            .abs("a")
            .difference("b", periods: 2)
    }
    let cpuTransformed = try BandalRuntime.withExecutionPolicy(.cpu) {
        try df
            .combining("a", "b", using: .add, as: "sum")
            .scaling("a", by: 2.5, using: .multiply, as: "scaled")
            .cumulative("b", using: .sum)
            .clipped("a", lower: -20, upper: 20)
            .abs("a")
            .difference("b", periods: 2)
    }

    for col in ["sum", "scaled", "b", "a"] {
        for (lhs, rhs) in zip(try nativeTransformed.values(in: col), try cpuTransformed.values(in: col)) {
            switch (lhs, rhs) {
            case (.missing, .missing): continue
            case (.number(let x), .number(let y)): expectClose(x, y)
            default: Issue.record("mismatch in \(col): \(lhs) vs \(rhs)")
            }
        }
    }

    let nativeCorr = try BandalRuntime.withExecutionPolicy(.native) { try df.correlation() }
    let cpuCorr = try BandalRuntime.withExecutionPolicy(.cpu) { try df.correlation() }
    let gpuCorr = try BandalRuntime.withExecutionPolicy(.gpu) { try df.correlation() }

    for (x, y) in zip(try nativeCorr.column("a"), try cpuCorr.column("a")) {
        expectClose(x, y, tolerance: 1e-7)
    }
    for (x, y) in zip(try nativeCorr.column("a"), try gpuCorr.column("a")) {
        expectClose(x, y, tolerance: 1e-4)
    }
}
