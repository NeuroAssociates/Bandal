import Foundation
import MLX

#if canImport(Metal)
import Metal
#endif

/// Arithmetic operators supported by vectorized column and series operations.
public enum ArithmeticOperator: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case add = "+"
    case subtract = "-"
    case multiply = "*"
    case divide = "/"

    public var description: String { rawValue }
}

/// Where numeric kernels execute.
///
/// Scope a policy for a call tree with
/// ``BandalRuntime/withExecutionPolicy(_:operation:)`` without mutating global
/// MLX device state.
public enum ExecutionPolicy: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    /// Use MLX (`float64` on the CPU stream) when it is available and the input
    /// is large enough (`>= 4,096` elements) to amortize conversion overhead.
    case automatic
    /// Always use native Swift `Double` loops.
    case native
    /// Force MLX execution on the CPU stream in `float64` whenever MLX is available.
    case cpu
    /// Use MLX on the GPU stream in `float32` when every input value is
    /// losslessly representable as `Float32`; otherwise fall back to `.cpu`
    /// (`float64`) so `Double` inputs are never silently rounded.
    case gpu

    /// Alias for `.cpu` used by internal tests.
    static let mlx: ExecutionPolicy = .cpu

    public var description: String { rawValue }
}

/// Runtime detection and dispatch rules for the MLX backend.
///
/// MLX needs a Metal device and its `default.metallib` even for CPU streams,
/// because arrays are allocated by the Metal allocator. When the library
/// cannot be found (for example a plain `swift test` run, where the SwiftPM
/// resource bundle is not next to the test host), touching MLX aborts the
/// process and cannot be recovered. `isAvailable` therefore checks the same
/// lookup locations MLX uses *before* any MLX resource is created.
///
/// By default, computation runs on MLX's CPU stream in `float64`, so results
/// keep full `Double` precision. When `ExecutionPolicy.gpu` is selected, inputs
/// that are exactly representable in `Float32` may use the GPU stream, while
/// inputs requiring `Double` precision automatically fall back to `.cpu`
/// `float64`.
enum MLXBackend {
    /// Smallest element count at which `ExecutionPolicy.automatic` uses MLX.
    static let automaticMinimumCount = 4_096

    @TaskLocal static var policy: ExecutionPolicy = .automatic

    /// Whether MLX can be used in this process. Computed once.
    static let isAvailable: Bool = detectAvailability()

    /// `true` when `BANDAL_REQUIRE_MLX=1` is set in the process environment.
    static var requireMLXForTesting: Bool {
        ProcessInfo.processInfo.environment["BANDAL_REQUIRE_MLX"] == "1"
    }

    /// Describes the backend for diagnostics and benchmark output.
    static var description: String {
        guard isAvailable else { return "native Swift" }
        switch policy {
        case .native:
            return "native Swift"
        case .gpu:
            return "MLX (GPU float32 exact / CPU float64 fallback)"
        case .automatic, .cpu:
            return "MLX (CPU stream, float64)"
        }
    }

    static func shouldUse(elementCount: Int) -> Bool {
        guard isAvailable, elementCount > 0 else { return false }
        switch policy {
        case .native:
            return false
        case .cpu, .gpu:
            return true
        case .automatic:
            return elementCount >= automaticMinimumCount
        }
    }

    /// Returns `true` when every value in `values` is finite and losslessly
    /// representable as an IEEE-754 `Float` (`Float32`).
    static func isExactFloat32(_ values: [Double]) -> Bool {
        for value in values {
            guard value.isFinite else { return false }
            let floatValue = Float(value)
            guard floatValue.isFinite, Double(floatValue) == value else {
                return false
            }
        }
        return true
    }

    private static let bundleName = "mlx-swift_Cmlx"

    private static func detectAvailability() -> Bool {
        if ProcessInfo.processInfo.environment["BANDAL_DISABLE_MLX"] == "1" {
            return false
        }
        #if canImport(Metal) && arch(arm64)
        guard MTLCreateSystemDefaultDevice() != nil else { return false }
        return metallibIsDiscoverable()
        #else
        return false
        #endif
    }

    private static func metallibIsDiscoverable() -> Bool {
        let fileManager = FileManager.default

        if let directory = currentBinaryDirectory() {
            let candidates = [
                directory.appendingPathComponent("mlx.metallib"),
                directory.appendingPathComponent("Resources/mlx.metallib"),
                directory.appendingPathComponent("Resources/default.metallib"),
            ]
            if candidates.contains(where: { fileManager.fileExists(atPath: $0.path) }) {
                return true
            }
        }

        var bundleRoots: [URL] = [Bundle.main.bundleURL]
        for bundle in [Bundle.main] + Bundle.allBundles {
            if let resourceURL = bundle.resourceURL {
                bundleRoots.append(resourceURL)
            }
        }
        for root in bundleRoots {
            let bundleURL = root.appendingPathComponent("\(bundleName).bundle")
            if let bundle = Bundle(url: bundleURL),
               bundle.url(forResource: "default", withExtension: "metallib") != nil
            {
                return true
            }
        }

        for framework in Bundle.allFrameworks where framework.bundleIdentifier == bundleName {
            if framework.url(forResource: "default", withExtension: "metallib") != nil {
                return true
            }
        }
        return false
    }

    private static func currentBinaryDirectory() -> URL? {
        var info = Dl_info()
        let marker: @convention(c) () -> Void = {}
        let address = unsafeBitCast(marker, to: UnsafeRawPointer.self)
        guard dladdr(address, &info) != 0, let name = info.dli_fname else { return nil }
        return URL(fileURLWithPath: String(cString: name)).deletingLastPathComponent()
    }
}

/// A value-type numeric column vector backed by `Double` storage and an
/// associated missing-value mask that converts directly to/from `MLXArray`.
public struct MLXVector: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    /// Underlying `Double` values in row order (`0.0` at missing positions).
    public let values: [Double]
    /// Missing-value mask (`true` when the corresponding row is `.missing`).
    public let isMissing: [Bool]

    /// Total number of rows.
    public var count: Int { values.count }

    /// Number of non-missing numeric rows.
    public var nonMissingCount: Int { isMissing.filter { !$0 }.count }

    /// Creates an `MLXVector` from `Double` values and an optional missing mask.
    public init(values: [Double], isMissing: [Bool]? = nil) {
        self.values = values
        if let isMissing, isMissing.count == values.count {
            self.isMissing = isMissing
        } else {
            self.isMissing = Array(repeating: false, count: values.count)
        }
    }

    /// Creates an `MLXVector` from an array of `DataValue` cells, returning
    /// `nil` if any cell is non-numeric and non-missing.
    public init?(dataValues: [DataValue]) {
        guard let buffer = NumericBuffer(dataValues: dataValues) else { return nil }
        self.values = buffer.values
        self.isMissing = buffer.isMissing
    }

    /// Converts the vector back to `[DataValue]`, preserving `.missing` cells.
    public func toDataValues() -> [DataValue] {
        NumericBuffer.toDataValues(numbers: values, isMissing: isMissing)
    }

    /// Creates a 1-D `MLXArray` (`float64` on CPU stream) when MLX is available
    /// in the current process.
    public func makeMLXArray() -> MLXArray? {
        guard MLXBackend.isAvailable else { return nil }
        return Self.makeArray(values)
    }

    public var description: String {
        "MLXVector(count: \(count), nonMissing: \(nonMissingCount))"
    }

    public var debugDescription: String { description }

    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }

    static func makeArray(_ values: [Double]) -> MLXArray {
        values.withUnsafeBytes { MLXArray($0, [values.count], type: Double.self) }
    }

    static func makeMatrix(_ values: [Double], rows: Int, columns: Int) -> MLXArray {
        values.withUnsafeBytes { MLXArray($0, [rows, columns], type: Double.self) }
    }
}

/// An internal MLX vector evaluated on the CPU stream in `float64` (or on the
/// GPU stream in `float32` when `.gpu` policy is active and all values are
/// exact `Float32` and within a safe magnitude to avoid overflow).
///
/// Created only when the dispatch policy selects MLX, so every caller keeps a
/// native fallback for `nil`.
struct MLXDeviceVector {
    private let array: MLXArray
    private let usesFloat32GPU: Bool
    let count: Int

    init?(_ values: [Double]) {
        guard MLXBackend.shouldUse(elementCount: values.count),
              values.allSatisfy(\.isFinite)
        else {
            return nil
        }
        if MLXBackend.policy == .gpu,
           MLXBackend.isExactFloat32(values),
           values.allSatisfy({ Swift.abs($0) <= 1e6 })
        {
            let floats = values.map(Float.init)
            self.array = floats.withUnsafeBytes { MLXArray($0, [floats.count], type: Float.self) }
            self.usesFloat32GPU = true
        } else {
            self.array = MLXVector.makeArray(values)
            self.usesFloat32GPU = false
        }
        self.count = values.count
    }

    var sum: Double {
        if usesFloat32GPU {
            return Double(MLX.sum(array, stream: .gpu).item(Float.self))
        }
        return MLX.sum(array, stream: .cpu).item(Double.self)
    }

    var minimum: Double {
        if usesFloat32GPU {
            return Double(array.min(stream: .gpu).item(Float.self))
        }
        return array.min(stream: .cpu).item(Double.self)
    }

    var maximum: Double {
        if usesFloat32GPU {
            return Double(array.max(stream: .gpu).item(Float.self))
        }
        return array.max(stream: .cpu).item(Double.self)
    }

    var mean: Double { sum / Double(count) }

    /// Sample variance with `ddof` delta degrees of freedom.
    func variance(ddof: Int) -> Double {
        if usesFloat32GPU {
            return Double(MLX.variance(array, ddof: ddof, stream: .gpu).item(Float.self))
        }
        return MLX.variance(array, ddof: ddof, stream: .cpu).item(Double.self)
    }

    /// Sorted copy, read back as Swift values.
    func sortedValues() -> [Double] {
        if usesFloat32GPU {
            return MLX.sorted(array, stream: .gpu).asArray(Float.self).map(Double.init)
        }
        return MLX.sorted(array, stream: .cpu).asArray(Double.self)
    }
}

/// Native and MLX-backed numeric kernels shared by statistics, grouping,
/// window, and column/series operations. Every MLX kernel has a native
/// reference path.
enum NumericEngine {
    static func withPolicy<R>(_ policy: ExecutionPolicy, _ body: () throws -> R) rethrows -> R {
        try MLXBackend.$policy.withValue(policy, operation: body)
    }

    /// Per-column descriptive statistics.
    struct Moments {
        var count: Int
        var mean: Double?
        var standardDeviation: Double?
        var median: Double?
        var min: Double?
        var max: Double?
        var q1: Double?
        var q3: Double?
        var sorted: [Double] = []
    }

    static func moments(of numbers: [Double]) -> Moments {
        guard !numbers.isEmpty else {
            return Moments(count: 0)
        }
        let vector = MLXDeviceVector(numbers)
        let sorted = vector?.sortedValues() ?? numbers.sorted()

        let mean: Double
        let standardDeviation: Double?
        if let vector {
            mean = vector.mean
            standardDeviation = numbers.count > 1
                ? vector.variance(ddof: 1).squareRoot()
                : nil
        } else {
            mean = numbers.reduce(0, +) / Double(numbers.count)
            standardDeviation = numbers.count > 1
                ? (numbers.reduce(0) { $0 + pow($1 - mean, 2) }
                    / Double(numbers.count - 1)).squareRoot()
                : nil
        }

        return Moments(
            count: numbers.count,
            mean: mean,
            standardDeviation: standardDeviation,
            median: median(sorted: sorted),
            min: sorted.first,
            max: sorted.last,
            q1: quantile(sorted: sorted, probability: 0.25),
            q3: quantile(sorted: sorted, probability: 0.75),
            sorted: sorted
        )
    }

    /// Reduces a non-missing numeric array using MLX when beneficial and native
    /// Swift otherwise.
    static func reduce(_ numbers: [Double], using function: AggregateFunction) -> Double? {
        guard !numbers.isEmpty else {
            return function == .count ? 0 : nil
        }
        if let vector = MLXDeviceVector(numbers) {
            switch function {
            case .count:
                return Double(numbers.count)
            case .sum:
                return vector.sum
            case .mean:
                return vector.mean
            case .min:
                return vector.minimum
            case .max:
                return vector.maximum
            case .median:
                return median(sorted: vector.sortedValues())
            case .variance:
                return numbers.count > 1 ? vector.variance(ddof: 1) : nil
            case .standardDeviation:
                return numbers.count > 1 ? vector.variance(ddof: 1).squareRoot() : nil
            case .first:
                return numbers.first
            case .last:
                return numbers.last
            }
        }
        switch function {
        case .count:
            return Double(numbers.count)
        case .sum:
            return numbers.reduce(0, +)
        case .mean:
            return numbers.reduce(0, +) / Double(numbers.count)
        case .min:
            return numbers.min()
        case .max:
            return numbers.max()
        case .median:
            return median(sorted: numbers.sorted())
        case .variance:
            guard numbers.count > 1 else { return nil }
            let mean = numbers.reduce(0, +) / Double(numbers.count)
            return numbers.reduce(0) { $0 + pow($1 - mean, 2) } / Double(numbers.count - 1)
        case .standardDeviation:
            guard numbers.count > 1 else { return nil }
            let mean = numbers.reduce(0, +) / Double(numbers.count)
            let variance = numbers.reduce(0) { $0 + pow($1 - mean, 2) } / Double(numbers.count - 1)
            return variance.squareRoot()
        case .first:
            return numbers.first
        case .last:
            return numbers.last
        }
    }

    static func median(sorted: [Double]) -> Double {
        let count = sorted.count
        if count.isMultiple(of: 2) {
            return sorted[(count - 1) / 2] / 2 + sorted[count / 2] / 2
        }
        return sorted[count / 2]
    }

    /// Linear-interpolation quantile of ascending values (pandas default).
    static func quantile(sorted: [Double], probability: Double) -> Double {
        guard sorted.count > 1 else { return sorted.first ?? .nan }
        let clamped = Swift.min(Swift.max(probability, 0), 1)
        let position = clamped * Double(sorted.count - 1)
        let lowerIndex = Int(position)
        let upperIndex = Swift.min(lowerIndex + 1, sorted.count - 1)
        let fraction = position - Double(lowerIndex)
        return sorted[lowerIndex] * (1 - fraction) + sorted[upperIndex] * fraction
    }

    // MARK: Grouped reductions

    /// Reduces one numeric column for every group in a single pass.
    ///
    /// - Parameters:
    ///   - values: one entry per grouped row; `nil` marks a missing value.
    ///   - groupIDs: group index (`0..<groupCount`) of each entry.
    /// - Returns: one result per group; `nil` when a group has no numeric value.
    static func groupedReduction(
        values: [Double?],
        groupIDs: [Int],
        groupCount: Int,
        function: AggregateFunction
    ) -> [Double?]? {
        Device.withDefaultDevice(.cpu) {
            scatterReduction(
                values: values, groupIDs: groupIDs, groupCount: groupCount, function: function)
        }
    }

    private static func scatterReduction(
        values: [Double?],
        groupIDs: [Int],
        groupCount: Int,
        function: AggregateFunction
    ) -> [Double?]? {
        guard MLXBackend.shouldUse(elementCount: values.count),
              groupCount > 0,
              [.sum, .mean, .min, .max, .variance, .standardDeviation].contains(function)
        else {
            return nil
        }
        let valid = values.map { $0 != nil }
        let numbers = values.map { $0 ?? 0 }
        let ids = groupIDs.map(Int32.init)

        let counts: [Double]
        var result: [Double] = []
        let sums: [Double]
        switch function {
        case .min, .max:
            let sentinel = function == .min ? Double.infinity : -Double.infinity
            let masked = zip(numbers, valid).map { $1 ? $0 : sentinel }
            let output = MLX.full(
                [groupCount], values: MLXArray(float64: sentinel), type: Double.self,
                stream: .cpu)
            let indices = MLXArray(ids)
            let source = MLXVector.makeArray(masked)
            let reduced = function == .min
                ? output.at[indices, stream: .cpu].minimum(source)
                : output.at[indices, stream: .cpu].maximum(source)
            result = reduced.asArray(Double.self)
            counts = Self.scatterSum(
                valid.map { $0 ? 1.0 : 0.0 }, ids: ids, groupCount: groupCount)
            sums = []
        default:
            sums = Self.scatterSum(numbers, ids: ids, groupCount: groupCount)
            counts = Self.scatterSum(
                valid.map { $0 ? 1.0 : 0.0 }, ids: ids, groupCount: groupCount)
            switch function {
            case .sum:
                result = sums
            case .mean:
                result = zip(sums, counts).map { $1 > 0 ? $0 / $1 : 0 }
            default:
                let means = zip(sums, counts).map { $1 > 0 ? $0 / $1 : 0 }
                let deviations = zip(numbers, zip(ids, valid)).map { number, pair in
                    pair.1 ? pow(number - means[Int(pair.0)], 2) : 0
                }
                let squares = Self.scatterSum(deviations, ids: ids, groupCount: groupCount)
                result = zip(squares, counts).map { $1 > 1 ? $0 / ($1 - 1) : 0 }
                if function == .standardDeviation {
                    result = result.map { $0.squareRoot() }
                }
            }
        }

        return (0..<groupCount).map { group in
            switch function {
            case .variance, .standardDeviation:
                return counts[group] > 1 ? result[group] : nil
            default:
                return counts[group] > 0 ? result[group] : nil
            }
        }
    }

    private static func scatterSum(_ values: [Double], ids: [Int32], groupCount: Int) -> [Double] {
        let output = MLX.zeros([groupCount], type: Double.self, stream: .cpu)
        let updated = output.at[MLXArray(ids), stream: .cpu].add(MLXVector.makeArray(values))
        return updated.asArray(Double.self)
    }

    // MARK: Window reductions

    /// Prefix sums and prefix valid-counts for rolling/expanding sum, mean,
    /// and count. Missing values contribute zero to the sum and one to
    /// nothing, so windows containing missing values need no native fallback.
    static func prefixSums(values: [Double?]) -> (sums: [Double], counts: [Int])? {
        guard MLXBackend.shouldUse(elementCount: values.count) else { return nil }
        let numbers = values.map { $0 ?? 0 }
        let valid = values.map { $0 == nil ? 0.0 : 1.0 }
        let sums = MLX.cumsum(MLXVector.makeArray(numbers), stream: .cpu).asArray(Double.self)
        let counts = MLX.cumsum(MLXVector.makeArray(valid), stream: .cpu).asArray(Double.self)
        return (sums, counts.map { Int($0) })
    }

    // MARK: Vectorized column & series kernels (MLX CPU float64)

    /// Cumulative reduction over a `NumericBuffer`, preserving `.missing` positions.
    static func cumulative(
        _ buffer: NumericBuffer,
        using function: CumulativeFunction
    ) -> [DataValue]? {
        let nonMissing = buffer.numbers
        guard MLXBackend.shouldUse(elementCount: nonMissing.count) else { return nil }
        let array = MLXVector.makeArray(nonMissing)
        let reduced: MLXArray
        switch function {
        case .sum:
            reduced = MLX.cumsum(array, stream: .cpu)
        case .product:
            reduced = MLX.cumprod(array, stream: .cpu)
        case .minimum:
            reduced = MLX.cummin(array, stream: .cpu)
        case .maximum:
            reduced = MLX.cummax(array, stream: .cpu)
        }
        let values = reduced.asArray(Double.self)
        guard values.allSatisfy(\.isFinite) else { return nil }
        var output: [DataValue] = []
        output.reserveCapacity(buffer.count)
        var nonMissingIndex = 0
        for row in 0..<buffer.count {
            if buffer.isMissing[row] {
                output.append(.missing)
            } else {
                output.append(.number(values[nonMissingIndex]))
                nonMissingIndex += 1
            }
        }
        return output
    }

    /// Vectorized clipping over a `NumericBuffer`.
    static func clip(
        _ buffer: NumericBuffer,
        lower: Double?,
        upper: Double?
    ) -> [DataValue]? {
        guard MLXBackend.shouldUse(elementCount: buffer.count) else { return nil }
        let array = MLXVector.makeArray(buffer.values)
        let clippedArray: MLXArray
        switch (lower, upper) {
        case (let lo?, let hi?):
            clippedArray = MLX.clip(
                array,
                min: MLXArray(float64: lo),
                max: MLXArray(float64: hi),
                stream: .cpu
            )
        case (let lo?, nil):
            clippedArray = MLX.maximum(array, MLXArray(float64: lo), stream: .cpu)
        case (nil, let hi?):
            clippedArray = MLX.minimum(array, MLXArray(float64: hi), stream: .cpu)
        case (nil, nil):
            clippedArray = array
        }
        let clipped = clippedArray.asArray(Double.self)
        return NumericBuffer.toDataValues(numbers: clipped, isMissing: buffer.isMissing)
    }

    /// Vectorized rounding to `decimals` decimal places.
    static func round(
        _ buffer: NumericBuffer,
        decimals: Int
    ) -> [DataValue]? {
        guard MLXBackend.shouldUse(elementCount: buffer.count) else { return nil }
        let scale = pow(10.0, Double(decimals))
        guard scale.isFinite, scale > 0 else { return nil }
        let array = MLXVector.makeArray(buffer.values)
        let scaled = MLX.multiply(array, MLXArray(float64: scale), stream: .cpu)
        let rounded = MLX.divide(
            MLX.round(scaled, stream: .cpu),
            MLXArray(float64: scale),
            stream: .cpu
        ).asArray(Double.self)
        return NumericBuffer.toDataValues(numbers: rounded, isMissing: buffer.isMissing)
    }

    /// Vectorized absolute value over a `NumericBuffer`.
    static func abs(_ buffer: NumericBuffer) -> [DataValue]? {
        guard MLXBackend.shouldUse(elementCount: buffer.count) else { return nil }
        let array = MLXVector.makeArray(buffer.values)
        let result = MLX.abs(array, stream: .cpu).asArray(Double.self)
        return NumericBuffer.toDataValues(numbers: result, isMissing: buffer.isMissing)
    }

    /// Vectorized lag difference `x[i] - x[i - periods]`.
    static func difference(
        _ buffer: NumericBuffer,
        periods: Int
    ) -> [DataValue]? {
        let count = buffer.count
        let step = Swift.abs(periods)
        guard MLXBackend.shouldUse(elementCount: count),
              step > 0,
              step < count
        else {
            return nil
        }
        let diffs: [Double] = Device.withDefaultDevice(.cpu) {
            let array = MLXVector.makeArray(buffer.values)
            if periods > 0 {
                let current = array[step..<count]
                let previous = array[0..<(count - step)]
                return MLX.subtract(current, previous, stream: .cpu).asArray(Double.self)
            } else {
                let current = array[0..<(count - step)]
                let previous = array[step..<count]
                return MLX.subtract(current, previous, stream: .cpu).asArray(Double.self)
            }
        }

        var output = Array(repeating: DataValue.missing, count: count)
        for row in 0..<count {
            let prior = row - periods
            guard (0..<count).contains(prior),
                  !buffer.isMissing[row],
                  !buffer.isMissing[prior]
            else {
                continue
            }
            let diffIndex = periods > 0 ? (row - step) : row
            let value = diffs[diffIndex]
            output[row] = value.isFinite ? .number(value) : .missing
        }
        return output
    }

    /// Vectorized percentage change `(x[i] - x[i - periods]) / x[i - periods]`.
    static func percentChange(
        _ buffer: NumericBuffer,
        periods: Int
    ) -> [DataValue]? {
        let count = buffer.count
        let step = Swift.abs(periods)
        guard MLXBackend.shouldUse(elementCount: count),
              step > 0,
              step < count
        else {
            return nil
        }
        let changes: [Double] = Device.withDefaultDevice(.cpu) {
            let array = MLXVector.makeArray(buffer.values)
            if periods > 0 {
                let current = array[step..<count]
                let previous = array[0..<(count - step)]
                let delta = MLX.subtract(current, previous, stream: .cpu)
                return MLX.divide(delta, previous, stream: .cpu).asArray(Double.self)
            } else {
                let current = array[0..<(count - step)]
                let previous = array[step..<count]
                let delta = MLX.subtract(current, previous, stream: .cpu)
                return MLX.divide(delta, previous, stream: .cpu).asArray(Double.self)
            }
        }

        var output = Array(repeating: DataValue.missing, count: count)
        for row in 0..<count {
            let prior = row - periods
            guard (0..<count).contains(prior),
                  !buffer.isMissing[row],
                  !buffer.isMissing[prior],
                  buffer.values[prior] != 0
            else {
                continue
            }
            let changeIndex = periods > 0 ? (row - step) : row
            let value = changes[changeIndex]
            output[row] = value.isFinite ? .number(value) : .missing
        }
        return output
    }

    /// Element-wise binary arithmetic on two numeric buffers of equal length.
    static func binaryArithmetic(
        left: NumericBuffer,
        right: NumericBuffer,
        operation: ArithmeticOperator
    ) -> [DataValue]? {
        guard left.count == right.count,
              MLXBackend.shouldUse(elementCount: left.count)
        else {
            return nil
        }
        let lhs = MLXVector.makeArray(left.values)
        let rhs = MLXVector.makeArray(right.values)
        let computed: MLXArray
        switch operation {
        case .add:
            computed = MLX.add(lhs, rhs, stream: .cpu)
        case .subtract:
            computed = MLX.subtract(lhs, rhs, stream: .cpu)
        case .multiply:
            computed = MLX.multiply(lhs, rhs, stream: .cpu)
        case .divide:
            computed = MLX.divide(lhs, rhs, stream: .cpu)
        }
        let raw = computed.asArray(Double.self)
        var output: [DataValue] = []
        output.reserveCapacity(left.count)
        for row in 0..<left.count {
            if left.isMissing[row] || right.isMissing[row] || (operation == .divide && right.values[row] == 0) {
                output.append(.missing)
            } else {
                let value = raw[row]
                output.append(value.isFinite ? .number(value) : .missing)
            }
        }
        return output
    }

    /// Element-wise arithmetic between a numeric buffer and a scalar `Double`.
    static func scalarArithmetic(
        buffer: NumericBuffer,
        scalar: Double,
        operation: ArithmeticOperator
    ) -> [DataValue]? {
        guard scalar.isFinite,
              MLXBackend.shouldUse(elementCount: buffer.count)
        else {
            return nil
        }
        if operation == .divide && scalar == 0 {
            return Array(repeating: .missing, count: buffer.count)
        }
        let lhs = MLXVector.makeArray(buffer.values)
        let rhs = MLXArray(float64: scalar)
        let computed: MLXArray
        switch operation {
        case .add:
            computed = MLX.add(lhs, rhs, stream: .cpu)
        case .subtract:
            computed = MLX.subtract(lhs, rhs, stream: .cpu)
        case .multiply:
            computed = MLX.multiply(lhs, rhs, stream: .cpu)
        case .divide:
            computed = MLX.divide(lhs, rhs, stream: .cpu)
        }
        let raw = computed.asArray(Double.self)
        return NumericBuffer.toDataValues(numbers: raw, isMissing: buffer.isMissing)
    }

    // MARK: Covariance and correlation matrices (MLX CPU float64 / GPU float32)

    /// Computes the pairwise sample covariance (`ddof = 1`) or Pearson
    /// correlation matrix across `buffers` using MLX on the CPU stream (or GPU
    /// stream in `float32` when `.gpu` policy is active and inputs are exact
    /// `Float32` within safe magnitude).
    static func covarianceOrCorrelationMatrix(
        buffers: [NumericBuffer],
        rowCount: Int,
        correlation: Bool
    ) -> [[Double?]]? {
        let columnCount = buffers.count
        guard columnCount > 0,
              MLXBackend.shouldUse(elementCount: rowCount)
        else {
            return nil
        }

        // Fast matrix path when no column has missing values and rowCount >= 2.
        if rowCount >= 2, buffers.allSatisfy({ !$0.hasMissing }) {
            let canUseGPU = MLXBackend.policy == .gpu
                && buffers.allSatisfy {
                    MLXBackend.isExactFloat32($0.values)
                        && $0.values.allSatisfy { Swift.abs($0) <= 1e4 }
                }

            let covArray: [Double]
            if canUseGPU {
                var rowMajorFloats = [Float]()
                rowMajorFloats.reserveCapacity(rowCount * columnCount)
                for row in 0..<rowCount {
                    for col in 0..<columnCount {
                        rowMajorFloats.append(Float(buffers[col].values[row]))
                    }
                }
                let matrix = rowMajorFloats.withUnsafeBytes {
                    MLXArray($0, [rowCount, columnCount], type: Float.self)
                }
                let means = MLX.mean(matrix, axis: 0, keepDims: true, stream: .gpu)
                let centered = MLX.subtract(matrix, means, stream: .gpu)
                let transposed = MLX.transposed(centered, axes: [1, 0], stream: .gpu)
                let cross = MLX.matmul(transposed, centered, stream: .gpu)
                covArray = MLX.divide(
                    cross,
                    MLXArray(Float(rowCount - 1)),
                    stream: .gpu
                ).asArray(Float.self).map(Double.init)
            } else {
                var rowMajor = [Double]()
                rowMajor.reserveCapacity(rowCount * columnCount)
                for row in 0..<rowCount {
                    for col in 0..<columnCount {
                        rowMajor.append(buffers[col].values[row])
                    }
                }
                let matrix = MLXVector.makeMatrix(rowMajor, rows: rowCount, columns: columnCount)
                let means = MLX.mean(matrix, axis: 0, keepDims: true, stream: .cpu)
                let centered = MLX.subtract(matrix, means, stream: .cpu)
                let transposed = MLX.transposed(centered, axes: [1, 0], stream: .cpu)
                let cross = MLX.matmul(transposed, centered, stream: .cpu)
                covArray = MLX.divide(
                    cross,
                    MLXArray(float64: Double(rowCount - 1)),
                    stream: .cpu
                ).asArray(Double.self)
            }

            var result = Array(
                repeating: Array(repeating: Double?.none, count: columnCount),
                count: columnCount
            )
            if !correlation {
                for i in 0..<columnCount {
                    for j in 0..<columnCount {
                        let v = covArray[i * columnCount + j]
                        result[i][j] = v.isFinite ? v : nil
                    }
                }
                return result
            } else {
                let diagStds = (0..<columnCount).map { i -> Double in
                    let variance = covArray[i * columnCount + i]
                    return variance > 0 ? variance.squareRoot() : 0
                }
                for i in 0..<columnCount {
                    for j in 0..<columnCount {
                        let denom = diagStds[i] * diagStds[j]
                        guard denom > 0 else {
                            result[i][j] = nil
                            continue
                        }
                        if i == j {
                            result[i][j] = 1.0
                        } else {
                            let corr = covArray[i * columnCount + j] / denom
                            result[i][j] = corr.isFinite ? Swift.min(Swift.max(corr, -1.0), 1.0) : nil
                        }
                    }
                }
                return result
            }
        }

        // Pairwise MLX dot-product path when missing values exist.
        var result = Array(
            repeating: Array(repeating: Double?.none, count: columnCount),
            count: columnCount
        )
        for i in 0..<columnCount {
            for j in i..<columnCount {
                let left = buffers[i]
                let right = buffers[j]
                var xs: [Double] = []
                var ys: [Double] = []
                xs.reserveCapacity(rowCount)
                ys.reserveCapacity(rowCount)
                for row in 0..<rowCount where !left.isMissing[row] && !right.isMissing[row] {
                    xs.append(left.values[row])
                    ys.append(right.values[row])
                }
                guard xs.count >= 2 else { continue }
                let xArr = MLXVector.makeArray(xs)
                let yArr = MLXVector.makeArray(ys)
                let dx = MLX.subtract(xArr, MLX.mean(xArr, stream: .cpu), stream: .cpu)
                let dy = MLX.subtract(yArr, MLX.mean(yArr, stream: .cpu), stream: .cpu)
                let xySum = MLX.sum(MLX.multiply(dx, dy, stream: .cpu), stream: .cpu).item(Double.self)
                let value: Double?
                if correlation {
                    let xxSum = MLX.sum(MLX.multiply(dx, dx, stream: .cpu), stream: .cpu).item(Double.self)
                    let yySum = MLX.sum(MLX.multiply(dy, dy, stream: .cpu), stream: .cpu).item(Double.self)
                    let denom = (xxSum * yySum).squareRoot()
                    value = denom > 0 ? Swift.min(Swift.max(xySum / denom, -1.0), 1.0) : nil
                } else {
                    value = xySum / Double(xs.count - 1)
                }
                result[i][j] = value
                result[j][i] = value
            }
        }
        return result
    }
}

extension DataFrame {
    /// Extracts a numeric column as an ``MLXVector`` (preserving missing-value
    /// positions).
    ///
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` or
    ///   ``DataFrameError/nonNumericColumn(_:)`` if the column contains
    ///   non-numeric values.
    public func toMLXVector(_ column: String) throws -> MLXVector {
        let buffer = try numericBuffer(in: column)
        return MLXVector(values: buffer.values, isMissing: buffer.isMissing)
    }

    /// Creates a `DataFrame` from ordered named ``MLXVector`` columns.
    public init(mlxColumns: [(name: String, vector: MLXVector)]) throws {
        try self.init(
            columns: mlxColumns.map { (name: $0.name, values: $0.vector.toDataValues()) }
        )
    }
}

/// Reports and scopes which numeric engine Bandal uses in the current process.
public enum BandalRuntime {
    /// `true` when MLX is available and not disabled with `BANDAL_DISABLE_MLX=1`.
    public static var isMLXAvailable: Bool { MLXBackend.isAvailable }

    /// A short human-readable description of the active numeric engine.
    public static var executionBackend: String { MLXBackend.description }

    /// The active task-local execution policy (`.automatic` by default).
    public static var executionPolicy: ExecutionPolicy { MLXBackend.policy }

    /// Runs `operation` with the specified `ExecutionPolicy` scoped to the
    /// current task, without mutating global MLX device state.
    public static func withExecutionPolicy<R>(
        _ policy: ExecutionPolicy,
        operation: () throws -> R
    ) rethrows -> R {
        try MLXBackend.$policy.withValue(policy, operation: operation)
    }

    /// Runs an asynchronous `operation` with the specified `ExecutionPolicy`
    /// scoped to the current task.
    public static func withExecutionPolicy<R>(
        _ policy: ExecutionPolicy,
        operation: () async throws -> R
    ) async rethrows -> R {
        try await MLXBackend.$policy.withValue(policy, operation: operation)
    }
}
