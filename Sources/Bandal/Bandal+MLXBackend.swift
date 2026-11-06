import Accelerate
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

    /// The raw string value of the case.
    public var description: String { rawValue }
}

/// Unary mathematical and transcendental functions supported by vectorized
/// column and series operations.
public enum UnaryMathFunction: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case sqrt
    case exp
    case log
    case log10
    case log1p
    case sin
    case cos
    case tan
    case sign

    /// The raw string value of the case.
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
    /// Force MLX execution on the Metal GPU stream in `float32` for any finite
    /// floating-point input within safe `Float32` range, even when inputs have
    /// non-exact binary fractions (such as `0.1` or `19.99`).
    case gpuFloat32

    /// Alias for `.cpu` used by internal tests.
    static let mlx: ExecutionPolicy = .cpu

    /// The raw string value of the case.
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
/// `float64`. When `ExecutionPolicy.gpuFloat32` is selected, all finite values
/// within safe `Float32` range execute directly on the GPU in `float32`.
enum MLXBackend {
    /// Smallest element count at which `ExecutionPolicy.automatic` uses MLX for simple scans.
    static let automaticMinimumCount = 4_096
    /// Smallest element count at which `ExecutionPolicy.automatic` uses MLX for heavier kernels (moments/sort/matmul/rolling).
    static let automaticComplexMinimumCount = 1_024

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
        case .gpuFloat32:
            return "MLX (GPU stream, float32)"
        case .automatic, .cpu:
            return "MLX (CPU stream, float64)"
        }
    }

    static func shouldUse(elementCount: Int, minimumCount: Int = automaticMinimumCount) -> Bool {
        guard isAvailable, elementCount > 0 else { return false }
        switch policy {
        case .native:
            return false
        case .cpu, .gpu, .gpuFloat32:
            return true
        case .automatic:
            return elementCount >= minimumCount
        }
    }

    static func shouldUseFloat32GPU(for values: [Double], maxMagnitude: Double = 1e6) -> Bool {
        switch policy {
        case .gpu:
            return isExactFloat32(values) && values.allSatisfy { Swift.abs($0) <= maxMagnitude }
        case .gpuFloat32:
            if values.isEmpty { return true }
            var minVal = 0.0
            var maxVal = 0.0
            values.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                vDSP_minvD(base, 1, &minVal, vDSP_Length(buf.count))
                vDSP_maxvD(base, 1, &maxVal, vDSP_Length(buf.count))
            }
            return minVal.isFinite && maxVal.isFinite
                && Swift.abs(minVal) <= maxMagnitude
                && Swift.abs(maxVal) <= maxMagnitude
        default:
            return false
        }
    }

    static func shouldUseFloat32GPU(for values: ContiguousArray<Double>, maxMagnitude: Double = 1e6) -> Bool {
        switch policy {
        case .gpu:
            return values.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return true }
                for i in 0..<buf.count {
                    let value = base[i]
                    guard value.isFinite, Swift.abs(value) <= maxMagnitude else { return false }
                    let f = Float(value)
                    guard f.isFinite, Double(f) == value else { return false }
                }
                return true
            }
        case .gpuFloat32:
            if values.isEmpty { return true }
            var minVal = 0.0
            var maxVal = 0.0
            values.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                vDSP_minvD(base, 1, &minVal, vDSP_Length(buf.count))
                vDSP_maxvD(base, 1, &maxVal, vDSP_Length(buf.count))
            }
            return minVal.isFinite && maxVal.isFinite
                && Swift.abs(minVal) <= maxMagnitude
                && Swift.abs(maxVal) <= maxMagnitude
        default:
            return false
        }
    }

    /// Returns `true` when every value in `values` is finite and losslessly
    /// representable as an IEEE-754 `Float` (`Float32`).
    static func isExactFloat32(_ values: [Double]) -> Bool {
        values.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return true }
            for i in 0..<buf.count {
                let value = base[i]
                guard value.isFinite else { return false }
                let floatValue = Float(value)
                guard floatValue.isFinite, Double(floatValue) == value else {
                    return false
                }
            }
            return true
        }
    }

    static func isExactFloat32(_ values: ContiguousArray<Double>) -> Bool {
        values.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return true }
            for i in 0..<buf.count {
                let value = base[i]
                guard value.isFinite else { return false }
                let floatValue = Float(value)
                guard floatValue.isFinite, Double(floatValue) == value else {
                    return false
                }
            }
            return true
        }
    }

    static func makeFloat32Array(_ values: [Double]) -> MLXArray {
        let count = values.count
        let floats = ContiguousArray<Float>(unsafeUninitializedCapacity: count) { outBuf, initCount in
            values.withUnsafeBufferPointer { inBuf in
                guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                vDSP_vdpsp(inBase, 1, outBase, 1, vDSP_Length(count))
            }
            initCount = count
        }
        return floats.withUnsafeBytes { MLXArray($0, [count], type: Float.self) }
    }

    static func makeFloat32Array(_ values: ContiguousArray<Double>) -> MLXArray {
        let count = values.count
        let floats = ContiguousArray<Float>(unsafeUninitializedCapacity: count) { outBuf, initCount in
            values.withUnsafeBufferPointer { inBuf in
                guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                vDSP_vdpsp(inBase, 1, outBase, 1, vDSP_Length(count))
            }
            initCount = count
        }
        return floats.withUnsafeBytes { MLXArray($0, [count], type: Float.self) }
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

    /// A summary showing the element count and non-missing count.
    public var description: String {
        "MLXVector(count: \(count), nonMissing: \(nonMissingCount))"
    }

    /// A debugging representation; identical to ``description``.
    public var debugDescription: String { description }

    /// Writes ``description`` to the given output stream.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }

    static func makeArray(_ values: [Double]) -> MLXArray {
        values.withUnsafeBytes { MLXArray($0, [values.count], type: Double.self) }
    }

    static func makeArray(_ values: ContiguousArray<Double>) -> MLXArray {
        values.withUnsafeBytes { MLXArray($0, [values.count], type: Double.self) }
    }

    static func makeMatrix(_ values: [Double], rows: Int, columns: Int) -> MLXArray {
        values.withUnsafeBytes { MLXArray($0, [rows, columns], type: Double.self) }
    }
}

/// An internal MLX vector evaluated on the CPU stream in `float64` (or on the
/// GPU stream in `float32` when `.gpu` / `.gpuFloat32` policy is active and all
/// values are within a safe magnitude to avoid overflow).
///
/// Created only when the dispatch policy selects MLX, so every caller keeps a
/// native fallback for `nil`.
struct MLXDeviceVector {
    let array: MLXArray
    let usesFloat32GPU: Bool
    let count: Int

    init?(_ values: [Double], minimumCount: Int = MLXBackend.automaticMinimumCount) {
        guard MLXBackend.shouldUse(elementCount: values.count, minimumCount: minimumCount) else {
            return nil
        }
        var checkSum = 0.0
        values.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            vDSP_sveD(base, 1, &checkSum, vDSP_Length(buf.count))
        }
        if !checkSum.isFinite {
            guard values.allSatisfy(\.isFinite) else { return nil }
        }
        if MLXBackend.shouldUseFloat32GPU(for: values, maxMagnitude: 1e6) {
            self.array = MLXBackend.makeFloat32Array(values)
            self.usesFloat32GPU = true
        } else {
            self.array = MLXVector.makeArray(values)
            self.usesFloat32GPU = false
        }
        self.count = values.count
    }

    init?(_ values: ContiguousArray<Double>, minimumCount: Int = MLXBackend.automaticMinimumCount, assumeFinite: Bool = false) {
        guard MLXBackend.shouldUse(elementCount: values.count, minimumCount: minimumCount) else {
            return nil
        }
        if !assumeFinite {
            var checkSum = 0.0
            values.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                vDSP_sveD(base, 1, &checkSum, vDSP_Length(buf.count))
            }
            if !checkSum.isFinite {
                guard values.allSatisfy(\.isFinite) else { return nil }
            }
        }
        if MLXBackend.shouldUseFloat32GPU(for: values, maxMagnitude: 1e6) {
            self.array = MLXBackend.makeFloat32Array(values)
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

    /// Evaluates sum, variance(ddof: 1), and sorted values in a single fused MLX graph pass.
    func fusedMoments() -> (mean: Double, standardDeviation: Double?, sorted: [Double]) {
        let stream: StreamOrDevice = usesFloat32GPU ? .gpu : .cpu
        let sumArr = MLX.sum(array, stream: stream)
        let sortedArr = MLX.sorted(array, stream: stream)
        if count > 1 {
            let varArr = MLX.variance(array, ddof: 1, stream: stream)
            MLX.eval([sumArr, varArr, sortedArr])
            if usesFloat32GPU {
                let s = Double(sumArr.item(Float.self))
                let v = Double(varArr.item(Float.self))
                let srt = sortedArr.asArray(Float.self).map(Double.init)
                return (s / Double(count), Swift.max(v, 0).squareRoot(), srt)
            } else {
                let s = sumArr.item(Double.self)
                let v = varArr.item(Double.self)
                let srt = sortedArr.asArray(Double.self)
                return (s / Double(count), Swift.max(v, 0).squareRoot(), srt)
            }
        } else {
            MLX.eval([sumArr, sortedArr])
            if usesFloat32GPU {
                let s = Double(sumArr.item(Float.self))
                let srt = sortedArr.asArray(Float.self).map(Double.init)
                return (s / Double(count), nil, srt)
            } else {
                let s = sumArr.item(Double.self)
                let srt = sortedArr.asArray(Double.self)
                return (s / Double(count), nil, srt)
            }
        }
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

    /// Fast O(N) summary statistics (`count`, `mean`, `standardDeviation`, `median`, `min`, `max`)
    /// operating directly on `NumericBuffer` without full sorting on CPU.
    static func summaryMoments(of buffer: NumericBuffer) -> ColumnSummary {
        guard buffer.nonMissingCount > 0 else {
            return ColumnSummary(
                count: 0,
                mean: nil,
                standardDeviation: nil,
                median: nil,
                min: nil,
                max: nil
            )
        }
        let contiguous: ContiguousArray<Double>
        if !buffer.hasMissing {
            contiguous = buffer.floatColumn.values
        } else {
            contiguous = ContiguousArray(buffer.numbers)
        }
        let n = contiguous.count

        // Use GPU Float32 fused reduction when `.gpu` or `.gpuFloat32` is active and eligible.
        if (MLXBackend.policy == .gpu || MLXBackend.policy == .gpuFloat32),
           let vector = MLXDeviceVector(
               contiguous,
               minimumCount: MLXBackend.automaticComplexMinimumCount,
               assumeFinite: true
           ),
           vector.usesFloat32GPU
        {
            let fused = vector.fusedMoments()
            return ColumnSummary(
                count: n,
                mean: fused.mean,
                standardDeviation: fused.standardDeviation,
                median: median(sorted: fused.sorted),
                min: fused.sorted.first,
                max: fused.sorted.last
            )
        }

        var minVal = Double.infinity
        var maxVal = -Double.infinity
        var meanVal = 0.0
        var stdVal: Double?

        contiguous.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            let len = vDSP_Length(n)
            vDSP_minvD(base, 1, &minVal, len)
            vDSP_maxvD(base, 1, &maxVal, len)
            vDSP_meanvD(base, 1, &meanVal, len)
            if n > 1 {
                var s0 = 0.0
                var s1 = 0.0
                var s2 = 0.0
                var s3 = 0.0
                let limit = n & ~3
                var i = 0
                while i < limit {
                    let d0 = base[i] - meanVal
                    let d1 = base[i + 1] - meanVal
                    let d2 = base[i + 2] - meanVal
                    let d3 = base[i + 3] - meanVal
                    s0 += d0 * d0
                    s1 += d1 * d1
                    s2 += d2 * d2
                    s3 += d3 * d3
                    i += 4
                }
                var sqDiff = (s0 + s1) + (s2 + s3)
                while i < n {
                    let d = base[i] - meanVal
                    sqDiff += d * d
                    i += 1
                }
                stdVal = (sqDiff / Double(n - 1)).squareRoot()
            }
        }

        let medVal = quickselectMedian(contiguous)
        return ColumnSummary(
            count: n,
            mean: meanVal,
            standardDeviation: stdVal,
            median: medVal,
            min: minVal,
            max: maxVal
        )
    }

    /// Computes the exact median in O(N) time using 3-way Dutch National Flag Quickselect
    /// with Tukey's ninther pivot selection.
    static func quickselectMedian(_ values: ContiguousArray<Double>) -> Double {
        let n = values.count
        if n == 0 { return 0.0 }
        if n == 1 { return values[0] }
        if n == 2 { return (values[0] + values[1]) * 0.5 }

        var work = values
        return work.withUnsafeMutableBufferPointer { buf -> Double in
            guard let base = buf.baseAddress else { return 0.0 }

            @inline(__always)
            func median3(_ a: Double, _ b: Double, _ c: Double) -> Double {
                if a < b {
                    return b < c ? b : (a < c ? c : a)
                } else {
                    return a < c ? a : (b < c ? c : b)
                }
            }

            let k = n / 2
            var lo = 0
            var hi = n - 1
            var matchedEqualRange = false
            var equalLower = 0

            while lo < hi {
                let len = hi - lo + 1
                if len <= 32 {
                    var i = lo + 1
                    while i <= hi {
                        let key = base[i]
                        var j = i - 1
                        while j >= lo && base[j] > key {
                            base[j + 1] = base[j]
                            j -= 1
                        }
                        base[j + 1] = key
                        i += 1
                    }
                    break
                }

                let step = len / 8
                let mid = lo + (len >> 1)
                let m1 = median3(base[lo], base[lo + step], base[lo + step * 2])
                let m2 = median3(base[mid - step], base[mid], base[mid + step])
                let m3 = median3(base[hi - step * 2], base[hi - step], base[hi])
                let pivot = median3(m1, m2, m3)

                var lt = lo
                var idx = lo
                var gt = hi
                while idx <= gt {
                    let v = base[idx]
                    if v < pivot {
                        base[idx] = base[lt]
                        base[lt] = v
                        lt += 1
                        idx += 1
                    } else if v > pivot {
                        base[idx] = base[gt]
                        base[gt] = v
                        gt -= 1
                    } else {
                        idx += 1
                    }
                }

                if k < lt {
                    hi = lt - 1
                } else if k > gt {
                    lo = gt + 1
                } else {
                    matchedEqualRange = true
                    equalLower = lt
                    break
                }
            }

            let upperMedian = base[k]
            if (n & 1) == 1 {
                return upperMedian
            }
            if matchedEqualRange && equalLower < k {
                return upperMedian
            }
            if !matchedEqualRange && (k - 1) >= lo {
                return (base[k - 1] + upperMedian) * 0.5
            }
            var leftMax = -Double.infinity
            vDSP_maxvD(base, 1, &leftMax, vDSP_Length(k))
            return (leftMax + upperMedian) * 0.5
        }
    }

    static func moments(of numbers: [Double]) -> Moments {
        guard !numbers.isEmpty else {
            return Moments(count: 0)
        }
        let sorted: [Double]
        let mean: Double
        let standardDeviation: Double?

        if (MLXBackend.policy == .gpu || MLXBackend.policy == .gpuFloat32),
           let vector = MLXDeviceVector(numbers, minimumCount: MLXBackend.automaticComplexMinimumCount),
           vector.usesFloat32GPU
        {
            let fused = vector.fusedMoments()
            mean = fused.mean
            standardDeviation = fused.standardDeviation
            sorted = fused.sorted
        } else {
            sorted = numbers.sorted()
            var avg = 0.0
            let n = numbers.count
            numbers.withUnsafeBufferPointer { buf in
                guard let base = buf.baseAddress else { return }
                vDSP_meanvD(base, 1, &avg, vDSP_Length(n))
            }
            mean = avg
            if n > 1 {
                var sqDiff = 0.0
                numbers.withUnsafeBufferPointer { buf in
                    guard let base = buf.baseAddress else { return }
                    for i in 0..<n {
                        let d = base[i] - avg
                        sqDiff += d * d
                    }
                }
                standardDeviation = (sqDiff / Double(n - 1)).squareRoot()
            } else {
                standardDeviation = nil
            }
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
        if function == .median && MLXBackend.policy != .gpu && MLXBackend.policy != .gpuFloat32 {
            return quickselectMedian(ContiguousArray(numbers))
        }
        let minCount = function == .median
            ? MLXBackend.automaticComplexMinimumCount
            : MLXBackend.automaticMinimumCount
        if let vector = MLXDeviceVector(numbers, minimumCount: minCount) {
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
            return quickselectMedian(ContiguousArray(numbers))
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
        let mid = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[mid - 1] + sorted[mid]) / 2.0
            : sorted[mid]
    }

    static func quantile(sorted: [Double], probability: Double) -> Double {
        let p = Swift.min(Swift.max(probability, 0), 1)
        let position = p * Double(sorted.count - 1)
        let lower = Int(floor(position))
        let upper = Int(ceil(position))
        guard lower != upper else { return sorted[lower] }
        let weight = position - Double(lower)
        return sorted[lower] * (1 - weight) + sorted[upper] * weight
    }

    // MARK: Grouped reductions

    /// Vectorized grouped reduction for `count`, `sum`, `mean`, `min`, `max`,
    /// `variance`, and `standardDeviation` over preassigned group indices
    /// `0..<groupCount` (or `-1` for rows with missing keys).
    static func reduceGroups(
        values: [Double?],
        groupIds: [Int],
        groupCount: Int,
        using function: AggregateFunction
    ) -> [Double?]? {
        Device.withDefaultDevice(.cpu) {
            reduceGroupsOnCPU(
                values: values, groupIds: groupIds, groupCount: groupCount, using: function)
        }
    }

    private static func reduceGroupsOnCPU(
        values: [Double?],
        groupIds: [Int],
        groupCount: Int,
        using function: AggregateFunction
    ) -> [Double?]? {
        guard groupCount > 0,
              values.count == groupIds.count,
              MLXBackend.shouldUse(elementCount: values.count)
        else {
            return nil
        }

        switch function {
        case .count, .sum, .mean, .min, .max, .variance, .standardDeviation:
            break
        case .median, .first, .last:
            return nil
        }

        var valid = [Bool]()
        valid.reserveCapacity(values.count)
        var numbers = [Double]()
        numbers.reserveCapacity(values.count)
        var ids = [Int32]()
        ids.reserveCapacity(values.count)

        for (value, group) in zip(values, groupIds) {
            if let number = value, number.isFinite, group >= 0 {
                valid.append(true)
                numbers.append(number)
                ids.append(Int32(group))
            } else {
                valid.append(false)
                numbers.append(0)
                ids.append(0)
            }
        }

        let ones = valid.map { $0 ? 1.0 : 0.0 }
        let counts = Self.scatterSum(ones, ids: ids, groupCount: groupCount)
        if function == .count {
            return counts.map( Optional.some )
        }

        var result: [Double]
        switch function {
        case .min:
            let fill = numbers.enumerated().map { valid[$0.offset] ? $0.element : Double.infinity }
            let initial = MLXArray(Array(repeating: Double.infinity, count: groupCount), [groupCount])
            let updated = initial.at[MLXArray(ids), stream: .cpu].minimum(MLXVector.makeArray(fill))
            result = updated.asArray(Double.self)
        case .max:
            let fill = numbers.enumerated().map { valid[$0.offset] ? $0.element : -Double.infinity }
            let initial = MLXArray(Array(repeating: -Double.infinity, count: groupCount), [groupCount])
            let updated = initial.at[MLXArray(ids), stream: .cpu].maximum(MLXVector.makeArray(fill))
            result = updated.asArray(Double.self)
        default:
            let sums = Self.scatterSum(numbers, ids: ids, groupCount: groupCount)
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

    /// Vectorized 2D strided window reduction for non-null contiguous `Double` columns.
    static func rollingWindowNoNulls(
        _ column: Float64Column,
        window: Int,
        minPeriods: Int,
        using function: AggregateFunction
    ) -> Float64Column? {
        let count = column.count
        // Note: `.sum`, `.mean`, and `.count` are O(N) via sliding-window accumulators in `DataFrame.rolling`,
        // which does Wx less work than 2D strided reductions (`O(N * W)`).
        guard !column.hasNulls,
              window >= 1,
              window <= count,
              minPeriods == window,
              MLXBackend.shouldUse(elementCount: count, minimumCount: MLXBackend.automaticComplexMinimumCount),
              [.min, .max, .variance, .standardDeviation].contains(function)
        else {
            return nil
        }

        let numWindows = count - window + 1
        let windowValues: [Double] = Device.withDefaultDevice(.cpu) {
            let arr = MLXVector.makeArray(column.values)
            let strided = MLX.asStrided(arr, [numWindows, window], strides: [1, 1], offset: 0, stream: .cpu)
            let reduced: MLXArray
            switch function {
            case .min:
                reduced = MLX.min(strided, axis: 1, stream: .cpu)
            case .max:
                reduced = MLX.max(strided, axis: 1, stream: .cpu)
            case .variance:
                reduced = MLX.variance(strided, axis: 1, ddof: 1, stream: .cpu)
            case .standardDeviation:
                reduced = MLX.sqrt(MLX.variance(strided, axis: 1, ddof: 1, stream: .cpu), stream: .cpu)
            default:
                reduced = strided
            }
            return reduced.asArray(Double.self)
        }

        if (function == .variance || function == .standardDeviation) && window < 2 {
            return Float64Column(
                name: column.name,
                values: ContiguousArray(repeating: 0.0, count: count),
                nullBitmap: BitSet(count: count, repeating: true)
            )
        }

        var outVals = ContiguousArray<Double>(repeating: 0.0, count: count)
        var outNulls = BitSet(count: count)
        for i in 0..<(window - 1) {
            outNulls[i] = true
        }
        for i in 0..<numWindows {
            let v = windowValues[i]
            let row = i + window - 1
            if v.isFinite {
                outVals[row] = v
            } else {
                outNulls[row] = true
            }
        }
        return Float64Column(name: column.name, values: outVals, nullBitmap: outNulls)
    }

    // MARK: Zero-Boxing Float64Column <-> MLXArray Kernels

    static func cumulativeColumn(
        _ buffer: NumericBuffer,
        name: String,
        using function: CumulativeFunction
    ) -> Float64Column? {
        guard MLXBackend.shouldUse(elementCount: buffer.nonMissingCount) else { return nil }
        if !buffer.hasMissing && function == .sum && MLXBackend.policy != .gpu && MLXBackend.policy != .gpuFloat32 {
            let count = buffer.count
            var anyNonFinite = false
            let outVals = ContiguousArray<Double>(unsafeUninitializedCapacity: count) { outBuf, initCount in
                buffer.floatColumn.values.withUnsafeBufferPointer { inBuf in
                    guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                    var acc = 0.0
                    for i in 0..<count {
                        acc += inBase[i]
                        outBase[i] = acc
                    }
                    if !acc.isFinite {
                        anyNonFinite = true
                    }
                }
                initCount = count
            }
            if !anyNonFinite {
                return Float64Column(name: name, values: outVals, nullBitmap: BitSet(count: count))
            }
            return nil
        }

        let nonMissing = buffer.numbers
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
        if !buffer.hasMissing {
            return Float64Column(
                name: name,
                values: ContiguousArray(values),
                nullBitmap: BitSet(count: values.count)
            )
        }
        let count = buffer.count
        let nulls = buffer.floatColumn.nullBitmap
        var outVals = ContiguousArray<Double>(repeating: 0.0, count: count)
        var nonMissingIndex = 0
        for row in 0..<count where !nulls[row] {
            outVals[row] = values[nonMissingIndex]
            nonMissingIndex += 1
        }
        return Float64Column(name: name, values: outVals, nullBitmap: nulls)
    }

    static func clipColumn(
        _ buffer: NumericBuffer,
        name: String,
        lower: Double?,
        upper: Double?
    ) -> Float64Column? {
        guard MLXBackend.shouldUse(elementCount: buffer.count) else { return nil }
        let array = MLXVector.makeArray(buffer.floatColumn.values)
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
        let clipped = ContiguousArray(clippedArray.asArray(Double.self))
        return NumericBuffer.makeColumn(name: name, values: clipped, nullBitmap: buffer.floatColumn.nullBitmap)
    }

    static func roundColumn(
        _ buffer: NumericBuffer,
        name: String,
        decimals: Int
    ) -> Float64Column? {
        guard MLXBackend.shouldUse(elementCount: buffer.count) else { return nil }
        let scale = pow(10.0, Double(decimals))
        guard scale.isFinite, scale > 0 else { return nil }
        let array = MLXVector.makeArray(buffer.floatColumn.values)
        let scaled = MLX.multiply(array, MLXArray(float64: scale), stream: .cpu)
        let rounded = ContiguousArray(
            MLX.divide(
                MLX.round(scaled, stream: .cpu),
                MLXArray(float64: scale),
                stream: .cpu
            ).asArray(Double.self)
        )
        return NumericBuffer.makeColumn(name: name, values: rounded, nullBitmap: buffer.floatColumn.nullBitmap)
    }

    static func absColumn(_ buffer: NumericBuffer, name: String) -> Float64Column? {
        guard MLXBackend.shouldUse(elementCount: buffer.count) else { return nil }
        let count = buffer.count
        let outVals = ContiguousArray<Double>(unsafeUninitializedCapacity: count) { outBuf, initCount in
            buffer.floatColumn.values.withUnsafeBufferPointer { inBuf in
                guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                vDSP_vabsD(inBase, 1, outBase, 1, vDSP_Length(count))
            }
            initCount = count
        }
        return Float64Column(name: name, values: outVals, nullBitmap: buffer.floatColumn.nullBitmap)
    }

    static func differenceColumn(
        _ buffer: NumericBuffer,
        name: String,
        periods: Int
    ) -> Float64Column? {
        let count = buffer.count
        let step = Swift.abs(periods)
        guard MLXBackend.shouldUse(elementCount: count),
              step > 0,
              step < count
        else {
            return nil
        }
        let diffs: [Double] = Device.withDefaultDevice(.cpu) {
            let array = MLXVector.makeArray(buffer.floatColumn.values)
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

        let srcNulls = buffer.floatColumn.nullBitmap
        var outVals = ContiguousArray<Double>(repeating: 0.0, count: count)
        var outNulls = BitSet(count: count, repeating: true)
        for row in 0..<count {
            let prior = row - periods
            guard (0..<count).contains(prior),
                  !srcNulls[row],
                  !srcNulls[prior]
            else {
                continue
            }
            let diffIndex = periods > 0 ? (row - step) : row
            let value = diffs[diffIndex]
            if value.isFinite {
                outVals[row] = value
                outNulls[row] = false
            }
        }
        return Float64Column(name: name, values: outVals, nullBitmap: outNulls)
    }

    static func percentChangeColumn(
        _ buffer: NumericBuffer,
        name: String,
        periods: Int
    ) -> Float64Column? {
        let count = buffer.count
        let step = Swift.abs(periods)
        guard MLXBackend.shouldUse(elementCount: count),
              step > 0,
              step < count
        else {
            return nil
        }
        let changes: [Double] = Device.withDefaultDevice(.cpu) {
            let array = MLXVector.makeArray(buffer.floatColumn.values)
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

        let srcVals = buffer.floatColumn.values
        let srcNulls = buffer.floatColumn.nullBitmap
        var outVals = ContiguousArray<Double>(repeating: 0.0, count: count)
        var outNulls = BitSet(count: count, repeating: true)
        for row in 0..<count {
            let prior = row - periods
            guard (0..<count).contains(prior),
                  !srcNulls[row],
                  !srcNulls[prior],
                  srcVals[prior] != 0
            else {
                continue
            }
            let changeIndex = periods > 0 ? (row - step) : row
            let value = changes[changeIndex]
            if value.isFinite {
                outVals[row] = value
                outNulls[row] = false
            }
        }
        return Float64Column(name: name, values: outVals, nullBitmap: outNulls)
    }

    /// Single-pass element-wise kernel: writes `body(left, right)` and reports
    /// whether every result is finite (`v - v` is zero only for finite `v`),
    /// so no separate validation pass over the output is needed.
    @inline(__always)
    private static func simdBinary(
        _ left: UnsafePointer<Double>,
        _ right: UnsafePointer<Double>,
        _ out: UnsafeMutablePointer<Double>,
        _ count: Int,
        _ body: (SIMD4<Double>, SIMD4<Double>) -> SIMD4<Double>
    ) -> Bool {
        var accumulator = SIMD4<Double>(repeating: 0)
        var index = 0
        while index + 4 <= count {
            let l = UnsafeRawPointer(left + index).loadUnaligned(as: SIMD4<Double>.self)
            let r = UnsafeRawPointer(right + index).loadUnaligned(as: SIMD4<Double>.self)
            let v = body(l, r)
            UnsafeMutableRawPointer(out + index).storeBytes(of: v, as: SIMD4<Double>.self)
            accumulator += v - v
            index += 4
        }
        var tail = 0.0
        while index < count {
            let v = body(SIMD4(repeating: left[index]), SIMD4(repeating: right[index]))[0]
            out[index] = v
            tail += v - v
            index += 1
        }
        return accumulator.sum() + tail == 0
    }

    /// Single-pass `sqrt` with the same finite-result reporting as ``simdBinary``.
    @inline(__always)
    private static func simdSquareRoot(
        _ input: UnsafePointer<Double>,
        _ out: UnsafeMutablePointer<Double>,
        _ count: Int
    ) -> Bool {
        var accumulator = SIMD4<Double>(repeating: 0)
        var index = 0
        while index + 4 <= count {
            let x = UnsafeRawPointer(input + index).loadUnaligned(as: SIMD4<Double>.self)
            let v = x.squareRoot()
            UnsafeMutableRawPointer(out + index).storeBytes(of: v, as: SIMD4<Double>.self)
            accumulator += v - v
            index += 4
        }
        var tail = 0.0
        while index < count {
            let v = input[index].squareRoot()
            out[index] = v
            tail += v - v
            index += 1
        }
        return accumulator.sum() + tail == 0
    }

    static func binaryArithmeticColumn(
        left: NumericBuffer,
        right: NumericBuffer,
        name: String,
        operation: ArithmeticOperator
    ) -> Float64Column? {
        let count = left.count
        guard count == right.count,
              MLXBackend.shouldUse(elementCount: count)
        else {
            return nil
        }
        return nativeBinaryArithmeticColumn(left: left, right: right, name: name, operation: operation)
    }

    /// Accelerate/SIMD element-wise arithmetic that does not depend on the MLX policy.
    static func nativeBinaryArithmeticColumn(
        left: NumericBuffer,
        right: NumericBuffer,
        name: String,
        operation: ArithmeticOperator
    ) -> Float64Column {
        let count = left.count
        var allFinite = false
        let raw = ContiguousArray<Double>(unsafeUninitializedCapacity: count) { outBuf, initCount in
            left.floatColumn.values.withUnsafeBufferPointer { lBuf in
                right.floatColumn.values.withUnsafeBufferPointer { rBuf in
                    guard let lBase = lBuf.baseAddress,
                          let rBase = rBuf.baseAddress,
                          let outBase = outBuf.baseAddress else { return }
                    switch operation {
                    case .add:
                        allFinite = simdBinary(lBase, rBase, outBase, count, +)
                    case .subtract:
                        allFinite = simdBinary(lBase, rBase, outBase, count, -)
                    case .multiply:
                        allFinite = simdBinary(lBase, rBase, outBase, count, *)
                    case .divide:
                        allFinite = simdBinary(lBase, rBase, outBase, count, /)
                    }
                }
            }
            initCount = count
        }

        let combinedNulls = left.floatColumn.nullBitmap.bitwiseOr(right.floatColumn.nullBitmap)
        if !combinedNulls.mayHaveTrueBits && allFinite {
            return Float64Column(name: name, values: raw, nullBitmap: combinedNulls)
        }

        var updatedNulls = combinedNulls
        if operation == .divide {
            let rVals = right.floatColumn.values
            for i in 0..<rVals.count where rVals[i] == 0 {
                updatedNulls[i] = true
            }
        }
        return NumericBuffer.makeColumn(name: name, values: raw, nullBitmap: updatedNulls)
    }

    static func scalarArithmeticColumn(
        buffer: NumericBuffer,
        name: String,
        scalar: Double,
        operation: ArithmeticOperator
    ) -> Float64Column? {
        let count = buffer.count
        guard scalar.isFinite,
              MLXBackend.shouldUse(elementCount: count)
        else {
            return nil
        }
        if operation == .divide && scalar == 0 {
            return Float64Column(
                name: name,
                values: ContiguousArray(repeating: 0.0, count: count),
                nullBitmap: BitSet(count: count, repeating: true)
            )
        }

        var checkSum = Double.nan
        let raw = ContiguousArray<Double>(unsafeUninitializedCapacity: count) { outBuf, initCount in
            buffer.floatColumn.values.withUnsafeBufferPointer { inBuf in
                guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                let len = vDSP_Length(count)
                var s = scalar
                switch operation {
                case .add:
                    vDSP_vsaddD(inBase, 1, &s, outBase, 1, len)
                case .subtract:
                    var negS = -scalar
                    vDSP_vsaddD(inBase, 1, &negS, outBase, 1, len)
                case .multiply:
                    vDSP_vsmulD(inBase, 1, &s, outBase, 1, len)
                case .divide:
                    vDSP_vsdivD(inBase, 1, &s, outBase, 1, len)
                }
                vDSP_sveD(outBase, 1, &checkSum, len)
            }
            initCount = count
        }

        let nulls = buffer.floatColumn.nullBitmap
        if !nulls.mayHaveTrueBits && checkSum.isFinite {
            return Float64Column(name: name, values: raw, nullBitmap: nulls)
        }
        return NumericBuffer.makeColumn(name: name, values: raw, nullBitmap: nulls)
    }

    static func unaryMathColumn(
        buffer: NumericBuffer,
        name: String,
        function: UnaryMathFunction
    ) -> Float64Column? {
        let count = buffer.count
        guard MLXBackend.shouldUse(elementCount: count) else { return nil }

        // `sqrt` and `sign` are memory-bound: Double/Float conversion costs more than the GPU saves.
        if (MLXBackend.policy == .gpu || MLXBackend.policy == .gpuFloat32),
           function != .sqrt, function != .sign,
           MLXBackend.shouldUseFloat32GPU(for: buffer.floatColumn.values, maxMagnitude: 1e6)
        {
            let array = MLXBackend.makeFloat32Array(buffer.floatColumn.values)
            let computed: MLXArray
            switch function {
            case .sqrt:
                computed = MLX.sqrt(array, stream: .gpu)
            case .exp:
                computed = MLX.exp(array, stream: .gpu)
            case .log:
                computed = MLX.log(array, stream: .gpu)
            case .log10:
                computed = MLX.log10(array, stream: .gpu)
            case .log1p:
                computed = MLX.log1p(array, stream: .gpu)
            case .sin:
                computed = MLX.sin(array, stream: .gpu)
            case .cos:
                computed = MLX.cos(array, stream: .gpu)
            case .tan:
                computed = MLX.tan(array, stream: .gpu)
            case .sign:
                computed = MLX.sign(array, stream: .gpu)
            }
            let floats = computed.asArray(Float.self)
            var checkSum = Double.nan
            let raw = ContiguousArray<Double>(unsafeUninitializedCapacity: count) { outBuf, initCount in
                floats.withUnsafeBufferPointer { inBuf in
                    guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                    let len = vDSP_Length(count)
                    vDSP_vspdp(inBase, 1, outBase, 1, len)
                    vDSP_sveD(outBase, 1, &checkSum, len)
                }
                initCount = count
            }
            let nulls = buffer.floatColumn.nullBitmap
            if !nulls.mayHaveTrueBits && checkSum.isFinite {
                return Float64Column(name: name, values: raw, nullBitmap: nulls)
            }
            return NumericBuffer.makeColumn(name: name, values: raw, nullBitmap: nulls)
        }

        return nativeUnaryMathColumn(buffer: buffer, name: name, function: function)
    }

    /// Accelerate/SIMD unary math that does not depend on the MLX policy.
    static func nativeUnaryMathColumn(
        buffer: NumericBuffer,
        name: String,
        function: UnaryMathFunction
    ) -> Float64Column {
        let count = buffer.count
        if function == .sqrt {
            var allFinite = false
            let raw = ContiguousArray<Double>(unsafeUninitializedCapacity: count) { outBuf, initCount in
                buffer.floatColumn.values.withUnsafeBufferPointer { inBuf in
                    guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                    allFinite = simdSquareRoot(inBase, outBase, count)
                }
                initCount = count
            }
            let nulls = buffer.floatColumn.nullBitmap
            if !nulls.mayHaveTrueBits && allFinite {
                return Float64Column(name: name, values: raw, nullBitmap: nulls)
            }
            return NumericBuffer.makeColumn(name: name, values: raw, nullBitmap: nulls)
        }

        var checkSum = Double.nan
        let raw = ContiguousArray<Double>(unsafeUninitializedCapacity: count) { outBuf, initCount in
            buffer.floatColumn.values.withUnsafeBufferPointer { inBuf in
                guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                var n32 = Int32(count)
                switch function {
                case .sqrt:
                    vvsqrt(outBase, inBase, &n32)
                case .exp:
                    vvexp(outBase, inBase, &n32)
                case .log:
                    vvlog(outBase, inBase, &n32)
                case .log10:
                    vvlog10(outBase, inBase, &n32)
                case .log1p:
                    vvlog1p(outBase, inBase, &n32)
                case .sin:
                    vvsin(outBase, inBase, &n32)
                case .cos:
                    vvcos(outBase, inBase, &n32)
                case .tan:
                    vvtan(outBase, inBase, &n32)
                case .sign:
                    for i in 0..<count {
                        let x = inBase[i]
                        outBase[i] = x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0)
                    }
                }
                vDSP_sveD(outBase, 1, &checkSum, vDSP_Length(count))
            }
            initCount = count
        }

        let nulls = buffer.floatColumn.nullBitmap
        if !nulls.mayHaveTrueBits && checkSum.isFinite {
            return Float64Column(name: name, values: raw, nullBitmap: nulls)
        }
        return NumericBuffer.makeColumn(name: name, values: raw, nullBitmap: nulls)
    }

    static func powerColumn(
        buffer: NumericBuffer,
        name: String,
        exponent: Double
    ) -> Float64Column? {
        guard exponent.isFinite,
              MLXBackend.shouldUse(elementCount: buffer.count)
        else {
            return nil
        }
        let lhs = MLXVector.makeArray(buffer.floatColumn.values)
        let rhs = MLXArray(float64: exponent)
        let raw = ContiguousArray(MLX.pow(lhs, rhs, stream: .cpu).asArray(Double.self))
        return NumericBuffer.makeColumn(name: name, values: raw, nullBitmap: buffer.floatColumn.nullBitmap)
    }

    // MARK: Legacy [DataValue] wrappers (maintained for internal test compatibility)

    static func cumulative(
        _ buffer: NumericBuffer,
        using function: CumulativeFunction
    ) -> [DataValue]? {
        cumulativeColumn(buffer, name: "", using: function)?.toDataValues()
    }

    static func clip(
        _ buffer: NumericBuffer,
        lower: Double?,
        upper: Double?
    ) -> [DataValue]? {
        clipColumn(buffer, name: "", lower: lower, upper: upper)?.toDataValues()
    }

    static func round(
        _ buffer: NumericBuffer,
        decimals: Int
    ) -> [DataValue]? {
        roundColumn(buffer, name: "", decimals: decimals)?.toDataValues()
    }

    static func abs(_ buffer: NumericBuffer) -> [DataValue]? {
        absColumn(buffer, name: "")?.toDataValues()
    }

    static func difference(
        _ buffer: NumericBuffer,
        periods: Int
    ) -> [DataValue]? {
        differenceColumn(buffer, name: "", periods: periods)?.toDataValues()
    }

    static func percentChange(
        _ buffer: NumericBuffer,
        periods: Int
    ) -> [DataValue]? {
        percentChangeColumn(buffer, name: "", periods: periods)?.toDataValues()
    }

    static func binaryArithmetic(
        left: NumericBuffer,
        right: NumericBuffer,
        operation: ArithmeticOperator
    ) -> [DataValue]? {
        binaryArithmeticColumn(left: left, right: right, name: "", operation: operation)?.toDataValues()
    }

    static func scalarArithmetic(
        buffer: NumericBuffer,
        scalar: Double,
        operation: ArithmeticOperator
    ) -> [DataValue]? {
        scalarArithmeticColumn(buffer: buffer, name: "", scalar: scalar, operation: operation)?.toDataValues()
    }

    // MARK: Covariance and correlation matrices (MLX CPU float64 / GPU float32)

    /// Computes the pairwise sample covariance (`ddof = 1`) or Pearson
    /// correlation matrix across `buffers` using MLX on the CPU stream (or GPU
    /// stream in `float32` when `.gpu` / `.gpuFloat32` policy is active and
    /// inputs are within safe magnitude).
    static func covarianceOrCorrelationMatrix(
        buffers: [NumericBuffer],
        rowCount: Int,
        correlation: Bool
    ) -> [[Double?]]? {
        let columnCount = buffers.count
        guard columnCount > 0,
              MLXBackend.shouldUse(elementCount: rowCount, minimumCount: MLXBackend.automaticComplexMinimumCount)
        else {
            return nil
        }

        // Fast matrix path when no column has missing values and rowCount >= 2.
        if rowCount >= 2, buffers.allSatisfy({ !$0.hasMissing }) {
            let canUseGPU = buffers.allSatisfy {
                MLXBackend.shouldUseFloat32GPU(for: $0.floatColumn.values, maxMagnitude: 1e6)
            }

            var covMatrix = [Double](repeating: 0.0, count: columnCount * columnCount)

            if canUseGPU {
                // Pack [columnCount, rowCount] contiguously without CPU transposition using vDSP_vdpsp.
                let totalCount = columnCount * rowCount
                let colMajorFloats = ContiguousArray<Float>(unsafeUninitializedCapacity: totalCount) { outBuf, initCount in
                    guard let outBase = outBuf.baseAddress else { return }
                    let len = vDSP_Length(rowCount)
                    for col in 0..<columnCount {
                        buffers[col].floatColumn.values.withUnsafeBufferPointer { inBuf in
                            guard let inBase = inBuf.baseAddress else { return }
                            vDSP_vdpsp(inBase, 1, outBase.advanced(by: col * rowCount), 1, len)
                        }
                    }
                    initCount = totalCount
                }
                let matrix = colMajorFloats.withUnsafeBytes {
                    MLXArray($0, [columnCount, rowCount], type: Float.self)
                }
                let means = MLX.mean(matrix, axis: 1, keepDims: true, stream: .gpu)
                let centered = MLX.subtract(matrix, means, stream: .gpu)
                let transposed = MLX.transposed(centered, axes: [1, 0], stream: .gpu)
                let cross = MLX.matmul(centered, transposed, stream: .gpu)
                covMatrix = MLX.divide(
                    cross,
                    MLXArray(Float(rowCount - 1)),
                    stream: .gpu
                ).asArray(Float.self).map(Double.init)
            } else {
                // Hardware-accelerated vDSP centering + upper-triangular dot products in Double precision.
                let len = vDSP_Length(rowCount)
                let invDenom = 1.0 / Double(rowCount - 1)
                var centeredColumns: [ContiguousArray<Double>] = []
                centeredColumns.reserveCapacity(columnCount)

                for col in 0..<columnCount {
                    let centered = ContiguousArray<Double>(unsafeUninitializedCapacity: rowCount) { outBuf, initCount in
                        buffers[col].floatColumn.values.withUnsafeBufferPointer { inBuf in
                            guard let inBase = inBuf.baseAddress, let outBase = outBuf.baseAddress else { return }
                            var mean = 0.0
                            vDSP_meanvD(inBase, 1, &mean, len)
                            var negMean = -mean
                            vDSP_vsaddD(inBase, 1, &negMean, outBase, 1, len)
                        }
                        initCount = rowCount
                    }
                    centeredColumns.append(centered)
                }

                for i in 0..<columnCount {
                    centeredColumns[i].withUnsafeBufferPointer { iBuf in
                        guard let iBase = iBuf.baseAddress else { return }
                        for j in i..<columnCount {
                            centeredColumns[j].withUnsafeBufferPointer { jBuf in
                                guard let jBase = jBuf.baseAddress else { return }
                                var dot = 0.0
                                vDSP_dotprD(iBase, 1, jBase, 1, &dot, len)
                                let cov = dot * invDenom
                                covMatrix[i * columnCount + j] = cov
                                covMatrix[j * columnCount + i] = cov
                            }
                        }
                    }
                }
            }

            var result = Array(
                repeating: Array(repeating: Double?.none, count: columnCount),
                count: columnCount
            )
            if !correlation {
                for i in 0..<columnCount {
                    for j in 0..<columnCount {
                        let v = covMatrix[i * columnCount + j]
                        result[i][j] = v.isFinite ? v : nil
                    }
                }
                return result
            } else {
                let diagStds = (0..<columnCount).map { i -> Double in
                    let variance = covMatrix[i * columnCount + i]
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
                            let corr = covMatrix[i * columnCount + j] / denom
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
                let lVals = left.floatColumn.values
                let rVals = right.floatColumn.values
                let lNulls = left.floatColumn.nullBitmap
                let rNulls = right.floatColumn.nullBitmap
                var xs: [Double] = []
                var ys: [Double] = []
                xs.reserveCapacity(rowCount)
                ys.reserveCapacity(rowCount)
                for row in 0..<rowCount where !lNulls[row] && !rNulls[row] {
                    xs.append(lVals[row])
                    ys.append(rVals[row])
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
            storageColumns: mlxColumns.map { item in
                NumericBuffer.makeColumn(
                    name: item.name,
                    numbers: item.vector.values,
                    isMissing: item.vector.isMissing
                )
            }
        )
    }
}

extension Series {
    /// Extracts a numeric series as an ``MLXVector`` (preserving missing-value
    /// positions).
    ///
    /// - Throws: ``DataFrameError/nonNumericColumn(_:)`` if the series contains
    ///   non-numeric values.
    public func toMLXVector() throws -> MLXVector {
        guard let buffer = numericBuffer else {
            throw DataFrameError.nonNumericColumn(name)
        }
        return MLXVector(values: buffer.values, isMissing: buffer.isMissing)
    }

    /// Creates a `Series` from an ``MLXVector``.
    public init(name: String, mlxVector: MLXVector) {
        let col = NumericBuffer.makeColumn(
            name: name,
            numbers: mlxVector.values,
            isMissing: mlxVector.isMissing
        )
        self.init(
            storage: col,
            numericBuffer: NumericBuffer(floatColumn: col)
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
