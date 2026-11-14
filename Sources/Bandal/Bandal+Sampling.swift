import Foundation

extension DataFrame {
    /// Whether the frame has no rows.
    public var isEmpty: Bool { rowCount == 0 }

    /// Returns `count` distinct rows chosen at random, keeping source order.
    ///
    /// Pass `seed` for a reproducible selection. `count` is clamped to
    /// `0...rowCount`.
    ///
    /// - Parameters:
    ///   - count: Number of distinct rows to sample.
    ///   - seed: Optional 64-bit seed for deterministic sampling via SplitMix64.
    /// - Returns: A sampled ``DataFrame`` preserving original relative row order.
    /// - Throws: ``DataFrameError`` if row slicing fails.
    public func sample(_ count: Int, seed: UInt64? = nil) throws -> DataFrame {
        let size = min(max(count, 0), rowCount)
        var indices = Array(0..<rowCount)
        if let seed {
            var generator = SplitMix64(seed: seed)
            indices.shuffle(using: &generator)
        } else {
            indices.shuffle()
        }
        return try rows(at: indices.prefix(size).sorted())
    }

    /// Returns a copy with `column` replaced by `transform` applied to each value.
    ///
    /// - Parameters:
    ///   - column: Name of the column to transform.
    ///   - transform: Closure mapping each source ``DataValue`` to a replacement ``DataValue``.
    /// - Returns: A new ``DataFrame`` with the transformed column.
    /// - Throws: ``DataFrameError/columnNotFound(_:)`` if `column` does not exist, or any error thrown by `transform`.
    public func mapped(
        _ column: String,
        _ transform: (DataValue) throws -> DataValue
    ) throws -> DataFrame {
        let replaced = try values(in: column).map(transform)
        return try withColumn(name: column, values: replaced)
    }
}

/// Deterministic 64-bit pseudo-random number generator implementing SplitMix64.
private struct SplitMix64: RandomNumberGenerator {
    /// Current 64-bit internal state.
    var state: UInt64

    /// Creates a SplitMix64 generator seeded with `seed`.
    init(seed: UInt64) { state = seed }

    /// Advances the internal state and returns the next 64-bit pseudo-random integer.
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
