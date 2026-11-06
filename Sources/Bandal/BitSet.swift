import Foundation

// MARK: - BooleanMask

/// A row-aligned Boolean mask backed by a contiguous ``BitSet``.
///
/// Combine masks with `&&`, `||`, and `!`, and pass them to
/// `df[mask: mask]` or `df.filtered(by: mask)` for zero-allocation row
/// filtering.
public struct BooleanMask: Equatable, Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable {
    @usableFromInline
    let bitSet: BitSet

    /// Total number of rows in the mask.
    @inlinable
    public var count: Int { bitSet.count }

    /// Whether the mask has zero elements.
    @inlinable
    public var isEmpty: Bool { bitSet.isEmpty }

    /// Number of `true` positions in the mask.
    @inlinable
    public var trueCount: Int { bitSet.setBitCount }

    /// Number of `false` positions in the mask.
    @inlinable
    public var falseCount: Int { bitSet.count - bitSet.setBitCount }

    /// All values as a Swift `[Bool]` array.
    public var values: [Bool] {
        (0..<bitSet.count).map { bitSet[$0] }
    }

    /// Creates a mask that wraps an existing bit set.
    ///
    /// - Parameter bitSet: The bits to use; a set bit means `true`.
    @inlinable
    public init(bitSet: BitSet) {
        self.bitSet = bitSet
    }

    /// Creates a mask from an array of Boolean values.
    ///
    /// - Parameter values: The values, one per row.
    /// - Complexity: O(*n*), where *n* is `values.count`.
    public init(_ values: [Bool]) {
        var bits = BitSet(count: values.count)
        for (index, flag) in values.enumerated() where flag {
            bits[index] = true
        }
        self.bitSet = bits
    }

    /// Creates a mask of the given length with every element set to one value.
    ///
    /// - Parameters:
    ///   - count: The number of rows.
    ///   - value: The value for every element. The default is `false`.
    public init(count: Int, repeating value: Bool = false) {
        self.bitSet = BitSet(count: count, repeating: value)
    }

    /// Accesses the Boolean value at the given row.
    ///
    /// - Parameter index: A position in `0..<count`; out-of-range values trap.
    @inlinable
    public subscript(index: Int) -> Bool {
        bitSet[index]
    }

    /// Logical conjunction (`AND`) of two masks.
    @inlinable
    public func and(_ other: BooleanMask) -> BooleanMask {
        BooleanMask(bitSet: bitSet.bitwiseAnd(other.bitSet))
    }

    /// Logical disjunction (`OR`) of two masks.
    @inlinable
    public func or(_ other: BooleanMask) -> BooleanMask {
        BooleanMask(bitSet: bitSet.bitwiseOr(other.bitSet))
    }

    /// Logical negation (`NOT`) of the mask.
    @inlinable
    public var negated: BooleanMask {
        BooleanMask(bitSet: bitSet.inverted())
    }

    /// Returns the element-wise logical `AND` of two masks.
    @inlinable
    public static func && (lhs: BooleanMask, rhs: BooleanMask) -> BooleanMask {
        lhs.and(rhs)
    }

    /// Returns the element-wise logical `OR` of two masks.
    @inlinable
    public static func || (lhs: BooleanMask, rhs: BooleanMask) -> BooleanMask {
        lhs.or(rhs)
    }

    /// Returns the element-wise logical negation of a mask.
    @inlinable
    public static prefix func ! (mask: BooleanMask) -> BooleanMask {
        mask.negated
    }

    enum CodingKeys: String, CodingKey {
        case values
    }

    /// Creates a mask by decoding a `values` array of Booleans.
    ///
    /// - Parameter decoder: The decoder to read data from.
    /// - Throws: A decoding error if the `values` key is missing or malformed.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decoded = try container.decode([Bool].self, forKey: .values)
        self.init(decoded)
    }

    /// Encodes the mask as a `values` array of Booleans.
    ///
    /// - Parameter encoder: The encoder to write data to.
    /// - Throws: An encoding error if the values cannot be encoded.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(values, forKey: .values)
    }

    /// A one-line summary showing the mask's count, true count, and false count.
    public func describe() -> String {
        "BooleanMask(count: \(count), trueCount: \(trueCount), falseCount: \(falseCount))"
    }

    /// A textual representation of the mask, equal to ``describe()``.
    public var description: String { describe() }

    /// A textual representation of the mask for debugging, equal to ``description``.
    public var debugDescription: String { description }

    /// Writes the mask's ``description`` to the given output stream.
    ///
    /// - Parameter target: The stream to write to.
    public func write<Target: TextOutputStream>(to target: inout Target) {
        target.write(description)
    }
}

/// A high-performance contiguous bitset for null tracking and boolean filtering masks.
public struct BitSet: Equatable, Hashable, Sendable {
    @usableFromInline
    var words: ContiguousArray<UInt64>

    @usableFromInline
    var _count: Int

    /// Conservative fast-path flag: when `false`, all bits in `words` are guaranteed to be `0`.
    @usableFromInline
    var mayHaveTrueBits: Bool

    /// The number of bits in the set.
    @inlinable
    public var count: Int {
        _count
    }

    /// Creates a bit set of the given length with every bit set to one value.
    ///
    /// - Parameters:
    ///   - count: The number of bits.
    ///   - initialValue: The value for every bit. The default is `false`.
    /// - Complexity: O(*n* / 64), where *n* is `count`.
    @inlinable
    public init(count: Int, repeating initialValue: Bool = false) {
        self._count = count
        self.mayHaveTrueBits = initialValue && count > 0
        let wordCount = (count + 63) / 64
        let initialWord: UInt64 = initialValue ? ~0 : 0
        self.words = ContiguousArray(repeating: initialWord, count: wordCount)
        if initialValue && count > 0 {
            let remainder = count % 64
            if remainder > 0 {
                let mask = (UInt64(1) << remainder) - 1
                self.words[wordCount - 1] &= mask
            }
        }
    }

    /// Creates a bit set directly from precomputed 64-bit words.
    @inlinable
    init(words: ContiguousArray<UInt64>, count: Int, mayHaveTrueBits: Bool = true) {
        self.words = words
        self._count = count
        self.mayHaveTrueBits = mayHaveTrueBits && count > 0
    }

    /// Creates an empty bit set.
    @inlinable
    public init() {
        self._count = 0
        self.words = []
        self.mayHaveTrueBits = false
    }

    @inlinable
    public static func == (lhs: BitSet, rhs: BitSet) -> Bool {
        guard lhs._count == rhs._count else { return false }
        if !lhs.mayHaveTrueBits && !rhs.mayHaveTrueBits { return true }
        return lhs.words == rhs.words
    }

    @inlinable
    public func hash(into hasher: inout Hasher) {
        hasher.combine(_count)
        hasher.combine(words)
    }

    /// Accesses the bit at the given position.
    ///
    /// - Parameter index: A position in `0..<count`; out-of-range values trap.
    /// - Complexity: O(1).
    @inlinable
    public subscript(index: Int) -> Bool {
        @inline(__always)
        get {
            precondition(index >= 0 && index < _count, "BitSet index out of bounds: \(index)")
            if !mayHaveTrueBits { return false }
            let wordIndex = index / 64
            let bitIndex = index % 64
            return (words[wordIndex] & (UInt64(1) << bitIndex)) != 0
        }
        @inline(__always)
        set {
            precondition(index >= 0 && index < _count, "BitSet index out of bounds: \(index)")
            let wordIndex = index / 64
            let bitIndex = index % 64
            if newValue {
                words[wordIndex] |= (UInt64(1) << bitIndex)
                mayHaveTrueBits = true
            } else if mayHaveTrueBits {
                words[wordIndex] &= ~(UInt64(1) << bitIndex)
            }
        }
    }

    /// Appends a bit to the end of the set.
    ///
    /// - Parameter bit: The value to append.
    /// - Complexity: Amortized O(1).
    @inlinable
    public mutating func append(_ bit: Bool) {
        let bitIndex = _count % 64
        if bitIndex == 0 {
            words.append(bit ? 1 : 0)
        } else if bit {
            words[words.count - 1] |= (UInt64(1) << bitIndex)
        }
        if bit {
            mayHaveTrueBits = true
        }
        _count += 1
    }

    /// The number of bits that are set to `true`.
    ///
    /// - Complexity: O(*n* / 64), where *n* is `count` (or O(1) when all bits are known to be `false`).
    @inlinable
    public var setBitCount: Int {
        if !mayHaveTrueBits { return 0 }
        var total = 0
        for word in words {
            total += word.nonzeroBitCount
        }
        return total
    }

    /// A Boolean value indicating whether the set contains no bits.
    @inlinable
    public var isEmpty: Bool {
        _count == 0
    }

    /// A Boolean value indicating whether no bit is set; `true` for an empty set.
    @inlinable
    public var allFalse: Bool {
        if !mayHaveTrueBits { return true }
        for word in words where word != 0 {
            return false
        }
        return true
    }

    /// A Boolean value indicating whether every bit is set; `true` for an empty set.
    @inlinable
    public var allTrue: Bool {
        if _count == 0 { return true }
        if !mayHaveTrueBits { return false }
        return setBitCount == _count
    }

    /// Returns a new bit set containing the bits in the given range.
    ///
    /// - Parameter range: The positions to copy; must lie within `0..<count`.
    /// - Returns: A bit set of `range.count` bits.
    /// - Complexity: O(*k*), where *k* is `range.count`.
    @inlinable
    public func slice(range: Range<Int>) -> BitSet {
        let length = range.count
        if !mayHaveTrueBits || length == 0 {
            return BitSet(count: length)
        }
        var result = BitSet(count: length)
        for i in 0..<length {
            if self[range.lowerBound + i] {
                result[i] = true
            }
        }
        return result
    }

    /// Returns the bitwise `AND` of this set and another.
    ///
    /// - Parameter other: The set to combine with.
    /// - Returns: A set whose length is the shorter of the two counts.
    /// - Complexity: O(*n* / 64), where *n* is the result's count.
    @inlinable
    public func bitwiseAnd(_ other: BitSet) -> BitSet {
        let minCount = min(_count, other._count)
        if !self.mayHaveTrueBits || !other.mayHaveTrueBits || minCount == 0 {
            return BitSet(count: minCount)
        }
        let wordCount = (minCount + 63) / 64
        var nonZero: UInt64 = 0
        let outWords = ContiguousArray<UInt64>(unsafeUninitializedCapacity: wordCount) { buf, initCount in
            self.words.withUnsafeBufferPointer { lBuf in
                other.words.withUnsafeBufferPointer { rBuf in
                    guard let outBase = buf.baseAddress,
                          let lBase = lBuf.baseAddress,
                          let rBase = rBuf.baseAddress else { return }
                    for i in 0..<wordCount {
                        let w = lBase[i] & rBase[i]
                        outBase[i] = w
                        nonZero |= w
                    }
                }
            }
            initCount = wordCount
        }
        return BitSet(words: outWords, count: minCount, mayHaveTrueBits: nonZero != 0)
    }

    /// Returns the bitwise `OR` of this set and another.
    ///
    /// - Parameter other: The set to combine with.
    /// - Returns: A set whose length is the longer of the two counts; missing bits count as `false`.
    /// - Complexity: O(*n* / 64), where *n* is the result's count.
    @inlinable
    public func bitwiseOr(_ other: BitSet) -> BitSet {
        let maxCount = max(_count, other._count)
        if (!self.mayHaveTrueBits && !other.mayHaveTrueBits) || maxCount == 0 {
            return BitSet(count: maxCount)
        }
        if !self.mayHaveTrueBits && other._count == maxCount {
            return other
        }
        if !other.mayHaveTrueBits && self._count == maxCount {
            return self
        }
        let wordCount = (maxCount + 63) / 64
        var nonZero: UInt64 = 0
        let outWords = ContiguousArray<UInt64>(unsafeUninitializedCapacity: wordCount) { buf, initCount in
            let lCount = self.words.count
            let rCount = other.words.count
            for i in 0..<wordCount {
                let w1 = i < lCount ? self.words[i] : 0
                let w2 = i < rCount ? other.words[i] : 0
                let w = w1 | w2
                buf[i] = w
                nonZero |= w
            }
            initCount = wordCount
        }
        return BitSet(words: outWords, count: maxCount, mayHaveTrueBits: nonZero != 0)
    }

    /// Returns a copy of the set with every bit flipped.
    ///
    /// - Complexity: O(*n* / 64), where *n* is `count`.
    @inlinable
    public func inverted() -> BitSet {
        if _count == 0 { return BitSet() }
        var result = BitSet(count: _count)
        for i in 0..<words.count {
            result.words[i] = ~self.words[i]
        }
        let remainder = _count % 64
        if remainder > 0 {
            let mask = (UInt64(1) << remainder) - 1
            result.words[words.count - 1] &= mask
        }
        result.mayHaveTrueBits = true
        return result
    }

    /// Returns the positions of all set bits in ascending order.
    ///
    /// - Complexity: O(*n* / 64 + *k*), where *k* is `setBitCount`.
    @inlinable
    public func trueIndices() -> [Int] {
        if !mayHaveTrueBits { return [] }
        let countTrue = setBitCount
        if countTrue == 0 { return [] }
        let limit = _count
        return Array<Int>(unsafeUninitializedCapacity: countTrue) { buffer, initializedCount in
            var writeIdx = 0
            words.withUnsafeBufferPointer { wBuf in
                guard let wBase = wBuf.baseAddress, let outBase = buffer.baseAddress else { return }
                let wCount = wBuf.count
                for wordIndex in 0..<wCount {
                    var word = wBase[wordIndex]
                    let baseIndex = wordIndex * 64
                    while word != 0 {
                        let bit = word.trailingZeroBitCount
                        let index = baseIndex + bit
                        if index < limit {
                            outBase[writeIdx] = index
                            writeIdx += 1
                        }
                        word &= word - 1
                    }
                }
            }
            initializedCount = writeIdx
        }
    }
}
