import Foundation

/// A high-performance contiguous bitset for null tracking and boolean filtering masks.
public struct BitSet: Equatable, Hashable, Sendable {
    @usableFromInline
    var words: ContiguousArray<UInt64>

    @usableFromInline
    var _count: Int

    @inlinable
    public var count: Int {
        _count
    }

    @inlinable
    public init(count: Int, repeating initialValue: Bool = false) {
        self._count = count
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

    @inlinable
    public init() {
        self._count = 0
        self.words = []
    }

    @inlinable
    public subscript(index: Int) -> Bool {
        @inline(__always)
        get {
            precondition(index >= 0 && index < _count, "BitSet index out of bounds: \(index)")
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
            } else {
                words[wordIndex] &= ~(UInt64(1) << bitIndex)
            }
        }
    }

    @inlinable
    public mutating func append(_ bit: Bool) {
        let bitIndex = _count % 64
        if bitIndex == 0 {
            words.append(bit ? 1 : 0)
        } else if bit {
            words[words.count - 1] |= (UInt64(1) << bitIndex)
        }
        _count += 1
    }

    @inlinable
    public var setBitCount: Int {
        var total = 0
        for word in words {
            total += word.nonzeroBitCount
        }
        return total
    }

    @inlinable
    public var isEmpty: Bool {
        _count == 0
    }

    @inlinable
    public var allFalse: Bool {
        setBitCount == 0
    }

    @inlinable
    public var allTrue: Bool {
        setBitCount == _count
    }


    @inlinable
    public func slice(range: Range<Int>) -> BitSet {
        let length = range.count
        var result = BitSet(count: length)
        for i in 0..<length {
            if self[range.lowerBound + i] {
                result[i] = true
            }
        }
        return result
    }

    @inlinable
    public func bitwiseAnd(_ other: BitSet) -> BitSet {
        let minCount = min(_count, other._count)
        var result = BitSet(count: minCount)
        let wordCount = (minCount + 63) / 64
        for i in 0..<wordCount {
            result.words[i] = self.words[i] & other.words[i]
        }
        return result
    }

    @inlinable
    public func bitwiseOr(_ other: BitSet) -> BitSet {
        let maxCount = max(_count, other._count)
        var result = BitSet(count: maxCount)
        let wordCount = (maxCount + 63) / 64
        for i in 0..<wordCount {
            let w1 = i < self.words.count ? self.words[i] : 0
            let w2 = i < other.words.count ? other.words[i] : 0
            result.words[i] = w1 | w2
        }
        return result
    }

    @inlinable
    public func inverted() -> BitSet {
        var result = BitSet(count: _count)
        for i in 0..<words.count {
            result.words[i] = ~self.words[i]
        }
        if _count > 0 {
            let remainder = _count % 64
            if remainder > 0 {
                let mask = (UInt64(1) << remainder) - 1
                result.words[words.count - 1] &= mask
            }
        }
        return result
    }

    /// Fast extraction of indices where bit is 1.
    @inlinable
    public func trueIndices() -> [Int] {
        var indices: [Int] = []
        indices.reserveCapacity(setBitCount)
        for (wordIndex, var word) in words.enumerated() {
            let baseIndex = wordIndex * 64
            while word != 0 {
                let bit = word.trailingZeroBitCount
                let index = baseIndex + bit
                if index < _count {
                    indices.append(index)
                }
                word &= word - 1
            }
        }
        return indices
    }
}
