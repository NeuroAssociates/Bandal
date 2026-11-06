import Foundation

/// Protocol representing a typed columnar buffer (Apache Arrow / Polars model).
public protocol AnyColumn: Sendable {
    /// The column's name.
    var name: String { get }

    /// The number of elements, including nulls.
    var count: Int { get }

    /// A bit set whose set bits mark null (missing) elements.
    var nullBitmap: BitSet { get }

    /// The kind of data the column holds, or `.unknown` if it has no valid elements.
    var dataType: DataType { get }

    /// Returns the element at the given position as a ``DataValue``.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: The element, or `.missing` if it is null.
    func value(at index: Int) -> DataValue

    /// Returns every element as a ``DataValue``, using `.missing` for nulls.
    func toDataValues() -> [DataValue]

    /// Returns a new column containing the elements at the given positions, in order.
    ///
    /// - Parameter indices: Positions in `0..<count`; repeats are allowed.
    func slice(at indices: [Int]) -> any AnyColumn

    /// Returns a new column containing the elements at the given positions, with nulls for `nil` entries.
    ///
    /// - Parameter indices: Positions in `0..<count`, or `nil` to produce a null element.
    func slice(atOptional indices: [Int?]) -> any AnyColumn

    /// Returns a new column containing the elements in the given range.
    ///
    /// - Parameter range: A range within `0..<count`.
    func slice(range: Range<Int>) -> any AnyColumn

    /// Returns a new column containing the elements whose mask bit is set.
    ///
    /// - Parameter mask: A bit set with at least `count` bits.
    func slice(mask: BitSet) -> any AnyColumn

    /// Returns a copy of the column with a different name.
    ///
    /// - Parameter newName: The name for the copy.
    func renamed(to newName: String) -> any AnyColumn
}

extension AnyColumn {
    /// Returns a Boolean value indicating whether the element at the given position is null.
    ///
    /// - Parameter index: A position in `0..<count`.
    @inlinable
    public func isNull(at index: Int) -> Bool {
        nullBitmap[index]
    }

    /// Returns a Boolean value indicating whether the element at the given position is not null.
    ///
    /// - Parameter index: A position in `0..<count`.
    @inlinable
    public func isValid(at index: Int) -> Bool {
        !nullBitmap[index]
    }

    /// A Boolean value indicating whether the column contains at least one null.
    @inlinable
    public var hasNulls: Bool {
        !nullBitmap.allFalse
    }

    /// The number of null elements.
    @inlinable
    public var nullCount: Int {
        nullBitmap.setBitCount
    }

    /// The number of elements that are not null.
    @inlinable
    public var validCount: Int {
        count - nullCount
    }
}

// MARK: - Float64Column

/// A column of `Double` values with a null bitmap.
public final class Float64Column: AnyColumn, @unchecked Sendable {
    /// The column's name.
    public let name: String

    /// The stored values; entries at null positions are placeholders.
    public var values: ContiguousArray<Double>

    /// A bit set whose set bits mark null elements.
    public var nullBitmap: BitSet

    /// The number of elements, including nulls.
    @inlinable
    public var count: Int { values.count }

    /// `.number` if the column has any valid element; otherwise `.unknown`.
    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .number }

    /// Creates a column from raw storage.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - values: The stored values.
    ///   - nullBitmap: Set bits mark null positions in `values`.
    public init(name: String, values: ContiguousArray<Double>, nullBitmap: BitSet) {
        self.name = name
        self.values = values
        self.nullBitmap = nullBitmap
    }

    /// Creates a column with no nulls from an array of numbers.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - numbers: The values.
    public convenience init(name: String, numbers: [Double]) {
        self.init(name: name, values: ContiguousArray(numbers), nullBitmap: BitSet(count: numbers.count))
    }

    /// Creates a column from ``DataValue``s, treating every non-number as null.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - dataValues: The values to convert.
    public convenience init(name: String, dataValues: [DataValue]) {
        var vals = ContiguousArray<Double>()
        vals.reserveCapacity(dataValues.count)
        var nulls = BitSet(count: dataValues.count)
        for (i, v) in dataValues.enumerated() {
            if case .number(let n) = v {
                vals.append(n)
            } else {
                vals.append(0.0)
                nulls[i] = true
            }
        }
        self.init(name: name, values: vals, nullBitmap: nulls)
    }

    /// Returns the element at the given position as a ``DataValue``.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: `.number`, or `.missing` if the element is null.
    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .number(values[index])
    }

    /// Returns every element as a ``DataValue``, using `.missing` for nulls.
    public func toDataValues() -> [DataValue] {
        var result: [DataValue] = []
        result.reserveCapacity(count)
        for i in 0..<count {
            if nullBitmap[i] {
                result.append(.missing)
            } else {
                result.append(.number(values[i]))
            }
        }
        return result
    }

    /// Returns a new column containing the elements at the given positions, in order.
    ///
    /// - Parameter indices: Positions in `0..<count`; repeats are allowed.
    public func slice(at indices: [Int]) -> any AnyColumn {
        let n = indices.count
        if !hasNulls {
            let newValues = ContiguousArray<Double>(unsafeUninitializedCapacity: n) { outBuf, initCount in
                values.withUnsafeBufferPointer { inBuf in
                    indices.withUnsafeBufferPointer { idxBuf in
                        guard let inBase = inBuf.baseAddress,
                              let idxBase = idxBuf.baseAddress,
                              let outBase = outBuf.baseAddress else { return }
                        for k in 0..<n {
                            outBase[k] = inBase[idxBase[k]]
                        }
                    }
                }
                initCount = n
            }
            return Float64Column(name: name, values: newValues, nullBitmap: BitSet(count: n))
        }
        var newValues = ContiguousArray<Double>()
        newValues.reserveCapacity(n)
        var newBitmap = BitSet(count: n)
        for (newIdx, i) in indices.enumerated() {
            newValues.append(values[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return Float64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements at the given positions, with nulls for `nil` entries.
    ///
    /// - Parameter indices: Positions in `0..<count`, or `nil` to produce a null element.
    public func slice(atOptional indices: [Int?]) -> any AnyColumn {
        var newValues = ContiguousArray<Double>()
        newValues.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (i, opt) in indices.enumerated() {
            if let idx = opt, !nullBitmap[idx] {
                newValues.append(values[idx])
            } else {
                newValues.append(0.0)
                newBitmap[i] = true
            }
        }
        return Float64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements in the given range.
    ///
    /// - Parameter range: A range within `0..<count`.
    public func slice(range: Range<Int>) -> any AnyColumn {
        let newValues = ContiguousArray(values[range])
        let newBitmap = nullBitmap.slice(range: range)
        return Float64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements whose mask bit is set.
    ///
    /// - Parameter mask: A bit set with at least `count` bits.
    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
        if targetCount == 0 {
            return Float64Column(name: name, values: [], nullBitmap: BitSet(count: 0))
        }
        if !hasNulls {
            let n = count
            let newValues = ContiguousArray<Double>(unsafeUninitializedCapacity: targetCount) { outBuf, initCount in
                values.withUnsafeBufferPointer { inBuf in
                    mask.words.withUnsafeBufferPointer { wordBuf in
                        guard let inBase = inBuf.baseAddress,
                              let wordBase = wordBuf.baseAddress,
                              let outBase = outBuf.baseAddress else { return }
                        var outIdx = 0
                        let fullWords = n >> 6
                        let wordLimit = Swift.min(fullWords, wordBuf.count)
                        for wIdx in 0..<wordLimit {
                            var w = wordBase[wIdx]
                            let baseRow = wIdx &<< 6
                            while w != 0 {
                                let bit = w.trailingZeroBitCount
                                outBase[outIdx] = inBase[baseRow + bit]
                                outIdx += 1
                                w &= w &- 1
                            }
                        }
                        if fullWords < wordBuf.count {
                            var w = wordBase[fullWords]
                            let baseRow = fullWords &<< 6
                            while w != 0 {
                                let bit = w.trailingZeroBitCount
                                let row = baseRow + bit
                                if row < n {
                                    outBase[outIdx] = inBase[row]
                                    outIdx += 1
                                }
                                w &= w &- 1
                            }
                        }
                    }
                }
                initCount = targetCount
            }
            return Float64Column(name: name, values: newValues, nullBitmap: BitSet(count: targetCount))
        }
        var newValues = ContiguousArray<Double>()
        newValues.reserveCapacity(targetCount)
        var newBitmap = BitSet(count: targetCount)
        var newIdx = 0
        for i in 0..<count where mask[i] {
            newValues.append(values[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
            newIdx += 1
        }
        return Float64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    /// Returns a copy of the column with a different name.
    ///
    /// - Parameter newName: The name for the copy.
    public func renamed(to newName: String) -> any AnyColumn {
        Float64Column(name: newName, values: values, nullBitmap: nullBitmap)
    }
}

// MARK: - Int64Column

/// A column of `Int64` values with a null bitmap.
public final class Int64Column: AnyColumn, @unchecked Sendable {
    /// The column's name.
    public let name: String

    /// The stored values; entries at null positions are placeholders.
    public var values: ContiguousArray<Int64>

    /// A bit set whose set bits mark null elements.
    public var nullBitmap: BitSet

    /// The number of elements, including nulls.
    @inlinable
    public var count: Int { values.count }

    /// `.number` if the column has any valid element; otherwise `.unknown`.
    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .number }

    /// Creates a column from raw storage.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - values: The stored values.
    ///   - nullBitmap: Set bits mark null positions in `values`.
    public init(name: String, values: ContiguousArray<Int64>, nullBitmap: BitSet) {
        self.name = name
        self.values = values
        self.nullBitmap = nullBitmap
    }

    /// Returns the element at the given position as a ``DataValue``.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: `.number` holding the value converted to `Double`, or `.missing` if the element is null.
    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .number(Double(values[index]))
    }

    /// Returns every element as a ``DataValue``, using `.missing` for nulls.
    public func toDataValues() -> [DataValue] {
        var result: [DataValue] = []
        result.reserveCapacity(count)
        for i in 0..<count {
            if nullBitmap[i] {
                result.append(.missing)
            } else {
                result.append(.number(Double(values[i])))
            }
        }
        return result
    }

    /// Returns a new column containing the elements at the given positions, in order.
    ///
    /// - Parameter indices: Positions in `0..<count`; repeats are allowed.
    public func slice(at indices: [Int]) -> any AnyColumn {
        let n = indices.count
        if !hasNulls {
            let newValues = ContiguousArray<Int64>(unsafeUninitializedCapacity: n) { outBuf, initCount in
                values.withUnsafeBufferPointer { inBuf in
                    indices.withUnsafeBufferPointer { idxBuf in
                        guard let inBase = inBuf.baseAddress,
                              let idxBase = idxBuf.baseAddress,
                              let outBase = outBuf.baseAddress else { return }
                        for k in 0..<n {
                            outBase[k] = inBase[idxBase[k]]
                        }
                    }
                }
                initCount = n
            }
            return Int64Column(name: name, values: newValues, nullBitmap: BitSet(count: n))
        }
        var newValues = ContiguousArray<Int64>()
        newValues.reserveCapacity(n)
        var newBitmap = BitSet(count: n)
        for (newIdx, i) in indices.enumerated() {
            newValues.append(values[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return Int64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements at the given positions, with nulls for `nil` entries.
    ///
    /// - Parameter indices: Positions in `0..<count`, or `nil` to produce a null element.
    public func slice(atOptional indices: [Int?]) -> any AnyColumn {
        var newValues = ContiguousArray<Int64>()
        newValues.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (i, opt) in indices.enumerated() {
            if let idx = opt, !nullBitmap[idx] {
                newValues.append(values[idx])
            } else {
                newValues.append(0)
                newBitmap[i] = true
            }
        }
        return Int64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements in the given range.
    ///
    /// - Parameter range: A range within `0..<count`.
    public func slice(range: Range<Int>) -> any AnyColumn {
        let newValues = ContiguousArray(values[range])
        let newBitmap = nullBitmap.slice(range: range)
        return Int64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements whose mask bit is set.
    ///
    /// - Parameter mask: A bit set with at least `count` bits.
    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
        if targetCount == 0 {
            return Int64Column(name: name, values: [], nullBitmap: BitSet(count: 0))
        }
        if !hasNulls {
            let n = count
            let newValues = ContiguousArray<Int64>(unsafeUninitializedCapacity: targetCount) { outBuf, initCount in
                values.withUnsafeBufferPointer { inBuf in
                    mask.words.withUnsafeBufferPointer { wordBuf in
                        guard let inBase = inBuf.baseAddress,
                              let wordBase = wordBuf.baseAddress,
                              let outBase = outBuf.baseAddress else { return }
                        var outIdx = 0
                        let fullWords = n >> 6
                        let wordLimit = Swift.min(fullWords, wordBuf.count)
                        for wIdx in 0..<wordLimit {
                            var w = wordBase[wIdx]
                            let baseRow = wIdx &<< 6
                            while w != 0 {
                                let bit = w.trailingZeroBitCount
                                outBase[outIdx] = inBase[baseRow + bit]
                                outIdx += 1
                                w &= w &- 1
                            }
                        }
                        if fullWords < wordBuf.count {
                            var w = wordBase[fullWords]
                            let baseRow = fullWords &<< 6
                            while w != 0 {
                                let bit = w.trailingZeroBitCount
                                let row = baseRow + bit
                                if row < n {
                                    outBase[outIdx] = inBase[row]
                                    outIdx += 1
                                }
                                w &= w &- 1
                            }
                        }
                    }
                }
                initCount = targetCount
            }
            return Int64Column(name: name, values: newValues, nullBitmap: BitSet(count: targetCount))
        }
        var newValues = ContiguousArray<Int64>()
        newValues.reserveCapacity(targetCount)
        var newBitmap = BitSet(count: targetCount)
        var newIdx = 0
        for i in 0..<count where mask[i] {
            newValues.append(values[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
            newIdx += 1
        }
        return Int64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    /// Returns a copy of the column with a different name.
    ///
    /// - Parameter newName: The name for the copy.
    public func renamed(to newName: String) -> any AnyColumn {
        Int64Column(name: newName, values: values, nullBitmap: nullBitmap)
    }
}

// MARK: - StringColumn

/// A column of `String` values with a null bitmap.
public final class StringColumn: AnyColumn, @unchecked Sendable {
    /// The column's name.
    public let name: String

    /// The stored strings; entries at null positions are empty placeholders.
    public var stringData: ContiguousArray<String>

    /// A bit set whose set bits mark null elements.
    public var nullBitmap: BitSet

    /// The number of elements, including nulls.
    @inlinable
    public var count: Int { stringData.count }

    /// `.string` if the column has any valid element; otherwise `.unknown`.
    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .string }

    /// Creates a column from raw storage.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - stringData: The stored strings.
    ///   - nullBitmap: Set bits mark null positions in `stringData`.
    public init(name: String, stringData: ContiguousArray<String>, nullBitmap: BitSet) {
        self.name = name
        self.stringData = stringData
        self.nullBitmap = nullBitmap
    }

    /// Creates a column with no nulls from an array of strings.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - strings: The values.
    public convenience init(name: String, strings: [String]) {
        self.init(name: name, stringData: ContiguousArray(strings), nullBitmap: BitSet(count: strings.count))
    }

    /// Creates a column from ``DataValue``s, treating every non-string as null.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - dataValues: The values to convert.
    public convenience init(name: String, dataValues: [DataValue]) {
        var strings = ContiguousArray<String>()
        strings.reserveCapacity(dataValues.count)
        var nulls = BitSet(count: dataValues.count)
        for (i, v) in dataValues.enumerated() {
            if case .string(let s) = v {
                strings.append(s)
            } else {
                strings.append("")
                nulls[i] = true
            }
        }
        self.init(name: name, stringData: strings, nullBitmap: nulls)
    }

    /// Returns the string at the given position.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: The stored string; an empty placeholder if the element is null.
    @inlinable
    public func string(at index: Int) -> String {
        stringData[index]
    }

    /// Returns the element at the given position as a ``DataValue``.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: `.string`, or `.missing` if the element is null.
    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .string(string(at: index))
    }

    /// Returns every element as a ``DataValue``, using `.missing` for nulls.
    public func toDataValues() -> [DataValue] {
        var result: [DataValue] = []
        result.reserveCapacity(count)
        for i in 0..<count {
            if nullBitmap[i] {
                result.append(.missing)
            } else {
                result.append(.string(stringData[i]))
            }
        }
        return result
    }

    /// Returns a new column containing the elements at the given positions, in order.
    ///
    /// - Parameter indices: Positions in `0..<count`; repeats are allowed.
    public func slice(at indices: [Int]) -> any AnyColumn {
        var newStrings = ContiguousArray<String>()
        newStrings.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (newIdx, i) in indices.enumerated() {
            newStrings.append(stringData[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return StringColumn(name: name, stringData: newStrings, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements at the given positions, with nulls for `nil` entries.
    ///
    /// - Parameter indices: Positions in `0..<count`, or `nil` to produce a null element.
    public func slice(atOptional indices: [Int?]) -> any AnyColumn {
        var newStrings = ContiguousArray<String>()
        newStrings.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (i, opt) in indices.enumerated() {
            if let idx = opt, !nullBitmap[idx] {
                newStrings.append(stringData[idx])
            } else {
                newStrings.append("")
                newBitmap[i] = true
            }
        }
        return StringColumn(name: name, stringData: newStrings, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements in the given range.
    ///
    /// - Parameter range: A range within `0..<count`.
    public func slice(range: Range<Int>) -> any AnyColumn {
        let newStrings = ContiguousArray(stringData[range])
        let newBitmap = nullBitmap.slice(range: range)
        return StringColumn(name: name, stringData: newStrings, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements whose mask bit is set.
    ///
    /// - Parameter mask: A bit set with at least `count` bits.
    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
        var newStrings = ContiguousArray<String>()
        newStrings.reserveCapacity(targetCount)
        var newBitmap = BitSet(count: targetCount)
        var newIdx = 0
        for i in 0..<count where mask[i] {
            newStrings.append(stringData[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
            newIdx += 1
        }
        return StringColumn(name: name, stringData: newStrings, nullBitmap: newBitmap)
    }

    /// Returns a copy of the column with a different name.
    ///
    /// - Parameter newName: The name for the copy.
    public func renamed(to newName: String) -> any AnyColumn {
        StringColumn(name: newName, stringData: stringData, nullBitmap: nullBitmap)
    }
}

// MARK: - CategoricalColumn

/// A dictionary-encoded column of strings, storing each distinct string once and a `UInt32` code per element.
public final class CategoricalColumn: AnyColumn, @unchecked Sendable {
    /// The column's name.
    public let name: String

    /// The distinct strings; each code indexes into this array.
    public var dictionary: [String]

    /// The per-element indices into ``dictionary``; entries at null positions are placeholders.
    public var codes: ContiguousArray<UInt32>

    /// A bit set whose set bits mark null elements.
    public var nullBitmap: BitSet

    /// The number of elements, including nulls.
    @inlinable
    public var count: Int { codes.count }

    /// `.string` if the column has any valid element; otherwise `.unknown`.
    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .string }

    /// Creates a column from raw storage.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - dictionary: The distinct strings.
    ///   - codes: Indices into `dictionary`, one per element.
    ///   - nullBitmap: Set bits mark null positions in `codes`.
    public init(name: String, dictionary: [String], codes: ContiguousArray<UInt32>, nullBitmap: BitSet) {
        self.name = name
        self.dictionary = dictionary
        self.codes = codes
        self.nullBitmap = nullBitmap
    }

    /// Creates a column with no nulls by dictionary-encoding an array of strings.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - strings: The values; the dictionary lists them in first-seen order.
    public convenience init(name: String, strings: [String]) {
        var dict: [String] = []
        var lookup: [String: UInt32] = [:]
        var codes = ContiguousArray<UInt32>()
        codes.reserveCapacity(strings.count)

        for s in strings {
            if let c = lookup[s] {
                codes.append(c)
            } else {
                let c = UInt32(dict.count)
                dict.append(s)
                lookup[s] = c
                codes.append(c)
            }
        }
        self.init(name: name, dictionary: dict, codes: codes, nullBitmap: BitSet(count: strings.count))
    }

    /// Returns the string at the given position.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: The dictionary entry for the element's code.
    @inlinable
    public func string(at index: Int) -> String {
        dictionary[Int(codes[index])]
    }

    /// Returns the element at the given position as a ``DataValue``.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: `.string`, or `.missing` if the element is null.
    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .string(string(at: index))
    }

    /// Returns every element as a ``DataValue``, using `.missing` for nulls.
    public func toDataValues() -> [DataValue] {
        var result: [DataValue] = []
        result.reserveCapacity(count)
        for i in 0..<count {
            if nullBitmap[i] {
                result.append(.missing)
            } else {
                result.append(.string(dictionary[Int(codes[i])]))
            }
        }
        return result
    }

    /// Returns a new column containing the elements at the given positions, in order.
    ///
    /// The result shares this column's dictionary.
    ///
    /// - Parameter indices: Positions in `0..<count`; repeats are allowed.
    public func slice(at indices: [Int]) -> any AnyColumn {
        let n = indices.count
        if !hasNulls {
            let newCodes = ContiguousArray<UInt32>(unsafeUninitializedCapacity: n) { outBuf, initCount in
                codes.withUnsafeBufferPointer { inBuf in
                    indices.withUnsafeBufferPointer { idxBuf in
                        guard let inBase = inBuf.baseAddress,
                              let idxBase = idxBuf.baseAddress,
                              let outBase = outBuf.baseAddress else { return }
                        for k in 0..<n {
                            outBase[k] = inBase[idxBase[k]]
                        }
                    }
                }
                initCount = n
            }
            return CategoricalColumn(name: name, dictionary: dictionary, codes: newCodes, nullBitmap: BitSet(count: n))
        }
        var newCodes = ContiguousArray<UInt32>()
        newCodes.reserveCapacity(n)
        var newBitmap = BitSet(count: n)
        for (newIdx, i) in indices.enumerated() {
            newCodes.append(codes[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return CategoricalColumn(name: name, dictionary: dictionary, codes: newCodes, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements at the given positions, with nulls for `nil` entries.
    ///
    /// - Parameter indices: Positions in `0..<count`, or `nil` to produce a null element.
    public func slice(atOptional indices: [Int?]) -> any AnyColumn {
        var newCodes = ContiguousArray<UInt32>()
        newCodes.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (i, opt) in indices.enumerated() {
            if let idx = opt, !nullBitmap[idx] {
                newCodes.append(codes[idx])
            } else {
                newCodes.append(0)
                newBitmap[i] = true
            }
        }
        return CategoricalColumn(name: name, dictionary: dictionary, codes: newCodes, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements in the given range.
    ///
    /// - Parameter range: A range within `0..<count`.
    public func slice(range: Range<Int>) -> any AnyColumn {
        let newCodes = ContiguousArray(codes[range])
        let newBitmap = nullBitmap.slice(range: range)
        return CategoricalColumn(name: name, dictionary: dictionary, codes: newCodes, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements whose mask bit is set.
    ///
    /// - Parameter mask: A bit set with at least `count` bits.
    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
        if targetCount == 0 {
            return CategoricalColumn(name: name, dictionary: dictionary, codes: [], nullBitmap: BitSet(count: 0))
        }
        if !hasNulls {
            let n = count
            let newCodes = ContiguousArray<UInt32>(unsafeUninitializedCapacity: targetCount) { outBuf, initCount in
                codes.withUnsafeBufferPointer { inBuf in
                    mask.words.withUnsafeBufferPointer { wordBuf in
                        guard let inBase = inBuf.baseAddress,
                              let wordBase = wordBuf.baseAddress,
                              let outBase = outBuf.baseAddress else { return }
                        var outIdx = 0
                        let fullWords = n >> 6
                        let wordLimit = Swift.min(fullWords, wordBuf.count)
                        for wIdx in 0..<wordLimit {
                            var w = wordBase[wIdx]
                            let baseRow = wIdx &<< 6
                            while w != 0 {
                                let bit = w.trailingZeroBitCount
                                outBase[outIdx] = inBase[baseRow + bit]
                                outIdx += 1
                                w &= w &- 1
                            }
                        }
                        if fullWords < wordBuf.count {
                            var w = wordBase[fullWords]
                            let baseRow = fullWords &<< 6
                            while w != 0 {
                                let bit = w.trailingZeroBitCount
                                let row = baseRow + bit
                                if row < n {
                                    outBase[outIdx] = inBase[row]
                                    outIdx += 1
                                }
                                w &= w &- 1
                            }
                        }
                    }
                }
                initCount = targetCount
            }
            return CategoricalColumn(name: name, dictionary: dictionary, codes: newCodes, nullBitmap: BitSet(count: targetCount))
        }
        var newCodes = ContiguousArray<UInt32>()
        newCodes.reserveCapacity(targetCount)
        var newBitmap = BitSet(count: targetCount)
        var newIdx = 0
        for i in 0..<count where mask[i] {
            newCodes.append(codes[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
            newIdx += 1
        }
        return CategoricalColumn(name: name, dictionary: dictionary, codes: newCodes, nullBitmap: newBitmap)
    }

    /// Returns a copy of the column with a different name.
    ///
    /// - Parameter newName: The name for the copy.
    public func renamed(to newName: String) -> any AnyColumn {
        CategoricalColumn(name: newName, dictionary: dictionary, codes: codes, nullBitmap: nullBitmap)
    }
}

// MARK: - BooleanColumn

/// A column of Boolean values stored as a bit set, with a null bitmap.
public final class BooleanColumn: AnyColumn, @unchecked Sendable {
    /// The column's name.
    public let name: String

    /// The stored values, one bit per element; bits at null positions are placeholders.
    public var bitmap: BitSet

    /// A bit set whose set bits mark null elements.
    public var nullBitmap: BitSet

    /// The number of elements, including nulls.
    @inlinable
    public var count: Int { bitmap.count }

    /// `.boolean` if the column has any valid element; otherwise `.unknown`.
    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .boolean }

    /// Creates a column from raw storage.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - bitmap: The values, one bit per element.
    ///   - nullBitmap: Set bits mark null positions in `bitmap`.
    public init(name: String, bitmap: BitSet, nullBitmap: BitSet) {
        self.name = name
        self.bitmap = bitmap
        self.nullBitmap = nullBitmap
    }

    /// Creates a column with no nulls from an array of Booleans.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - booleans: The values.
    public convenience init(name: String, booleans: [Bool]) {
        var b = BitSet(count: booleans.count)
        for (i, val) in booleans.enumerated() where val {
            b[i] = true
        }
        self.init(name: name, bitmap: b, nullBitmap: BitSet(count: booleans.count))
    }

    /// Returns the Boolean at the given position.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: The stored bit; a placeholder if the element is null.
    @inlinable
    public func bool(at index: Int) -> Bool {
        bitmap[index]
    }

    /// Returns the element at the given position as a ``DataValue``.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: `.boolean`, or `.missing` if the element is null.
    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .boolean(bitmap[index])
    }

    /// Returns every element as a ``DataValue``, using `.missing` for nulls.
    public func toDataValues() -> [DataValue] {
        var result: [DataValue] = []
        result.reserveCapacity(count)
        for i in 0..<count {
            if nullBitmap[i] {
                result.append(.missing)
            } else {
                result.append(.boolean(bitmap[i]))
            }
        }
        return result
    }

    /// Returns a new column containing the elements at the given positions, in order.
    ///
    /// - Parameter indices: Positions in `0..<count`; repeats are allowed.
    public func slice(at indices: [Int]) -> any AnyColumn {
        var newBits = BitSet(count: indices.count)
        var newNulls = BitSet(count: indices.count)
        for (newIdx, i) in indices.enumerated() {
            if bitmap[i] { newBits[newIdx] = true }
            if nullBitmap[i] { newNulls[newIdx] = true }
        }
        return BooleanColumn(name: name, bitmap: newBits, nullBitmap: newNulls)
    }

    /// Returns a new column containing the elements at the given positions, with nulls for `nil` entries.
    ///
    /// - Parameter indices: Positions in `0..<count`, or `nil` to produce a null element.
    public func slice(atOptional indices: [Int?]) -> any AnyColumn {
        var newBits = BitSet(count: indices.count)
        var newNulls = BitSet(count: indices.count)
        for (i, opt) in indices.enumerated() {
            if let idx = opt, !nullBitmap[idx] {
                if bitmap[idx] { newBits[i] = true }
            } else {
                newNulls[i] = true
            }
        }
        return BooleanColumn(name: name, bitmap: newBits, nullBitmap: newNulls)
    }

    /// Returns a new column containing the elements in the given range.
    ///
    /// - Parameter range: A range within `0..<count`.
    public func slice(range: Range<Int>) -> any AnyColumn {
        BooleanColumn(name: name, bitmap: bitmap.slice(range: range), nullBitmap: nullBitmap.slice(range: range))
    }

    /// Returns a new column containing the elements whose mask bit is set.
    ///
    /// - Parameter mask: A bit set with at least `count` bits.
    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
        var newBits = BitSet(count: targetCount)
        var newNulls = BitSet(count: targetCount)
        var newIdx = 0
        for i in 0..<count where mask[i] {
            if bitmap[i] { newBits[newIdx] = true }
            if nullBitmap[i] { newNulls[newIdx] = true }
            newIdx += 1
        }
        return BooleanColumn(name: name, bitmap: newBits, nullBitmap: newNulls)
    }

    /// Returns a copy of the column with a different name.
    ///
    /// - Parameter newName: The name for the copy.
    public func renamed(to newName: String) -> any AnyColumn {
        BooleanColumn(name: newName, bitmap: bitmap, nullBitmap: nullBitmap)
    }
}

// MARK: - DateColumn

/// A column of dates stored as time intervals since the reference date, with a null bitmap.
public final class DateColumn: AnyColumn, @unchecked Sendable {
    /// The column's name.
    public let name: String

    /// The stored values in seconds since `Date.timeIntervalSinceReferenceDate`; entries at null positions are placeholders.
    public var timeIntervals: ContiguousArray<Double>

    /// A bit set whose set bits mark null elements.
    public var nullBitmap: BitSet

    /// The number of elements, including nulls.
    @inlinable
    public var count: Int { timeIntervals.count }

    /// `.date` if the column has any valid element; otherwise `.unknown`.
    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .date }

    /// Creates a column from raw storage.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - timeIntervals: Seconds since the reference date, one per element.
    ///   - nullBitmap: Set bits mark null positions in `timeIntervals`.
    public init(name: String, timeIntervals: ContiguousArray<Double>, nullBitmap: BitSet) {
        self.name = name
        self.timeIntervals = timeIntervals
        self.nullBitmap = nullBitmap
    }

    /// Creates a column with no nulls from an array of dates.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - dates: The values.
    public convenience init(name: String, dates: [Date]) {
        var intervals = ContiguousArray<Double>()
        intervals.reserveCapacity(dates.count)
        for d in dates {
            intervals.append(d.timeIntervalSinceReferenceDate)
        }
        self.init(name: name, timeIntervals: intervals, nullBitmap: BitSet(count: dates.count))
    }

    /// Returns the date at the given position.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: The stored date; a placeholder if the element is null.
    @inlinable
    public func date(at index: Int) -> Date {
        Date(timeIntervalSinceReferenceDate: timeIntervals[index])
    }

    /// Returns the element at the given position as a ``DataValue``.
    ///
    /// - Parameter index: A position in `0..<count`.
    /// - Returns: `.date`, or `.missing` if the element is null.
    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .date(date(at: index))
    }

    /// Returns every element as a ``DataValue``, using `.missing` for nulls.
    public func toDataValues() -> [DataValue] {
        var result: [DataValue] = []
        result.reserveCapacity(count)
        for i in 0..<count {
            if nullBitmap[i] {
                result.append(.missing)
            } else {
                result.append(.date(Date(timeIntervalSinceReferenceDate: timeIntervals[i])))
            }
        }
        return result
    }

    /// Returns a new column containing the elements at the given positions, in order.
    ///
    /// - Parameter indices: Positions in `0..<count`; repeats are allowed.
    public func slice(at indices: [Int]) -> any AnyColumn {
        let n = indices.count
        if !hasNulls {
            let newValues = ContiguousArray<Double>(unsafeUninitializedCapacity: n) { outBuf, initCount in
                timeIntervals.withUnsafeBufferPointer { inBuf in
                    indices.withUnsafeBufferPointer { idxBuf in
                        guard let inBase = inBuf.baseAddress,
                              let idxBase = idxBuf.baseAddress,
                              let outBase = outBuf.baseAddress else { return }
                        for k in 0..<n {
                            outBase[k] = inBase[idxBase[k]]
                        }
                    }
                }
                initCount = n
            }
            return DateColumn(name: name, timeIntervals: newValues, nullBitmap: BitSet(count: n))
        }
        var newValues = ContiguousArray<Double>()
        newValues.reserveCapacity(n)
        var newBitmap = BitSet(count: n)
        for (newIdx, i) in indices.enumerated() {
            newValues.append(timeIntervals[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return DateColumn(name: name, timeIntervals: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements at the given positions, with nulls for `nil` entries.
    ///
    /// - Parameter indices: Positions in `0..<count`, or `nil` to produce a null element.
    public func slice(atOptional indices: [Int?]) -> any AnyColumn {
        var newValues = ContiguousArray<Double>()
        newValues.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (i, opt) in indices.enumerated() {
            if let idx = opt, !nullBitmap[idx] {
                newValues.append(timeIntervals[idx])
            } else {
                newValues.append(0.0)
                newBitmap[i] = true
            }
        }
        return DateColumn(name: name, timeIntervals: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements in the given range.
    ///
    /// - Parameter range: A range within `0..<count`.
    public func slice(range: Range<Int>) -> any AnyColumn {
        DateColumn(
            name: name,
            timeIntervals: ContiguousArray(timeIntervals[range]),
            nullBitmap: nullBitmap.slice(range: range)
        )
    }

    /// Returns a new column containing the elements whose mask bit is set.
    ///
    /// - Parameter mask: A bit set with at least `count` bits.
    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
        if targetCount == 0 {
            return DateColumn(name: name, timeIntervals: [], nullBitmap: BitSet(count: 0))
        }
        if !hasNulls {
            let n = count
            let newValues = ContiguousArray<Double>(unsafeUninitializedCapacity: targetCount) { outBuf, initCount in
                timeIntervals.withUnsafeBufferPointer { inBuf in
                    mask.words.withUnsafeBufferPointer { wordBuf in
                        guard let inBase = inBuf.baseAddress,
                              let wordBase = wordBuf.baseAddress,
                              let outBase = outBuf.baseAddress else { return }
                        var outIdx = 0
                        let fullWords = n >> 6
                        let wordLimit = Swift.min(fullWords, wordBuf.count)
                        for wIdx in 0..<wordLimit {
                            var w = wordBase[wIdx]
                            let baseRow = wIdx &<< 6
                            while w != 0 {
                                let bit = w.trailingZeroBitCount
                                outBase[outIdx] = inBase[baseRow + bit]
                                outIdx += 1
                                w &= w &- 1
                            }
                        }
                        if fullWords < wordBuf.count {
                            var w = wordBase[fullWords]
                            let baseRow = fullWords &<< 6
                            while w != 0 {
                                let bit = w.trailingZeroBitCount
                                let row = baseRow + bit
                                if row < n {
                                    outBase[outIdx] = inBase[row]
                                    outIdx += 1
                                }
                                w &= w &- 1
                            }
                        }
                    }
                }
                initCount = targetCount
            }
            return DateColumn(name: name, timeIntervals: newValues, nullBitmap: BitSet(count: targetCount))
        }
        var newValues = ContiguousArray<Double>()
        newValues.reserveCapacity(targetCount)
        var newBitmap = BitSet(count: targetCount)
        var newIdx = 0
        for i in 0..<count where mask[i] {
            newValues.append(timeIntervals[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
            newIdx += 1
        }
        return DateColumn(name: name, timeIntervals: newValues, nullBitmap: newBitmap)
    }

    /// Returns a copy of the column with a different name.
    ///
    /// - Parameter newName: The name for the copy.
    public func renamed(to newName: String) -> any AnyColumn {
        DateColumn(name: newName, timeIntervals: timeIntervals, nullBitmap: nullBitmap)
    }
}

// MARK: - AnyValueColumn (Fallback for mixed DataValues)

/// A fallback column that stores heterogeneous ``DataValue``s.
public final class AnyValueColumn: AnyColumn, @unchecked Sendable {
    /// The column's name.
    public let name: String

    /// The stored values; null positions hold `.missing`.
    public var dataValues: ContiguousArray<DataValue>

    /// A bit set whose set bits mark null elements.
    public var nullBitmap: BitSet

    /// The number of elements, including nulls.
    @inlinable
    public var count: Int { dataValues.count }

    /// The single type of all valid elements, `.mixed` if they differ, or `.unknown` if there are none.
    ///
    /// - Complexity: O(*n*), where *n* is `count`.
    public var dataType: DataType {
        var types = Set<DataType>()
        for v in dataValues where v != .missing {
            switch v {
            case .number: types.insert(.number)
            case .string: types.insert(.string)
            case .boolean: types.insert(.boolean)
            case .date: types.insert(.date)
            case .missing: break
            }
        }
        guard !types.isEmpty else { return .unknown }
        guard types.count == 1, let type = types.first else { return .mixed }
        return type
    }

    /// Creates a column from raw storage.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - dataValues: The stored values.
    ///   - nullBitmap: Set bits mark null positions in `dataValues`.
    public init(name: String, dataValues: ContiguousArray<DataValue>, nullBitmap: BitSet) {
        self.name = name
        self.dataValues = dataValues
        self.nullBitmap = nullBitmap
    }

    /// Creates a column from ``DataValue``s, marking every `.missing` value as null.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - values: The values to store.
    public convenience init(name: String, values: [DataValue]) {
        var nulls = BitSet(count: values.count)
        for (i, v) in values.enumerated() where v == .missing {
            nulls[i] = true
        }
        self.init(name: name, dataValues: ContiguousArray(values), nullBitmap: nulls)
    }

    /// Returns the element at the given position.
    ///
    /// - Parameter index: A position in `0..<count`.
    @inlinable
    public func value(at index: Int) -> DataValue {
        dataValues[index]
    }

    /// Returns every element as a ``DataValue``.
    public func toDataValues() -> [DataValue] {
        Array(dataValues)
    }

    /// Returns a new column containing the elements at the given positions, in order.
    ///
    /// - Parameter indices: Positions in `0..<count`; repeats are allowed.
    public func slice(at indices: [Int]) -> any AnyColumn {
        var newValues = ContiguousArray<DataValue>()
        newValues.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (newIdx, i) in indices.enumerated() {
            newValues.append(dataValues[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return AnyValueColumn(name: name, dataValues: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements at the given positions, with nulls for `nil` entries.
    ///
    /// - Parameter indices: Positions in `0..<count`, or `nil` to produce a null element.
    public func slice(atOptional indices: [Int?]) -> any AnyColumn {
        var newValues = ContiguousArray<DataValue>()
        newValues.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (i, opt) in indices.enumerated() {
            if let idx = opt, !nullBitmap[idx] {
                newValues.append(dataValues[idx])
            } else {
                newValues.append(.missing)
                newBitmap[i] = true
            }
        }
        return AnyValueColumn(name: name, dataValues: newValues, nullBitmap: newBitmap)
    }

    /// Returns a new column containing the elements in the given range.
    ///
    /// - Parameter range: A range within `0..<count`.
    public func slice(range: Range<Int>) -> any AnyColumn {
        AnyValueColumn(
            name: name,
            dataValues: ContiguousArray(dataValues[range]),
            nullBitmap: nullBitmap.slice(range: range)
        )
    }

    /// Returns a new column containing the elements whose mask bit is set.
    ///
    /// - Parameter mask: A bit set with at least `count` bits.
    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
        var newValues = ContiguousArray<DataValue>()
        newValues.reserveCapacity(targetCount)
        var newBitmap = BitSet(count: targetCount)
        var newIdx = 0
        for i in 0..<count where mask[i] {
            newValues.append(dataValues[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
            newIdx += 1
        }
        return AnyValueColumn(name: name, dataValues: newValues, nullBitmap: newBitmap)
    }

    /// Returns a copy of the column with a different name.
    ///
    /// - Parameter newName: The name for the copy.
    public func renamed(to newName: String) -> any AnyColumn {
        AnyValueColumn(name: newName, dataValues: dataValues, nullBitmap: nullBitmap)
    }
}

// MARK: - Factory

/// A namespace for building the most specific column type for a set of values.
public enum AnyColumnFactory {
    /// Creates a column whose concrete type matches the given values.
    ///
    /// Uses ``Float64Column``, ``StringColumn``, ``CategoricalColumn`` (strings, at least 1,000 values and at most 256 distinct),
    /// ``BooleanColumn``, or ``DateColumn`` when all non-missing values share one kind, and ``AnyValueColumn`` otherwise.
    ///
    /// - Parameters:
    ///   - name: The column's name.
    ///   - values: The values; `.missing` entries become nulls.
    /// - Returns: A column holding `values`.
    /// - Complexity: O(*n*), where *n* is `values.count`.
    public static func make(name: String, values: [DataValue]) -> any AnyColumn {
        var isAllDouble = true
        var isAllString = true
        var isAllBool = true
        var isAllDate = true
        var hasValid = false

        for v in values {
            if v != .missing {
                hasValid = true
            }
            switch v {
            case .number:
                isAllString = false
                isAllBool = false
                isAllDate = false
            case .string:
                isAllDouble = false
                isAllBool = false
                isAllDate = false
            case .boolean:
                isAllDouble = false
                isAllString = false
                isAllDate = false
            case .date:
                isAllDouble = false
                isAllString = false
                isAllBool = false
            case .missing:
                break
            }
        }

        guard hasValid else {
            return AnyValueColumn(name: name, values: values)
        }

        if isAllDouble {
            return Float64Column(name: name, dataValues: values)
        }
        if isAllString {
            // If large column, categorical can save memory, but string column is safe
            if values.count >= 1000 {
                // check uniqueness
                var unique = Set<String>()
                for v in values {
                    if case .string(let s) = v {
                        unique.insert(s)
                        if unique.count > 1000 { break }
                    }
                }
                if unique.count <= 256 {
                    return CategoricalColumn(name: name, strings: values.map { $0.stringValue ?? "" })
                }
            }
            return StringColumn(name: name, dataValues: values)
        }
        if isAllBool {
            var b = BitSet(count: values.count)
            var nulls = BitSet(count: values.count)
            for (i, v) in values.enumerated() {
                if case .boolean(let boolVal) = v {
                    if boolVal { b[i] = true }
                } else {
                    nulls[i] = true
                }
            }
            return BooleanColumn(name: name, bitmap: b, nullBitmap: nulls)
        }
        if isAllDate {
            var intervals = ContiguousArray<Double>()
            intervals.reserveCapacity(values.count)
            var nulls = BitSet(count: values.count)
            for (i, v) in values.enumerated() {
                if case .date(let d) = v {
                    intervals.append(d.timeIntervalSinceReferenceDate)
                } else {
                    intervals.append(0.0)
                    nulls[i] = true
                }
            }
            return DateColumn(name: name, timeIntervals: intervals, nullBitmap: nulls)
        }

        return AnyValueColumn(name: name, values: values)
    }
}
