import Foundation

/// Protocol representing a typed columnar buffer (Apache Arrow / Polars model).
public protocol AnyColumn: Sendable {
    var name: String { get }
    var count: Int { get }
    var nullBitmap: BitSet { get }
    var dataType: DataType { get }

    func value(at index: Int) -> DataValue
    func toDataValues() -> [DataValue]
    func slice(at indices: [Int]) -> any AnyColumn
    func slice(atOptional indices: [Int?]) -> any AnyColumn
    func slice(range: Range<Int>) -> any AnyColumn
    func slice(mask: BitSet) -> any AnyColumn
    func renamed(to newName: String) -> any AnyColumn
}

extension AnyColumn {
    @inlinable
    public func isNull(at index: Int) -> Bool {
        nullBitmap[index]
    }

    @inlinable
    public func isValid(at index: Int) -> Bool {
        !nullBitmap[index]
    }

    @inlinable
    public var hasNulls: Bool {
        !nullBitmap.allFalse
    }

    @inlinable
    public var nullCount: Int {
        nullBitmap.setBitCount
    }

    @inlinable
    public var validCount: Int {
        count - nullCount
    }
}

// MARK: - Float64Column

public final class Float64Column: AnyColumn, @unchecked Sendable {
    public let name: String
    public var values: ContiguousArray<Double>
    public var nullBitmap: BitSet

    @inlinable
    public var count: Int { values.count }

    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .number }

    public init(name: String, values: ContiguousArray<Double>, nullBitmap: BitSet) {
        self.name = name
        self.values = values
        self.nullBitmap = nullBitmap
    }

    public convenience init(name: String, numbers: [Double]) {
        self.init(name: name, values: ContiguousArray(numbers), nullBitmap: BitSet(count: numbers.count))
    }

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

    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .number(values[index])
    }

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

    public func slice(at indices: [Int]) -> any AnyColumn {
        var newValues = ContiguousArray<Double>()
        newValues.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (newIdx, i) in indices.enumerated() {
            newValues.append(values[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return Float64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

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

    public func slice(range: Range<Int>) -> any AnyColumn {
        let newValues = ContiguousArray(values[range])
        let newBitmap = nullBitmap.slice(range: range)
        return Float64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
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

    public func renamed(to newName: String) -> any AnyColumn {
        Float64Column(name: newName, values: values, nullBitmap: nullBitmap)
    }
}

// MARK: - Int64Column

public final class Int64Column: AnyColumn, @unchecked Sendable {
    public let name: String
    public var values: ContiguousArray<Int64>
    public var nullBitmap: BitSet

    @inlinable
    public var count: Int { values.count }

    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .number }

    public init(name: String, values: ContiguousArray<Int64>, nullBitmap: BitSet) {
        self.name = name
        self.values = values
        self.nullBitmap = nullBitmap
    }

    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .number(Double(values[index]))
    }

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

    public func slice(at indices: [Int]) -> any AnyColumn {
        var newValues = ContiguousArray<Int64>()
        newValues.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (newIdx, i) in indices.enumerated() {
            newValues.append(values[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return Int64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

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

    public func slice(range: Range<Int>) -> any AnyColumn {
        let newValues = ContiguousArray(values[range])
        let newBitmap = nullBitmap.slice(range: range)
        return Int64Column(name: name, values: newValues, nullBitmap: newBitmap)
    }

    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
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

    public func renamed(to newName: String) -> any AnyColumn {
        Int64Column(name: newName, values: values, nullBitmap: nullBitmap)
    }
}

// MARK: - StringColumn

public final class StringColumn: AnyColumn, @unchecked Sendable {
    public let name: String
    public var stringData: ContiguousArray<String>
    public var nullBitmap: BitSet

    @inlinable
    public var count: Int { stringData.count }

    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .string }

    public init(name: String, stringData: ContiguousArray<String>, nullBitmap: BitSet) {
        self.name = name
        self.stringData = stringData
        self.nullBitmap = nullBitmap
    }

    public convenience init(name: String, strings: [String]) {
        self.init(name: name, stringData: ContiguousArray(strings), nullBitmap: BitSet(count: strings.count))
    }

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

    @inlinable
    public func string(at index: Int) -> String {
        stringData[index]
    }

    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .string(string(at: index))
    }

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

    public func slice(range: Range<Int>) -> any AnyColumn {
        let newStrings = ContiguousArray(stringData[range])
        let newBitmap = nullBitmap.slice(range: range)
        return StringColumn(name: name, stringData: newStrings, nullBitmap: newBitmap)
    }

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

    public func renamed(to newName: String) -> any AnyColumn {
        StringColumn(name: newName, stringData: stringData, nullBitmap: nullBitmap)
    }
}

// MARK: - CategoricalColumn

public final class CategoricalColumn: AnyColumn, @unchecked Sendable {
    public let name: String
    public var dictionary: [String]
    public var codes: ContiguousArray<UInt32>
    public var nullBitmap: BitSet

    @inlinable
    public var count: Int { codes.count }

    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .string }

    public init(name: String, dictionary: [String], codes: ContiguousArray<UInt32>, nullBitmap: BitSet) {
        self.name = name
        self.dictionary = dictionary
        self.codes = codes
        self.nullBitmap = nullBitmap
    }

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

    @inlinable
    public func string(at index: Int) -> String {
        dictionary[Int(codes[index])]
    }

    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .string(string(at: index))
    }

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

    public func slice(at indices: [Int]) -> any AnyColumn {
        var newCodes = ContiguousArray<UInt32>()
        newCodes.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (newIdx, i) in indices.enumerated() {
            newCodes.append(codes[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return CategoricalColumn(name: name, dictionary: dictionary, codes: newCodes, nullBitmap: newBitmap)
    }

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

    public func slice(range: Range<Int>) -> any AnyColumn {
        let newCodes = ContiguousArray(codes[range])
        let newBitmap = nullBitmap.slice(range: range)
        return CategoricalColumn(name: name, dictionary: dictionary, codes: newCodes, nullBitmap: newBitmap)
    }

    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
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

    public func renamed(to newName: String) -> any AnyColumn {
        CategoricalColumn(name: newName, dictionary: dictionary, codes: codes, nullBitmap: nullBitmap)
    }
}

// MARK: - BooleanColumn

public final class BooleanColumn: AnyColumn, @unchecked Sendable {
    public let name: String
    public var bitmap: BitSet
    public var nullBitmap: BitSet

    @inlinable
    public var count: Int { bitmap.count }

    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .boolean }

    public init(name: String, bitmap: BitSet, nullBitmap: BitSet) {
        self.name = name
        self.bitmap = bitmap
        self.nullBitmap = nullBitmap
    }

    public convenience init(name: String, booleans: [Bool]) {
        var b = BitSet(count: booleans.count)
        for (i, val) in booleans.enumerated() where val {
            b[i] = true
        }
        self.init(name: name, bitmap: b, nullBitmap: BitSet(count: booleans.count))
    }

    @inlinable
    public func bool(at index: Int) -> Bool {
        bitmap[index]
    }

    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .boolean(bitmap[index])
    }

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

    public func slice(at indices: [Int]) -> any AnyColumn {
        var newBits = BitSet(count: indices.count)
        var newNulls = BitSet(count: indices.count)
        for (newIdx, i) in indices.enumerated() {
            if bitmap[i] { newBits[newIdx] = true }
            if nullBitmap[i] { newNulls[newIdx] = true }
        }
        return BooleanColumn(name: name, bitmap: newBits, nullBitmap: newNulls)
    }

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

    public func slice(range: Range<Int>) -> any AnyColumn {
        BooleanColumn(name: name, bitmap: bitmap.slice(range: range), nullBitmap: nullBitmap.slice(range: range))
    }

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

    public func renamed(to newName: String) -> any AnyColumn {
        BooleanColumn(name: newName, bitmap: bitmap, nullBitmap: nullBitmap)
    }
}

// MARK: - DateColumn

public final class DateColumn: AnyColumn, @unchecked Sendable {
    public let name: String
    public var timeIntervals: ContiguousArray<Double>
    public var nullBitmap: BitSet

    @inlinable
    public var count: Int { timeIntervals.count }

    @inlinable
    public var dataType: DataType { validCount == 0 ? .unknown : .date }

    public init(name: String, timeIntervals: ContiguousArray<Double>, nullBitmap: BitSet) {
        self.name = name
        self.timeIntervals = timeIntervals
        self.nullBitmap = nullBitmap
    }

    public convenience init(name: String, dates: [Date]) {
        var intervals = ContiguousArray<Double>()
        intervals.reserveCapacity(dates.count)
        for d in dates {
            intervals.append(d.timeIntervalSinceReferenceDate)
        }
        self.init(name: name, timeIntervals: intervals, nullBitmap: BitSet(count: dates.count))
    }

    @inlinable
    public func date(at index: Int) -> Date {
        Date(timeIntervalSinceReferenceDate: timeIntervals[index])
    }

    @inlinable
    public func value(at index: Int) -> DataValue {
        if nullBitmap[index] { return .missing }
        return .date(date(at: index))
    }

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

    public func slice(at indices: [Int]) -> any AnyColumn {
        var newValues = ContiguousArray<Double>()
        newValues.reserveCapacity(indices.count)
        var newBitmap = BitSet(count: indices.count)
        for (newIdx, i) in indices.enumerated() {
            newValues.append(timeIntervals[i])
            if nullBitmap[i] {
                newBitmap[newIdx] = true
            }
        }
        return DateColumn(name: name, timeIntervals: newValues, nullBitmap: newBitmap)
    }

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

    public func slice(range: Range<Int>) -> any AnyColumn {
        DateColumn(
            name: name,
            timeIntervals: ContiguousArray(timeIntervals[range]),
            nullBitmap: nullBitmap.slice(range: range)
        )
    }

    public func slice(mask: BitSet) -> any AnyColumn {
        let targetCount = mask.setBitCount
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

    public func renamed(to newName: String) -> any AnyColumn {
        DateColumn(name: newName, timeIntervals: timeIntervals, nullBitmap: nullBitmap)
    }
}

// MARK: - AnyValueColumn (Fallback for mixed DataValues)

public final class AnyValueColumn: AnyColumn, @unchecked Sendable {
    public let name: String
    public var dataValues: ContiguousArray<DataValue>
    public var nullBitmap: BitSet

    @inlinable
    public var count: Int { dataValues.count }

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

    public init(name: String, dataValues: ContiguousArray<DataValue>, nullBitmap: BitSet) {
        self.name = name
        self.dataValues = dataValues
        self.nullBitmap = nullBitmap
    }

    public convenience init(name: String, values: [DataValue]) {
        var nulls = BitSet(count: values.count)
        for (i, v) in values.enumerated() where v == .missing {
            nulls[i] = true
        }
        self.init(name: name, dataValues: ContiguousArray(values), nullBitmap: nulls)
    }

    @inlinable
    public func value(at index: Int) -> DataValue {
        dataValues[index]
    }

    public func toDataValues() -> [DataValue] {
        Array(dataValues)
    }

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

    public func slice(range: Range<Int>) -> any AnyColumn {
        AnyValueColumn(
            name: name,
            dataValues: ContiguousArray(dataValues[range]),
            nullBitmap: nullBitmap.slice(range: range)
        )
    }

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

    public func renamed(to newName: String) -> any AnyColumn {
        AnyValueColumn(name: newName, dataValues: dataValues, nullBitmap: nullBitmap)
    }
}

// MARK: - Factory

public enum AnyColumnFactory {
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
