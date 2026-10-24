extension DataFrame {
    /// Returns a new frame sorted by one or more columns.
    ///
    /// Sort keys are applied in order, ties preserve their input order, and
    /// missing values remain last in both ascending and descending order.
    public func sorted(by descriptors: [SortDescriptor]) throws -> DataFrame {
        guard !descriptors.isEmpty else { return self }
        let sortColumns = try descriptors.map { descriptor in
            (descriptor.order, try values(in: descriptor.column))
        }
        let indices = (0..<rowCount).sorted { left, right in
            for (order, values) in sortColumns {
                let lhs = values[left]
                let rhs = values[right]
                guard lhs != rhs else { continue }
                if lhs == .missing { return false }
                if rhs == .missing { return true }
                switch order {
                case .ascending:
                    return isOrderedBefore(lhs, rhs)
                case .descending:
                    return isOrderedBefore(rhs, lhs)
                }
            }
            return left < right
        }
        return try rows(at: indices)
    }
}

func isOrderedBefore(_ lhs: DataValue, _ rhs: DataValue) -> Bool {
    switch (lhs, rhs) {
    case (.number(let left), .number(let right)):
        left < right
    case (.string(let left), .string(let right)):
        left < right
    case (.boolean(let left), .boolean(let right)):
        !left && right
    case (.date(let left), .date(let right)):
        left < right
    case (.missing, _), (_, .missing):
        false
    case _:
        valueOrder(lhs) < valueOrder(rhs)
    }
}

private func valueOrder(_ value: DataValue) -> Int {
    switch value {
    case .number: 0
    case .boolean: 1
    case .date: 2
    case .string: 3
    case .missing: 4
    }
}
