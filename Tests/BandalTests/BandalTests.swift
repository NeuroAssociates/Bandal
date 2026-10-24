import Foundation
import Testing
@testable import Bandal

func expectDataFrameError(
    _ expected: DataFrameError,
    operation: () throws -> Void
) {
    do {
        try operation()
        Issue.record("Expected error \(expected).")
    } catch let error as DataFrameError {
        #expect(error == expected)
    } catch {
        Issue.record("Unexpected error: \(error).")
    }
}
