import Foundation
import Testing
@testable import Bandal

@Test func rendersPandasStylePreviewInfoAndDescribe() throws {
    let data = try DataFrame(columns: [
        (name: "temperature", values: [.number(10), .number(20), .number(30), .missing]),
        (name: "city", values: [.string("Seoul"), .string("Busan"), .string("Seoul"), .missing]),
    ])

    #expect(data.description.contains("temperature"))
    #expect(data.description.contains("[4 rows x 2 columns]"))
    #expect(data.info().contains("3 non-null"))
    #expect(data.info().contains("float64(1), object(1)"))

    let statistics = data.describe()
    #expect(statistics.contains("count"))
    #expect(statistics.contains("std"))
    #expect(statistics.contains("25%"))
    #expect(statistics.contains("std     10"))
    #expect(statistics.contains("min     10"))
    #expect(statistics.contains("25%     15"))
    #expect(statistics.contains("50%     20"))
    #expect(statistics.contains("75%     25"))
    #expect(statistics.contains("max     30"))
    #expect(statistics.contains("Seoul"))
    #expect(statistics.contains("freq"))
}

@Test func truncatesRenderedFramesWithEllipses() throws {
    let data = try DataFrame(columns: [
        (name: "value", values: [1.0, 2.0, 3.0, 4.0, 5.0]),
    ])

    let rendered = data.render(maxRows: 3)
    #expect(rendered.contains("…"))
    #expect(rendered.contains("[5 rows x 1 columns]"))
}
