import Foundation
import Testing
@testable import Bandal

@Test func readsCSVWithQuotedNumbersAndMissingValues() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("csv")
    try "Country,Sales,Note\n\"United Kingdom\",\"1,100,000\",good\nJapan,$980000,\n"
        .write(to: url, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: url) }

    let data = try DataFrame(contentsOfCSV: url)

    #expect(data.shape == [2, 3])
    #expect(try data.head(1).rowCount == 1)
    #expect(try data.tail(1).rowCount == 1)
    #expect(try data.values(in: "Country") == [.string("United Kingdom"), .string("Japan")])
    #expect(try data.column("Sales") == [1_100_000, 980_000])
    #expect(try data.values(in: "Note") == [.string("good"), .missing])

    let limited = try DataFrame(contentsOfCSV: url, maximumRows: 1)
    #expect(limited.rowCount == 1)
    #expect(try limited.values(in: "Country") == [.string("United Kingdom")])
}

@Test func readsCSVWithExplicitHeaderSelectionMissingAndInferenceOptions() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("csv")
    try "Name;Amount;Comment\nAda;0012;NA\nLinus;;unknown\n"
        .write(to: url, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: url) }

    let data = try DataFrame(
        contentsOfCSV: url,
        options: CSVReadOptions(
            delimiter: ";",
            selectedColumns: ["Comment", "Amount"],
            missingTokens: ["", "NA"],
            infersNumbers: false
        )
    )
    #expect(data.columnNames == ["Comment", "Amount"])
    #expect(try data.values(in: "Comment") == [.missing, .string("unknown")])
    #expect(try data.values(in: "Amount") == [.string("0012"), .missing])
}

@Test func readsCSVWithExplicitBooleanAndDateTypes() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("csv")
    try "enabled,created\ntrue,2024/02/01\nfalse,2024/02/02\n"
        .write(to: url, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: url) }

    let data = try DataFrame(
        contentsOfCSV: url,
        options: CSVReadOptions(
            columnTypes: ["enabled": .boolean, "created": .date],
            dateFormat: "yyyy/MM/dd"
        )
    )
    #expect(try data.values(in: "enabled") == [.boolean(true), .boolean(false)])
    #expect(try data.dataType(of: "created") == .date)

    let roundTripURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("csv")
    defer { try? FileManager.default.removeItem(at: roundTripURL) }
    try data.writeCSV(to: roundTripURL)
    let roundTripped = try DataFrame(
        contentsOfCSV: roundTripURL,
        options: CSVReadOptions(columnTypes: ["enabled": .boolean, "created": .date])
    )
    #expect(try roundTripped.values(in: "enabled") == [.boolean(true), .boolean(false)])
    #expect(try roundTripped.dataType(of: "created") == .date)

    try "enabled\nsometimes\n".write(to: url, atomically: true, encoding: .utf8)
    expectDataFrameError(.invalidCSV("cannot parse 'sometimes' as Boolean")) {
        _ = try DataFrame(
            contentsOfCSV: url,
            options: CSVReadOptions(columnTypes: ["enabled": .boolean])
        )
    }
}

@Test func readsCSVWithoutHeaderAndUsesConfiguredNamesAndRowLimit() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("csv")
    try "a,1\nb,2\nc,3\n"
        .write(to: url, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: url) }

    let data = try DataFrame(
        contentsOfCSV: url,
        options: CSVReadOptions(
            maximumRows: 1,
            hasHeader: false,
            columnNames: ["label", "value"]
        )
    )
    #expect(data.shape == [1, 2])
    #expect(try data.values(in: "label") == [.string("a")])
    #expect(try data.column("value") == [1])

    expectDataFrameError(
        .invalidCSV("received 1 column names; expected 2")
    ) {
        _ = try DataFrame(
            contentsOfCSV: url,
            options: CSVReadOptions(hasHeader: false, columnNames: ["only"])
        )
    }
}

@Test func streamsCSVRecordsAcrossChunksAndStopsAtRowLimit() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("csv")
    let longValue = String(repeating: "x", count: 65_530)
    try "id,note\n1,\"\(longValue)\ncontinued\"\n2,ignored\n3,\"unterminated"
        .write(to: url, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: url) }

    let data = try DataFrame(contentsOfCSV: url, maximumRows: 1)
    #expect(data.rowCount == 1)
    #expect(try data.values(in: "note") == [.string("\(longValue)\ncontinued")])
}

@Test func writesCSVWithEscapedFieldsAndReadsItBack() throws {
    let original = try DataFrame(columns: [
        (name: "name", values: [
            .string("plain"), .string("two,fields"), .string("say \"hello\""),
            .string("two\nlines"),
        ]),
        (name: "value", values: [
            .number(1.25), .missing, .number(-4), .number(9),
        ]),
    ])
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("csv")
    defer { try? FileManager.default.removeItem(at: url) }

    try original.writeCSV(to: url)
    let roundTrip = try DataFrame(
        contentsOfCSV: url,
        options: CSVReadOptions(infersNumbers: false)
    )

    #expect(try roundTrip.values(in: "name") == [
        .string("plain"), .string("two,fields"), .string("say \"hello\""),
        .string("two\nlines"),
    ])
    #expect(try roundTrip.values(in: "value") == [
        .string("1.25"), .missing, .string("-4.0"), .string("9.0"),
    ])

    try original.writeCSV(to: url, delimiter: ";", includeHeader: false)
    let headerless = try DataFrame(
        contentsOfCSV: url,
        options: CSVReadOptions(
            delimiter: ";",
            hasHeader: false,
            columnNames: ["name", "value"],
            infersNumbers: false
        )
    )
    #expect(headerless.shape == original.shape)

    #expect(try headerless.values(in: "name") == original.values(in: "name"))
}

@Test func readsCSVWithBOMAndRejectsMalformedInput() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("csv")
    defer { try? FileManager.default.removeItem(at: url) }

    try "\u{FEFF}name,value\r\n\"Ada, Lovelace\",1\r\n".write(
        to: url,
        atomically: true,
        encoding: .utf8
    )
    let data = try DataFrame(contentsOfCSV: url)
    #expect(data.columnNames == ["name", "value"])
    #expect(try data.values(in: "name") == [.string("Ada, Lovelace")])
    #expect(try data.column("value") == [1])

    try "name,value\nAda,\"unterminated".write(
        to: url,
        atomically: true,
        encoding: .utf8
    )
    expectDataFrameError(.invalidCSV("unterminated quoted field")) {
        _ = try DataFrame(contentsOfCSV: url)
    }

    try "name,value\nAda,1\nGrace,2,extra\n".write(
        to: url,
        atomically: true,
        encoding: .utf8
    )
    expectDataFrameError(.invalidCSV("record 3 has 3 fields; expected 2")) {
        _ = try DataFrame(contentsOfCSV: url)
    }

    try Data([0xFF]).write(to: url)
    expectDataFrameError(.invalidCSV("input is not valid UTF-8")) {
        _ = try DataFrame(contentsOfCSV: url)
    }
}
