# Bandal API 레퍼런스

[English](API.md)

Bandal은 Python의 [pandas](https://pandas.pydata.org/)에서 영감을 받아 [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/)에 따라 설계된 테이블형 데이터 처리용 Swift 패키지입니다. 모든 데이터프레임과 결과 구조체는 불변(immutable) 값 타입(value type)이며, `Sendable`, `Equatable`, `Hashable`, `Codable`, `CustomStringConvertible`, `CustomDebugStringConvertible`, `TextOutputStreamable`을 준수하여 `print(...)`에 전달하면 pandas 스타일의 정렬된 텍스트를 바로 출력할 수 있습니다.

수치 연산에서는 [MLX Swift](https://github.com/ml-explore/mlx-swift)를 벡터화 수치 엔진으로 활용하며(기본적으로 MLX CPU 스트림의 `float64`를 사용하고, `ExecutionPolicy.gpu` 지정 시 GPU 스트림의 `float32`를 사용), 입력 크기가 작거나 MLX Metal 런타임을 사용할 수 없는 환경에서는 네이티브 Swift 구현으로 자동 폴백(fallback)합니다.

---

## 목차

1. [데이터 모델 (`DataFrame`, `Series`, `DataRow`, `DataValue`, `DataType`)](#1-데이터-모델)
2. [DataFrame 생성하기](#2-dataframe-생성하기)
3. [CSV 및 JSON 읽기와 쓰기](#3-csv-및-json-읽기와-쓰기)
4. [데이터 확인, 렌더링 및 Markdown 내보내기](#4-데이터-확인-렌더링-및-markdown-내보내기)
5. [행, 컬럼 및 셀 선택](#5-행-컬럼-및-셀-선택)
6. [필터링 및 쿼리 (`filterRows`, `RowPredicate`, `query`)](#6-필터링-및-쿼리)
7. [컬럼 추가, 계산, 이름 변경 및 타입 캐스팅](#7-컬럼-추가-계산-이름-변경-및-타입-캐스팅)
8. [날짜 및 문자열 컬럼 연산](#8-날짜-및-문자열-컬럼-연산)
9. [결측치(Missing Value) 처리](#9-결측치missing-value-처리)
10. [중복 제거, 정렬, 샘플링 및 행 연결](#10-중복-제거-정렬-샘플링-및-행-연결)
11. [Series 및 컬럼 단위 수치 연산](#11-series-및-컬럼-단위-수치-연산)
12. [기술 통계 및 구조화된 결과 타입](#12-기술-통계-및-구조화된-결과-타입)
13. [그룹화 (`groupBy`, `aggregate`, `transform`)](#13-그룹화)
14. [조인 (`leftJoin`, `join`)](#14-조인)
15. [재구조화 (`melt`, `pivotTable`, `exploded`, `transposed`)](#15-재구조화)
16. [윈도우 함수 (`rolling`, 시간 기반 `rolling`, `expanding`)](#16-윈도우-함수)
17. [MLX 런타임, `ExecutionPolicy` 및 `MLXVector` 상호운용](#17-mlx-런타임-executionpolicy-및-mlxvector-상호운용)
18. [오류 (`DataFrameError`)](#18-오류)

---

## 1. 데이터 모델

### `DataFrame`

```swift
public struct DataFrame: Equatable, Hashable, Sendable, Codable,
    CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable
```

모든 컬럼이 동일한 행 개수를 갖는 순서가 보존된 명명된 컬럼 컬렉션입니다. 모든 연산은 원본 `DataFrame`을 변경하지 않고 새로운 `DataFrame`을 반환하거나 `DataFrameError`를 던집니다.

### `Series`

```swift
public struct Series: Equatable, Hashable, Sendable, Codable,
    RandomAccessCollection, CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable
```

`try df.series("col")` 또는 `df["col"]`로 추출하는 1차원 명명된 컬럼 타입입니다. `RandomAccessCollection`(`Element == DataValue`, `Index == Int`)을 준수하며 벡터화 사칙연산, 누적 연산, 순위 계산, 분위수 계산, 결측치 처리, 빈도 계산을 지원합니다.

### `DataRow`

```swift
public struct DataRow: Equatable, Hashable, Sendable, Codable,
    CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable
```

`df.row(at:)`, `df[row: i]` 또는 `df.rows`가 반환하는 단일 행 뷰 구조체입니다.

| 멤버 | 타입 | 설명 |
| --- | --- | --- |
| `index` | `Int` | 원본 프레임에서의 0 기반 행 인덱스입니다. |
| `columnOrder` | `[String]` | 순서가 유지된 컬럼 이름 배열입니다. |
| `values` | `[String: DataValue]` | 컬럼 이름과 셀 값의 딕셔너리입니다. |
| `subscript(_ column: String)` | `DataValue?` | 해당 컬럼의 셀 값을 반환하며, 컬럼이 없으면 `nil`을 반환합니다. |
| `description` | `String` | 컬럼 이름과 값을 정렬한 2열 요약 텍스트와 `Row: <index>`를 반환합니다. |

```swift
let df = try DataFrame(columns: [
    (name: "city", values: [.string("Seoul"), .string("Busan")]),
    (name: "temp", values: [.number(21.5), .number(24.0)]),
])
if let firstRow = df[row: 0] {
    print(firstRow["city"]!) // Seoul
    print(firstRow)
}
```

### `DataValue`

```swift
public enum DataValue: Equatable, Hashable, Sendable, Codable,
    CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable,
    ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByStringLiteral, ExpressibleByBooleanLiteral, ExpressibleByNilLiteral
```

단일 셀 값을 나타냅니다.

- `.number(Double)` — 유한한 `Double` 값(`NaN` 및 `±infinity`는 `.nonFiniteValue` 오류로 거부됨).
- `.string(String)` — UTF-8 문자열.
- `.boolean(Bool)` — `true` 또는 `false`.
- `.date(Date)` — UTC ISO 8601(`yyyy-MM-dd'T'HH:mm:ss'Z'`)로 포맷되는 `Foundation.Date`.
- `.missing` — 결측치(출력 시 pandas의 `NaN`/`None`처럼 `NaN`으로 표시됨).

`DataValue`는 Swift 리터럴 프로토콜을 준수하므로 `[DataValue]` 배열 안에서 리터럴을 직접 사용할 수 있습니다:

```swift
let values: [DataValue] = [10, 20.5, "text", true, nil]
// [.number(10), .number(20.5), .string("text"), .boolean(true), .missing]과 동일
```

`DataValue`의 타입별 편의 프로퍼티:

- `numberValue: Double?`
- `stringValue: String?`
- `boolValue: Bool?`
- `dateValue: Date?`
- `isMissing: Bool`

### `DataType`

```swift
public enum DataType: String, Equatable, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case number
    case string
    case boolean
    case date
    case mixed
    case unknown
}
```

`try df.dataType(of:)` 및 `df.dtypes` / `df.columnTypes()`에서 반환됩니다:

- `.number`, `.string`, `.boolean`, `.date`: 컬럼 내 결측이 아닌 모든 값이 동일한 구체 타입일 때.
- `.mixed`: 컬럼 내에 서로 다른 구체 타입의 값이 섞여 있을 때.
- `.unknown`: 모든 값이 `.missing`이거나 행이 0개일 때.

---

## 2. DataFrame 생성하기

### 수치형 컬럼 딕셔너리로부터 생성

컬럼 이름을 사전순으로 정렬하여 실행할 때마다 동일한 컬럼 순서를 보장합니다:

```swift
import Bandal

let weather = try DataFrame(columns: [
    "temperature": [18.0, 21.5, 19.0],
    "humidity": [0.45, 0.50, 0.48],
])
print(weather)
```

### 순서가 있는 수치형 컬럼 배열로부터 생성

전달한 컬럼 순서를 그대로 유지합니다:

```swift
let ordered = try DataFrame(columns: [
    (name: "z", values: [1.0, 2.0]),
    (name: "a", values: [3.0, 4.0]),
])
print(ordered.columnNames) // ["z", "a"]
```

### 순서가 있는 혼합 타입(`DataValue`) 컬럼 배열로부터 생성

```swift
let mixed = try DataFrame(columns: [
    (name: "city", values: [.string("Seoul"), .string("Busan"), .missing]),
    (name: "temp", values: [.number(18.0), .missing, .number(22.5)]),
    (name: "active", values: [.boolean(true), .boolean(false), .missing]),
])
print(mixed)
```

### MLX 벡터(`MLXVector`)로부터 생성

`try DataFrame(mlxColumns:)`에 대한 자세한 설명은 [17장](#17-mlx-런타임-executionpolicy-및-mlxvector-상호운용)을 참고하세요.

---

## 3. CSV 및 JSON 읽기와 쓰기

### CSV 읽기 (`init(contentsOfCSV:...)` 및 `CSVReadOptions`)

```swift
public struct CSVReadOptions: Equatable, Hashable, Sendable, Codable {
    public var delimiter: Character
    public var maximumRows: Int?
    public var selectedColumns: [String]?
    public var hasHeader: Bool
    public var columnNames: [String]?
    public var missingTokens: Set<String>
    public var infersNumbers: Bool
    public var columnTypes: [String: DataType]
    public var dateFormat: String?
}
```

기본값:

- `delimiter = ","`
- `hasHeader = true`
- `missingTokens = ["", "NA", "N/A", "NaN", "null", "NULL", "nil"]`
- `infersNumbers = true` (수치 토큰 내부의 `,` 및 선행 `$` 기호를 자동 제거하여 파싱)

```swift
let df = try DataFrame(
    contentsOfCSV: url,
    options: CSVReadOptions(
        delimiter: ",",
        maximumRows: 1_000,
        selectedColumns: ["Country", "Sales", "Created"],
        columnTypes: ["Created": .date],
        dateFormat: "yyyy-MM-dd"
    )
)
```

행 개수 제한만 필요한 경우 축약 이니셜라이저를 사용할 수 있습니다:

```swift
let preview = try DataFrame(contentsOfCSV: url, maximumRows: 10)
```

### CSV 쓰기 (`writeCSV(to:delimiter:includeHeader:)`)

RFC 4180 인용 규칙을 준수하며 64 KB 청크 단위로 디스크에 스트리밍 기록합니다:

```swift
try df.writeCSV(to: outputURL, delimiter: ",", includeHeader: true)
```

### JSON 읽기 및 쓰기 (`init(jsonData:)`, `init(contentsOfJSON:)`, `jsonData`, `writeJSON`)

행 객체들로 이루어진 JSON 배열(`[{"col": value, ...}, ...]`)을 읽고 씁니다. 숫자, 문자열, 불리언, `null`(`.missing`), 그리고 ISO 8601(또는 사용자 지정 `dateFormat`) 날짜 문자열을 지원합니다.

```swift
let json = """
[
  {"city": "Seoul", "temp": 21.5, "capital": true},
  {"city": "Busan", "temp": null, "capital": false}
]
""".data(using: .utf8)!

let fromJSON = try DataFrame(jsonData: json)
print(fromJSON)

// Data로 직렬화하거나 파일 URL로 저장 후 다시 읽기
let encodedData = try fromJSON.jsonData(prettyPrinted: true)
try fromJSON.writeJSON(to: jsonFileURL, prettyPrinted: true)
let reloaded = try DataFrame(contentsOfJSON: jsonFileURL)
```

---

## 4. 데이터 확인, 렌더링 및 Markdown 내보내기

### 주요 프로퍼티

| 프로퍼티 / 메서드 | 반환 타입 | 설명 |
| --- | --- | --- |
| `rowCount` | `Int` | 전체 행 개수입니다. |
| `columnNames` | `[String]` | 순서가 유지된 컬럼 이름 목록입니다. |
| `shape` | `[Int]` | `[rowCount, columnNames.count]` 배열입니다. |
| `isEmpty` | `Bool` | `rowCount == 0`이면 `true`입니다. |
| `dtypes` / `columnTypes()` | `ColumnTypes` | 컬럼별 추론된 `DataType`을 담은 순서 보장 구조체입니다. |
| `dataType(of:)` | `DataType` | 특정 컬럼의 추론된 `DataType`을 반환합니다(없으면 `.columnNotFound` 발생). |

### 테이블 렌더링 (`description`, `render`, `RenderOptions`)

`print(df)`를 호출하면 `df.description`을 통해 기본적으로 최대 20행, 10컬럼까지 `…` 생략 기호와 `[R rows x C columns]` 푸터를 포함한 표를 출력합니다.

`RenderOptions`를 사용하면 최대 행 수, 최대 컬럼 수, 셀 최대 너비를 세밀하게 조절할 수 있습니다:

```swift
public struct RenderOptions: Equatable, Hashable, Sendable, Codable {
    public var maxRows: Int
    public var maxColumns: Int
    public var maxColumnWidth: Int?

    public init(maxRows: Int = 20, maxColumns: Int = 10, maxColumnWidth: Int? = nil)
}
```

```swift
print(df.render(maxRows: 6, maxColumns: 4))
print(df.render(options: RenderOptions(maxRows: 10, maxColumns: 8, maxColumnWidth: 24)))
```

### Markdown 표 내보내기 (`toMarkdown`)

README, 노트북, 리포트 등에 바로 붙여넣을 수 있는 GitHub Flavored Markdown 표 문자열을 생성합니다:

```swift
let md = df.toMarkdown(maxRows: 10, maxColumns: 6)
print(md)
// |  | city | temp |
// | --- | --- | --- |
// | 0 | Seoul | 21.5 |
// | 1 | Busan | NaN |
```

### 스키마 및 메모리 요약 (`info()` 및 `infoSummary()`)

```swift
print(df.info())
let info: DataFrameInfo = df.infoSummary()
print(info.rowCount, info.estimatedMemoryBytes)
print(info) // CustomStringConvertible을 통해 동일한 요약 텍스트 출력
```

---

## 5. 행, 컬럼 및 셀 선택

### 컬럼 선택 및 삭제

```swift
// 가변 인자 또는 배열로 컬럼 선택
let sub1 = try df.select("city", "temp")
let sub2 = try df.select(columns: ["city", "temp"])

// 특정 DataType의 컬럼만 선택
let numericOnly = try df.select(columnsOfType: .number)

// 이름으로 컬럼 삭제
let withoutTemp = try df.drop(columns: ["temp"])
```

### 컬럼 값 및 `Series` 추출

```swift
// [Double] 추출 (결측치나 비수치 값이 포함되어 있으면 .nonNumericColumn 발생)
let numbers: [Double] = try df.column("temp")

// 모든 컬럼 타입에 대한 [DataValue] 추출
let rawValues: [DataValue] = try df.values(in: "city")

// Series 추출 (오류 던지기 메서드 또는 옵셔널 서브스크립트)
let s1: Series = try df.series("temp")
let s2: Series? = df["temp"]

// 전체 DataFrame을 [String: [DataValue]] 딕셔너리로 변환
let dict: [String: [DataValue]] = df.toDictionary()
```

### 행 및 셀 접근 (`row(at:)`, `rows`, 서브스크립트, `head`, `tail`, `largest`, `smallest`)

```swift
// 단일 행 조회 (DataRow?)
let firstRow = df.row(at: 0)
let sameRow = df[row: 0]

// 전체 행 순회 ([DataRow])
for row in df.rows {
    print(row.index, row["city"] ?? .missing)
}

// 연속된 행 범위 슬라이싱 (DataFrame?)
if let slice = df[rows: 0..<2] {
    print(slice)
}

// 단일 셀 서브스크립트 조회 (DataValue?)
let cell = df["city", 0] // .string("Seoul")

// 처음 / 마지막 N개 행
let top3 = try df.head(3)
let bottom3 = try df.tail(3)

// 특정 컬럼 기준 상위 / 하위 N개 행
let highest = try df.largest(2, by: "temp")
let lowest = try df.smallest(2, by: "temp")
```

---

## 6. 필터링 및 쿼리

### 클로저 기반 필터링 (`filterRows`)

```swift
let warm = try df.filterRows { row in
    guard case .number(let t) = row["temp"] else { return false }
    return t >= 20.0
}
```

### 조합 가능한 타입 안전 조건식 (`RowPredicate` 및 `query(_ predicate:)`)

`RowPredicate`는 읽기 쉬운 정적 빌더 메서드와 논리 연산자(`&&`, `||`, `!`, `.and(_:)`, `.or(_:)`, `.negated`)를 제공합니다:

| 빌더 메서드 | 설명 |
| --- | --- |
| `.column(name, equals: value)` | `row[name] == value` |
| `.column(name, notEquals: value)` | `row[name] != value` |
| `.column(name, greaterThan: x)` | 수치형 `>` 비교 |
| `.column(name, greaterThanOrEqual: x)` | 수치형 `>=` 비교 |
| `.column(name, lessThan: x)` | 수치형 `<` 비교 |
| `.column(name, lessThanOrEqual: x)` | 수치형 `<=` 비교 |
| `.column(name, between: a, and: b, inclusive: true)` | 수치형 범위 포함 여부 검사 |
| `.column(name, in: Set<DataValue>)` | 집합 포함 여부(`isin`) 검사 |
| `.column(name, contains: substring, caseSensitive: true)` | 부분 문자열 포함 여부 검사 |
| `.isMissing(name)` / `.notMissing(name)` | 결측치 여부 검사 |

```swift
let predicate: RowPredicate =
    .column("temp", greaterThanOrEqual: 20)
    && .notMissing("city")
    && !.column("city", equals: "Jeju")

let filtered = try df.query(predicate)
```

### 문자열 표현식 쿼리 (`query(_ expression:)`)

`==`, `!=`, `>=`, `<=`, `>`, `<` 비교 연산자와 `&&`, `||` 조합을 지원합니다:

```swift
let result = try df.query("temp >= 20 && active == true")
let cities = try df.query("city == 'Seoul' || city == 'Busan'")
```

---

## 7. 컬럼 추가, 계산, 이름 변경 및 타입 캐스팅

### 컬럼 추가 또는 교체 (`withColumn`)

```swift
// 명시적 [DataValue] 배열로 추가/교체
let withCountry = try df.withColumn(
    name: "country",
    values: [.string("KR"), .string("KR"), .string("KR")]
)

// DataRow 클로저로 행마다 값을 계산하여 추가/교체 (pandas의 df.assign과 유사)
let withFahrenheit = try df.withColumn(name: "temp_f") { row in
    guard let c = row["temp"]?.numberValue else { return .missing }
    return .number(c * 9 / 5 + 32)
}
```

### 단일 컬럼 값 변환 (`mapped`)

```swift
let doubled = try df.mapped("temp") { value in
    guard case .number(let x) = value else { return value }
    return .number(x * 2)
}
```

### 컬럼 이름 변경 (`renamedColumns`)

```swift
let renamed = try df.renamedColumns(["temp": "temperature_c"])
```

### 명시적 타입 캐스팅 (`castColumn`)

컬럼을 `.number`, `.string`, `.boolean`, `.date`로 변환합니다. 변환할 수 없는 셀이 있으면 정확한 행 인덱스를 담은 `.invalidConversion(column:row:type:)` 오류를 던집니다.

```swift
let casted = try rawFrame
    .castColumn("amount", to: .number)
    .castColumn("enabled", to: .boolean)
    .castColumn("created", to: .date, dateFormat: "yyyy/MM/dd")
```

---

## 8. 날짜 및 문자열 컬럼 연산

### UTC 날짜 구성요소 추출 (`DateComponent` 및 `dateComponent(_:from:as:)`)

`DateComponent`는 `.year`, `.month`, `.day`, `.weekday`, `.hour`, `.minute`, `.second`를 지원합니다:

```swift
let withYearAndMonth = try df
    .dateComponent(.year, from: "created", as: "year")
    .dateComponent(.month, from: "created") // 기본 출력 컬럼 이름: "created_month"
```

### 문자열 컬럼 연산

pandas의 `df["col"].str.*` 접근자에 대응하는 문자열 변환 메서드들을 제공합니다:

| 메서드 | 설명 |
| --- | --- |
| `lowercased(_:as:)` | 문자열을 소문자로 변환합니다. |
| `uppercased(_:as:)` | 문자열을 대문자로 변환합니다. |
| `trimmed(_:as:)` | 앞뒤 공백 및 줄바꿈 문자를 제거합니다. |
| `stringLength(_:as:)` | 문자열의 글자 수를 수치형 컬럼으로 계산합니다(기본 이름: `"<col>_length"`). |
| `replacingSubstring(in:target:with:as:)` | `target` 부분 문자열을 `replacement`로 치환합니다. |
| `containsSubstring(_:_:caseSensitive:as:)` | 부분 문자열 포함 여부를 나타내는 불리언 컬럼을 생성합니다(기본 이름: `"<col>_contains"`). |

```swift
let cleaned = try df
    .trimmed("city")
    .uppercased("city", as: "city_upper")
    .stringLength("city", as: "name_len")
    .replacingSubstring(in: "city", target: "SEOUL", with: "Seoul")
    .containsSubstring("city", "ul", caseSensitive: false, as: "has_ul")
```

---

## 9. 결측치(Missing Value) 처리

| 메서드 | 반환 타입 | 설명 |
| --- | --- | --- |
| `isMissing(in:)` | `[Bool]` | 행 순서에 맞춘 불리언 배열(`.missing`인 위치가 `true`). |
| `missingMask()` | `DataFrame` | 동일한 크기의 불리언 `DataFrame`(결측 셀이 `true`). |
| `missingCounts()` | `[String: Int]` | 컬럼별 결측치 개수 딕셔너리. |
| `missingSummary()` | `MissingCounts` | 컬럼 순서를 보존하며 `counts`, `totalMissing`, 서브스크립트 및 출력용 `description`을 제공하는 구조체. |
| `fillMissing(in:with:)` | `DataFrame` | 지정된(또는 전체) 컬럼의 `.missing` 값을 상수 `DataValue`로 채움. |
| `forwardFilled()` | `DataFrame` | 이전의 가장 가까운 비결측 값으로 결측치를 채움(`ffill`). |
| `backwardFilled()` | `DataFrame` | 이후의 가장 가까운 비결측 값으로 결측치를 채움(`bfill`). |
| `dropMissing(in:)` | `DataFrame` | 지정된(또는 전체) 컬럼에 `.missing`이 있는 행을 제거(`dropna`). |
| `replacing(_:with:in:)` | `DataFrame` | 지정된(또는 전체) 컬럼에서 특정 `DataValue`를 다른 값으로 치환. |

```swift
print(df.missingSummary())
// city    1
// temp    1
// active  1
// Total missing: 3 (rows: 3)

let filled = try df.fillMissing(in: ["temp"], with: .number(20.0))
let ffilled = try df.forwardFilled()
let bfilled = try df.backwardFilled()
let complete = try df.dropMissing()
```

---

## 10. 중복 제거, 정렬, 샘플링 및 행 연결

### 중복 행 탐지 및 제거 (`duplicatedRows`, `dropDuplicates`, `DuplicateKeep`)

`DuplicateKeep` 케이스: `.first`(기본값), `.last`, `.none`.

```swift
let mask: [Bool] = try df.duplicatedRows(in: ["city"], keep: .first)
let deduped = try df.dropDuplicates(in: ["city"], keep: .last)
```

### 다중 컬럼 정렬 (`sorted(by:)`, `SortDescriptor`, `SortOrder`)

결측치는 `.ascending`이나 `.descending`에 상관없이 항상 마지막에 배치됩니다:

```swift
let sorted = try df.sorted(by: [
    SortDescriptor("city", order: .ascending),
    SortDescriptor("temp", order: .descending),
])
```

### 재현 가능한 무작위 샘플링 (`sample(_:seed:)`)

비복원 추출로 최대 `count`개의 행을 원본 행 순서대로 샘플링합니다. `seed`를 지정하면 항상 동일한 결과를 얻습니다:

```swift
let sample = try df.sample(10, seed: 42)
```

### 수직 행 연결 (`DataFrame.concatenatingRows`)

```swift
let combined = try DataFrame.concatenatingRows([part1, part2])
```

---

## 11. Series 및 컬럼 단위 수치 연산

### `Series` 활용하기

```swift
let s = try df.series("temp")
print(s)

// 축소 연산 및 분위수
let sum: DataValue = s.reduced(using: .sum)
let q90: Double? = s.quantile(0.9)
let distinctCount: Int = s.uniqueCount()
let distinctValues: [DataValue] = s.unique()
let frequencies: DataFrame = try s.valueCounts()

// 벡터화 사칙연산 (+, -, *, /: 다른 Series 또는 Double 스칼라와 연산)
let centered = try s - 20.0
let scaled = try s * 1.8 + 32.0
let combined = try s + s

// 원소별 변환
let absSeries = try s.abs()
let clipped = try s.clipped(lower: 0, upper: 30)
let rounded = try s.rounded(decimals: 1)
let shifted = s.shifted(by: 1)
let diff = try s.difference(periods: 1)
let pct = try s.percentChange(periods: 1)
let cumSum = try s.cumulative(using: .sum)
let ranked = try s.ranked(ascending: false)

// Series 결측치 처리 및 필터링
let noMissing = s.droppedMissing()
let filled = try s.filledMissing(with: .number(0))
let ffill = s.forwardFilled()
let bfill = s.backwardFilled()
let replaced = try s.replacing(.missing, with: .number(0))
let inSet = s.isin([.number(18.0), .number(21.5)])
```

### `DataFrame`의 컬럼 단위 수치 연산

| 메서드 | 설명 |
| --- | --- |
| `combining(_:_:using:as:)` | 두 수치형 컬럼 간 원소별 `.add`, `.subtract`, `.multiply`, `.divide` 결과를 `outputColumn`에 기록(대규모 컬럼에서 MLX 가속). |
| `scaling(_:by:using:as:)` | 수치형 컬럼에 스칼라 `.add`, `.subtract`, `.multiply`, `.divide` 적용(대규모 컬럼에서 MLX 가속). |
| `cumulative(_:using:)` | 누적 `.sum`, `.product`, `.minimum`, `.maximum` 계산(`CumulativeFunction`). |
| `shifted(_:by:)` | 컬럼 값을 `periods` 행만큼 이동하고 빈 위치를 `.missing`으로 채움. |
| `difference(_:periods:)` | `periods` 행 이전 값과의 차분(`x[i] - x[i - periods]`) 계산. |
| `percentChange(_:periods:)` | `periods` 행 이전 값 대비 변화율(`(x[i] - x[i - periods]) / x[i - periods]`) 계산. |
| `clipped(_:lower:upper:)` | 수치 값을 `lower...upper` 범위로 제한. |
| `rounded(_:decimals:)` | 소수점 `decimals` 자리로 반올림. |
| `abs(_:)` | 수치형 컬럼의 절댓값 계산. |
| `ranked(_:ascending:as:)` | 동점 시 평균 순위를 부여하는 1 기반 순위 계산. |
| `reduced(using:)` | 모든 수치형 컬럼을 `AggregateFunction`으로 축소한 단일 행 `DataFrame` 반환. |
| `quantile(_:)` | 모든 수치형 컬럼의 `p` 분위수(`0...1`)를 담은 단일 행 `DataFrame` 반환. |
| `covariance()` | 수치형 컬럼 간 표본 공분산 행렬 `DataFrame` 계산(대규모 결측 없는 프레임에서 `MLX.matmul` 사용). |
| `correlation()` | 수치형 컬럼 간 피어슨 상관계수 행렬 `DataFrame` 계산(대규모 결측 없는 프레임에서 `MLX.matmul` 사용). |
| `valueCounts(of:)` | 빈도 내림차순으로 정렬된 2열 `DataFrame`(`value`, `count`) 반환. |
| `uniqueCounts()` | 각 컬럼의 고유한 비결측 값 개수를 담은 `[String: Int]` 반환. |
| `uniqueSummary()` | 컬럼 순서가 보존되며 서브스크립트와 출력용 `description`을 제공하는 `UniqueCounts` 구조체 반환. |

```swift
let enriched = try salesFrame
    .combining("revenue", "cost", using: .subtract, as: "profit")
    .scaling("profit", by: 100.0, using: .multiply, as: "profit_cents")
    .cumulative("revenue", using: .sum)
    .ranked("profit", ascending: false, as: "profit_rank")

let corrMatrix = try enriched.correlation()
print(corrMatrix)
```

---

## 12. 기술 통계 및 구조화된 결과 타입

모든 통계 및 요약 API는 프로그래밍 방식으로 프로퍼티에 접근할 수 있을 뿐 아니라 `print(...)`로 전달했을 때 pandas 스타일의 요약 표를 출력하는 Swift 구조체를 반환합니다.

| 메서드 | 반환 타입 | 주요 프로퍼티 및 메서드 |
| --- | --- | --- |
| `statistics()` | `DataFrameStatistics` | `columns: [String: ColumnStatistics]`, `rowCount`, `columnCount`, `subscript(_ column: String)`, `description` |
| `describeStructured(percentiles:include:)` | `DataFrameDescription` | `columnOrder`, `numeric: [String: ColumnStatistics]`, `nonNumeric: [String: NonNumericSummary]`, `percentileLabels`, `describe()`, `description` |
| `describe(percentiles:include:)` | `String` | `describeStructured(percentiles:include:).describe()`와 동일한 문자열 반환 |
| `summaryTable()` | `DataFrameSummary` | `columnOrder`, `summaries: [String: ColumnSummary]`, `subscript`, `describe()`, `description` |
| `summary()` | `[String: ColumnSummary]` | `ColumnSummary`(`count`, `mean`, `standardDeviation`, `median`, `min`, `max`) 딕셔너리 |
| `infoSummary()` | `DataFrameInfo` | `rowCount`, `columns: [DataFrameInfo.Column]`, `estimatedMemoryBytes`, `describe()`, `description` |
| `missingSummary()` | `MissingCounts` | `columnOrder`, `counts: [String: Int]`, `totalMissing`, `rowCount`, `describe()`, `description` |
| `uniqueSummary()` | `UniqueCounts` | `columnOrder`, `counts: [String: Int]`, `rowCount`, `describe()`, `description` |
| `columnTypes()` / `dtypes` | `ColumnTypes` | `columnOrder`, `types: [String: DataType]`, `describe()`, `description` |

### 예제: 구조화된 통계 조회 및 사용자 지정 분위수

```swift
let stats: DataFrameStatistics = df.statistics()
if let tempStats: ColumnStatistics = stats["temp"] {
    print("Mean:", tempStats.mean ?? .nan)
    print("IQR:", (tempStats.q3 ?? 0) - (tempStats.q1 ?? 0))
}

// 사용자 지정 분위수 및 포함할 타입 필터링 (pandas의 df.describe(percentiles=[0.1, 0.5, 0.9], include=['number'])와 동일)
let desc: DataFrameDescription = df.describeStructured(
    percentiles: [0.1, 0.5, 0.9],
    include: [.number]
)
print(desc)
print(df.summaryTable())
print(df.dtypes)
print(df.uniqueSummary())
```

---

## 13. 그룹화

### `groupBy`, `GroupedDataFrame`, `Aggregation`, `AggregateFunction`

`AggregateFunction` 케이스:

- `.sum`
- `.mean`
- `.min`
- `.max`
- `.count`
- `.median`
- `.variance` (표본 분산, `n - 1`)
- `.standardDeviation` (표본 표준편차, `n - 1`)
- `.first`
- `.last`

그룹 키 컬럼 중 하나라도 `.missing`인 행은 그룹 결과에서 제외되며(pandas의 기본 `dropna=True`와 동일), 그룹 순서는 항상 결정론적으로 정렬됩니다.

```swift
// 단일 집계
let byRegion = try sales
    .groupBy("Region")
    .aggregate("Sales", using: .sum)

// 다중 키 및 다중 집계
let detailed = try sales
    .groupBy("Region", "Product")
    .aggregate([
        Aggregation("Sales", using: .sum),
        Aggregation("Sales", using: .mean),
        Aggregation("Sales", using: .standardDeviation),
        Aggregation("Sales", using: .count),
    ])
print(detailed)

// 그룹 집계 결과를 원본 행 순서에 맞춰 브로드캐스트 (pandas의 groupby.transform과 동일)
let withGroupMean = try sales
    .groupBy("Region")
    .transform("Sales", using: .mean)
```

---

## 14. 조인

단일 키 또는 이름이 서로 다른 다중 키에 대해 `.left`, `.inner`, `.right`, `.full` 조인(`JoinKind`)과 중복 컬럼 이름 접미사(`suffixes`) 설정을 지원합니다:

```swift
// 단일 키 조인
let joined = try orders.join(customers, on: "customer_id", kind: .left)

// 좌/우 키 이름이 다른 다중 키 조인 및 접미사 지정
let multiJoined = try leftFrame.join(
    rightFrame,
    leftOn: ["region_id", "year"],
    rightOn: ["id", "fiscal_year"],
    kind: .inner,
    suffixes: ("_left", "_right")
)
```

---

## 15. 재구조화

### `melt` (Wide에서 Long으로 변환)

```swift
let longFrame = try wideFrame.melt(
    idColumns: ["region"],
    variableName: "metric",
    valueName: "amount"
)
```

### `pivotTable` (Long에서 Wide로 집계 변환)

```swift
let pivoted = try longFrame.pivotTable(
    index: ["region"],
    columns: "metric",
    values: "amount",
    using: .sum
)
```

### `exploded` (구분자로 분리된 문자열을 여러 행으로 전개)

지정한 `column`의 문자열을 `separator`(기본값 `","`)로 분리하여 각 토큰마다 나머지 컬럼 값을 복제한 새로운 행을 생성합니다(pandas의 `explode()`와 유사):

```swift
let tags = try DataFrame(columns: [
    (name: "id", values: [.number(1), .number(2)]),
    (name: "tags", values: [.string("swift,mlx,data"), .string("pandas")]),
])
let exploded = try tags.exploded("tags", separator: ",")
print(exploded)
// 4개 행: (1, "swift"), (1, "mlx"), (1, "data"), (2, "pandas")
```

### `transposed` (행과 열 전치)

원본 컬럼 이름들을 첫 번째 컬럼(`indexColumnName`, 기본값 `"column"`)으로 옮기고 원본의 각 행을 컬럼(`row_0`, `row_1`, ...)으로 변환합니다:

```swift
let t = try df.transposed(indexColumnName: "feature", rowPrefix: "sample_")
print(t)
```

---

## 16. 윈도우 함수

### 행 개수 기반 롤링 윈도우 (`rolling(_:window:using:)`)

직전 `window`개 행에 대해 `AggregateFunction`을 계산합니다. 첫 윈도우가 채워지기 전 위치(`0 ..< window - 1`)는 `.missing`이 됩니다. 대규모 `.sum`, `.mean`, `.count` 윈도우는 MLX 누적합(prefix sum)으로 벡터화됩니다.

```swift
let rolled = try df.rolling("temp", window: 3, using: .mean)
```

### 시간 간격 기반 롤링 윈도우 (`rolling(_:on:interval:using:)`)

각 행 `i`에 대해 `dateColumn`의 타임스탬프가 `(t_i - interval) ... t_i` 범위 안에 들어오는 이전 행 `j <= i`들을 모아 윈도우 집계를 수행합니다:

```swift
let timeRolled = try events.rolling(
    "latency_ms",
    on: "timestamp",
    interval: 60.0, // 직전 60초 윈도우
    using: .mean
)
```

### 확장 윈도우 (`expanding(_:using:)`)

모든 행에 대해 `0...row` 범위의 누적 윈도우에 `AggregateFunction`을 적용합니다:

```swift
let cumulativeMean = try df.expanding("temp", using: .mean)
```

---

## 17. MLX 런타임, `ExecutionPolicy` 및 `MLXVector` 상호운용

### Bandal의 MLX 활용 범위

Bandal의 `NumericEngine`은 다음 연산들에 MLX Swift를 사용합니다:

- 기술 통계 모멘트 계산 (`summary`, `statistics`, `describe`, `describeStructured`)
- 단일 컬럼 및 전체 프레임 축소 연산 (`sum`, `mean`, `min`, `max`, `count`, `variance`, `standardDeviation`)
- 그룹별 축소 연산 (`groupBy.aggregate`, `groupBy.transform`, `pivotTable`)
- 롤링 및 확장 윈도우 (`sum`, `mean`, `count`)
- 원소별 단항/이항 수치 연산 (`combining`, `scaling`, `Series`의 `+`, `-`, `*`, `/`, `abs`, `clipped`, `rounded`, `difference`, `percentChange`, `cumulative`)
- `MLX.matmul`을 활용한 쌍별 `covariance()` 및 `correlation()` 행렬 계산

### `ExecutionPolicy` 및 `BandalRuntime`

```swift
public enum ExecutionPolicy: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case automatic // 임계값 이상일 때 MLX CPU 스트림(float64) 사용, 미만일 때 네이티브 Swift 사용
    case native    // 순수 네이티브 Swift(Double) 실행
    case cpu       // 비어 있지 않은 모든 입력에 대해 MLX CPU 스트림(float64) 강제 사용
    case gpu       // 값이 Float32로 정확히 표현 가능할 때 MLX GPU 스트림(float32) 사용, 아니면 CPU float64 사용
}
```

런타임 실행 정책 확인 및 스코프별 전환:

```swift
print(BandalRuntime.isMLXAvailable)
print(BandalRuntime.executionBackend)
print(BandalRuntime.executionPolicy)

// 특정 블록을 GPU 스트림이나 CPU/네이티브 정책으로 실행
let corr = try BandalRuntime.withExecutionPolicy(.gpu) {
    try largeFrame.correlation()
}
```

프로세스 환경 변수에 `BANDAL_DISABLE_MLX=1`을 설정하면 전역적으로 네이티브 Swift 백엔드를 강제할 수 있습니다.

### `MLXVector` 상호운용 (`toMLXVector` 및 `init(mlxColumns:)`)

수치형 컬럼을 결측치 마스크와 함께 MLX의 `MLXArray` 텐서로 직접 변환하거나 다시 `DataFrame`으로 복원할 수 있습니다:

```swift
let vector: MLXVector = try df.toMLXVector("temp")
print(vector)
// 비결측 Double 배열을 확인하거나 MLXArray 텐서 생성:
let array = vector.makeMLXArray() // MLXArray? (CPU에서는 float64, GPU에서는 float32)

// MLXVector 배열로부터 DataFrame 재구성
let roundTripped = try DataFrame(mlxColumns: [
    (name: "temp", vector: vector)
])
```

---

## 18. 오류

오류를 던지는 모든 API는 `Error`, `Equatable`, `Hashable`, `Sendable`, `LocalizedError`, `CustomStringConvertible`을 준수하는 `DataFrameError`를 던집니다:

| 케이스 | 발생 조건 |
| --- | --- |
| `.emptyColumns` | 컬럼이 0개인 프레임을 생성하거나 선택하려 할 때. |
| `.emptyColumnName` | 컬럼 이름이 `""`일 때. |
| `.duplicateColumnName(String)` | 두 컬럼의 이름이 중복될 때. |
| `.unequalColumnLengths(column:expected:actual:)` | 컬럼의 길이가 프레임의 행 개수와 일치하지 않을 때. |
| `.columnNotFound(String)` | 존재하지 않는 컬럼 이름을 참조할 때. |
| `.nonNumericColumn(String)` | 문자열, 불리언, 날짜 또는 결측치가 섞인 컬럼에 수치 전용 연산을 호출할 때. |
| `.nonFiniteValue(column:row:)` | `Double` 값이 `NaN`이거나 `±infinity`일 때. |
| `.columnNameConflict(String)` | 조인이나 `transform` 결과 컬럼 이름이 기존 컬럼과 충돌할 때. |
| `.invalidCSV(String)` | 잘못된 형식의 CSV/JSON 입력이거나 잘못된 쿼리/캐스팅 구문일 때. |
| `.invalidWindowSize(Int)` | 롤링 윈도우 크기가 `<= 0`일 때. |
| `.incompatibleSchema(expected:actual:)` | `concatenatingRows` 호출 시 컬럼 이름이나 순서가 일치하지 않을 때. |
| `.invalidConversion(column:row:type:)` | `castColumn`, `dateComponent`, 문자열 연산 중 `row` 행에서 호환되지 않는 값을 만났을 때. |
