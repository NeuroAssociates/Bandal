# 가이드 02: 데이터 입출력(CSV / JSON) 및 스키마 탐색

> **언어**: [English](02-Data-IO-and-Inspection.md) | **한국어**  
> **네비게이션**: [← 가이드 01: 핵심 데이터 구조](01-Core-Data-Structures.ko.md) | [API 레퍼런스](../API.ko.md) | **다음**: [가이드 03: 인덱싱, 필터링 및 쿼리 →](03-Indexing-Filtering-and-Querying.ko.md)

---

## 개요

모든 데이터 분석 워크플로는 외부 데이터를 불러오고 데이터의 크기(Shape), 컬럼 타입, 결측치 현황, 샘플 행을 점검하는 탐색적 데이터 분석(EDA)에서 시작됩니다. `pandas`에서는 `pd.read_csv`, `pd.read_json`, `df.info()`, `df.dtypes`, `df.to_markdown()`이 이 역할을 담당합니다.

이 가이드에서는 Bandal의 스트리밍 CSV/JSON 입출력 엔진과 구조화된 스키마 진단 도구들을 상세히 다룹니다.

- **스트리밍 CSV 로드 및 저장** (`init(contentsOfCSV:options:)`, `CSVReadOptions`, `writeCSV`)
- **수치, 통화(`$`), 천 단위 쉼표, 불리언, 날짜 자동 파싱 규칙**
- **JSON 로드 및 저장** (`init(contentsOfJSON:dateFormat:)`, `init(jsonData:dateFormat:)`, `writeJSON`, `jsonData`)
- **스키마 및 메모리 진단** (`shape`, `dtypes`, `columnTypes()`, `info()`, `infoSummary()`, `DataFrameInfo`)
- **테이블 출력 제어 및 마크다운 변환** (`render(options:)`, `RenderOptions`, `toMarkdown(maxRows:maxColumns:)`)

---

## 1. 한눈에 보는 비교: `pandas` vs. `Bandal`

| 작업 | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| 기본 CSV 읽기 | `pd.read_csv("data.csv")` | `try DataFrame(contentsOfCSV: url)` |
| 상세 옵션으로 CSV 읽기 | `pd.read_csv("data.csv", sep="\t", nrows=100, usecols=["a","b"], na_values=["N/A"], dtype={"id": str})` | `try DataFrame(contentsOfCSV: url, options: CSVReadOptions(...))` |
| CSV 파일 쓰기 | `df.to_csv("out.csv", index=False)` | `try df.writeCSV(to: url, delimiter: ",", includeHeader: true)` |
| JSON 파일 읽기 | `pd.read_json("data.json")` | `try DataFrame(contentsOfJSON: url, dateFormat: nil)` |
| JSON 파일 쓰기 | `df.to_json("out.json", orient="records")` | `try df.writeJSON(to: url, prettyPrinted: true)` |
| 행/열 차원 확인 | `df.shape` | `df.shape` (`[rowCount, columnNames.count]`) |
| 컬럼별 데이터 타입 확인 | `df.dtypes` | `df.dtypes` (`ColumnTypes` 구조체 반환) |
| 스키마 및 메모리 요약 | `df.info()` | `df.info()` (문자열) 또는 `df.infoSummary()` (`DataFrameInfo`) |
| 마크다운 표 내보내기 | `df.to_markdown()` | `df.toMarkdown(maxRows: nil, maxColumns: nil)` |

---

## 2. CSV 파일 읽기 및 쓰기

Bandal의 CSV 리더는 RFC 4180 인용 규칙을 준수하며 64 KB 청크 단위로 파일을 스트리밍 파싱합니다. `maximumRows`를 지정하면 요청한 행 수만큼 읽은 즉시 파일 읽기를 중단하므로, 수 기가바이트 크기의 대용량 CSV 파일도 순식간에 미리보기할 수 있습니다.

### 2.1 기본 CSV 로드

```swift
import Foundation
import Bandal

let csvURL = URL(fileURLWithPath: "/path/to/sales.csv")

// 전체 CSV 로드 (쉼표 구분자, 첫 번째 레코드를 헤더로 사용)
let df = try DataFrame(contentsOfCSV: csvURL)

// 탭 구분자(TSV) 파일에서 상위 500개 행만 빠르게 미리보기
let preview = try DataFrame(
    contentsOfCSV: URL(fileURLWithPath: "/path/to/logs.tsv"),
    delimiter: "\t",
    maximumRows: 500
)
```

### 2.2 `CSVReadOptions`를 활용한 정밀 파싱 제어

실무 CSV 데이터에는 다양한 결측치 표기(`"NA"`, `"NULL"`, `"-"`), 선행 0이 유지되어야 하는 코드 컬럼(`"00124"`), 통화 기호나 천 단위 쉼표가 포함된 금액(`"$1,250.50"`), 또는 날짜 문자열이 섞여 있는 경우가 많습니다. `CSVReadOptions`를 사용하면 단일 패스로 이를 모두 처리할 수 있습니다.

```swift
public struct CSVReadOptions: Equatable, Sendable, CustomStringConvertible {
    public var delimiter: Character           // 기본값: ","
    public var maximumRows: Int?              // 기본값: nil (전체 행 로드)
    public var hasHeader: Bool                // 기본값: true
    public var columnNames: [String]?         // 기본값: nil (헤더 이름을 덮어쓰거나 직접 지정)
    public var selectedColumns: [String]?     // 기본값: nil (필요한 컬럼만 선택적으로 로드)
    public var missingTokens: Set<String>     // 기본값: [""]
    public var infersNumbers: Bool            // 기본값: true
    public var columnTypes: [String: DataType]// 기본값: [:] (컬럼별 명시적 타입 지정)
    public var dateFormat: String?            // 기본값: nil (예: "yyyy-MM-dd")
}
```

#### 수치 및 통화 포맷 자동 파싱 규칙
`infersNumbers == true`(기본값)일 때 Bandal은 아래 형태의 문자열을 자동으로 `.number(Double)`로 파싱합니다.
- 일반 정수 및 부동소수점 (`42`, `-3.14`, `1.5e4`)
- 따옴표로 감싸진 천 단위 쉼표 숫자 (`"1,250,000.75"` → `.number(1250000.75)`)
- 달러(`$`) 통화 기호가 붙은 숫자 (`"$1,250.50"` 또는 `"-$45.00"` → `.number(1250.5)`, `.number(-45.0)`)

#### `CSVReadOptions` 종합 실습 예제

```swift
import Foundation
import Bandal

// 임시 디렉토리에 실무 형태의 샘플 CSV 생성
let rawCSV = """
order_id,order_date,region,amount,is_vip,notes
00101,2026-03-01,Seoul,"$1,250.50",true,Fast shipping
00102,2026-03-02,Busan,N/A,false,NULL
00103,2026-03-03,Seoul,"$890.00",true,-
"""

let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("orders_demo.csv")
try rawCSV.write(to: tempURL, atomically: true, encoding: .utf8)

let options = CSVReadOptions(
    delimiter: ",",
    maximumRows: nil,
    hasHeader: true,
    selectedColumns: ["order_id", "order_date", "region", "amount", "is_vip"], // 로드 단계에서 "notes" 제외
    missingTokens: ["", "N/A", "NULL", "-"],
    infersNumbers: true,
    columnTypes: [
        "order_id": .string,      // 선행 0("00101")이 101.0으로 변환되지 않도록 문자열로 고정
        "order_date": .date,      // 날짜 타입으로 파싱
        "is_vip": .boolean        // "true"/"false"를 불리언 타입으로 파싱
    ],
    dateFormat: "yyyy-MM-dd"      // .date 컬럼 파싱에 사용할 UTC DateFormatter 패턴
)

let orders = try DataFrame(contentsOfCSV: tempURL, options: options)
print(orders)
print("\nColumn Types:")
print(orders.dtypes)
```

**콘솔 출력**:
```text
  order_id            order_date  region  amount  is_vip
0    00101  2026-03-01T00:00:00Z   Seoul  1250.5    true
1    00102  2026-03-02T00:00:00Z   Busan     NaN   false
2    00103  2026-03-03T00:00:00Z   Seoul     890    true

[3 rows x 5 columns]

Column Types:
order_id      string
order_date    date
region        string
amount        number
is_vip        boolean
Columns: 5
```

### 2.3 CSV 파일로 내보내기 (`writeCSV`)

`writeCSV(to:delimiter:includeHeader:)`는 데이터를 UTF-8 CSV 포맷으로 디스크에 스트리밍 기록합니다. 결측치(`.missing`)는 빈 필드(`""`)로, 날짜(`.date`)는 UTC ISO-8601 문자열로 기록되며, 구분자·따옴표·줄바꿈이 포함된 문자열은 RFC 4180 규칙에 따라 자동으로 이스케이프됩니다.

```swift
let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("cleaned_orders.csv")
try orders.writeCSV(to: outURL, delimiter: ",", includeHeader: true)
```

---

## 3. JSON 데이터 읽기 및 쓰기

Bandal은 데이터 분석에서 가장 널리 쓰이는 두 가지 JSON 구조를 모두 지원합니다.
1. **레코드 배열 형태 (`records` orientation)**: `[{"colA": 1, "colB": "x"}, {"colA": 2, "colB": "y"}]`
2. **컬럼 배열 딕셔너리 형태 (`columns` orientation)**: `{"colA": [1, 2], "colB": ["x", "y"]}`

JSON의 `null` 값은 자동으로 `.missing`으로 변환되며, `dateFormat` 인자(예: `"yyyy-MM-dd"`)를 전달하면 해당 포맷과 일치하는 문자열을 자동으로 `.date(Date)` 셀로 파싱합니다.

```swift
let jsonText = """
[
  {"date": "2026-04-01", "sensor": "S1", "reading": 41.2},
  {"date": "2026-04-02", "sensor": "S1", "reading": null},
  {"date": "2026-04-03", "sensor": "S2", "reading": 39.8}
]
"""

let sensorDF = try DataFrame(
    jsonData: Data(jsonText.utf8),
    dateFormat: "yyyy-MM-dd"
)
print(sensorDF)

// 레코드 배열 형태의 JSON Data로 직렬화 (들여쓰기 포맷팅 활성화)
let exportedData = try sensorDF.jsonData(prettyPrinted: true)
print(String(decoding: exportedData, as: UTF8.self))
```

**콘솔 출력**:
```text
                   date  reading  sensor
0  2026-04-01T00:00:00Z     41.2      S1
1  2026-04-02T00:00:00Z      NaN      S1
2  2026-04-03T00:00:00Z     39.8      S2

[3 rows x 3 columns]
[
  {
    "date" : "2026-04-01T00:00:00Z",
    "reading" : 41.2,
    "sensor" : "S1"
  },
  {
    "date" : "2026-04-02T00:00:00Z",
    "reading" : null,
    "sensor" : "S1"
  },
  {
    "date" : "2026-04-03T00:00:00Z",
    "reading" : 39.8,
    "sensor" : "S2"
  }
]
```

> **`Codable` 지원 안내**: `DataFrame`, `Series`, `DataRow`, `DataValue`는 모두 Swift 표준 `Codable` 프로토콜도 준수하므로, 앱이나 서버의 `Codable` 모델 구조체 내부 프로퍼티로 직접 포함하여 직렬화/역직렬화할 수도 있습니다.

---

## 4. 스키마 및 메모리 진단 (`dtypes`, `info()`, `infoSummary()`)

Bandal의 중요한 설계 원칙 중 하나는 **모든 진단 및 통계 API가 구조화된 Swift 인스턴스를 반환**한다는 점입니다. 반환된 인스턴스를 `print(...)`에 넣으면 `pandas` 스타일의 표가 출력되고, 코드에서는 프로퍼티와 서브스크립트로 값을 즉시 활용할 수 있습니다.

### 4.1 차원(`shape`) 및 컬럼 타입(`dtypes`) 확인

```swift
print("Rows: \(orders.rowCount), Columns: \(orders.columnNames.count)")
print("Shape: \(orders.shape)") // [3, 5]

// 구조화된 ColumnTypes 인스턴스 반환
let dtypes: ColumnTypes = orders.dtypes
print(dtypes) // 정렬된 컬럼별 타입 표 출력

// 서브스크립트를 통한 프로그래밍 방식의 타입 검사
if dtypes["amount"] == .number {
    print("The 'amount' column is ready for numerical analysis.")
}
```

### 4.2 `info()` vs. `infoSummary()` (`DataFrameInfo`)

- `df.info()`는 `pandas`의 `df.info()`와 동일한 레이아웃의 요약 문자열을 반환합니다.
- `df.infoSummary()`는 `rowCount`, `columns: [DataFrameInfo.Column]`, `estimatedMemoryBytes` 프로퍼티를 가진 `DataFrameInfo` 구조체를 반환합니다.

```swift
// 1. Pandas 스타일의 info() 텍스트 리포트 출력
print(orders.info())

print("\n--- Structured Inspection ---")
// 2. infoSummary()를 활용한 프로그래밍 방식의 스키마 점검
let info: DataFrameInfo = orders.infoSummary()
print("Estimated payload memory: \(info.estimatedMemoryBytes) bytes")
for col in info.columns where col.nonMissingCount < info.rowCount {
    let missing = info.rowCount - col.nonMissingCount
    print("Column '\(col.name)' (\(col.type)) has \(missing) missing value(s).")
}
```

**콘솔 출력**:
```text
DataFrame
RangeIndex: 3 entries, 0 to 2
Data columns (total 5 columns):
 #  Column      Non-Null Count  Dtype   
 0  order_id    3 non-null      object  
 1  order_date  3 non-null      datetime
 2  region      3 non-null      object  
 3  amount      2 non-null      float64 
 4  is_vip      3 non-null      bool    
---
dtypes: bool(1), datetime(1), float64(1), object(2)
memory usage: ~70 bytes (estimated payload)

--- Structured Inspection ---
Estimated payload memory: 70 bytes
Column 'amount' (number) has 1 missing value(s).
```

---

## 5. 콘솔 출력 포맷 제어 및 마크다운 표 생성

### 5.1 `RenderOptions`로 출력 행/열 개수 및 셀 너비 제어하기

기본적으로 `print(df)`는 최대 `10`개의 행과 `8`개의 컬럼을 표시하며, 이를 초과하는 프레임은 중간에 생략 기호(`…`)를 삽입합니다. 더 많은 행/열을 출력하거나 긴 문자열 셀을 일정 길이로 자르고 싶다면 `df.render(options:)` 또는 `df.render(maxRows:maxColumns:)`를 사용합니다.

```swift
let customView = orders.render(
    options: RenderOptions(
        maxRows: 20,
        maxColumns: 10,
        maxColumnWidth: 12 // 12자를 초과하는 셀 텍스트를 "…"로 축약
    )
)
print(customView)
```

```text
  order_id    order_date  region  amount  is_vip
0    00101  2026-03-01T…   Seoul  1250.5    true
1    00102  2026-03-02T…   Busan     NaN   false
2    00103  2026-03-03T…   Seoul     890    true

[3 rows x 5 columns]
```

### 5.2 GitHub 스타일 마크다운 표 생성 (`toMarkdown`)

분석 결과를 보고서, 문서, 또는 LLM 프롬프트에 바로 붙여넣을 수 있도록 `toMarkdown(maxRows:maxColumns:)` 메서드를 제공합니다.

```swift
print(orders.toMarkdown())
```

**콘솔 출력**:
```markdown
| order_id | order_date | region | amount | is_vip |
| --- | --- | --- | --- | --- |
| 00101 | 2026-03-01T00:00:00Z | Seoul | 1250.5 | true |
| 00102 | 2026-03-02T00:00:00Z | Busan | NaN | false |
| 00103 | 2026-03-03T00:00:00Z | Seoul | 890 | true |
```

---

> **다음 단계**: 데이터를 불러오고 스키마를 점검하는 방법을 익혔다면 **[가이드 03: 인덱싱, 필터링 및 쿼리 →](03-Indexing-Filtering-and-Querying.ko.md)**로 이동하여 원하는 행과 열을 추출하고 조건으로 필터링하는 방법을 학습하세요.
