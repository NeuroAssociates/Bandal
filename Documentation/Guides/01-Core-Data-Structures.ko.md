# 가이드 01: 핵심 데이터 구조 및 Swift 값 타입 모델

> **언어**: [English](01-Core-Data-Structures.md) | **한국어**  
> **네비게이션**: [API 레퍼런스](../API.ko.md) | **다음**: [가이드 02: 데이터 입출력 및 탐색 →](02-Data-IO-and-Inspection.ko.md)

---

## 개요

Python의 `pandas`를 사용해 본 경험이 있다면 테이블 형태 데이터 분석의 핵심 개념에 이미 익숙할 것입니다. **`DataFrame`**은 이름이 지정된 컬럼들로 이루어진 2차원 테이블이고, **`Series`**는 1차원 단일 컬럼 벡터입니다.

**Bandal**은 이러한 `pandas`의 멘탈 모델을 그대로 유지하면서, Swift의 정적 타입 안전성, 값 시맨틱(Value Semantics), 그리고 하드웨어 가속 컬럼형 메모리 구조(`mlx-swift`)를 결합한 라이브러리입니다. 이 가이드에서는 `pandas`의 핵심 개념이 Bandal의 핵심 타입에 어떻게 매핑되는지 학습합니다.

- **`DataValue`** — 기본 Swift 리터럴(`Double`, `Int`, `String`, `Bool`, `nil`)을 직접 사용할 수 있는 타입 안전한 셀 값(`Double`, `String`, `Bool`, `Date`, `.missing`).
- **`DataType`** — 컬럼의 논리적 데이터 타입(`.number`, `.string`, `.boolean`, `.date`, `.mixed`, `.unknown`).
- **`Series`** — 연속된(Contiguous) 타입별 컬럼 버퍼(`Float64Column`, `Int64Column`, `StringColumn`, `CategoricalColumn`, `BoolColumn`, `DateColumn`) 기반의 1차원 명명 컬럼.
- **`DataFrame`** — 변경되지 않은 컬럼을 `O(1)`로 공유하는 불변(Immutable) 2차원 테이블.
- **`DataRow`** — 컬럼 이름 또는 정수 위치 서브스크립트로 접근 가능한 단일 행 읽기 전용 뷰.
- **`DataFrameError`** — 스키마 불일치, 존재하지 않는 컬럼 참조, 타입 변환 실패 시 발생하는 구조화된 에러 타입.

---

## 1. 패러다임 비교: Python `pandas` vs. Swift `Bandal`

| 개념 | Python `pandas` | Swift `Bandal` | 데이터 분석가에게 주는 이점 |
| --- | --- | --- | --- |
| **가변성 (Mutability)** | 가변 객체 (`df["a"] = 1`, `inplace=True`) | 불변 값 타입 (`let df2 = try df.withColumn(...)`) | 원본 데이터가 의도치 않게 변형되는 부작용(`SettingWithCopyWarning` 등)을 원천 차단하며, 변경되지 않은 컬럼은 `O(1)`로 메모리를 공유합니다. |
| **결측치 (Missing Values)** | `np.nan`, `None`, `pd.NA` 혼재 | `DataValue.missing` (또는 리터럴 `nil`) | 수치, 문자열, 불리언, 날짜 전 타입에서 단일한 결측 표현을 사용하며, 내부적으로 64비트 `BitSet` 널 비트맵으로 고속 추적됩니다. |
| **혼합 셀 표현** | `dtype: object` (박싱된 파이썬 객체) | `DataValue` 열거형 & 타입별 전용 컬럼 스토리지 | 순수 수치/문자열/불리언/날짜 컬럼은 박싱 없는 연속 메모리 버퍼에 저장되며, 개별 셀 조회 시에만 `DataValue`로 편리하게 다룹니다. |
| **에러 처리** | 런타임 `KeyError`, `ValueError` | Swift `throws` (`DataFrameError`) | 오타로 인한 컬럼 누락이나 잘못된 타입 변환을 `try` 구문을 통해 안전하게 제어할 수 있습니다. |
| **콘솔 출력** | `print(df)`, `df.describe()` | `print(df)`, `print(try df.describeStructured())` | 모든 데이터 구조와 통계 결과 타입이 `CustomStringConvertible` 및 `.describe()`를 지원하여 즉시 표 형태로 출력됩니다. |

---

## 2. `DataValue`: 개별 셀 값의 표현

`pandas`에서 하나의 셀은 실수, 정수, 문자열, 불리언, 타임스탬프, 또는 `NaN`/`None`을 담을 수 있습니다. 정적 타입 언어인 Swift에서 이를 자연스럽게 표현하기 위해 Bandal은 `DataValue` 열거형을 제공합니다.

```swift
public enum DataValue: Hashable, Sendable, Codable,
    CustomStringConvertible, CustomDebugStringConvertible, TextOutputStreamable,
    ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByStringLiteral, ExpressibleByBooleanLiteral, ExpressibleByNilLiteral
{
    case number(Double)
    case string(String)
    case boolean(Bool)
    case date(Date)
    case missing
}
```

### 2.1 리터럴을 사용한 `DataValue` 생성

`DataValue`는 Swift의 `ExpressibleBy*Literal` 프로토콜들을 준수하므로, 매번 `.number(10.5)`나 `.string("Seoul")`처럼 열거형 케이스를 길게 적을 필요가 없습니다. `[DataValue]` 배열 안에서 기본 Swift 리터럴을 그대로 작성할 수 있습니다.

```swift
import Foundation
import Bandal

// [DataValue] 배열 내에 기본 Swift 리터럴 직접 사용
let cells: [DataValue] = [
    42,                         // .number(42.0) — 정수 리터럴 자동 변환
    19.95,                      // .number(19.95) — 실수 리터럴 자동 변환
    "Seoul",                    // .string("Seoul") — 문자열 리터럴 자동 변환
    true,                       // .boolean(true) — 불리언 리터럴 자동 변환
    nil,                        // .missing — nil 리터럴 자동 변환
    .date(Date(timeIntervalSince1970: 1_700_000_000)) // 명시적 Date 케이스
]

print(cells)
// 출력: [42.0, 19.95, Seoul, true, nil, 2023-11-14T22:13:20Z]
```

### 2.2 `DataValue`에서 Swift 기본 타입 값 추출하기

`DataFrame`, `Series`, 또는 `DataRow`에서 셀 값을 꺼낼 때는 복잡한 `switch` 문 대신 아래의 옵셔널 편의 프로퍼티를 사용합니다.

| 프로퍼티 | 반환 타입 | 설명 |
| --- | --- | --- |
| `value.numberValue` | `Double?` | `.number`인 경우 `Double` 반환, 아니면 `nil` |
| `value.stringValue` | `String?` | `.string`인 경우 `String` 반환, 아니면 `nil` |
| `value.boolValue` | `Bool?` | `.boolean`인 경우 `Bool` 반환, 아니면 `nil` |
| `value.dateValue` | `Date?` | `.date`인 경우 `Date` 반환, 아니면 `nil` |
| `value.isMissing` | `Bool` | `value == .missing`이면 `true` |
| `value.description` | `String` | 문자열 표현 (`.missing`은 `"nil"`, `.date`는 ISO-8601 포맷) |

```swift
let priceCell: DataValue = 1250.5
let missingCell: DataValue = nil

if let price = priceCell.numberValue {
    print("Price with tax: \(price * 1.1)")
}
print("Is missing? \(missingCell.isMissing)")
```

```text
Price with tax: 1375.5500000000002
Is missing? true
```

---

## 3. `DataType`: 컬럼 데이터 타입 추론

`DataFrame`이나 `Series`의 모든 컬럼은 결측치(`.missing`)를 제외한 실제 값들로부터 자동 추론되는 논리적 `DataType`을 가집니다.

```swift
public enum DataType: String, Hashable, Sendable, Codable, CaseIterable, CustomStringConvertible {
    case number   // 결측치가 아닌 모든 값이 .number
    case string   // 결측치가 아닌 모든 값이 .string
    case boolean  // 결측치가 아닌 모든 값이 .boolean
    case date     // 결측치가 아닌 모든 값이 .date
    case mixed    // 서로 다른 두 개 이상의 타입이 섞여 있는 경우 (예: .number와 .string 혼재)
    case unknown  // 빈 컬럼(0행)이거나 모든 행이 .missing인 경우
}
```

> **`pandas` 사용자를 위한 팁**: `pandas`에서는 결측치가 포함되면 정수 컬럼이 `float64`로 바뀌거나 문자열 컬럼이 `object` 타입이 되는 경우가 많았습니다. Bandal에서는 `.missing` 여부가 별도의 `BitSet` 널 비트맵에 기록되므로 **결측치가 컬럼의 `DataType`을 오염시키지 않습니다**. 예를 들어 `[10, nil, 30]` 컬럼의 타입은 그대로 `DataType.number`입니다.

---

## 4. `Series`: 1차원 명명 컬럼 벡터

`Series`는 `pd.Series`에 대응하는 단일 명명 컬럼 구조체입니다. Swift의 `RandomAccessCollection`(`Element == DataValue`, 인덱스 범위 `0..<count`)을 준수하므로 `for-in` 루프나 고차 함수(`map`, `filter`)를 그대로 사용할 수 있습니다.

### 4.1 `Series` 생성하기

**Pandas (Python)**
```python
s_num = pd.Series([10.0, 20.0, 30.0], name="revenue")
s_opt = pd.Series([10.0, None, 30.0], name="revenue")
s_mix = pd.Series(["A", "B", None], name="grade")
```

**Bandal (Swift)**
```swift
import Bandal

// 1. 박싱 없는 고속 수치 배열 생성자 ([Double])
let sNum = Series(name: "revenue", numbers: [10.0, 20.0, 30.0])

// 2. 옵셔널 수치 배열 생성자 ([Double?], nil은 자동으로 .missing 처리)
let sOpt = Series(name: "revenue", values: [10.0, nil, 30.0])

// 3. 범용 DataValue 배열 생성자 (리터럴 활용)
let sMix = Series(name: "grade", values: ["A", "B", nil])

print(sOpt)
```

**콘솔 출력 (`print(sOpt)`)**:
```text
   revenue
0     10.0
1      nil
2     30.0
Name: revenue, dtype: number
```

### 4.2 `Series` 주요 프로퍼티 및 조회

| 프로퍼티 / 서브스크립트 | 타입 | Pandas 대응 | 설명 |
| --- | --- | --- | --- |
| `s.name` | `String` | `s.name` | 컬럼 이름 |
| `s.count` | `Int` | `len(s)` / `s.size` | 전체 행 개수 (`.missing` 포함) |
| `s.isEmpty` | `Bool` | `s.empty` | `count == 0` 여부 |
| `s.nonMissingCount` | `Int` | `s.count()` | 결측치가 아닌 유효 원소 개수 |
| `s.missingCount` | `Int` | `s.isna().sum()` | `.missing` 원소 개수 |
| `s.dtype` | `DataType` | `s.dtype` | 추론된 데이터 타입 (`.number`, `.string` 등) |
| `s.values` | `[DataValue]` | `s.tolist()` | 행 순서대로 정렬된 전체 `DataValue` 배열 |
| `s.numbers` | `[Double]` | `s.dropna().to_numpy()` | 결측치를 제외한 순수 `Double` 값 배열 |
| `s[index]` | `DataValue` | `s.iloc[i]` | 0-기반 정수 위치 `index`의 셀 값 |

```swift
let scores = Series(name: "score", values: [85.0, 92.5, nil, 78.0, 92.5])

print("Count: \(scores.count), Non-missing: \(scores.nonMissingCount), Missing: \(scores.missingCount)")
print("Dtype: \(scores.dtype)")
print("First element: \(scores[0])")
print("Non-missing numbers: \(scores.numbers)")
print("Unique values: \(scores.unique())")
```

```text
Count: 5, Non-missing: 4, Missing: 1
Dtype: number
First element: 85.0
Non-missing numbers: [85.0, 92.5, 78.0, 92.5]
Unique values: [85.0, 92.5, 78.0]
```

---

## 5. `DataFrame`: 2차원 컬럼 지향 테이블

`DataFrame`은 Bandal의 중심 데이터 구조입니다. 모든 컬럼은 비어 있지 않은 고유한 이름을 가져야 하며, 모든 컬럼의 행 길이(`rowCount`)가 동일해야 합니다.

### 5.1 메모리에서 `DataFrame`을 생성하는 4가지 방법

#### A. 순서가 보장되는 혼합 타입 컬럼 배열 (`[(name: String, values: [DataValue])]`) — *가장 많이 사용*
지정한 컬럼 순서를 그대로 유지하며, 수치·문자열·불리언·날짜·결측치(`nil`)를 자유롭게 조합할 수 있습니다.

**Pandas (Python)**
```python
df = pd.DataFrame({
    "city": ["Seoul", "Busan", "Incheon", "Daegu"],
    "temp": [21.5, 24.0, 20.1, None],
    "rain": [False, True, False, False]
})
```

**Bandal (Swift)**
```swift
let df = try DataFrame(columns: [
    (name: "city", values: ["Seoul", "Busan", "Incheon", "Daegu"]),
    (name: "temp", values: [21.5, 24.0, 20.1, nil]),
    (name: "rain", values: [false, true, false, false]),
])
print(df)
```

**콘솔 출력**:
```text
      city  temp   rain
0    Seoul  21.5  false
1    Busan    24   true
2  Incheon  20.1  false
3    Daegu   nil  false
```

#### B. 순서가 보장되는 순수 수치 컬럼 배열 (`[(name: String, values: [Double])]`)
모든 컬럼이 결측치 없는 `Double` 배열일 때 박싱 없이 즉시 `Float64Column`으로 생성하는 고속 생성자입니다.

```swift
let matrixDF = try DataFrame(numericColumns: [
    (name: "x", values: [1.0, 2.0, 3.0]),
    (name: "y", values: [10.0, 20.0, 30.0]),
])
print(matrixDF)
```

```text
   x   y
0  1  10
1  2  20
2  3  30
```

#### C. 수치 컬럼 딕셔너리 (`[String: [Double]]`)
간단한 수치 테이블을 빠르게 만들 때 유용합니다. Swift의 `Dictionary`는 순서가 없으므로, Bandal은 항상 일관된 결과를 보장하기 위해 **컬럼 이름을 알파벳 사전순(`A` → `Z`)으로 자동 정렬**합니다.

```swift
let dictDF = try DataFrame([
    "revenue": [100.0, 200.0, 150.0],
    "cost": [60.0, 110.0, 90.0],
])
print(dictDF.columnNames) // ["cost", "revenue"] (사전순 정렬됨)
print(dictDF)
```

```text
["cost", "revenue"]
   cost  revenue
0    60      100
1   110      200
2    90      150
```

#### D. `MLXVector` 컬럼 배열로부터 생성 (`[(name: String, vector: MLXVector)]`)
`MLXVector` 버퍼로부터 직접 `DataFrame`을 생성합니다(자세한 MLX 텐서 연동은 [가이드 08](08-MLX-Acceleration-and-Workflows.ko.md) 참고).

```swift
let vec = MLXVector(values: [1.5, 0.0, 3.5], isMissing: [false, true, false])
let mlxDF = try DataFrame(mlxColumns: [
    (name: "signal", vector: vec)
])
print(mlxDF)
```

```text
   signal
0     1.5
1     nil
2     3.5
```

---

## 6. `DataRow`: 개별 행(Row) 조회 및 다루기

서브스크립트 `df[row: index]`로 특정 행을 조회하면 `pandas`의 `df.iloc[i]`에 해당하는 `DataRow` 구조체가 반환됩니다(딕셔너리 형태가 필요한 경우 `try df.row(at: index)`를 사용할 수도 있습니다).

```swift
public struct DataRow: Equatable, Hashable, Sendable, Codable, CustomStringConvertible {
    public let index: Int
    public let columnOrder: [String]
    public let values: [String: DataValue]

    public subscript(_ column: String) -> DataValue?
    public subscript(position: Int) -> DataValue
}
```

### 예제: 컬럼 이름 및 컬럼 위치 인덱스로 행 데이터 접근하기

```swift
let row0 = df[row: 0]
print(row0)
print("---")
print("City by name: \(row0["city"]?.stringValue ?? "Unknown")")
print("Temp by position 1: \(row0[1].numberValue ?? .nan)")
```

**콘솔 출력**:
```text
city    Seoul
temp    21.5
rain    false
Row: 0, Columns: 3
---
City by name: Seoul
Temp by position 1: 21.5
```

---

## 7. 불변 값 시맨틱과 함수형 메서드 체이닝 (`pipe`)

`pandas`에서는 슬라이싱된 데이터프레임을 수정할 때 원본이 수정되는지 복사본이 수정되는지 모호하여 `SettingWithCopyWarning`이 자주 발생합니다.

반면 Bandal에서 **`DataFrame`과 `Series`는 불변 값 타입(`struct`)**입니다. 모든 변환 메서드는 새로운 `DataFrame`을 반환하며, 수정되지 않은 기존 컬럼들의 내부 스토리지(`AnyColumn`)는 복사 없이 `O(1)`로 안전하게 공유됩니다.

### 메서드 체이닝을 통한 선언적 데이터 파이프라인

```swift
let raw = try DataFrame(columns: [
    (name: "product", values: ["Laptop", "Mouse", "Monitor", "Keyboard"]),
    (name: "price", values: [1200.0, 25.0, nil, 80.0]),
    (name: "qty", values: [2, 10, 3, 5]),
])

// 다단계 변환 체이닝
let processed = try raw
    .fillMissing(in: ["price"], with: 300.0)
    .combining("price", "qty", using: .multiply, as: "total")
    .query(.column("total", greaterThanOrEqual: 300.0))
    .sorted(by: [SortDescriptor("total", order: .descending)])

print(processed)
```

**콘솔 출력**:
```text
    product  price  qty  total
0    Laptop   1200    2   2400
1   Monitor    300    3    900
2  Keyboard     80    5    400
```

또한 `pandas`의 `df.pipe(fn)`과 동일한 `.pipe(_:)` 메서드를 사용하면 사용자 정의 변환 함수를 체인 중간에 깔끔하게 결합할 수 있습니다.

```swift
func addTax(_ frame: DataFrame) throws -> DataFrame {
    try frame.scaling("total", by: 1.1, using: .multiply, as: "total_with_tax")
}

let finalReport = try processed.pipe(addTax)
print(finalReport)
```

```text
    product  price  qty  total  total_with_tax
0    Laptop   1200    2   2400            2640
1   Monitor    300    3    900             990
2  Keyboard     80    5    400             440
```

---

## 8. `DataFrameError`를 통한 에러 처리

스키마 검증, 컬럼 이름 조회, 수치 연산 조건 검증에 실패하면 `DataFrameError`가 던져집니다(`throws`). `LocalizedError`와 `Equatable`을 준수하므로 에러 메시지 출력과 단위 테스트 검증이 모두 간편합니다.

| `DataFrameError` 케이스 | 발생 조건 |
| --- | --- |
| `.emptyColumns` | 컬럼이 0개인 `DataFrame`을 생성하거나 대상 컬럼 없이 집계를 호출한 경우 |
| `.emptyColumnName` | 빈 문자열(`""`)을 컬럼 이름으로 사용한 경우 |
| `.duplicateColumnName(String)` | 동일한 컬럼 이름이 중복되거나 `pivoted`에서 중복된 `(index, column)` 쌍이 발견된 경우 |
| `.unequalColumnLengths(column:expected:actual:)` | 컬럼 배열이나 `BooleanMask`의 길이가 `rowCount`와 일치하지 않는 경우 |
| `.columnNotFound(String)` | 존재하지 않는 컬럼 이름을 참조한 경우 |
| `.nonFiniteValue(column:row:)` | 수치 컬럼에 `NaN`이나 `Infinity` 값을 입력한 경우 |
| `.nonNumericColumn(String)` | 비수치 컬럼이나 결측치가 포함된 컬럼에 순수 수치 전용 API(`column(_:)`, `standardized` 등)를 호출한 경우 |
| `.invalidCSV(String)` | CSV 따옴표가 닫히지 않았거나 행별 필드 개수가 맞지 않는 경우 |
| `.columnNameConflict(String)` | 조인 시 접미사 없이 비-키 컬럼 이름이 충돌하거나 이미 존재하는 컬럼 이름으로 추가하려 한 경우 |
| `.incompatibleSchema(expected:actual:)` | `concatenatingRows` 호출 시 데이터프레임 간 컬럼 스키마가 다른 경우 |
| `.invalidConversion(column:row:type:)` | `castColumn`이나 `resample`에서 특정 셀을 목표 `DataType`으로 변환할 수 없는 경우 |
| `.invalidWindowSize(Int)` | 윈도우 크기가 `0` 이하이거나, `alpha`가 `(0, 1]` 범위를 벗어났거나, 구간(`edges`) 설정이 잘못된 경우 |

```swift
do {
    _ = try df.series("non_existent_column")
} catch let error as DataFrameError {
    print("Caught DataFrameError: \(error.localizedDescription)")
}
```

```text
Caught DataFrameError: Column 'non_existent_column' does not exist.
```

---

> **다음 단계**: **[가이드 02: 데이터 입출력 및 탐색 →](02-Data-IO-and-Inspection.ko.md)**로 이동하여 CSV 및 JSON 파일 읽기/쓰기, `infoSummary()`를 통한 스키마 및 메모리 진단, 그리고 콘솔/마크다운 테이블 출력 포맷 설정 방법을 학습하세요.
