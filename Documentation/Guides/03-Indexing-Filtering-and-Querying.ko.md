# 가이드 03: 인덱싱, 슬라이싱, 불리언 마스크 및 쿼리

> **언어**: [English](03-Indexing-Filtering-and-Querying.md) | **한국어**  
> **네비게이션**: [← 가이드 02: 데이터 입출력 및 탐색](02-Data-IO-and-Inspection.ko.md) | [API 레퍼런스](../API.ko.md) | **다음**: [가이드 04: 데이터 정제 및 피처 엔지니어링 →](04-Cleaning-and-Feature-Engineering.ko.md)

---

## 개요

`pandas`를 사용하는 데이터 분석가가 가장 빈번하게 수행하는 작업은 원하는 컬럼을 선택(`df[["a", "b"]]`)하고, 행 범위를 슬라이싱(`df.iloc[0:5]`)하며, 조건에 맞는 행을 필터링(`df[df["score"] >= 80]`, `df.query(...)`)하는 것입니다.

Bandal은 무복사(Zero-Copy) 컬럼 프로젝션부터 64비트 워드 단위로 동작하는 벡터화 비트맵 마스크(`BooleanMask`), 그리고 타입 안전 조건식(`RowPredicate`)까지 총 4가지의 직관적인 선택 및 필터링 방식을 제공합니다.

이 가이드에서 다룰 핵심 내용은 다음과 같습니다.
- **컬럼 선택 및 제거** (`select`, `select(columnsOfType:)`, `drop(columns:)`, `series(_:)`, `df["col"]`)
- **위치 기반 행/셀 슬라이싱 및 극값 탐색** (`df[row:]`, `df[rows:]`, `df["col", row]`, `head`, `tail`, `largest`, `smallest`, `indexOfMax`, `indexOfMin`)
- **벡터화 불리언 마스크 필터링** (`BooleanMask`, `Series` 비교 연산자, `&&`, `||`, `!`, `df[mask]`, `filtered(by:)`)
- **조합 가능한 타입 안전 조건자** (`RowPredicate` + `df.query(_:)`)
- **간결한 문자열 쿼리 표현식** (`df.query("score >= 80 && active == true")`)
- **사용자 정의 행 클로저 필터링** (`df.filterRows { row in ... }`)

---

## 1. 한눈에 보는 비교: `pandas` vs. `Bandal`

| 작업 | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| 단일 컬럼을 `Series`로 추출 | `df["price"]` | `try df.series("price")` 또는 `df["price"]` (`Series?`) |
| 단일 수치 컬럼을 `[Double]`로 추출 | `df["price"].to_numpy()` | `try df.column("price")` |
| 다중 컬럼 선택 | `df[["city", "price"]]` | `try df.select("city", "price")` |
| 데이터 타입별 컬럼 선택 | `df.select_dtypes(include="number")` | `try df.select(columnsOfType: .number)` |
| 특정 컬럼 제거 | `df.drop(columns=["notes"])` | `try df.drop(columns: ["notes"])` |
| 정수 위치로 단일 행 조회 | `df.iloc[2]` | `df[row: 2]` 또는 `df.row(at: 2)` (`DataRow?`) |
| 정수 범위로 행 슬라이싱 | `df.iloc[1:4]` | `df[rows: 1..<4]` (`DataFrame?`) |
| 단일 셀(열, 행) 조회 | `df.loc[0, "price"]` | `df["price", 0]` (`DataValue?`) |
| 상위 / 하위 $N$개 행 | `df.head(5)` / `df.tail(5)` | `try df.head(5)` / `try df.tail(5)` |
| 특정 컬럼 기준 상위 / 하위 $N$개 행 | `df.nlargest(3, "price")` / `df.nsmallest(3, "price")` | `try df.largest(3, by: "price")` / `try df.smallest(3, by: "price")` |
| 최댓값 / 최솟값의 행 인덱스 | `df["price"].idxmax()` / `idxmin()` | `try df.indexOfMax("price")` / `try df.indexOfMin("price")` |
| 벡터화 불리언 마스크 인덱싱 | `df[(df["price"] > 50) & (df["qty"] >= 2)]` | `let mask = try (df.series("price") > 50) && (df.series("qty") >= 2)`<br>`let sub = df[mask]` |
| 집합 포함 여부 (`isin`) | `s.isin(["Seoul", "Busan"])` | `s.isin(["Seoul", "Busan"])` |
| 구간 포함 여부 (`between`) | `s.between(10, 50)` | `s.between(lower: 10, upper: 50)` |
| 조건식 쿼리 | `df.query("price > 50 and region == 'Seoul'")` | `try df.query(.column("price", greaterThan: 50) && .column("region", equals: "Seoul"))`<br>또는 `try df.query("price > 50 && region == 'Seoul'")` |

---

## 2. 컬럼 선택 및 제거

실습을 위해 직원 인사 및 급여 데이터프레임을 생성해 보겠습니다.

```swift
import Bandal

let employees = try DataFrame(columns: [
    (name: "id", values: ["E01", "E02", "E03", "E04", "E05"]),
    (name: "dept", values: ["AI", "Sales", "AI", "Finance", "Sales"]),
    (name: "salary", values: [95000.0, 72000.0, 110000.0, 88000.0, 68000.0]),
    (name: "bonus", values: [12000.0, nil, 18000.0, 9000.0, 5000.0]),
    (name: "remote", values: [true, false, true, false, false]),
])
```

### 2.1 단일 컬럼 추출 (`series`, `subscript`, `values`, `column`)

목적에 따라 4가지 방법으로 단일 컬럼을 꺼낼 수 있습니다.

```swift
// 1. O(1) 무복사 Series 추출 (존재하지 않는 컬럼이면 에러 throw — 파이프라인 권장)
let salarySeries: Series = try employees.series("salary")

// 2. 옵셔널 서브스크립트 (컬럼이 없으면 nil 반환)
if let deptSeries = employees["dept"] {
    print("Departments: \(deptSeries.unique())")
}

// 3. [DataValue] 셀 배열로 추출
let bonusCells: [DataValue] = try employees.values(in: "bonus")

// 4. 언박싱된 [Double] 배열로 추출 (.missing이나 비수치 값이 있으면 .nonNumericColumn 에러 throw)
let salaryDoubles: [Double] = try employees.column("salary")
```

### 2.2 다중 컬럼 선택(`select`) 및 제거(`drop`)

`DataFrame`은 내부 컬럼 스토리지를 참조로 공유하므로, `select`와 `drop`은 행 데이터를 복사하지 않고 `O(컬럼 수)` 시간에 즉시 완료됩니다.

```swift
// 지정한 순서대로 특정 컬럼만 선택
let summaryCols = try employees.select("id", "dept", "salary")
print(summaryCols)

// 특정 데이터 타입(.number)인 컬럼만 자동 선택 (pandas의 df.select_dtypes(include="number"))
let numericOnly = try employees.select(columnsOfType: .number)
print("\nNumeric columns: \(numericOnly.columnNames)")

// 불필요한 컬럼 제거
let withoutBonus = try employees.drop(columns: ["bonus", "remote"])
```

**콘솔 출력**:
```text
    id     dept  salary
0  E01       AI   95000
1  E02    Sales   72000
2  E03       AI  110000
3  E04  Finance   88000
4  E05    Sales   68000

[5 rows x 3 columns]

Numeric columns: ["salary", "bonus"]
```

---

## 3. 위치 기반 행/셀 슬라이싱 및 극값 탐색 (`iloc` & Extremes)

### 3.1 행, 행 범위, 개별 셀 서브스크립트

```swift
// 1. 0-기반 인덱스로 단일 행 조회 (DataRow? 반환, 범위 밖이면 nil)
if let secondRow = employees[row: 1] {
    print("Row 1 dept: \(secondRow["dept"] ?? .missing)")
}

// 2. 연속된 행 범위 슬라이싱 (DataFrame? 반환, 자동으로 0..<rowCount 범위로 안전하게 클램핑)
if let middleSlice = employees[rows: 1..<4] {
    print(middleSlice)
}

// 3. (컬럼명, 행 인덱스)로 단일 셀 직접 조회 (DataValue? 반환)
let cell: DataValue? = employees["salary", 2] // .number(110000.0)
```

### 3.2 `head`, `tail`, `largest`, `smallest`, `indexOfMax`, `indexOfMin`

```swift
// 상위 2개 행 및 하위 2개 행
let firstTwo = try employees.head(2)
let lastTwo = try employees.tail(2)

// 급여(salary) 기준 상위 2명 추출 (pandas: df.nlargest(2, "salary"))
let topEarners = try employees.largest(2, by: "salary")
print("Top 2 Earners:")
print(topEarners)

// 보너스(bonus) 최댓값 및 최솟값이 위치한 행 인덱스 찾기 (.missing은 자동 무시)
if let maxBonusRow = try employees.indexOfMax("bonus"),
   let minBonusRow = try employees.indexOfMin("bonus") {
    print("\nMax bonus is at row \(maxBonusRow) (\(employees["id", maxBonusRow]!))")
    print("Min bonus is at row \(minBonusRow) (\(employees["id", minBonusRow]!))")
}
```

**콘솔 출력**:
```text
Top 2 Earners:
    id  dept  salary  bonus  remote
0  E03    AI  110000  18000    true
1  E01    AI   95000  12000    true

[2 rows x 5 columns]

Max bonus is at row 2 (E03)
Min bonus is at row 4 (E05)
```

---

## 4. 벡터화 불리언 마스크 인덱싱 (`BooleanMask`)

`pandas`에서 가장 직관적인 필터링 방식은 불리언 인덱싱(`df[df["salary"] > 80000]`)입니다. Bandal은 이를 64비트 워드 단위로 압축된 비트맵 구조체인 **`BooleanMask`(`BitSet`)**로 구현하여, 한 번의 CPU 비트 연산으로 64개 행의 조건을 동시에 결합합니다.

### 4.1 `Series` 비교 연산으로 `BooleanMask` 생성하기

| `Series` 비교 연산 / 메서드 | 반환 타입 | 결측치(`.missing`) 처리 규칙 |
| --- | --- | --- |
| `series > threshold` (`>=`, `<`, `<=` 동일) | `BooleanMask` (비수치 컬럼이면 throw) | `.missing` 셀은 항상 `false`로 평가됨 |
| `series.equals(value)` | `BooleanMask` | `series.equals(.missing)` 호출 시 결측 행만 `true` |
| `series.notEquals(value)` | `BooleanMask` | `.missing != 일반값`은 `true`로 평가됨 |

### 4.2 마스크 논리 결합 (`&&`, `||`, `!`) 및 `df[mask]` 인덱싱

`BooleanMask`는 논리 연산자(`&&`, `||`, `!`)와 메서드(`.and(_:)`, `.or(_:)`, `.negated`), 그리고 참/거짓 개수를 즉시 확인하는 프로퍼티(`trueCount`, `falseCount`, `count`)를 제공합니다.

**Pandas (Python)**
```python
mask = (df["salary"] >= 85000) & (df["dept"] == "AI")
ai_seniors = df[mask]
```

**Bandal (Swift)**
```swift
let salary = try employees.series("salary")
let dept = try employees.series("dept")

// 벡터화 마스크를 &&, ||, ! 연산자로 결합
let mask: BooleanMask = try (salary >= 85000) && dept.equals("AI")

print("Matching rows: \(mask.trueCount) out of \(mask.count)")

// 방법 A: 서브스크립트 문법 (마스크 길이가 다르면 빈 DataFrame 반환)
let aiSeniors = employees[mask]
print(aiSeniors)

// 방법 B: 에러 검증 메서드 (마스크 길이가 다르면 .unequalColumnLengths throw)
let aiSeniorsChecked = try employees.filtered(by: mask)
```

**콘솔 출력**:
```text
Matching rows: 2 out of 5
    id  dept  salary  bonus  remote
0  E01    AI   95000  12000    true
1  E03    AI  110000  18000    true

[2 rows x 5 columns]
```

### 4.3 `Series`의 `isin` 및 `between` 활용

`Series`의 `.isin(_:)`과 `.between(lower:upper:inclusive:)` 메서드는 `[Bool]` 배열을 반환하며, `BooleanMask([Bool])` 생성자에 전달하여 다른 마스크와 자유롭게 결합할 수 있습니다.

```swift
let targetDepts = BooleanMask(dept.isin(["AI", "Finance"]))
let midSalary = BooleanMask(salary.between(lower: 70000, upper: 100000, inclusive: true))

let filtered = employees[targetDepts && midSalary]
print(filtered)
```

```text
    id     dept  salary  bonus  remote
0  E01       AI   95000  12000    true
1  E04  Finance   88000   9000   false

[2 rows x 5 columns]
```

---

## 5. `RowPredicate`를 사용한 타입 안전 조합형 쿼리

별도의 `Series` 변수를 꺼내지 않고 `DataFrame` 메서드 체인 안에서 바로 조건 필터링을 수행하고 싶다면 **`df.query(_ predicate: RowPredicate)`**를 사용합니다.

내장 `RowPredicate` 생성자들은 내부적으로 컬럼 스토리지에 직접 접근하는 고속 비트맵 스캔(`fastMask`)으로 컴파일되므로, 행 딕셔너리를 생성하지 않고 벡터화 속도로 실행됩니다.

### 5.1 내장 `RowPredicate` 생성자 목록

| 조건자 생성자 | 설명 |
| --- | --- |
| `.column(name, equals: value)` / `.column(name, notEquals: value)` | 특정 `DataValue`와의 일치 / 불일치 비교 |
| `.column(name, greaterThan: x)` / `.column(name, greaterThanOrEqual: x)` | 수치 컬럼 `>` / `>=` 비교 |
| `.column(name, lessThan: x)` / `.column(name, lessThanOrEqual: x)` | 수치 컬럼 `<` / `<=` 비교 |
| `.column(name, between: low, and: high, inclusive: true)` | 수치 컬럼 범위(`low...high`) 포함 여부 |
| `.column(name, in: ["A", "B"])` | 후보 집합(`Set<DataValue>`) 포함 여부 |
| `.column(name, contains: "sub", caseSensitive: true)` | 문자열 컬럼 부분 문자열 포함 여부 |
| `.isMissing(name)` / `.notMissing(name)` | 결측치 여부 검사 (`O(1)` 널 비트맵 즉시 참조) |
| `p1 && p2`, `p1 || p2`, `!p1` | 논리 `AND`, `OR`, `NOT` 결합 |

### 5.2 예제: 복합 조건 `RowPredicate` 쿼리

```swift
let result = try employees.query(
    .notMissing("bonus")
    && (.column("dept", in: ["AI", "Finance"]) || .column("salary", greaterThanOrEqual: 70000))
    && !.column("id", equals: "E04")
)
print(result)
```

**콘솔 출력**:
```text
    id  dept  salary  bonus  remote
0  E01    AI   95000  12000    true
1  E03    AI  110000  18000    true

[2 rows x 5 columns]
```

---

## 6. 문자열 쿼리 표현식 및 커스텀 행 클로저 필터링

### 6.1 간결한 문자열 쿼리 (`df.query(_ expression: String)`)

`pandas`의 `df.query("...")`와 동일하게, 문자열 표현식 하나로 직관적인 필터링이 가능합니다.
- **비교 연산자**: `==`, `!=`, `>=`, `<=`, `>`, `<`
- **논리 연산자**: `&&` (또는 `and`), `||` (또는 `or`)
- **우변 리터럴**: 숫자(`85000`), 불리언(`true`, `false`), 결측치(`nil`, `null`, `NaN`), 따옴표 문자열(`'AI'`, `"Sales"`)

```swift
let queried = try employees.query("salary >= 80000 && remote == true && bonus != nil")
print(queried)
```

```text
    id  dept  salary  bonus  remote
0  E01    AI   95000  12000    true
1  E03    AI  110000  18000    true

[2 rows x 5 columns]
```

### 6.2 `filterRows`를 활용한 자유로운 Swift 클로저 필터링

동일 행 내의 여러 컬럼을 상호 계산하여 비교하거나 복잡한 도메인 로직이 필요한 경우에는 `df.filterRows(_:)`를 사용합니다.

```swift
// 보너스(bonus)가 기본급(salary)의 15% 이상인 직원만 필터링
let highBonusRatio = try employees.filterRows { row in
    guard let sal = row["salary"]?.numberValue,
          let bon = row["bonus"]?.numberValue,
          sal > 0
    else {
        return false
    }
    return (bon / sal) >= 0.15
}
print(highBonusRatio)
```

**콘솔 출력**:
```text
    id  dept  salary  bonus  remote
0  E03    AI  110000  18000    true

[1 rows x 5 columns]
```

---

> **다음 단계**: 데이터 선택과 필터링을 마스터했다면 **[가이드 04: 데이터 정제 및 피처 엔지니어링 →](04-Cleaning-and-Feature-Engineering.ko.md)**으로 이동하여 결측치 보간, 중복 제거, 정렬, 파생 변수 생성, 정규화 및 원-핫 인코딩 방법을 학습하세요.
