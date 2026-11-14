# 가이드 04: 데이터 정제, 결측치 처리 및 피처 엔지니어링

> **언어**: [English](04-Cleaning-and-Feature-Engineering.md) | **한국어**  
> **네비게이션**: [← 가이드 03: 인덱싱, 필터링 및 쿼리](03-Indexing-Filtering-and-Querying.ko.md) | [API 레퍼런스](../API.ko.md) | **다음**: [가이드 05: 벡터화 수학 연산 및 기술 통계 →](05-Vectorized-Math-and-Statistics.ko.md)

---

## 개요

통계 분석이나 머신러닝 모델링에 앞서 실무 데이터는 반드시 정제(Data Cleaning)와 피처 엔지니어링(Feature Engineering) 과정을 거쳐야 합니다. 결측치를 진단 및 보간하고, 중복 레코드를 제거하며, 문자열 포맷을 통일하고, 날짜에서 달력 변수를 추출하고, 수치 변수를 스케일링하거나 구간화(Binning)하고, 범주형 변수를 원-핫 인코딩하는 작업이 여기에 포함됩니다.

이 가이드에서는 Bandal이 제공하는 데이터 전처리 및 피처 엔지니어링 API 전체를 상세히 다룹니다.
- **결측치 진단 및 보간** (`missingSummary()`, `MissingCounts`, `fillMissing`, `forwardFilled`, `backwardFilled`, `interpolated`, `dropMissing`, `replacing`, `Series.where`)
- **중복 제거, 다중 컬럼 정렬 및 재현 가능한 샘플링** (`duplicatedRows`, `dropDuplicates`, `sorted(by:)`, `sample`)
- **파생 컬럼 추가, 변환, 이름 변경 및 타입 캐스팅** (`withColumn`, `mapped`, `renamedColumns`, `castColumn`)
- **문자열 정제 및 날짜 구성요소 추출** (`lowercased`, `uppercased`, `trimmed`, `stringLength`, `replacingSubstring`, `containsSubstring`, `dateComponent`)
- **머신러닝 피처 엔지니어링** (`standardized`, `minMaxScaled`, `binned`, `quantileBinned`, `oneHotEncoded`)

---

## 1. 한눈에 보는 비교: `pandas` vs. `Bandal`

| 작업 | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| 컬럼별 결측치 개수 요약 | `df.isna().sum()` | `df.missingSummary()` (`MissingCounts`) 또는 `df.missingCounts()` |
| 결측 여부 불리언 DataFrame | `df.isna()` | `try df.missingMask()` |
| 상수로 결측치 채우기 | `df.fillna(0)` / `df[["a"]].fillna(0)` | `try df.fillMissing(in: ["a"], with: 0.0)` |
| 앞/뒤 값으로 결측치 채우기 | `df.ffill()` / `df.bfill()` | `try df.forwardFilled()` / `try df.backwardFilled()` |
| 선형 보간 (`interpolate`) | `df["temp"].interpolate()` | `try df.interpolated("temp", as: "temp_interp")` |
| 결측치가 있는 행 제거 | `df.dropna(subset=["a"])` | `try df.dropMissing(in: ["a"])` |
| 조건부 값 치환 (`where`) | `s.where(mask, other=0)` | `try s.where(mask, other: 0.0)` |
| 중복 행 제거 | `df.drop_duplicates(subset=["id"], keep="first")` | `try df.dropDuplicates(in: ["id"], keep: .first)` |
| 다중 컬럼 정렬 | `df.sort_values(["dept", "salary"], ascending=[True, False])` | `try df.sorted(by: [SortDescriptor("dept"), SortDescriptor("salary", order: .descending)])` |
| 무작위 샘플링 (개수 / 비율) | `df.sample(n=10, random_state=42)` / `df.sample(frac=0.2)` | `try df.sample(10, seed: 42)` / `try df.sample(fraction: 0.2, seed: 42)` |
| 컬럼 타입 변환 | `df["a"].astype(float)` | `try df.castColumn("a", to: .number)` |
| Z-점수 표준화 | `(s - s.mean()) / s.std()` | `try df.standardized("price", as: "price_z")` |
| Min-Max 정규화 | `(s - s.min()) / (s.max() - s.min())` | `try df.minMaxScaled("price", to: 0.0...1.0, as: "price_norm")` |
| 고정 경계 / 분위수 구간화 | `pd.cut(...)` / `pd.qcut(...)` | `try df.binned(...)` / `try df.quantileBinned(...)` |
| 원-핫 인코딩 | `pd.get_dummies(df, columns=["city"], drop_first=True)` | `try df.oneHotEncoded("city", dropFirst: true)` |

---

## 2. 결측치(`.missing`) 진단 및 처리

### 2.1 결측치 현황 진단 (`missingSummary`, `missingCounts`, `missingMask`, `isMissing`)

`df.missingSummary()`는 구조화된 `MissingCounts` 인스턴스를 반환합니다. 이를 바로 `print(...)`하면 전체 결측치 합계가 포함된 정렬된 표가 출력되며, 서브스크립트(`missing["price"]`)나 `totalMissing` 프로퍼티로 코드에서 직접 값을 조회할 수도 있습니다.

```swift
import Bandal

let sensorData = try DataFrame(columns: [
    (name: "hour", values: [1, 2, 3, 4, 5]),
    (name: "temp", values: [20.0, nil, nil, 26.0, 28.0]),
    (name: "status", values: ["OK", "WARN", nil, "OK", "ERR"]),
])

let missing: MissingCounts = sensorData.missingSummary()
print(missing)
print("Missing in 'temp': \(missing["temp"] ?? 0), Total: \(missing.totalMissing)")
```

**콘솔 출력**:
```text
hour      0
temp      2
status    1
Total missing: 3 (rows: 5)
Missing in 'temp': 2, Total: 3
```

### 2.2 결측치 채우기 및 보간 (`fillMissing`, `forwardFilled`, `backwardFilled`, `interpolated`)

Bandal은 4가지 결측치 보간 전략을 제공합니다.
1. **고정 상수 대체**: `df.fillMissing(in: selectedColumns, with: replacement)`
2. **앞 방향 채우기 (`ffill`)**: `df.forwardFilled()` — 직전의 유효한 관측값으로 결측치를 채웁니다.
3. **뒤 방향 채우기 (`bfill`)**: `df.backwardFilled()` — 직후의 유효한 관측값으로 결측치를 채웁니다.
4. **선형 보간 (`interpolate`)**: `df.interpolated(_:as:)` (또는 `Series.interpolated()`) — 양 끝 유효 관측값 사이의 연속된 결측 구간을 등간격 선형 보간합니다(시작과 끝의 결측치는 그대로 유지).

```swift
// 1. 특정 컬럼의 결측치를 상수로 대체
let filledStatus = try sensorData.fillMissing(in: ["status"], with: "UNKNOWN")

// 2. 앞 방향 채우기 (Forward Fill)
let ffilled = try sensorData.forwardFilled()
print("--- Forward Filled ---")
print(ffilled)

// 3. 'temp' 컬럼 선형 보간 (20.0 -> [22.0, 24.0] -> 26.0 -> 28.0)
let interpolated = try sensorData.interpolated("temp", as: "temp_linear")
print("\n--- Linearly Interpolated ---")
print(interpolated)
```

**콘솔 출력**:
```text
--- Forward Filled ---
   hour  temp  status
0     1    20      OK
1     2    20    WARN
2     3    20    WARN
3     4    26      OK
4     5    28     ERR

[5 rows x 3 columns]

--- Linearly Interpolated ---
   hour  temp  status  temp_linear
0     1    20      OK           20
1     2   NaN    WARN           22
2     3   NaN     NaN           24
3     4    26      OK           26
4     5    28     ERR           28

[5 rows x 4 columns]
```

### 2.3 결측 행 제거 및 조건부 치환 (`dropMissing`, `replacing`, `where`)

```swift
// "status" 컬럼이 결측인 행만 제거 (`in:` 생략 시 모든 컬럼 중 하나라도 결측이면 제거)
let completeStatus = try sensorData.dropMissing(in: ["status"])

// 특정 값(예: 에러 코드 "ERR")을 .missing으로 일괄 치환
let cleanedStatus = try sensorData.replacing("ERR", with: .missing, in: ["status"])

// 조건부 Series 치환 (`pd.Series.where`와 동일):
// 마스크가 true인 위치는 원래 값을 유지하고, false인 위치는 `other`로 대체
let tempSeries = try interpolated.series("temp_linear")
let normalRangeMask = try (tempSeries >= 21.0) && (tempSeries <= 27.0)
let cappedSeries = try tempSeries.where(normalRangeMask, other: .missing)
print(cappedSeries)
```

**콘솔 출력**:
```text
   temp_linear
0          NaN
1           22
2           24
3           26
4          NaN
Name: temp_linear, Length: 5
```

---

## 3. 중복 제거, 다중 정렬 및 샘플링

### 3.1 중복 행 탐지 및 제거 (`duplicatedRows`, `dropDuplicates`)

`DuplicateKeep` 열거형으로 중복 키 발생 시 어느 행을 남길지 지정할 수 있습니다.
- `.first` (기본값): 첫 번째 등장 행을 유지하고 이후 중복 행을 제거합니다.
- `.last`: 마지막 등장 행을 유지하고 이전 중복 행을 제거합니다.
- `.none`: 한 번이라도 중복된 키를 가진 모든 행을 제거합니다.

```swift
let logs = try DataFrame(columns: [
    (name: "user", values: ["u1", "u2", "u1", "u3", "u2"]),
    (name: "event", values: ["login", "view", "login", "buy", "buy"]),
    (name: "seq", values: [1, 2, 3, 4, 5]),
])

// "user" 컬럼 기준으로 각 사용자의 가장 마지막 이벤트(.last)만 유지
let latestPerUser = try logs.dropDuplicates(in: ["user"], keep: .last)
print(latestPerUser)
```

**콘솔 출력**:
```text
   user  event  seq
0    u1  login    3
1    u3    buy    4
2    u2    buy    5

[3 rows x 3 columns]
```

### 3.2 다중 컬럼 안정 정렬 (`sorted(by:)`)

`sorted(by: [SortDescriptor])`는 입력 순서를 보존하는 안정 정렬(Stable Sort)을 수행합니다(512행 이상의 수치/날짜 단일 컬럼은 $O(N)$ LSD 기수 정렬로 가속). 결측치(`.missing`)는 오름차순과 내림차순 모두에서 항상 맨 뒤에 배치됩니다.

```swift
let sortedLogs = try logs.sorted(by: [
    SortDescriptor("user", order: .ascending),
    SortDescriptor("seq", order: .descending),
])
print(sortedLogs)
```

```text
   user  event  seq
0    u1  login    3
1    u1  login    1
2    u2    buy    5
3    u2   view    2
4    u3    buy    4

[5 rows x 3 columns]
```

### 3.3 시드(Seed) 기반 재현 가능한 샘플링 (`sample`)

`seed: UInt64`를 전달하면 결정론적 `SplitMix64` 난수 생성기를 사용하여 언제 실행해도 동일한 샘플을 얻을 수 있습니다.

```swift
// 정확한 행 개수로 비복원 추출
let threeRows = try logs.sample(3, seed: 2026)

// 비율(40%)로 비복원 추출
let fortyPercent = try logs.sample(fraction: 0.4, seed: 2026)
print(fortyPercent)
```

```text
   user  event  seq
0    u1  login    3
1    u3    buy    4

[2 rows x 3 columns]
```

---

## 4. 파생 컬럼 추가, 변환, 이름 변경 및 타입 캐스팅

```swift
let rawProducts = try DataFrame(columns: [
    (name: "sku_code", values: [" ab-10 ", " cd-20 ", " ef-30 "]),
    (name: "raw_price", values: ["$120.00", "$250.50", "$80.00"]),
])

let transformed = try rawProducts
    // 1. 컬럼 이름 변경
    .renamedColumns(["sku_code": "sku", "raw_price": "price"])
    // 2. 통화 문자열("$120.00") 컬럼을 수치(.number) 컬럼으로 캐스팅
    .castColumn("price", to: .number)
    // 3. 특정 컬럼의 각 셀에 변환 함수 적용
    .mapped("price") { val in
        guard let p = val.numberValue else { return .missing }
        return .number(p * 1.1) // 부가세 10% 가산
    }
    // 4. 각 DataRow로부터 파생 컬럼 생성 (pandas의 df.assign 역할)
    .withColumn(name: "tier") { row in
        guard let p = row["price"]?.numberValue else { return .missing }
        return p >= 200.0 ? "Premium" : "Standard"
    }

print(transformed)
```

**콘솔 출력**:
```text
       sku   price      tier
0   ab-10      132  Standard
1   cd-20   275.55   Premium
2   ef-30       88  Standard

[3 rows x 3 columns]
```

---

## 5. 문자열 및 날짜 컬럼 전용 연산

### 5.1 문자열 정제 및 파생 변수 추출 (`trimmed`, `uppercased`, `lowercased`, `replacingSubstring`, `stringLength`, `containsSubstring`)

모든 문자열 메서드는 선택적 `as outputColumn: String? = nil` 인자를 받습니다. `as`를 생략하면 원본 컬럼을 덮어쓰거나 기본 파생 컬럼명(`"<col>_length"`, `"<col>_contains"`)으로 추가됩니다.

```swift
let cleanedStrings = try transformed
    .trimmed("sku")
    .uppercased("sku")
    .replacingSubstring(in: "sku", target: "-", with: "_")
    .stringLength("sku", as: "sku_len")
    .containsSubstring("sku", "CD", caseSensitive: true, as: "is_cd")

print(cleanedStrings)
```

**콘솔 출력**:
```text
     sku   price      tier  sku_len  is_cd
0  AB_10     132  Standard        5  false
1  CD_20  275.55   Premium        5   true
2  EF_30      88  Standard        5  false

[3 rows x 5 columns]
```

### 5.2 날짜(`.date`) 컬럼에서 달력 구성요소 추출 (`dateComponent`)

`dateComponent(_:from:as:)`는 UTC 기준 달력 구성요소(`.year`, `.month`, `.day`, `.weekday`, `.hour`, `.minute`, `.second`)를 수치 컬럼으로 추출합니다.

```swift
let events = try DataFrame(columns: [
    (name: "ts", values: ["2026-03-15", "2026-07-04", "2026-12-25"]),
])
.castColumn("ts", to: .date, dateFormat: "yyyy-MM-dd")
.dateComponent(.year, from: "ts", as: "year")
.dateComponent(.month, from: "ts", as: "month")
.dateComponent(.weekday, from: "ts", as: "weekday")

print(events)
```

**콘솔 출력**:
```text
                     ts  year  month  weekday
0  2026-03-15T00:00:00Z  2026      3        1
1  2026-07-04T00:00:00Z  2026      7        7
2  2026-12-25T00:00:00Z  2026     12        6

[3 rows x 4 columns]
```

---

## 6. 피처 스케일링, 구간화(Binning) 및 원-핫 인코딩

Bandal은 별도의 외부 라이브러리 없이 `pandas`와 `scikit-learn`의 핵심 전처리 기능들을 기본 내장하고 있습니다.

### 6.1 Z-점수 표준화(`standardized`) 및 Min-Max 정규화(`minMaxScaled`)

- **`standardized(_:as:)`**: 평균 `0.0`, 표본 표준편차 `1.0`($\frac{x - \mu}{s}$)으로 표준화합니다.
- **`minMaxScaled(_:to:as:)`**: 최솟값과 최댓값을 지정한 범위(기본값 `0.0...1.0`)로 선형 스케일링합니다.

```swift
let scores = try DataFrame(columns: [
    (name: "city", values: ["Seoul", "Busan", "Incheon", "Seoul"]),
    (name: "score", values: [50.0, 70.0, 80.0, 100.0]),
])
.standardized("score", as: "score_z")
.minMaxScaled("score", to: 0.0...1.0, as: "score_minmax")

print(scores)
```

**콘솔 출력**:
```text
      city  score    score_z  score_minmax
0    Seoul     50  -1.183216             0
1    Busan     70  -0.236643           0.4
2  Incheon     80   0.236643           0.6
3    Seoul    100   1.183216             1

[4 rows x 4 columns]
```

### 6.2 연속형 변수 구간화: 고정 경계(`binned`) 및 분위수 구간화(`quantileBinned`)

- **`binned(_:edges:labels:rightClosed:as:)`** (`pd.cut`): 단조 증가하는 `edges` 경계값 배열에 따라 구간 라벨을 부여합니다. 경계 밖의 값은 `.missing`이 됩니다.
- **`quantileBinned(_:bins:labels:as:)`** (`pd.qcut`): 데이터를 동일한 확률 분위수 크기를 갖는 `bins`개의 구간으로 나눕니다.

```swift
let binnedDF = try scores
    .select("city", "score")
    .binned(
        "score",
        edges: [0.0, 60.0, 85.0, 100.0],
        labels: ["Low", "Mid", "High"],
        rightClosed: true,
        as: "grade"
    )
    .quantileBinned(
        "score",
        bins: 2,
        labels: ["Bottom50%", "Top50%"],
        as: "half"
    )

print(binnedDF)
```

**콘솔 출력**:
```text
      city  score  grade       half
0    Seoul     50    Low  Bottom50%
1    Busan     70    Mid  Bottom50%
2  Incheon     80    Mid     Top50%
3    Seoul    100   High     Top50%

[4 rows x 4 columns]
```

### 6.3 범주형 변수 원-핫 인코딩 (`oneHotEncoded` / `pd.get_dummies`)

`oneHotEncoded(_:prefix:dropFirst:includeMissing:dropOriginal:)`는 범주형/문자열 컬럼의 고유값들을 첫 등장 순서대로 `0.0` / `1.0` 더미(Dummy) 수치 컬럼으로 전개합니다.

- `dropFirst: true`: 다중공선성(Multicollinearity) 방지를 위해 첫 번째 범주 컬럼을 생략합니다.
- `includeMissing: true`: 결측치 여부를 나타내는 `<prefix>_nan` 컬럼을 함께 생성합니다.
- `dropOriginal: false`: 원본 범주형 컬럼을 제거하지 않고 유지합니다.

```swift
let encoded = try binnedDF
    .select("city", "score")
    .oneHotEncoded("city", prefix: "city", dropFirst: false, dropOriginal: true)

print(encoded)
```

**콘솔 출력**:
```text
   score  city_Seoul  city_Busan  city_Incheon
0     50           1           0             0
1     70           0           1             0
2     80           0           0             1
3    100           1           0             0

[4 rows x 4 columns]
```

---

> **다음 단계**: 데이터 정제와 전처리를 마쳤다면 **[가이드 05: 벡터화 수학 연산 및 기술 통계 →](05-Vectorized-Math-and-Statistics.ko.md)**로 이동하여 벡터화 산술/수학 함수, 누적 연산, 그리고 구조화된 기술 통계 API를 학습하세요.
