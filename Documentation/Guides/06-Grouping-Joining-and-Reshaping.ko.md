# 가이드 06: 그룹 집계(GroupBy), 관계형 조인 및 재구조화

> **언어**: [English](06-Grouping-Joining-and-Reshaping.md) | **한국어**  
> **네비게이션**: [← 가이드 05: 벡터화 수학 연산 및 기술 통계](05-Vectorized-Math-and-Statistics.ko.md) | [API 레퍼런스](../API.ko.md) | **다음**: [가이드 07: 시계열 분석 및 윈도우 함수 →](07-Time-Series-and-Window-Functions.ko.md)

---

## 개요

테이블 형태의 데이터를 분석할 때 비즈니스 인텔리전스(BI)와 파생 변수 생성의 핵심이 되는 3대 작업은 다음과 같습니다.
1. **분할-적용-결합 (`groupBy`)**: 하나 이상의 범주형 키로 행을 그룹화하여 요약 지표를 집계(`aggregate`)하거나, 그룹 통계량을 원본 행 위치로 브로드캐스팅(`transform`)하거나, 그룹 단위 조건으로 필터링(`filter`)합니다.
2. **관계형 조인 및 행 결합 (`join`, `leftJoin`, `concatenatingRows`)**: 공통 키를 기준으로 두 테이블을 수평 결합하거나 동일 스키마의 테이블들을 수직으로 쌓습니다.
3. **데이터 재구조화 (`melt`, `pivotTable`, `pivoted`, `crossTabulated`, `exploded`, `transposed`)**: 넓은(Wide) 포맷과 긴(Long) 포맷 간에 테이블 구조를 자유롭게 변환합니다.

이 가이드에서는 `pandas`와의 1:1 비교 표와 실행 가능한 Swift 예제를 통해 이 기능들을 완벽히 마스터합니다.

---

## 1. 한눈에 보는 비교: `pandas` vs. `Bandal`

| 작업 | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| 단일 그룹 집계 | `df.groupby("dept")["salary"].mean()` | `try df.groupBy("dept").aggregate("salary", using: .mean)` |
| 다중 컬럼 / 별칭 지정 집계 | `df.groupby(["dept", "tier"]).agg(total=("sales", "sum"), avg=("sales", "mean"))` | `try df.groupBy("dept", "tier").aggregate([Aggregation("sales", using: .sum, as: "total"), Aggregation("sales", using: .mean, as: "avg")])` |
| 그룹 통계량 브로드캐스트 (`transform`) | `df["dept_mean"] = df.groupby("dept")["salary"].transform("mean")` | `try df.groupBy("dept").transform("salary", using: .mean)` |
| 그룹 단위 필터링 (`filter`) | `df.groupby("dept").filter(lambda g: len(g) >= 2)` | `try df.groupBy("dept").filter { $0.rowCount >= 2 }` |
| Inner / Left / Right / Full 조인 | `df1.merge(df2, on="id", how="left")` | `try df1.join(df2, on: "id", kind: .left)` (또는 `try df1.leftJoin(df2, on: "id")`) |
| 다중 키 및 접미사 지정 조인 | `df1.merge(df2, left_on=["a"], right_on=["b"], suffixes=("_l", "_r"))` | `try df1.join(df2, leftOn: ["a"], rightOn: ["b"], kind: .inner, suffixes: ("_l", "_r"))` |
| 수직 행 연결 (`concat`) | `pd.concat([df1, df2], ignore_index=True)` | `try DataFrame.concatenatingRows([df1, df2])` |
| 넓은 표 → 긴 표 언피벗 (`melt`) | `df.melt(id_vars=["id"], var_name="metric", value_name="val")` | `try df.melt(idColumns: ["id"], variableName: "metric", valueName: "val")` |
| 집계형 피벗 테이블 (`pivot_table`) | `df.pivot_table(index=["region"], columns="qtr", values="rev", aggfunc="sum")` | `try df.pivotTable(index: ["region"], columns: "qtr", values: "rev", using: .sum)` |
| 비집계형 고유 쌍 피벗 (`pivot`) | `df.pivot(index="date", columns="ticker", values="close")` | `try df.pivoted(index: "date", columns: "ticker", values: "close")` |
| 교차 빈도표 (`crosstab`) | `pd.crosstab(df["dept"], df["remote"], normalize=True)` | `try df.crossTabulated(index: "dept", columns: "remote", normalize: true)` |
| 구분자 문자열 행 분리 (`explode`) | `df.assign(tag=df["tag"].str.split(",")).explode("tag")` | `try df.exploded("tag", separator: ",")` |
| 행/열 전치 (`df.T`) | `df.T` | `try df.transposed()` |

---

## 2. 분할-적용-결합 (`groupBy` & `GroupedDataFrame`)

### 2.1 지원하는 `AggregateFunction` 종류 (10종)

Bandal은 `AggregateFunction` 열거형을 통해 10가지 집계 함수를 기본 지원합니다.

| 케이스 | 기본 출력 컬럼명 | 설명 |
| --- | --- | --- |
| `.count` | `<col>_count` | 그룹 내 결측치(`.missing`)를 제외한 유효 값 개수 |
| `.sum` | `<col>_sum` | 수치 합계 |
| `.mean` | `<col>_mean` | 산술 평균 |
| `.min` / `.max` | `<col>_min` / `<col>_max` | 최솟값 / 최댓값 |
| `.median` | `<col>_median` | 중앙값 (50% 분위수) |
| `.variance` | `<col>_variance` | 표본 분산 (자유도 $N - 1$) |
| `.standardDeviation` | `<col>_std` | 표본 표준편차 (자유도 $N - 1$) |
| `.first` / `.last` | `<col>_first` / `<col>_last` | 그룹 내 첫 번째 / 마지막 비결측 값 (문자열·날짜 등 모든 컬럼 타입 지원!) |

> **그룹 키 결측치 및 정렬 규칙**: `pandas`의 `df.groupby(..., dropna=True)`와 동일하게, 그룹 키 컬럼 중 하나라도 `.missing`인 행은 그룹 집계에서 제외되며 결과 그룹 키는 오름차순으로 정렬됩니다.

### 2.2 단일 및 다중 집계와 별칭 지정 (`Aggregation`)

```swift
import Bandal

let sales = try DataFrame(columns: [
    (name: "region", values: ["Seoul", "Seoul", "Busan", "Seoul", "Busan"]),
    (name: "channel", values: ["Online", "Retail", "Online", "Online", "Retail"]),
    (name: "amount", values: [120.0, 80.0, 150.0, 200.0, 90.0]),
    (name: "rep", values: ["Kim", "Lee", "Park", "Choi", "Jung"]),
])

// 한 번의 그룹화 패스로 여러 집계 지표 계산 및 출력 컬럼명(as:) 지정
let byRegion = try sales
    .groupBy("region")
    .aggregate([
        Aggregation("amount", using: .count, as: "orders"),
        Aggregation("amount", using: .sum, as: "total_sales"),
        Aggregation("amount", using: .mean, as: "avg_order"),
        Aggregation("rep", using: .first, as: "lead_rep"),
    ])

print(byRegion)
```

**콘솔 출력**:
```text
   region  orders  total_sales  avg_order  lead_rep
0   Busan       2          240        120      Park
1   Seoul       3          400    133.333       Kim

[2 rows x 5 columns]
```

---

## 3. 그룹 통계 브로드캐스팅(`transform`) 및 그룹 필터링(`filter`)

### 3.1 `GroupedDataFrame.transform(_:using:)`

`pandas`에서 `df.groupby("region")["amount"].transform("mean")`은 그룹별 평균을 구한 뒤 원본 데이터프레임의 각 행에 맞춰 값을 브로드캐스팅합니다.

Bandal에서도 `groupBy(...).transform(_:using:)`을 호출하면 원본 행 순서와 길이를 그대로 유지한 채 `<column>_<operation>`(예: `"amount_sum"`) 컬럼이 추가됩니다.

```swift
// 각 주문 금액이 해당 지역 전체 매출 합계에서 차지하는 비중(regional_share) 계산
let withRegionalShare = try sales
    .groupBy("region")
    .transform("amount", using: .sum) // "amount_sum" 컬럼 추가
    .combining("amount", "amount_sum", using: .divide, as: "regional_share")
    .rounded("regional_share", decimals: 3)

print(withRegionalShare)
```

**콘솔 출력**:
```text
   region  channel  amount   rep  amount_sum  regional_share
0   Seoul   Online     120   Kim         400             0.3
1   Seoul   Retail      80   Lee         400             0.2
2   Busan   Online     150  Park         240           0.625
3   Seoul   Online     200  Choi         400             0.5
4   Busan   Retail      90  Jung         240           0.375

[5 rows x 6 columns]
```

### 3.2 `GroupedDataFrame.filter(_:)`

`filter(_:)`는 각 그룹의 서브 `DataFrame`을 클로저에 전달하고, 클로저가 `true`를 반환한 그룹에 속하는 행들만 원본 순서대로 남깁니다.

```swift
// 주문 건수가 3건 이상인 지역의 행들만 필터링
let highVolumeRegions = try sales
    .groupBy("region")
    .filter { groupDF in
        groupDF.rowCount >= 3
    }

print(highVolumeRegions)
```

**콘솔 출력**:
```text
   region  channel  amount   rep
0   Seoul   Online     120   Kim
1   Seoul   Retail      80   Lee
2   Seoul   Online     200  Choi

[3 rows x 4 columns]
```

---

## 4. 관계형 조인 (`join`, `leftJoin`) 및 수직 행 결합 (`concatenatingRows`)

### 4.1 단일 키 조인 (`JoinKind`: `.inner`, `.left`, `.right`, `.full`)

양쪽 테이블의 조인 키 컬럼명이 같고 나머지 컬럼명이 겹치지 않을 때는 `join(_:on:kind:)` 또는 단축 메서드인 `leftJoin(_:on:)`을 사용합니다.

```swift
let customers = try DataFrame(columns: [
    (name: "cust_id", values: ["C1", "C2", "C3"]),
    (name: "name", values: ["Alice", "Bob", "Charlie"]),
])

let purchases = try DataFrame(columns: [
    (name: "cust_id", values: ["C1", "C1", "C4"]),
    (name: "total", values: [50.0, 120.0, 300.0]),
])

// Left Join: 왼쪽 고객(C1, C2, C3)을 모두 유지하며 매칭되지 않는 C2, C3의 total은 .missing 처리
let leftJoined = try customers.leftJoin(purchases, on: "cust_id")
print("--- Left Join ---")
print(leftJoined)

// Full Outer Join: 오른쪽에만 존재하는 C4까지 모두 포함
let fullJoined = try customers.join(purchases, on: "cust_id", kind: .full)
print("\n--- Full Outer Join ---")
print(fullJoined)
```

**콘솔 출력**:
```text
--- Left Join ---
   cust_id     name  total
0       C1    Alice     50
1       C1    Alice    120
2       C2      Bob    NaN
3       C3  Charlie    NaN

[4 rows x 3 columns]

--- Full Outer Join ---
   cust_id     name  total
0       C1    Alice     50
1       C1    Alice    120
2       C2      Bob    NaN
3       C3  Charlie    NaN
4       C4      nil    300

[5 rows x 3 columns]
```

### 4.2 다중 키 조인 및 중복 컬럼 접미사 처리 (`join(_:leftOn:rightOn:kind:suffixes:)`)

양쪽 테이블의 키 컬럼 이름이 서로 다르거나, 키가 아닌 일반 컬럼 이름이 양쪽에 모두 존재할 때는 `join(_:leftOn:rightOn:kind:suffixes:)`를 사용합니다.

```swift
let targetQ1 = try DataFrame(columns: [
    (name: "dept_id", values: ["D1", "D2"]),
    (name: "revenue", values: [1000.0, 800.0]),
])

let actualQ1 = try DataFrame(columns: [
    (name: "id", values: ["D1", "D2"]),
    (name: "revenue", values: [1150.0, 750.0]),
])

let comparison = try targetQ1.join(
    actualQ1,
    leftOn: ["dept_id"],
    rightOn: ["id"],
    kind: .inner,
    suffixes: (left: "_target", right: "_actual")
)
print(comparison)
```

**콘솔 출력**:
```text
   dept_id  revenue_target   id  revenue_actual
0       D1            1000   D1            1150
1       D2             800   D2             750

[2 rows x 4 columns]
```

### 4.3 수직 행 연결 (`DataFrame.concatenatingRows`)

동일한 컬럼 이름과 순서를 가진 여러 `DataFrame`을 위아래로 결합합니다(`pd.concat([df1, df2], axis=0)`).

```swift
let jan = try DataFrame(columns: [
    (name: "month", values: ["Jan"]),
    (name: "rev", values: [100.0]),
])
let feb = try DataFrame(columns: [
    (name: "month", values: ["Feb"]),
    (name: "rev", values: [130.0]),
])

let combined = try DataFrame.concatenatingRows([jan, feb])
print(combined)
```

```text
   month  rev
0    Jan  100
1    Feb  130

[2 rows x 2 columns]
```

---

## 5. 데이터 재구조화 (Reshaping & Pivot Tables)

### 5.1 넓은 표를 긴 표로 언피벗 (`melt`)

`melt(idColumns:variableName:valueName:)`는 식별자 컬럼(`idColumns`)을 고정하고 나머지 측정값 컬럼들을 `(variable, value)` 쌍의 행으로 길게 풀어냅니다.

```swift
let wideMetrics = try DataFrame(columns: [
    (name: "city", values: ["Seoul", "Busan"]),
    (name: "q1", values: [100.0, 60.0]),
    (name: "q2", values: [120.0, 75.0]),
])

let longMetrics = try wideMetrics.melt(
    idColumns: ["city"],
    variableName: "quarter",
    valueName: "revenue"
)
print(longMetrics)
```

**콘솔 출력**:
```text
    city  quarter  revenue
0  Seoul       q1      100
1  Busan       q1       60
2  Seoul       q2      120
3  Busan       q2       75

[4 rows x 3 columns]
```

### 5.2 고유 쌍 재구조화(`pivoted`) vs. 집계형 피벗 테이블(`pivotTable`)

Bandal은 목적에 따라 두 가지 피벗 메서드를 구분하여 제공합니다.
1. **`pivoted(index:columns:values:)` (`df.pivot`)**: `(index, columns)` 조합이 고유할 때 집계 없이 값을 가로 컬럼(`"q1"`, `"q2"`)으로 즉시 펼칩니다. 중복된 `(index, columns)` 쌍이 있으면 `.duplicateColumnName` 에러를 던집니다.
2. **`pivotTable(index:columns:values:using:)` (`df.pivot_table`)**: 동일한 `(index, columns)` 조합에 여러 행이 존재할 때 지정한 `AggregateFunction`으로 집계한 2차원 요약표를 생성합니다.

```swift
// 1. (city, quarter) 조합이 고유할 때 melt의 역변환:
let unmelted = try longMetrics.pivoted(
    index: "city",
    columns: "quarter",
    values: "revenue"
)
print("--- pivoted (Unique Pairs) ---")
print(unmelted)

// 2. 동일한 (region, channel) 조합의 여러 주문을 합산(.sum)하는 피벗 테이블:
let summaryPivot = try sales.pivotTable(
    index: ["region"],
    columns: "channel",
    values: "amount",
    using: .sum
)
print("\n--- pivotTable (Aggregated Sum) ---")
print(summaryPivot)
```

**콘솔 출력**:
```text
--- pivoted (Unique Pairs) ---
    city   q1   q2
0  Busan   60   75
1  Seoul  100  120

[2 rows x 3 columns]

--- pivotTable (Aggregated Sum) ---
   region  amount_string_Online_sum  amount_string_Retail_sum
0   Busan                       150                        90
1   Seoul                       320                        80

[2 rows x 3 columns]
```

### 5.3 교차 빈도표 (`crossTabulated` / `pd.crosstab`)

`crossTabulated(index:columns:normalize:)`는 두 범주형 컬럼 간의 교차 빈도수(또는 `normalize: true` 시 전체 합이 `1.0`인 결합 확률 비율)를 계산합니다.

```swift
let ctCounts = try sales.crossTabulated(index: "region", columns: "channel", normalize: false)
print("--- Cross-Tabulation (Counts) ---")
print(ctCounts)

let ctProps = try sales.crossTabulated(index: "region", columns: "channel", normalize: true)
print("\n--- Cross-Tabulation (Normalized Proportions) ---")
print(ctProps)
```

**콘솔 출력**:
```text
--- Cross-Tabulation (Counts) ---
   region  Online  Retail
0   Busan       1       1
1   Seoul       2       1

[2 rows x 3 columns]

--- Cross-Tabulation (Normalized Proportions) ---
   region  Online  Retail
0   Busan     0.2     0.2
1   Seoul     0.4     0.2

[2 rows x 3 columns]
```

### 5.4 구분자 문자열 행 분리(`exploded`) 및 전치(`transposed`)

```swift
let articles = try DataFrame(columns: [
    (name: "id", values: [1, 2]),
    (name: "tags", values: ["swift, mlx, data", "pandas, python"]),
])

// 쉼표(,)로 구분된 태그 문자열을 개별 행으로 분리 (공백 자동 트림)
let explodedTags = try articles.exploded("tags", separator: ",")
print("--- Exploded ---")
print(explodedTags)

// 행과 열을 전치 (df.T)
let transposed = try wideMetrics.transposed(indexColumnName: "metric", rowPrefix: "city_")
print("\n--- Transposed ---")
print(transposed)
```

**콘솔 출력**:
```text
--- Exploded ---
   id    tags
0   1   swift
1   1     mlx
2   1    data
3   2  pandas
4   2  python

[5 rows x 2 columns]

--- Transposed ---
   metric  city_0  city_1
0    city   Seoul   Busan
1      q1     100      60
2      q2     120      75

[3 rows x 3 columns]
```

---

> **다음 단계**: **[가이드 07: 시계열 분석 및 윈도우 함수 →](07-Time-Series-and-Window-Functions.ko.md)**로 이동하여 행 기반/시간 간격 기반 이동 윈도우(`rolling`), 확장 윈도우(`expanding`), 지수 가중 이동평균(`exponentiallyWeighted`), 근접 시점 조인(`joinAsOf`), 시계열 리샘플링(`resample`)을 학습하세요.
