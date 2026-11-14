# 가이드 05: 벡터화 수학 연산, 누적 함수 및 기술 통계

> **언어**: [English](05-Vectorized-Math-and-Statistics.md) | **한국어**  
> **네비게이션**: [← 가이드 04: 데이터 정제 및 피처 엔지니어링](04-Cleaning-and-Feature-Engineering.ko.md) | [API 레퍼런스](../API.ko.md) | **다음**: [가이드 06: 그룹 집계, 조인 및 재구조화 →](06-Grouping-Joining-and-Reshaping.ko.md)

---

## 개요

`pandas`와 `NumPy`에서 고속 수치 분석의 핵심은 반복문 없는 벡터화 연산(`df["a"] + df["b"]`, `np.log(s)`, `df.cumsum()`, `df.describe()`, `df.corr()`)입니다.

Bandal의 수치 컬럼은 연속된 `Float64Column` 메모리 버퍼(`NumericBuffer`)로 저장되며, 중소 규모 데이터는 Apple의 `Accelerate`(`vDSP`) 프레임워크로, 대규모 데이터는 `mlx-swift`(`MLXArray`) 엔진으로 자동 가속됩니다. 또한 Bandal의 모든 통계 API는 **구조화된 결과 타입(Structured Result Types)** 설계 원칙을 따릅니다. 즉, `DataFrameDescription`, `DataFrameStatistics`, `ColumnStatistics`, `QuantileTable`, `UniqueCounts` 등의 구조체를 반환하여 `print(...)` 시에는 `pandas` 스타일의 표를 출력하고, 코드에서는 타입 안전 프로퍼티로 통계량에 즉시 접근할 수 있습니다.

이 가이드에서 다룰 핵심 내용은 다음과 같습니다.
- **벡터화 컬럼 및 스칼라 산술 연산** (`+`, `-`, `*`, `/`, `combining`, `scaling`)
- **초등 수학 함수 및 수치 변환** (`UnaryMathFunction`, `applying`, `power`, `abs`, `clipped`, `rounded`)
- **누적 연산, 시차(Lag), 차분, 증감률 및 순위** (`cumulative`, `shifted`, `difference`, `percentChange`, `ranked`)
- **구조화된 기술 통계량** (`describeStructured`, `DataFrameDescription`, `statistics`, `ColumnStatistics`, `summaryTable`, `quantiles`)
- **빈도표, 공분산 행렬 및 피어슨 상관계수 행렬** (`uniqueSummary`, `valueCounts`, `covariance`, `correlation`)

---

## 1. 한눈에 보는 비교: `pandas` / `NumPy` vs. `Bandal`

| 작업 | Python `pandas` / `NumPy` | Swift `Bandal` |
| --- | --- | --- |
| Series 간 / 스칼라 사칙연산 | `s1 + s2`, `s1 * 1.1` | `try s1 + s2`, `try s1 * 1.1` |
| DataFrame 두 컬럼 사칙연산 결합 | `df["total"] = df["p"] * df["q"]` | `try df.combining("p", "q", using: .multiply, as: "total")` |
| DataFrame 컬럼 스칼라 연산 | `df["p_tax"] = df["p"] * 1.1` | `try df.scaling("p", by: 1.1, using: .multiply, as: "p_tax")` |
| 단항 수학 함수 (`log`, `sqrt`, `exp`…) | `np.log(df["x"])` | `try df.applying("x", function: .log, as: "log_x")` |
| 거듭제곱 ($x^p$) | `df["x"] ** 2` | `try df.power("x", exponent: 2.0, as: "x_sq")` |
| 클리핑 / 반올림 / 절댓값 | `df["x"].clip(0, 100).round(2).abs()` | `try df.clipped("x", lower: 0, upper: 100).rounded("x", decimals: 2).abs("x")` |
| 누적 합 / 누적 최댓값 / 최솟값 / 곱 | `df["x"].cumsum()` | `try df.cumulative("x", using: .sum)` |
| 시차 이동 / 차분 / 변화율 | `s.shift(1)` / `s.diff(1)` / `s.pct_change(1)` | `try df.shifted("x", by: 1)` / `try df.difference("x", periods: 1)` / `try df.percentChange("x", periods: 1)` |
| 평균 동점 순위 (`rank`) | `df["x"].rank(ascending=False)` | `try df.ranked("x", ascending: false, as: "rank")` |
| 종합 기술 통계표 | `df.describe(percentiles=[0.1, 0.5, 0.9])` | `df.describeStructured(percentiles: [0.1, 0.5, 0.9])` |
| 다중 분위수 테이블 | `df.quantile([0.25, 0.5, 0.75])` | `try df.quantiles([0.25, 0.5, 0.75])` (`QuantileTable`) |
| 고유값 개수 / 빈도수 집계 | `df.nunique()` / `df["cat"].value_counts()` | `df.uniqueSummary()` / `try df.valueCounts(of: "cat")` |
| 공분산 행렬 / 상관계수 행렬 | `df.cov()` / `df.corr()` | `try df.covariance()` / `try df.correlation()` |

---

## 2. 벡터화 사칙연산 (`Series` & `DataFrame`)

### 2.1 `Series` 산술 연산자 (`+`, `-`, `*`, `/`)

길이가 같은 두 수치형 `Series`(또는 `Series`와 `Double` 스칼라)는 `+`, `-`, `*`, `/` 연산자로 직접 계산할 수 있습니다. 결측치(`.missing`)는 결과에서도 자동으로 `.missing`으로 전파되며, `0`으로 나누기 등 비유한(Non-finite) 결과가 발생하면 런타임 오류 대신 안전하게 `.missing`으로 처리됩니다.

```swift
import Bandal

let price = Series(name: "price", values: [100.0, 200.0, nil, 50.0])
let qty   = Series(name: "qty",   values: [2.0,   3.0,   5.0, 4.0])

let revenue = try price * qty
let discounted = try revenue * 0.9
print(discounted)
```

**콘솔 출력**:
```text
   price
0    180
1    540
2    NaN
3    180
Name: price, Length: 4
```

### 2.2 `DataFrame.combining` 및 `DataFrame.scaling`

`DataFrame` 파이프라인 내부에서는 `combining(_:_:using:as:)`과 `scaling(_:by:using:as:)`을 사용하여 메서드 체인을 끊지 않고 벡터화 컬럼 연산을 수행할 수 있습니다.

```swift
let orders = try DataFrame(columns: [
    (name: "item", values: ["A", "B", "C"]),
    (name: "price", values: [100.0, 250.0, 80.0]),
    (name: "cost", values: [60.0, 150.0, 50.0]),
    (name: "qty", values: [10.0, 4.0, 20.0]),
])
.combining("price", "cost", using: .subtract, as: "unit_margin")
.combining("unit_margin", "qty", using: .multiply, as: "total_profit")
.scaling("price", by: 1.1, using: .multiply, as: "price_with_vat")

print(orders)
```

**콘솔 출력**:
```text
   item  price  cost  qty  unit_margin  total_profit  price_with_vat
0     A    100    60   10           40           400             110
1     B    250   150    4          100           400             275
2     C     80    50   20           30           600              88

[3 rows x 7 columns]
```

---

## 3. 초등 수학 함수 및 수치 변환

Bandal은 `Series`와 `DataFrame` 모두에서 벡터화 수학 변환 함수를 제공합니다.

- **`UnaryMathFunction`**:
  - `.sqrt` ($\sqrt{x}$), `.exp` ($e^x$), `.log` ($\ln x$), `.log10` ($\log_{10} x$), `.log1p` ($\ln(1 + x)$)
  - `.sin`, `.cos`, `.tan`, `.sign` ($-1, 0, +1$ 부호 함수)
- **`power(_:exponent:as:)`**: 각 원소를 `exponent` 거듭제곱($x^p$)합니다.
- **`abs(_:)`**, **`clipped(_:lower:upper:)`**, **`rounded(_:decimals:)`**

> **정의역 안전성(Domain Safety)**: 음수에 대한 제곱근(`.sqrt`)이나 로그(`.log`)처럼 실수 정의역을 벗어나는 입력은 에러를 던지거나 `NaN`으로 버퍼를 오염시키는 대신, 해당 셀만 안전하게 `.missing`으로 변환합니다.

```swift
let mathDF = try DataFrame(columns: [
    (name: "x", values: [-4.0, 1.0, 4.0, 9.0, 100.0]),
])
.applying("x", function: .sqrt, as: "sqrt_x")     // -4.0 -> .missing (NaN)
.applying("x", function: .log10, as: "log10_x")   // -4.0 -> .missing (NaN)
.power("x", exponent: 2.0, as: "x_squared")
.clipped("x", lower: 0.0, upper: 50.0)            // "x"를 0...50 범위로 클리핑

print(mathDF)
```

**콘솔 출력**:
```text
     x  sqrt_x  log10_x  x_squared
0    0     NaN      NaN         16
1    1       1        0          1
2    4       2  0.60206         16
3    9       3  0.95424         81
4   50      10        2      10000

[5 rows x 4 columns]
```

---

## 4. 누적 연산, 시차(Lag), 차분, 변화율 및 순위

시계열 및 금융 데이터 분석에 필수적인 `cumulative`, `shifted`, `difference`, `percentChange`, `ranked` 메서드를 `DataFrame`과 `Series` 모두에서 사용할 수 있습니다.

- `df.cumulative("col", using: .sum)`: 누적 합(`.sum`), 누적 곱(`.product`), 누적 최솟값(`.minimum`), 누적 최댓값(`.maximum`)으로 컬럼을 갱신합니다.
- `df.shifted("col", by: 1)`: 지정한 행 수만큼 아래(음수면 위)로 이동하고 빈자리를 `.missing`으로 채웁니다.
- `df.difference("col", periods: 1)`: `periods` 이전 행과의 차분($x_t - x_{t-k}$)을 계산합니다.
- `df.percentChange("col", periods: 1)`: `periods` 이전 행 대비 변화율($\frac{x_t}{x_{t-k}} - 1$)을 계산합니다.
- `df.ranked("col", ascending: false, as: "rank")`: 1부터 시작하는 순위를 계산하며, 동점 값은 평균 순위(Average-tie rank)를 부여합니다.

```swift
let dailyRevenue = try DataFrame(columns: [
    (name: "day", values: [1, 2, 3, 4]),
    (name: "rev", values: [100.0, 120.0, 90.0, 120.0]),
])

let inPlaceExample = try dailyRevenue
    .ranked("rev", ascending: false, as: "rev_rank")
    .percentChange("rev", periods: 1)
    .rounded("rev", decimals: 3)

print(inPlaceExample)
```

**콘솔 출력**:
```text
   day     rev  rev_rank
0    1     NaN         3
1    2     0.2       1.5
2    3   -0.25         4
3    4   0.333       1.5

[4 rows x 3 columns]
```

---

## 5. 구조화된 기술 통계량 (Structured Descriptive Statistics)

`pandas`에서 `df.describe()`는 결과를 또 다른 `DataFrame`으로 반환하며, 기본적으로 범주형 컬럼은 생략됩니다.

Bandal은 두 가지 상호보완적인 기술 통계 API를 제공합니다.
1. **`df.describe(percentiles:include:) -> String`**: `pandas` 스타일의 요약 통계표를 문자열로 즉시 렌더링합니다.
2. **`df.describeStructured(percentiles:include:) -> DataFrameDescription`**: 수치형 통계(`numeric: [String: ColumnStatistics]`)와 비수치형 통계(`nonNumeric: [String: NonNumericSummary]`)를 모두 담은 구조체를 반환합니다. 이 구조체를 `print(desc)`에 전달하면 `df.describe()`와 완전히 동일한 표가 출력됩니다.

### 5.1 `describeStructured()` 및 `DataFrameDescription` 활용

```swift
let store = try DataFrame(columns: [
    (name: "branch", values: ["Seoul", "Seoul", "Busan", "Seoul", "Busan"]),
    (name: "sales", values: [120.0, 150.0, 90.0, 200.0, 140.0]),
    (name: "returns", values: [2.0, 5.0, 1.0, 4.0, 3.0]),
])

// 1. 구조화된 기술 통계 객체 생성 (원하는 백분위수 배열 지정 가능)
let desc: DataFrameDescription = store.describeStructured(percentiles: [0.25, 0.5, 0.75])

// print(desc) 호출 시 pandas 스타일의 전체 요약표 출력
print(desc)

// 2. Swift 코드에서 수치형 및 범주형 통계량 프로퍼티에 직접 접근!
if let salesStats: ColumnStatistics = desc.numeric["sales"] {
    print("\nSales Mean: \(salesStats.mean ?? .nan), IQR: \((salesStats.q3 ?? 0) - (salesStats.q1 ?? 0))")
}
if let branchSummary: NonNumericSummary = desc.nonNumeric["branch"] {
    print("Top Branch: \(branchSummary.top ?? .missing) (appeared \(branchSummary.freq) times out of \(branchSummary.count))")
}
```

**콘솔 출력**:
```text
        branch    sales  returns
count   5         5      5      
mean    NaN       140    3      
std     NaN       40.6202  1.58114
min     NaN       90     1      
25%     NaN       120    2      
50%     NaN       140    3      
75%     NaN       150    4      
max     NaN       200    5      
unique  2         NaN    NaN    
top     Seoul     NaN    NaN    
freq    3         NaN    NaN    

Sales Mean: 140.0, IQR: 30.0
Top Branch: Seoul (appeared 3 times out of 5)
```

### 5.2 `statistics()`, `summaryTable()`, `quantiles(_:)`

- **`df.statistics() -> DataFrameStatistics`**: 모든 수치형 컬럼에 대한 `ColumnStatistics` (`count`, `sum`, `mean`, `variance`, `standardDeviation`, `min`, `q1`, `median`, `q3`, `max`)를 담고 있으며 `stats["sales"]` 서브스크립트를 제공합니다.
- **`df.summaryTable() -> DataFrameSummary`**: 6대 핵심 요약 통계(`count`, `mean`, `std`, `min`, `50%`, `max`) 테이블을 반환합니다.
- **`df.quantiles(_ probabilities: [Double]) -> QuantileTable`**: 여러 확률 분위수를 한 번에 계산하며 `table[column, probability]` 서브스크립트와 `table.toDataFrame()` 변환을 지원합니다.

```swift
let qTable: QuantileTable = try store.quantiles([0.1, 0.5, 0.9])
print(qTable)

// (컬럼명, 확률) 서브스크립트로 특정 분위수 값 즉시 조회
let p90Sales: Double? = qTable["sales", 0.9]
print("\n90th percentile of sales: \(p90Sales ?? .nan)")
```

**콘솔 출력**:
```text
      sales  returns
0.1     102      1.4
0.5     140        3
0.9     180      4.6

90th percentile of sales: 180.0
```

---

## 6. 고유값 집계, 빈도표, 공분산 및 상관계수 행렬

### 6.1 고유값 개수(`uniqueSummary`) 및 빈도수 테이블(`valueCounts`)

```swift
// 1. 컬럼별 결측 제외 고유값 개수 (pandas의 df.nunique())
let uniques: UniqueCounts = store.uniqueSummary()
print(uniques)

// 2. 특정 컬럼의 값 빈도수 집계표 (pandas의 df["branch"].value_counts())
print("\nBranch Value Counts:")
print(try store.valueCounts(of: "branch"))
```

**콘솔 출력**:
```text
branch     2
sales      5
returns    5
Columns: 3 (rows: 5)

Branch Value Counts:
   value  count
0  Seoul      3
1  Busan      2

[2 rows x 2 columns]
```

### 6.2 표본 공분산(`covariance()`) 및 피어슨 상관계수(`correlation()`) 행렬

`covariance()`와 `correlation()`은 모든 수치형 컬럼 쌍에 대해 표본 공분산 및 피어슨 상관계수 행렬을 `DataFrame`으로 반환합니다(쌍별로 결측치가 있는 행은 자동 제외). 대규모 데이터프레임에서는 MLX(`float64` CPU 스트림) 또는 Apple `Accelerate`(`vDSP_dotprD`)를 통해 행렬 내적이 가속됩니다.

```swift
print("--- Covariance Matrix ---")
print(try store.covariance())

print("\n--- Pearson Correlation Matrix ---")
print(try store.correlation())
```

**콘솔 출력**:
```text
--- Covariance Matrix ---
   sales  returns
0   1650       55
1     55      2.5

[2 rows x 2 columns]

--- Pearson Correlation Matrix ---
      sales   returns
0         1  0.856349
1  0.856349         1

[2 rows x 2 columns]
```

---

> **다음 단계**: 기술 통계와 벡터화 수학 연산을 익혔다면 **[가이드 06: 그룹 집계, 조인 및 재구조화 →](06-Grouping-Joining-and-Reshaping.ko.md)**로 이동하여 `groupBy` 집계, 그룹 브로드캐스팅(`transform`), 테이블 조인, 피벗 테이블 및 재구조화 방법을 학습하세요.
