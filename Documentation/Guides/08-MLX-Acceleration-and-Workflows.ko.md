# 가이드 08: MLX 가속 엔진, 실행 정책 제어 및 실전 종합 워크플로우

> **언어**: [English](08-MLX-Acceleration-and-Workflows.md) | **한국어**  
> **네비게이션**: [← 가이드 07: 시계열 분석 및 윈도우 함수](07-Time-Series-and-Window-Functions.ko.md) | [API 레퍼런스](../API.ko.md) | [🏠 가이드 01로 돌아가기](01-Core-Data-Structures.ko.md)

---

## 개요

Python `pandas`가 내부적으로 `NumPy`의 C/Fortran 연속 배열을 수치 연산 엔진으로 삼아 고속 연산을 수행하듯, Bandal은 **타입별 컬럼 스토리지(Typed Columnar Storage)**, **Apple `Accelerate`(`vDSP`)**, 그리고 **`mlx-swift`(`MLXArray`)**를 결합한 Apple Silicon 맞춤형 하이브리드 실행 엔진을 탑재하고 있습니다.

이 마지막 가이드에서 다룰 내용은 다음과 같습니다.
- **Bandal 하이브리드 엔진의 내부 아키텍처**: 타입별 연속 메모리 버퍼(`Float64Column`, `CategoricalColumn`, `BitSet`)와 `vDSP` / `mlx-swift` 자동 분기 원리
- **실행 정책(`ExecutionPolicy`) 제어**: `.automatic`, `.nativeOnly`, `.mlxPreferred` 설정 및 `BandalRuntime.withExecutionPolicy` 활용법
- **`MLXVector` 및 `MLXArray` 텐서 직접 연동**: `DataFrame` 수치 컬럼을 `mlx-swift` 머신러닝 파이프라인의 `MLXArray`로 변환하고 결과를 다시 `DataFrame`에 결합하는 방법
- **실전 종합 케이스 스터디**: 원시 CSV 로드부터 정제, 선형 보간, 테이블 조인, 그룹 브로드캐스팅, 요약 집계 및 마크다운 리포트 생성까지 이어지는 엔드투엔드(End-to-End) 파이프라인

---

## 1. 아키텍처: 컬럼 기반 스토리지와 하이브리드 실행 엔진

### 1.1 타입별 컬럼 스토리지 + 64비트 널 비트맵(`BitSet`)

표를 단순한 `[[Any]]`나 `[[DataValue]]` 2차원 배열로 저장하면 매 연산마다 박싱/언박싱과 포인터 추적 비용이 발생합니다. Bandal은 각 컬럼을 데이터 타입에 최적화된 연속 버퍼와 64비트 워드 압축 널 비트맵(`BitSet`)의 쌍으로 저장합니다.

| 내부 컬럼 스토리지 타입 | 메모리 레이아웃 | 주요 최적화 효과 |
| --- | --- | --- |
| `Float64Column` | `ContiguousArray<Double>` + `BitSet` | `vDSP` SIMD 및 `MLX` 텐서 무복사 연산 |
| `Int64Column` | `ContiguousArray<Int64>` + `BitSet` | 64비트 정수 ID 및 카운터 고속 해싱/정렬 |
| `BoolColumn` | 64비트 압축 `BitSet` 값 + `BitSet` 널 | 메모리 사용량 1/8 절감 및 비트 단위 논리 연산 |
| `DateColumn` | `ContiguousArray<Double>` (UTC 초) + `BitSet` | $O(N)$ 기수 정렬, `joinAsOf` 이진 탐색, `resample` |
| `CategoricalColumn` | `ContiguousArray<UInt32>` 사전 코드 + `[String]` 사전 | 저카디널리티 문자열에 대한 초고속 $O(N)$ `groupBy` 및 `leftJoin` |
| `StringColumn` | `ContiguousArray<String>` + `BitSet` | 고카디널리티 일반 텍스트 저장 |

### 1.2 자동 디스패치 전략: 네이티브 `vDSP` vs. `mlx-swift`

`mlx-swift`의 텐서 그래프를 생성하고 디스패치하는 데는 미세한 고정 오버헤드가 존재합니다. 예를 들어 50행짜리 소규모 컬럼은 CPU 캐시 안에서 `vDSP` SIMD 루프로 수 나노초 만에 처리하는 것이 훨씬 빠르며, 반대로 수만 행 이상의 대규모 컬럼이나 2차원 스트라이드 이동 윈도우, 공분산 행렬 곱셈에서는 MLX의 통합 메모리(Unified Memory) 텐서 엔진이 압도적인 처리량을 발휘합니다.

기본 정책인 `.automatic(minimumElementCount: 4096)`에서 Bandal은 다음과 같이 최적 경로를 자동 선택합니다.
- **`4,096`개 미만 원소**: 네이티브 Swift 및 Apple `Accelerate` (`vDSP_vaddD`, `vDSP_vmulD`, `vDSP_dotprD` 등)로 즉시 실행합니다.
- **`4,096`개 이상 원소** (및 MLX/Metal 런타임 사용 가능 시): `mlx-swift`(`MLXArray`)의 CPU `float64` 스트림(64비트 `Double` 정밀도 100% 보존)을 사용하여 사칙연산, 초등 수학 함수, 스캐터 그룹 집계(`at.add`, `at.maximum`, `at.minimum`), 2D 스트라이드 이동 윈도우(`MLX.asStrided`), 공분산/상관계수 행렬 곱셈(`MLX.matmul`)을 가속합니다.
- **투명한 자동 폴백(Automatic Fallback)**: 비수치 타입 연산이거나 Metal 번들이 없는 커맨드라인 환경(`swift test` 등)에서는 사용자 코드 수정 없이 네이티브 `Accelerate` 엔진으로 자동 폴백하여 동일한 수치 결과를 보장합니다.

---

## 2. 실행 정책 제어 (`ExecutionPolicy` & `BandalRuntime`)

`MLXBackend`와 `BandalRuntime`을 통해 현재 환경의 MLX 가용 여부를 확인하거나, 전역 또는 특정 클로저 블록 범위에서만 실행 정책을 변경할 수 있습니다.

```swift
import Bandal

// 1. MLX 백엔드 가용 여부 및 기본 임계값 확인
print("MLX Available: \(MLXBackend.isAvailable)")
print("Default threshold: \(MLXBackend.defaultMinimumElements) elements")

let df = try DataFrame(columns: [
    (name: "x", values: [10.0, 20.0, 30.0, 40.0, 50.0]),
    (name: "y", values: [2.0,  4.0,  6.0,  8.0,  10.0]),
])

// 2. 특정 클로저 범위에서만 MLX 우선 실행 강제 (.mlxPreferred)
let mlxResult = try BandalRuntime.withExecutionPolicy(.mlxPreferred) {
    try df
        .combining("x", "y", using: .multiply, as: "xy")
        .applying("xy", function: .sqrt, as: "sqrt_xy")
}

// 3. 특정 클로저 범위에서만 네이티브 Swift/Accelerate 실행 강제 (.nativeOnly)
let nativeResult = try BandalRuntime.withExecutionPolicy(.nativeOnly) {
    try df.correlation()
}

// 4. 자동 분기 임계값 변경 (예: 1,024개 이상부터 MLX 사용)
BandalRuntime.executionPolicy = .automatic(minimumElementCount: 1024)
// 기본값(4,096)으로 복원
BandalRuntime.executionPolicy = .automatic()
```

---

## 3. `MLXVector` 및 `MLXArray` 텐서 직접 연동하기

Bandal로 전처리한 파생 변수들을 `mlx-swift` 기반의 머신러닝/딥러닝 모델 입력 텐서(`MLXArray`)로 넘기거나, 모델이 출력한 예측값 벡터를 다시 `DataFrame` 컬럼으로 합치고 싶을 때 **`MLXVector`**를 사용합니다.

```swift
public struct MLXVector: Equatable, Sendable {
    public let values: [Double]    // 연속된 Double 배열 (결측 위치는 0.0으로 채워짐)
    public let isMissing: [Bool]   // 원래 셀이 .missing이었는지 나타내는 마스크
    public var count: Int { values.count }
    public var hasMissing: Bool { ... }

    // 결측 위치를 `fillValue`로 치환하여 1차원 float32 MLXArray 텐서로 변환
    public func makeMLXArray(replacingMissingWith fillValue: Float = 0) -> MLXArray

    // 결측 여부 마스크를 1차원 Bool MLXArray 텐서로 변환
    public func missingMaskArray() -> MLXArray
}
```

### 예제: `MLXVector` / `MLXArray` 추출 및 `DataFrame` 재결합

```swift
import Bandal
import MLX

let features = try DataFrame(columns: [
    (name: "id", values: ["S1", "S2", "S3", "S4"]),
    (name: "signal", values: [1.0, nil, 3.0, 4.0]),
])

// 1. DataFrame 컬럼(또는 Series.mlxVector())으로부터 MLXVector 추출
let vec: MLXVector = try features.mlxVector(for: "signal")
print("Has missing: \(vec.hasMissing), Values buffer: \(vec.values)")

// 2. MLX 가용 환경에서 MLXArray 텐서로 변환 (결측치는 0.0으로 대체)
if MLXBackend.isAvailable {
    let tensor: MLXArray = vec.makeMLXArray(replacingMissingWith: 0.0)
    let maskTensor: MLXArray = vec.missingMaskArray()
    print("Tensor shape: \(tensor.shape), dtype: \(tensor.dtype)")
    print("Mask shape: \(maskTensor.shape)")
}

// 3. 연산 결과 배열과 기존 결측 마스크를 보존한 새 MLXVector를 DataFrame에 컬럼으로 추가
let scaledValues = vec.values.map { $0 * 10.0 }
let outputVec = MLXVector(values: scaledValues, isMissing: vec.isMissing)

let updatedDF = try features.withColumn(name: "signal_x10", vector: outputVec)
print(updatedDF)
```

**콘솔 출력**:
```text
Has missing: true, Values buffer: [1.0, 0.0, 3.0, 4.0]
   id  signal  signal_x10
0  S1       1          10
1  S2     NaN         NaN
2  S3       3          30
3  S4       4          40

[4 rows x 3 columns]
```

---

## 4. 실전 종합 케이스 스터디: 이커머스 매출 분석 파이프라인

지금까지 가이드 01~08에서 학습한 기능들을 하나의 완결된 실무 분석 파이프라인으로 결합해 보겠습니다.
1. 공백, 통화 기호(`$1,200.00`), 커스텀 결측 토큰(`N/A`), 중복 행이 섞여 있는 일별 매장 주문 CSV 데이터를 로드합니다.
2. 중복을 제거하고 문자열을 정규화한 뒤, 매장별 시간순 정렬 후 누락된 매출을 선형 보간(`interpolated`)합니다.
3. 매장 마스터 테이블(`storeDirectory`)과 `leftJoin`하여 도시 정보를 결합하고, `groupBy(...).transform`으로 매장 내 일별 매출 비중을 계산합니다.
4. 도시별 종합 성과를 집계(`aggregate`)하고 마크다운 표(`toMarkdown()`)로 리포트를 출력합니다.

```swift
import Bandal
import Foundation

// --- 1단계: 커스텀 옵션으로 원시 CSV 데이터 로드 ---
let rawOrdersCSV = """
order_id,store_id,date,revenue,items
ORD-01, s_01 ,2026-03-01,"$1,200.00",10
ORD-01, s_01 ,2026-03-01,"$1,200.00",10
ORD-02, s_01 ,2026-03-02,N/A,12
ORD-03, s_01 ,2026-03-03,"$1,800.00",15
ORD-04, s_02 ,2026-03-01,"$900.00",9
ORD-05, s_02 ,2026-03-02,"$1,100.00",11
"""

let csvOptions = CSVReadOptions(
    additionalNullLiterals: ["N/A"],
    dateFormat: "yyyy-MM-dd"
)
let rawOrders = try DataFrame(csv: rawOrdersCSV, options: csvOptions)

// --- 2단계: 중복 제거, 문자열 정규화, 선형 보간 및 파생 변수 생성 ---
let cleanedOrders = try rawOrders
    // 중복된 order_id 행 제거
    .dropDuplicates(in: ["order_id"], keep: .first)
    // store_id 공백 제거 및 대문자 통일 (" s_01 " -> "S_01")
    .trimmed("store_id")
    .uppercased("store_id")
    // 선형 보간 전 매장별·날짜별 오름차순 정렬
    .sorted(by: [
        SortDescriptor("store_id", order: .ascending),
        SortDescriptor("date", order: .ascending),
    ])
    // 결측된 매출(ORD-02)을 선형 보간 (1200 -> [1500] -> 1800)
    .interpolated("revenue")
    // 품목당 평균 단가(avg_item_price = revenue / items) 계산
    .combining("revenue", "items", using: .divide, as: "avg_item_price")
    .rounded("avg_item_price", decimals: 2)

print("=== 1. Cleaned & Interpolated Orders ===")
print(cleanedOrders)

// --- 3단계: 매장 메타데이터 Left Join 및 그룹 브로드캐스트(transform) ---
let storeDirectory = try DataFrame(columns: [
    (name: "store_id", values: ["S_01", "S_02"]),
    (name: "city", values: ["Seoul", "Busan"]),
])

let enrichedOrders = try cleanedOrders
    .leftJoin(storeDirectory, on: "store_id")
    // 각 매장의 전체 매출 합계(revenue_sum)를 모든 행에 브로드캐스트
    .groupBy("store_id")
    .transform("revenue", using: .sum)
    // 매장 전체 매출 대비 해당 일자의 매출 비중(store_rev_share) 계산
    .combining("revenue", "revenue_sum", using: .divide, as: "store_rev_share")
    .rounded("store_rev_share", decimals: 3)

print("\n=== 2. Enriched Orders with Store Share ===")
print(enrichedOrders)

// --- 4단계: 도시별 경영 요약 집계 및 Markdown 리포트 생성 ---
let citySummary = try enrichedOrders
    .groupBy("city")
    .aggregate([
        Aggregation("order_id", using: .count, as: "days_active"),
        Aggregation("revenue",  using: .sum,   as: "total_revenue"),
        Aggregation("revenue",  using: .mean,  as: "daily_avg_rev"),
        Aggregation("items",    using: .sum,   as: "total_items"),
    ])
    .sorted(by: [SortDescriptor("total_revenue", order: .descending)])

print("\n=== 3. Executive Summary (Markdown) ===")
print(citySummary.toMarkdown())
```

**콘솔 출력**:
```text
=== 1. Cleaned & Interpolated Orders ===
   order_id  store_id                  date  revenue  items  avg_item_price
0    ORD-01      S_01  2026-03-01T00:00:00Z     1200     10             120
1    ORD-02      S_01  2026-03-02T00:00:00Z     1500     12             125
2    ORD-03      S_01  2026-03-03T00:00:00Z     1800     15             120
3    ORD-04      S_02  2026-03-01T00:00:00Z      900      9             100
4    ORD-05      S_02  2026-03-02T00:00:00Z     1100     11             100

[5 rows x 6 columns]

=== 2. Enriched Orders with Store Share ===
   order_id  store_id                  date  revenue  items  avg_item_price   city  revenue_sum  store_rev_share
0    ORD-01      S_01  2026-03-01T00:00:00Z     1200     10             120  Seoul         4500            0.267
1    ORD-02      S_01  2026-03-02T00:00:00Z     1500     12             125  Seoul         4500            0.333
2    ORD-03      S_01  2026-03-03T00:00:00Z     1800     15             120  Seoul         4500              0.4
3    ORD-04      S_02  2026-03-01T00:00:00Z      900      9             100  Busan         2000             0.45
4    ORD-05      S_02  2026-03-02T00:00:00Z     1100     11             100  Busan         2000             0.55

[5 rows x 9 columns]

=== 3. Executive Summary (Markdown) ===
| city | days_active | total_revenue | daily_avg_rev | total_items |
| :--- | ---: | ---: | ---: | ---: |
| Seoul | 3 | 4500 | 1500 | 37 |
| Busan | 2 | 2000 | 1000 | 20 |
```

---

## 전체 학습 가이드 목차 요약

1. **[가이드 01: 핵심 데이터 구조 (`DataValue`, `Series`, `DataFrame`, `DataRow`)](01-Core-Data-Structures.ko.md)**
2. **[가이드 02: 데이터 입출력(CSV & JSON) 및 스키마 탐색 (`infoSummary`, `RenderOptions`)](02-Data-IO-and-Inspection.ko.md)**
3. **[가이드 03: 인덱싱, 슬라이싱, 불리언 마스크 및 쿼리 (`BooleanMask`, `RowPredicate`)](03-Indexing-Filtering-and-Querying.ko.md)**
4. **[가이드 04: 데이터 정제, 결측치 처리 및 피처 엔지니어링](04-Cleaning-and-Feature-Engineering.ko.md)**
5. **[가이드 05: 벡터화 수학 연산, 누적 함수 및 기술 통계](05-Vectorized-Math-and-Statistics.ko.md)**
6. **[가이드 06: 그룹 집계(`groupBy`), 관계형 조인 및 재구조화](06-Grouping-Joining-and-Reshaping.ko.md)**
7. **[가이드 07: 시계열 데이터 분석 및 윈도우 함수 (`rolling`, `joinAsOf`, `resample`)](07-Time-Series-and-Window-Functions.ko.md)**
8. **[가이드 08: MLX 가속 엔진, 실행 정책 제어 및 실전 종합 워크플로우](08-MLX-Acceleration-and-Workflows.ko.md)**
- **[전체 API 레퍼런스 (`Documentation/API.ko.md`)](../API.ko.md)**
