# Bandal

[![Swiftversions](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNeuroAssociates%2FBandal%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/NeuroAssociates/Bandal)
[![Platforms](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNeuroAssociates%2FBandal%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/NeuroAssociates/Bandal)

[English](README.md) · [API 레퍼런스](Documentation/API.ko.md) · [API Reference](Documentation/API.md) · [독학 가이드](Documentation/Guides/01-Core-Data-Structures.ko.md) · [Self-Study Guides](Documentation/Guides/01-Core-Data-Structures.md)

**Bandal**은 Python의 [pandas](https://pandas.pydata.org/)에서 영감을 받아 [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/)에 따라 설계된 테이블형 데이터 분석용 Swift 패키지입니다.

`DataFrame`, `Series`, `DataRow`와 모든 요약/통계 결과 타입은 불변(immutable) 값 타입(`Sendable`, `Equatable`, `Hashable`, `Codable`)으로 설계되었으며 `print(...)`에 전달하면 pandas 스타일의 표와 요약 텍스트를 출력합니다. 또한 대규모 수치형 컬럼에 대해 [MLX Swift](https://github.com/ml-explore/mlx-swift)를 벡터화 수치 엔진으로 사용하여(기본적으로 MLX CPU 스트림의 `float64`, `ExecutionPolicy.gpu` 지정 시 GPU 스트림의 `float32`) 축소 연산, 그룹별 집계, 윈도우 함수, 원소별 사칙연산, 공분산/상관계수 행렬 계산을 가속하며, 소규모 입력이거나 Metal 라이브러리를 사용할 수 없는 환경에서는 네이티브 Swift로 자동 폴백(fallback)합니다.

데모 프로젝트는 다음 저장소를 참고하세요. [Bandal-demo](https://github.com/NeuroAssociates/Bandal-demo)

---

## 주요 특징

- **값 타입 `DataFrame`, `Series` 및 `DataRow`**: 불변 구조체로서 `Sendable`, `Hashable`, `Codable`을 준수하며 리터럴로 바로 표현 가능한 `DataValue` 셀(`.number(Double)`, `.string(String)`, `.boolean(Bool)`, `.date(Date)`, `.missing`)을 지원합니다.
- **출력 가능한 구조화된 결과 타입**: 모든 확인 및 통계 API(`DataFrame`, `Series`, `DataRow`, `DataFrameDescription`, `DataFrameStatistics`, `DataFrameSummary`, `DataFrameInfo`, `MissingCounts`, `UniqueCounts`, `ColumnTypes`)는 타입 안전 프로퍼티와 pandas 스타일의 `description` / `describe()` 메서드를 갖춘 Swift 구조체를 반환합니다.
- **스트리밍 CSV 및 JSON 입출력**: 64 KB 청크 단위의 RFC 4180 스트리밍 CSV 리더/라이터와 JSON 행 배열 리더/라이터(`init(jsonData:)`, `init(contentsOfJSON:)`, `jsonData(prettyPrinted:)`, `writeJSON(to:)`)를 제공합니다.
- **풍부한 쿼리 및 변환 기능**: 조합 가능한 `RowPredicate`와 문자열 표현식 기반 `query("temp >= 20 && active == true")`, 행 단위 계산 컬럼(`withColumn(name:computing:)`), UTC `DateComponent` 추출, 문자열 컬럼 연산(`lowercased`, `uppercased`, `trimmed`, `stringLength`, `replacingSubstring`, `containsSubstring`)을 지원합니다.
- **그룹화, 조인, 재구조화 및 윈도우 함수**: 다중 키 `groupBy`(`aggregate`, `transform`), 단일/다중 키 조인(`.left`, `.inner`, `.right`, `.full` 및 `suffixes`), `melt`, `pivotTable`, `exploded`, `transposed`, 행 개수 및 시간 간격 기반 `rolling`, `expanding` 윈도우를 지원합니다.
- **MLX-Swift 수치 엔진**: 기술 통계 모멘트, 축소 연산, 그룹 집계, 누적합 윈도우, 원소별 수치 연산(`combining`, `scaling`, `cumulative`, `abs`, `clipped`, `rounded`, `difference`, `percentChange`), `MLX.matmul` 기반 공분산/상관계수 행렬, `ExecutionPolicy` 전환(`.automatic`, `.cpu`, `.gpu`, `.gpuFloat32`), `MLXVector` 텐서 상호운용을 제공합니다.

---

## 요구 사항

- 플랫폼: macOS 14+, iOS 17+, tvOS 17+, visionOS 1+ (MLX 가속을 위해 Apple Silicon 권장)
- Swift 6.0 이상
- MLX 가속 커널 실행 시 Metal 툴체인이 포함된 Xcode

---

## 설치 방법

`Package.swift`의 의존성에 Bandal을 추가하세요:

```swift
dependencies: [
    .package(url: "https://github.com/NeuroAssociates/Bandal.git", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "Bandal", package: "Bandal"),
        ]
    ),
]
```

---

## 빠른 시작

```swift
import Bandal

let sales = try DataFrame(columns: [
    (name: "region", values: ["East", "East", "West", "West"]),
    (name: "revenue", values: [120.0, 180.0, 90.0, nil]),
    (name: "cost", values: [70.0, 100.0, 50.0, 40.0]),
])

print(sales)
//    region  revenue  cost
// 0  East    120.0    70.0
// 1  East    180.0    100.0
// 2  West    90.0     50.0
// 3  West    NaN      40.0
//
// [4 rows x 3 columns]

let analyzed = try sales
    .fillMissing(in: ["revenue"], with: 110.0)
    .combining("revenue", "cost", using: .subtract, as: "profit")
    .query("profit >= 50")

print(analyzed)

let byRegion = try analyzed
    .groupBy("region")
    .aggregate([
        Aggregation("revenue", using: .sum),
        Aggregation("profit", using: .mean),
    ])
print(byRegion)

// 구조화된 기술 통계 — 프로퍼티로 직접 조회하거나 바로 출력 가능
let desc = analyzed.describeStructured()
print(desc.numeric["profit"]?.mean ?? .nan)
print(desc)
```

---

## pandas와 Bandal 비교표

| pandas | Bandal |
| --- | --- |
| `pd.DataFrame({...})` | `try DataFrame(columns: [...])` |
| `pd.read_csv(path, ...)` / `df.to_csv(path)` | `try DataFrame(contentsOfCSV: url, options: ...)` / `try df.writeCSV(to: url)` |
| `pd.read_json(path)` / `df.to_json(path)` | `try DataFrame(contentsOfJSON: url)` / `try df.writeJSON(to: url)` |
| `df["col"]` / `df[["a", "b"]]` | `df["col"]` (`Series?`) 또는 `try df.series("col")` / `try df.select("a", "b")` |
| `df.iloc[i]` / `df.iloc[a:b]` / `df.at[i, "col"]` | `df[row: i]` (`DataRow?`) / `df[rows: a..<b]` / `df["col", i]` |
| `df.head(n)` / `df.tail(n)` | `try df.head(n)` / `try df.tail(n)` |
| `df.nlargest(n, "col")` / `df.nsmallest(n, "col")` | `try df.largest(n, by: "col")` / `try df.smallest(n, by: "col")` |
| `df.drop(columns=[...])` / `df.rename(columns={...})` | `try df.drop(columns: [...])` / `try df.renamedColumns([...])` |
| `df.astype(...)` | `try df.castColumn("col", to: .number)` |
| `df[df["x"] >= 10]` / `df.query("x >= 10")` | `try df.filterRows { ... }`, `try df.query(.column("x", greaterThanOrEqual: 10))`, 또는 `try df.query("x >= 10")` |
| `df.assign(y=lambda r: ...)` | `try df.withColumn(name: "y") { row in ... }` |
| `df["col"].str.lower()` / `.upper()` / `.strip()` / `.len()` / `.contains()` | `try df.lowercased("col")` / `uppercased` / `trimmed` / `stringLength` / `containsSubstring` |
| `df["date"].dt.year` / `.month` / `.day` | `try df.dateComponent(.year, from: "date")` |
| `df.isna()` / `df.isna().sum()` | `try df.missingMask()` / `df.missingSummary()` (`MissingCounts`) |
| `df.fillna(v)` / `df.ffill()` / `df.bfill()` / `df.dropna()` | `try df.fillMissing(with: v)` / `try df.forwardFilled()` / `try df.backwardFilled()` / `try df.dropMissing()` |
| `df.duplicated()` / `df.drop_duplicates()` | `try df.duplicatedRows()` / `try df.dropDuplicates()` |
| `df.sort_values(by=[...])` | `try df.sorted(by: [SortDescriptor(...)])` |
| `df.sample(n, random_state=42)` | `try df.sample(n, seed: 42)` |
| `pd.concat([df1, df2])` | `try DataFrame.concatenatingRows([df1, df2])` |
| `df.describe()` / `df.info()` / `df.dtypes` / `df.nunique()` | `df.describeStructured()` / `df.infoSummary()` / `df.dtypes` (`ColumnTypes`) / `df.uniqueSummary()` (`UniqueCounts`) |
| `s.value_counts()` / `s.quantile(q)` / `s.rank()` | `try df.valueCounts(of: "col")` / `try df.quantile(q)` / `try df.ranked("col")` |
| `df.cumsum()` / `df.shift()` / `df.diff()` / `df.pct_change()` | `try df.cumulative("col", using: .sum)` / `shifted` / `difference` / `percentChange` |
| `df.corr()` / `df.cov()` | `try df.correlation()` / `try df.covariance()` |
| `df.groupby(...).agg(...)` / `.transform(...)` | `try df.groupBy(...).aggregate(...)` / `try df.groupBy(...).transform(...)` |
| `df.merge(other, left_on=..., right_on=..., how=...)` | `try df.join(other, leftOn: [...], rightOn: [...], kind: ...)` |
| `df.melt(...)` / `df.pivot_table(...)` | `try df.melt(...)` / `try df.pivotTable(...)` |
| `df.explode("col")` / `df.T` | `try df.exploded("col")` / `try df.transposed()` |
| `df.rolling(w).mean()` / `df.rolling("60s", on=...).mean()` / `df.expanding().sum()` | `try df.rolling("col", window: w, minPeriods: 1, using: .mean)` / `try df.rolling("col", on: "ts", interval: 60, using: .mean)` / `try df.expanding("col", using: .sum)` |
| `df[(df["x"] >= 10) & (df["y"] == "a")]` / `s.where(cond, other)` | `df[(try df.series("x") >= 10.0) && try df.series("y").equals("a")]` / `try s.where(mask, other: other)` |
| `np.log1p(df["x"])` / `df["x"] ** 2` / `pd.cut` / `pd.qcut` / `s.interpolate()` / `s.ewm(alpha=...).mean()` | `try df.applying("x", function: .log1p)` / `try df.power("x", exponent: 2)` / `binned` / `quantileBinned` / `interpolated` / `exponentiallyWeighted` |
| `df.pivot(...)` / `pd.crosstab(...)` / `pd.get_dummies(...)` / `pd.merge_asof(...)` / `df.resample(...)` | `try df.pivoted(...)` / `try df.crossTabulated(...)` / `try df.oneHotEncoded(...)` / `try df.joinAsOf(...)` / `try df.resample(...)` |
| `df.to_markdown()` | `df.toMarkdown()` |

---

## MLX 백엔드 및 실행 정책

Bandal은 [MLX Swift](https://github.com/ml-explore/mlx-swift)를 벡터화 수치 엔진으로 사용합니다. 기본 정책(`ExecutionPolicy.automatic`)에서는 내부 크기 임계값 이상의 수치형 컬럼을 MLX CPU 스트림의 `float64`로 연산하여 `Double` 정밀도를 그대로 유지하고, 소규모 입력은 텐서 할당 오버헤드 없이 네이티브 Swift로 처리합니다.

런타임에 실행 정책을 조회하거나 변경할 수 있습니다:

```swift
print(BandalRuntime.isMLXAvailable)
print(BandalRuntime.executionBackend)

// 특정 작업을 MLX GPU 스트림(무손실 변환 가능 시 float32) 또는 CPU 스트림으로 실행
let corr = try BandalRuntime.withExecutionPolicy(.gpu) {
    try frame.correlation()
}

// 컬럼을 MLXVector로 직접 변환하거나 다시 DataFrame으로 복원
let vector = try frame.toMLXVector("revenue")
let restored = try DataFrame(mlxColumns: [(name: "revenue", vector: vector)])
```

환경 변수에 `BANDAL_DISABLE_MLX=1`을 지정하면 모든 곳에서 네이티브 Swift 구현을 강제할 수 있습니다.

---

## 테스트 및 벤치마크

- **표준 테스트 스위트 (네이티브 경로)**:
  ```sh
  xcrun swift test
  ```
- **MLX Metal 라이브러리를 포함한 전체 테스트 스위트**:
  ```sh
  Scripts/test-with-mlx.sh
  ```
- **벤치마크 실행**:
  ```sh
  xcrun swift run -c release BandalBenchmarks 1000 10000 100000
  ```
- **pandas와 Bandal 성능 비교**: `uv` 환경(pandas, numpy)을 구성하고 릴리스 벤치마크를 빌드한 뒤, pandas·네이티브 Swift·MLX CPU·MLX GPU의 순수 계산 시간(데이터 준비, 컴파일, 워밍업, MLX 커널 컴파일 제외)을 표로 출력합니다. 체크섬으로 모든 엔진이 동일한 결과를 계산했는지 검증합니다.
  ```sh
  Scripts/benchmark.sh [--rows 100000,1000000,5000000] [--repeat 5] [--warmup 2]
  ```
  MLX는 집계·적률 계산(`stats_summary`), 롤링 윈도우, 상관 행렬에서 효과가 있으며, `groupby_sum`, `left_join` 같은 행 중심 연산은 MLX를 사용하지 않습니다.

---

## 문서

### API 레퍼런스 및 프로젝트 문서
- [한국어 API 레퍼런스 (`Documentation/API.ko.md`)](Documentation/API.ko.md)
- [영어 API 레퍼런스 (`Documentation/API.md`)](Documentation/API.md)
- [기여 가이드 (`CONTRIBUTING.md`)](CONTRIBUTING.md)
- [변경 로그 (`CHANGELOG.md`)](CHANGELOG.md)

### `pandas` 사용자를 위한 주제별 독학 가이드 (`Documentation/Guides/`)
1. [가이드 01: 핵심 데이터 구조 및 Swift 값 타입 모델](Documentation/Guides/01-Core-Data-Structures.ko.md) ([English](Documentation/Guides/01-Core-Data-Structures.md))
2. [가이드 02: 데이터 입출력(CSV & JSON) 및 스키마 탐색](Documentation/Guides/02-Data-IO-and-Inspection.ko.md) ([English](Documentation/Guides/02-Data-IO-and-Inspection.md))
3. [가이드 03: 인덱싱, 슬라이싱, 불리언 마스크 및 쿼리](Documentation/Guides/03-Indexing-Filtering-and-Querying.ko.md) ([English](Documentation/Guides/03-Indexing-Filtering-and-Querying.md))
4. [가이드 04: 데이터 정제, 결측치 처리 및 피처 엔지니어링](Documentation/Guides/04-Cleaning-and-Feature-Engineering.ko.md) ([English](Documentation/Guides/04-Cleaning-and-Feature-Engineering.md))
5. [가이드 05: 벡터화 수학 연산, 누적 함수 및 기술 통계](Documentation/Guides/05-Vectorized-Math-and-Statistics.ko.md) ([English](Documentation/Guides/05-Vectorized-Math-and-Statistics.md))
6. [가이드 06: 그룹 집계(`groupBy`), 관계형 조인 및 재구조화](Documentation/Guides/06-Grouping-Joining-and-Reshaping.ko.md) ([English](Documentation/Guides/06-Grouping-Joining-and-Reshaping.md))
7. [가이드 07: 시계열 데이터 분석 및 윈도우 함수](Documentation/Guides/07-Time-Series-and-Window-Functions.ko.md) ([English](Documentation/Guides/07-Time-Series-and-Window-Functions.md))
8. [가이드 08: MLX 가속 엔진, 실행 정책 제어 및 실전 종합 워크플로우](Documentation/Guides/08-MLX-Acceleration-and-Workflows.ko.md) ([English](Documentation/Guides/08-MLX-Acceleration-and-Workflows.md))

---

## 라이선스
Created by [Ned Park](https://github.com/nedpark) of Neuro Associates. All rights reserved.
[MIT License](LICENSE)에 따라 배포됩니다.
