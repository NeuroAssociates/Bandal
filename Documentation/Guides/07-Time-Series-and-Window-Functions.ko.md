# 가이드 07: 시계열 데이터 분석 및 윈도우 함수

> **언어**: [English](07-Time-Series-and-Window-Functions.md) | **한국어**  
> **네비게이션**: [← 가이드 06: 그룹 집계, 조인 및 재구조화](06-Grouping-Joining-and-Reshaping.ko.md) | [API 레퍼런스](../API.ko.md) | **다음**: [가이드 08: MLX 가속 엔진 및 실전 종합 워크플로우 →](08-MLX-Acceleration-and-Workflows.ko.md)

---

## 개요

금융 틱(Tick) 데이터, IoT 센서 텔레메트리, 서비스 이벤트 로그는 모두 시간 순서에 따라 기록됩니다. 이러한 시계열 데이터를 분석하려면 단순 정적 집계를 넘어선 전용 윈도우 함수와 시간축 정렬(Temporal Alignment) API가 필요합니다.

이 가이드에서 다룰 핵심 기능은 다음과 같습니다.
- **행 기반 이동 윈도우 (`rolling(_:window:minPeriods:using:)`)**: 직전 $W$개 행에 대한 이동평균, 이동합계, 이동표준편차(변동성), 최댓값/최솟값 계산
- **시간 간격 기반 이동 윈도우 (`rolling(_:on:interval:using:)`)**: 불규칙한 타임스탬프(`.date`) 상에서 물리적 시간 간격(`TimeInterval` 초)을 기준으로 한 이동 윈도우 집계
- **확장 윈도우 (`expanding(_:using:)`)**: 첫 번째 행(`0`)부터 현재 행(`i`)까지의 누적 프리픽스 집계
- **지수 가중 이동평균 (`exponentiallyWeighted` / `ewm`)**: 최근 관측값에 더 높은 가중치를 부여하는 지수 평활(`alpha` 또는 `span` 지정)
- **시계열 근접 조인 (`joinAsOf` / `pd.merge_asof`)**: 미세하게 어긋난 두 시계열 테이블을 직전(`.backward`), 직후(`.forward`), 또는 최근접(`.nearest`) 시점 기준으로 매칭(`by` 파티션 및 `tolerance` 허용 오차 지원)
- **고정 시간 간격 리샘플링 (`resample`)**: 불규칙한 시계열 데이터를 고정 간격(예: 1분, 1시간) 버킷으로 다운샘플링(예: OHLCV 캔들스틱 생성)

---

## 1. 한눈에 보는 비교: `pandas` vs. `Bandal`

| 작업 | Python `pandas` | Swift `Bandal` |
| --- | --- | --- |
| 행 개수 기반 이동 윈도우 | `df["price"].rolling(3, min_periods=1).mean()` | `try df.rolling("price", window: 3, minPeriods: 1, using: .mean)` |
| 시간 간격 기반 이동 윈도우 | `df.rolling("60s", on="ts")["price"].sum()` | `try df.rolling("price", on: "ts", interval: 60, using: .sum)` |
| 확장 윈도우 (`expanding`) | `df["price"].expanding().max()` | `try df.expanding("price", using: .max)` |
| `alpha` 기준 지수 가중 이동평균 | `df["price"].ewm(alpha=0.5, adjust=False).mean()` | `try df.exponentiallyWeighted("price", alpha: 0.5, as: "ewma")` |
| `span` 기준 지수 가중 이동평균 | `df["price"].ewm(span=3, adjust=False).mean()` | `try df.exponentiallyWeighted("price", span: 3.0, as: "ewma_span3")` |
| 근접 시점 조인 (`merge_asof`) | `pd.merge_asof(trades, quotes, on="ts", by="sym", direction="backward", tolerance=pd.Timedelta("5s"))` | `try trades.joinAsOf(quotes, on: "ts", by: "sym", direction: .backward, tolerance: 5.0)` |
| 시계열 리샘플링 (`resample`) | `df.resample("3600s", on="ts").agg({"price": ["first", "last"], "vol": "sum"})` | `try df.resample(on: "ts", interval: 3600, aggregations: [...])` |

---

## 2. 행 기반 이동 윈도우 (`rolling(_:window:minPeriods:using:)`)

`df.rolling(_:window:minPeriods:using:)`은 현재 행을 포함하여 최대 `window`개의 직전 행들에 대해 집계를 수행하고 **`<column>_<operation>_rolling_<window>`** 이름의 새 컬럼을 추가합니다.

- **`minPeriods`**: 윈도우 내에 최소 몇 개의 유효(비결측) 관측값이 있어야 결과를 산출할지 지정합니다(기본값은 `window`와 동일하므로 처음 `window - 1`개 행은 `.missing`이 됨). 첫 행부터 부분 윈도우 결과를 얻으려면 `minPeriods: 1`을 전달합니다.
- **하드웨어 가속**:
  - `.sum`, `.mean`, `.count`는 단일 패스 $O(N)$ 슬라이딩 누산기로 동작합니다.
  - 대규모 데이터의 `.min`, `.max`, `.variance`, `.standardDeviation`은 `MLX.asStrided`를 통한 2D 스트라이드 뷰로 병렬 감축됩니다.

```swift
import Bandal
import Foundation

let stock = try DataFrame(columns: [
    (name: "day", values: [1, 2, 3, 4, 5]),
    (name: "close", values: [100.0, 106.0, 103.0, 109.0, 112.0]),
])
.rolling("close", window: 3, using: .mean)               // 처음 2개 행은 .missing
.rolling("close", window: 3, minPeriods: 1, using: .max) // 첫 행부터 부분 윈도우 허용

print(stock)
```

**콘솔 출력**:
```text
   day  close  close_mean_rolling_3  close_max_rolling_3
0    1    100                   NaN                  100
1    2    106                   NaN                  106
2    3    103                   103                  106
3    4    109                   106                  109
4    5    112                   108                  112

[5 rows x 4 columns]
```

---

## 3. 시간 간격 기반 이동 윈도우 (`rolling(_:on:interval:using:)`)

실제 로그나 센서 데이터는 일정한 시간 간격으로 도착하지 않는 경우가 많습니다. `.date` 타입 컬럼이 있을 때 `df.rolling(_:on:interval:using:)`을 호출하면, 각 행 $i$(시각 $t_i$)에 대해 반열린 시간 구간 **$(t_i - \text{interval}, t_i]$** 안에 들어오는 모든 이전 행 $j \le i$를 대상으로 집계를 수행합니다.

출력 컬럼 이름은 **`<valueColumn>_<operation>_rolling_<interval>s`**로 생성됩니다.

```swift
let baseTime = Date(timeIntervalSince1970: 1_700_000_000)

// t = 0초, 10초, 25초, 70초에 발생한 이벤트
let requests = try DataFrame(columns: [
    (name: "ts", values: [
        .date(baseTime),
        .date(baseTime.addingTimeInterval(10)),
        .date(baseTime.addingTimeInterval(25)),
        .date(baseTime.addingTimeInterval(70)),
    ]),
    (name: "latency_ms", values: [12.0, 18.0, 30.0, 15.0]),
])
// 직전 30초 이동 윈도우 합계:
// - t=0s:  ( -30s,  0s] -> [12]         -> sum = 12
// - t=10s: ( -20s, 10s] -> [12, 18]     -> sum = 30
// - t=25s: (  -5s, 25s] -> [12, 18, 30] -> sum = 60
// - t=70s: (  40s, 70s] -> [15]         -> sum = 15
.rolling("latency_ms", on: "ts", interval: 30, using: .sum)

print(requests)
```

**콘솔 출력**:
```text
                     ts  latency_ms  latency_ms_sum_rolling_30s
0  2023-11-14T22:13:20Z          12                          12
1  2023-11-14T22:13:30Z          18                          30
2  2023-11-14T22:13:45Z          30                          60
3  2023-11-14T22:14:30Z          15                          15

[4 rows x 3 columns]
```

---

## 4. 확장 윈도우(`expanding`) 및 지수 가중 이동평균(`EWMA`)

### 4.1 확장 윈도우 (`expanding(_:using:)`)

`expanding(_:using:)`은 0번 행부터 현재 행 `i`까지의 누적 구간 전체에 대해 집계 함수를 적용하며, `<column>_<operation>_expanding` 컬럼을 추가합니다.

```swift
let expandingDemo = try DataFrame(columns: [
    (name: "step", values: [1, 2, 3, 4]),
    (name: "pnl", values: [10.0, -4.0, 18.0, 8.0]),
])
.expanding("pnl", using: .mean)
.expanding("pnl", using: .max)

print(expandingDemo)
```

**콘솔 출력**:
```text
   step  pnl  pnl_mean_expanding  pnl_max_expanding
0     1   10                  10                 10
1     2   -4                   3                 10
2     3   18                   8                 18
3     4    8                   8                 18

[4 rows x 4 columns]
```

### 4.2 지수 가중 이동평균 (`exponentiallyWeighted` / `ewm`)

`exponentiallyWeighted`는 `pandas`의 `ewm(..., adjust=False).mean()`과 동일한 점화식으로 지수 가중 이동평균을 계산합니다.

$$y_0 = x_0, \qquad y_t = \alpha x_t + (1 - \alpha) y_{t-1}$$

중간에 결측치(`.missing`)가 나타나면 이전 평활 상태를 유지하다가 다음 유효 관측값에서 갱신을 이어갑니다.
- **`alpha`**: 평활 계수 $\alpha \in (0, 1]$ 직접 지정
- **`span`**: 기간 $\text{span} \ge 1$ 지정 ($\alpha = \frac{2}{\text{span} + 1}$로 자동 변환)

```swift
let signalDF = try DataFrame(columns: [
    (name: "t", values: [1, 2, 3, 4]),
    (name: "price", values: [100.0, 110.0, 105.0, 115.0]),
])
.exponentiallyWeighted("price", alpha: 0.5, as: "ewma_alpha05")
.exponentiallyWeighted("price", span: 3.0, as: "ewma_span3") // span=3 -> alpha = 2/(3+1) = 0.5

print(signalDF)
```

**콘솔 출력**:
```text
   t  price  ewma_alpha05  ewma_span3
0  1    100           100         100
1  2    110           105         105
2  3    105         105.0       105.0
3  4    115           110         110

[4 rows x 4 columns]
```

---

## 5. 시계열 근접 조인 (`joinAsOf` / `pd.merge_asof`)

체결 내역(Trades)과 호가창(Quotes), 또는 사용자 클릭 로그와 시스템 설정 변경 이력처럼 서로 다른 주기로 기록된 두 시계열 테이블을 합칠 때는 타임스탬프가 밀리초 단위로 어긋나기 때문에 일반 동등 조인(`join(on:)`)으로는 매칭되지 않습니다.

`df.joinAsOf(_:on:by:direction:tolerance:suffixes:)`는 이진 탐색을 통해 $O(N_{\text{left}} + N_{\text{right}} \log N_{\text{right}})$ 시간에 근접 시점 조인을 수행합니다.
- **`on: String`**: 양쪽 테이블에 존재하는 `.date` 또는 `.number` 기준 시점 컬럼
- **`by: String? = nil`**: 시점 비교 전에 **정확히 일치**해야 하는 보조 그룹 키(예: 종목 코드 `"ticker"` 또는 센서 ID)
- **`direction: AsOfDirection = .backward`**:
  - `.backward`: $t_{\text{right}} \le t_{\text{left}}$를 만족하는 가장 최근의 오른쪽 행 매칭 (미래 참조 편향 방지!)
  - `.forward`: $t_{\text{right}} \ge t_{\text{left}}$를 만족하는 가장 빠른 오른쪽 행 매칭
  - `.nearest`: 절대 시간 차이 $|t_{\text{right}} - t_{\text{left}}|$가 가장 작은 오른쪽 행 매칭
- **`tolerance: TimeInterval? = nil`**: 허용 가능한 최대 시간(또는 수치) 격차. 이 범위를 벗어난 매칭은 `.missing`으로 처리됩니다.

```swift
let t0 = Date(timeIntervalSince1970: 1_700_000_000)

// t = +2초, +8초, +20초에 발생한 주식 체결(Trades)
let trades = try DataFrame(columns: [
    (name: "ts", values: [
        .date(t0.addingTimeInterval(2)),
        .date(t0.addingTimeInterval(8)),
        .date(t0.addingTimeInterval(20)),
    ]),
    (name: "ticker", values: ["AAPL", "AAPL", "AAPL"]),
    (name: "trade_qty", values: [100.0, 50.0, 200.0]),
])

// t = 0초, +6초에 기록된 매수 호가(Quotes)
let quotes = try DataFrame(columns: [
    (name: "ts", values: [
        .date(t0),
        .date(t0.addingTimeInterval(6)),
    ]),
    (name: "ticker", values: ["AAPL", "AAPL"]),
    (name: "bid", values: [185.0, 185.5]),
])

// 각 체결 시점 기준 최대 5초 이내의 직전(.backward) 호가 매칭:
// - +2초 체결 -> 0초 호가 매칭   (시간차 2초 <= 5초) -> bid = 185.0
// - +8초 체결 -> +6초 호가 매칭  (시간차 2초 <= 5초) -> bid = 185.5
// - +20초 체결 -> +6초 호가와 14초 차이 (> 5초 초과) -> bid = NaN (.missing)
let matchedTrades = try trades.joinAsOf(
    quotes,
    on: "ts",
    by: "ticker",
    direction: .backward,
    tolerance: 5.0
)

print(matchedTrades)
```

**콘솔 출력**:
```text
                     ts  ticker  trade_qty    bid
0  2023-11-14T22:13:22Z    AAPL        100    185
1  2023-11-14T22:13:28Z    AAPL         50  185.5
2  2023-11-14T22:13:40Z    AAPL        200    NaN

[3 rows x 4 columns]
```

---

## 6. 고정 시간 간격 리샘플링 (`resample`)

`df.resample(on:interval:aggregations:)`는 불규칙한 `.date` 컬럼을 UTC 에포크 기준 고정 폭 `interval`(초) 버킷($\lfloor t / \text{interval} \rfloor \times \text{interval}$)으로 그룹화하고, 데이터가 존재하는 각 시간 버킷에 대해 `[Aggregation]` 리스트를 시간순으로 집계합니다.

### 예제: 틱(Tick) 데이터로부터 1분봉 OHLCV 캔들스틱 생성하기

금융 틱 데이터에서 1분(`interval: 60`) 단위의 시가(`open`), 고가(`high`), 저가(`low`), 종가(`close`), 거래량(`volume`) 테이블을 단 한 번의 `resample` 호출로 생성할 수 있습니다.

```swift
let minute0 = Date(timeIntervalSince1970: 1_700_000_040) // 22:14:00 UTC

let ticks = try DataFrame(columns: [
    (name: "ts", values: [
        .date(minute0.addingTimeInterval(5)),   // 22:14:05 (1번 분봉)
        .date(minute0.addingTimeInterval(25)),  // 22:14:25 (1번 분봉)
        .date(minute0.addingTimeInterval(50)),  // 22:14:50 (1번 분봉)
        .date(minute0.addingTimeInterval(70)),  // 22:15:10 (2번 분봉)
        .date(minute0.addingTimeInterval(110)), // 22:15:50 (2번 분봉)
    ]),
    (name: "price", values: [100.0, 104.0, 99.0, 99.5, 102.0]),
    (name: "volume", values: [10.0, 25.0, 15.0, 30.0, 20.0]),
])

let ohlcv = try ticks.resample(
    on: "ts",
    interval: 60, // 60초 (1분봉)
    aggregations: [
        Aggregation("price", using: .first, as: "open"),
        Aggregation("price", using: .max,   as: "high"),
        Aggregation("price", using: .min,   as: "low"),
        Aggregation("price", using: .last,  as: "close"),
        Aggregation("volume", using: .sum,  as: "volume"),
    ]
)

print(ohlcv)
```

**콘솔 출력**:
```text
                     ts  open  high   low  close  volume
0  2023-11-14T22:14:00Z   100   104    99     99      50
1  2023-11-14T22:15:00Z  99.5   102  99.5    102      50

[2 rows x 6 columns]
```

---

> **다음 단계**: 마지막 가이드인 **[가이드 08: MLX 가속 엔진 및 실전 종합 워크플로우 →](08-MLX-Acceleration-and-Workflows.ko.md)**로 이동하여 Bandal의 하이브리드 실행 엔진(`Accelerate` + `mlx-swift`), `ExecutionPolicy` 제어, `MLXArray` 텐서 직접 연동, 그리고 실전 데이터 분석 전체 파이프라인 예제를 확인하세요.
