# 다음 조건을 만족하도록 프로젝트를 구성한다

Bandal 라이브러리는 Swift Package로 배포되는 프로젝트이다.

Bandal 라이브러리는 Pandas의 모든 기능을 최대한 Swift 언어답게 만든다.
https://github.com/pandas-dev/pandas

모든 API 구축에는 애플의 다음 가이드라인을 따른다. https://www.swift.org/documentation/api-design-guidelines/

모든 기능은 MLX-Swift를 최대한으로 사용한다. Pandas가 NumPy를 통해 얻는 성능 이득 및 수치해석 기반활용 관계를 이 프로젝트에서도 NumPy대신 MLX-Swift를 통해 기대한다.
https://github.com/ml-explore/mlx-swift
https://github.com/numpy/numpy

모든 API 응답은 Swift의 구조체 인스턴스로 응답되며 description과 같은 인스턴스의 메소드를 통해 print로 출력 가능한 텍스트 정보들을 Pandas 처럼 응답받을 수 있다.

API.md에 영어로 모든 기능을 나열하고 예제 코드를 제시하는 개발자 지원 문서를 작성한다.

API.ko.md에 API.md의 내용을 그대로 한국어로 번역한 내용을 작성한다.

README.md에 영어로 GitHub에 오픈소스로 공개하기에 적절한 안내글을 작성한다.

README.ko.md에 README.md의 내용을 그대로 한국어로 번역한 내용을 작성한다.

---

# 고성능 컬럼 기반 엔진 (High-Performance Columnar Engine)

- [x] **1. Typed Columnar Buffers (Apache Arrow / Polars 모델)**
  - 개별 원소 `[DataValue]` 박싱을 우회하고 타입별 연속 메모리 버퍼(`ContiguousArray`) 기반 컬럼 아키텍처 채택 (`AnyColumn`, `Float64Column`, `Int64Column`, `StringColumn`, `CategoricalColumn`, `BooleanColumn`, `DateColumn`, `AnyValueColumn`).
  - 초고속 64비트 워드 비트맵(`BitSet`)을 통한 zero-overhead null 및 결측값 추적 및 비트 슬라이싱 구현.
  - 기존 `[DataValue]`, `Series`, `DataRow`, `subscript` 등 100% 하위 호환성 유지 (63개 전체 단위 테스트 통과).

- [x] **2. 고속 필터링 및 비트셋 슬라이싱 (Fast-Path Filtering & Masking)**
  - `query` 및 `filterRows` 시 불필요한 행 단위 딕셔너리 생성 오버헤드 축소 및 `BitSet` 마스크 기반 컬럼 슬라이싱 지원.
  - 40,000행 필터링 벤치마크: **6.20ms** 달성 (목표치 < 10ms 및 Pandas 9ms 대비 우수).

- [x] **3. 스트림형 Zero-Copy CSV 파서 (Streamlined CSV Reader)**
  - 파일 크기 기반 버퍼 용량 사전 할당(capacity pre-allocation)으로 동적 재할당 오버헤드 제거.
  - 비인용(quote-free) 고속 라인 분해 및 `Substring` 기반 토큰화로 숫자 컬럼 파싱 시 중간 문자열 힙 할당 완전 제거.
  - UTF-8 BOM(Byte Order Mark) 자동 감지 및 처리. 40,000행 파싱 속도 대폭 개선 (130ms → **84.39ms**).

- [x] **4. 범주형(Categorical) 딕셔너리 인코딩 & GroupBy / 정렬 최적화**
  - 단일 컬럼 정렬 시 프리미티브 `Double`, 범주형 랭크, 타임스탬프 원시 버퍼 기반 직접 정렬 fast path 구축.
  - `groupBy` 시 단일 키에 대한 dense `Int32` 그룹 식별자 부여 및 단일 패스 스캐터(scatter) 집계 연산 적용. 40,000행 집계: **6.79ms** (목표치 < 10ms 달성).
  - 조인(`join`) 시 행별 딕셔너리 재구성을 정수 인덱스 수집 및 `slice(atOptional:)` zero-copy 슬라이싱으로 대체: 조인 시간 19.59ms → **6.95ms** (약 3배 향상).

- [x] **5. 성능 벤치마크 검증 (40,000행 릴리스 모드 실측)**
  - CSV Load: **84.39ms** (개선 전 130ms 대비 35% 단축)
  - Cleansing / Filter: **6.20ms** (목표치 < 10ms 초과 달성, Pandas 9ms 대비 31% 빠름)
  - Vector Math / Summary: **1.58ms** (목표치 < 5ms 초과 달성)
  - GroupBy Aggregation: **6.79ms** (목표치 < 10ms 초과 달성, 기존 12.01ms 대비 43% 단축)
  - Left Join: **6.95ms** (기존 19.59ms 대비 65% 단축, 약 3배 향상)
  - Describe: **9.42ms**
  - 전체 63개 유닛 테스트 100% 통과
