# 다음 조건을 만족하도록 프로젝트를 구성한다

Bandal 라이브러리는 Swift Package로 배포된든 프로젝트이다.

Bandal 라이브러리는 Pandas의 모든 기능을 최대한 Swift 언어답게 만든다.
https://github.com/pandas-dev/pandas

모든 API 구축에는 애플의 다음 가이드라인을 따른다. https://www.swift.org/documentation/api-design-guidelines/

모든 기능은 MLX-Swift를 최대한으로 사용한다.  Pandas가 NumPy를 통해 얻는 성능 이득 및 수치해석 기반활용 관계를 이 프로젝트에서도 NumPy대신 MLX-Swift를 통해 기대한다.
https://github.com/ml-explore/mlx-swift
https://github.com/numpy/numpy

모든 API 응답은 Swift의 구조체 인스턴스로 응답되며 description과 같은 인스턴스의 메소드를 통해 print로 출력 가능한 텍스트 정보들을 Pandas 처럼 응답받을 수 있다.

API.md에 영어로 모든 기능을 나열하고 예제 코드를 제시하는 개발자 지원 문서를 작성한다.

API.ko.md에 API.md의 내용을 그대로 한국어로 번역한 내용을 작성한다.

README.md에 영어로 GitHub에 오픈소스로 공개하기에 적절한 안내글을 작성한다.

README.ko.md에 README.md의 내용을 그대로 한국어로 번역한 내용을 작성한다.
