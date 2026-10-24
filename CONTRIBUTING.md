# Contributing to Bandal

Thanks for your interest in Bandal! Bug reports, focused feature proposals, and pull requests are welcome.

## Getting started

1. Install Swift 6.4+ and a full Xcode (the Metal compiler is required by MLX Swift).
2. Clone the repository and run `xcrun swift build` from the `Bandal` package directory.
3. Run the tests:
   - `xcrun swift test` — native path; MLX tests are skipped.
   - `Scripts/test-with-mlx.sh` — full suite with MLX active.

## Guidelines

- Follow the [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/).
- `DataFrame` and result types are value types. Operations return new values and report problems by throwing `DataFrameError`.
- Every public result type should conform to `CustomStringConvertible`.
- Numeric kernels live in `Bandal+MLXBackend.swift` (`NumericEngine`). A new MLX path needs a native reference implementation and a test that compares both.
- Add tests (Swift Testing) for behavior changes.
- Update `Documentation/API.md` and keep `Documentation/API.ko.md` a faithful translation. Update `README.md` and `README.ko.md` when user-facing behavior changes.

## Pull requests

Keep changes focused, describe the motivation, and make sure `swift build` and both test commands pass.
