# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-05-04

This release refreshes the Python package for renewed PyPI publishing and
brings the public API closer to the R `densratio` package.

### Added

- Added R-style public wrappers: `uLSIF()`, `RuLSIF()`, and `KLIEP()`.
- Added the KLIEP density ratio estimator.
- Added R compatibility regression tests for public API behavior and estimator
  outputs.
- Added a PEP 517 build configuration with `pyproject.toml`.
- Added GitHub Actions CI for Python 3.10 through 3.14.
- Added package extras for test, docs, and development environments.
- Added a Trusted Publishing release workflow and release checklist.
- Added a README section listing research papers that used the Python package.

### Changed

- Updated supported Python versions to Python 3.10 and newer.
- Refined README examples and API documentation for the 0.4.0 release.
- Replaced Travis CI with GitHub Actions.
- Simplified development dependency installation through `requirements.txt`
  and package extras.
- Replaced the placeholder license file with the full MIT License text.

### Fixed

- Fixed RuLSIF center sampling and two-dimensional sample handling.
- Fixed pandas `DataFrame` conversion in helper utilities.
- Fixed `DensityRatio` pickling support.

## Older Releases

Older release history is available from the Git commit history and published
PyPI releases.
