# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2022-mm-dd
### Added
- Adds a `fake_cuda` parameter to `fake_mode()` that allows constructing fake
  CUDA tensors even if CUDA is not available.

## [0.2.0] - 2022-06-23
### Added
 - Moves to PyTorch v1.12
 - Adds support for Python 3.10

### Fixed
 - Addresses a minor bug in Fake tensor caused by the API changes in PyTorch

## [0.1.0] - 2022-04-14
### Added
 - Initial release with Fake Tensor and Deferred Module Initialization
