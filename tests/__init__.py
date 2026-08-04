"""Tests package initializer to make test modules importable under Bazel.

This file intentionally left minimal; its presence makes `tests` a package
so absolute imports like `tests.model_test_support` resolve in the test
runfiles environment.
"""

__all__ = []
