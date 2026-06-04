#!/usr/bin/env bash

# Generate an LLVM source-based coverage HTML report for the C++ tests.

set -euo pipefail

LLVM_BIN="/opt/homebrew/opt/llvm/bin"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
CPP_DIR="$ROOT/x7_renderer"
BUILD_DIR="$CPP_DIR/build-coverage"
REPORT_DIR="${1:-$BUILD_DIR/report}"
PROFDATA="$BUILD_DIR/x7_combined.profdata"
TEST_BIN="$BUILD_DIR/tests/x7_tests"
ERROR_TEST_BIN="$BUILD_DIR/tests/x7_error_tests"

cd "$CPP_DIR"

cmake --preset coverage
cmake --build --preset coverage
LLVM_PROFILE_FILE="$BUILD_DIR/x7_tests.profraw" "$TEST_BIN"
LLVM_PROFILE_FILE="$BUILD_DIR/x7_error_tests.profraw" "$ERROR_TEST_BIN"
"$LLVM_BIN/llvm-profdata" merge -sparse \
    "$BUILD_DIR/x7_tests.profraw" \
    "$BUILD_DIR/x7_error_tests.profraw" \
    -o "$PROFDATA"

"$LLVM_BIN/llvm-cov" show \
    -object="$TEST_BIN" -object="$ERROR_TEST_BIN" \
    -instr-profile="$PROFDATA" \
    -format=html \
    -output-dir="$REPORT_DIR" \
    -show-line-counts-or-regions \
    -show-branches=count \
    -ignore-filename-regex='tests/.*' \
    --sources="$CPP_DIR/src"

echo "Summary:"
"$LLVM_BIN/llvm-cov" report \
    -object="$TEST_BIN" -object="$ERROR_TEST_BIN" \
    -instr-profile="$PROFDATA" \
    -ignore-filename-regex='tests/.*' \
    --sources="$CPP_DIR/src"
