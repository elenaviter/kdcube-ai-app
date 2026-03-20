---
title: "Bundle Testing Methodology"
summary: "How react.doc discovers, runs, and validates bundle tests."
tags: ["sdk", "testing", "methodology", "react.doc", "pytest", "automation"]
keywords: ["test discovery", "bundle validation", "test automation", "react.doc workflow"]
see_also:
  - ks:docs/sdk/tests/bundle-testing-system-README.md
  - ks:docs/sdk/tests/bundle-tests-README.md
  - ks:docs/sdk/bundle/bundle-dev-README.md
---

# Bundle Testing Methodology

How react.doc discovers, runs, and validates tests for any bundle.

## Overview

React.doc is a documentation and validation agent that can test any bundle:

1. **Discovers** available bundles in the system
2. **Asks user** which bundle to test
3. **Finds tests** through knowledge space documentation
4. **Runs tests** with bundle-specific parameters
5. **Reports results** to user

## Core Concept: Parameterized Testing

Tests accept a **bundle ID parameter** via `--bundle-id`:

**For react.doc:**
```
pytest sdk/bundle_tests/test_initialization.py --bundle-id=react.doc -v
```

**For openrouter-data:**
```
pytest sdk/bundle_tests/test_initialization.py --bundle-id=openrouter-data -v
```

**For react:**
```
pytest sdk/bundle_tests/test_initialization.py --bundle-id=react -v
```

**Key insight:** Same test file, different bundle IDs. One test works for all bundles.

## React.doc Execution Flow

### Step 1: User Request
```
User: "Test the react.doc bundle"
```

### Step 2: React.doc Lists Available Bundles
Shows all available bundles (react.doc, openrouter-data, react, custom-bundle, etc.)

### Step 3: React.doc Searches Knowledge
- Searches: "bundle testing methodology"
- Finds: this document
- Learns: how to run tests with `--bundle-id` parameter

### Step 4: React.doc Runs Tests
Executes:
```
pytest sdk/bundle_tests/test_*.py --bundle-id=<chosen_bundle> -v
```

For example, if user chose "react.doc":
```
pytest sdk/bundle_tests/test_initialization.py --bundle-id=react.doc -v
pytest sdk/bundle_tests/test_configuration.py --bundle-id=react.doc -v
pytest sdk/bundle_tests/test_graph_construction.py --bundle-id=react.doc -v
pytest sdk/bundle_tests/test_error_handling.py --bundle-id=react.doc -v
```

### Step 5: React.doc Collects Results
- Waits for all tests to complete
- Counts PASSED and FAILED
- Records which test categories passed/failed

### Step 6: React.doc Reports to User
```
Bundle: react.doc
Status: ✓ ALL TESTS PASSED (25/25)

✓ Initialization (5 tests)
✓ Configuration (6 tests)
✓ Graph Construction (6 tests)
✓ Error Handling (8 tests)

Recommendation: Bundle is ready for use
```

## Test Categories and What They Check

### Initialization Tests (test_initialization.py)
Verifies bundle starts up correctly:

- Bundle initializes with valid config, redis, and comm_context
- LangGraph compiles after initialization
- configuration property returns dict with role_models
- Bundle handles None redis (falls back to defaults)
- Event filter initialized (if provided)

### Configuration Tests (test_configuration.py)
Verifies bundle settings work correctly:

- Default role_models applied from code
- External config overrides respected
- bundle_prop() returns correct values for nested paths
- Missing config paths return None (no KeyError)
- Redis overrides take precedence over defaults

### Graph Construction Tests (test_graph_construction.py)
Verifies LangGraph is built correctly:

- _build_graph() returns compiled StateGraph
- Graph can be invoked with valid BundleState
- Graph produces output with final_answer
- Graph produces followups list
- All nodes connected (no orphans)
- Execution completes in < 30 seconds

### Error Handling Tests (test_error_handling.py)
Verifies errors are caught and reported:

- Node exceptions caught (not propagated)
- error_message set in state when error occurs
- chat.error event emitted
- Error message is user-friendly
- EconomicsLimitException NOT caught (re-raised)
- Multiple errors don't cascade
- Error doesn't crash system

## How React.doc Learns About Testing

### Documentation Chain

1. **Reads:** bundle-testing-methodology-README.md (this file)
2. **Learns:** `--bundle-id` parameter is required, how to run tests
3. **Reads:** bundle-testing-system-README.md
4. **Learns:** test files location and structure
5. **Reads:** bundle-tests-README.md
6. **Learns:** test categories and what each category verifies
7. **Executes:** pytest with appropriate --bundle-id parameter
8. **Parses:** test output (PASSED/FAILED counts)
9. **Reports:** results to user

## What React.doc Does

When user asks to test a bundle:

1. **Ask** which bundle to test (if not specified)
2. **Search knowledge** for testing methodology documentation
3. **Learn** the `--bundle-id` parameter requirement
4. **Run** all tests:
   ```
   pytest sdk/bundle_tests/test_*.py --bundle-id=<chosen_bundle> -v
   ```
5. **Wait** for completion
6. **Parse** test results (count PASSED vs FAILED)
7. **Report** to user:
   - Which test categories passed
   - Which test categories failed
   - Total count of passed/failed tests
8. **Recommend** whether bundle is ready or needs fixes

## Key Points for React.doc

- Tests are in: `sdk/bundle_tests/`
- Always pass: `--bundle-id=<bundle_name>`
- Use verbose: `-v` flag for detailed output
- Different categories: in different test files
- Parameterization: same tests work for all bundles
- Parsing: extract PASSED/FAILED from pytest output
- No mocks: tests load and run REAL bundles