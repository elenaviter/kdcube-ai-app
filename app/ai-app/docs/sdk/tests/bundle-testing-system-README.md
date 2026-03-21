---
title: "Bundle Testing System"
summary: "Auto-discovery and execution of pytest bundle tests by react.doc agent."
tags: ["sdk", "testing", "bundle", "architecture", "react.doc"]
keywords: ["bundle tests", "pytest", "test discovery", "test execution", "react.doc"]
see_also:
  - ks:docs/sdk/bundle-tests-README.md
  - ks:docs/sdk/bundle/bundle-dev-README.md
---

# Bundle Testing System

Actual pytest tests stored in the SDK codebase. React.doc agent auto-discovers and runs them.

## Architecture

```
Pytest Test Files (actual Python code)
  └─ kdcube_ai_app/apps/chat/sdk/bundle_tests/
       ├─ test_initialization.py
       ├─ test_configuration.py
       ├─ test_graph_construction.py
       ├─ test_error_handling.py
       ├─ test_accounting.py
       ├─ conftest.py (shared fixtures)
       └─ ...

          ↓

Test Registry/Discovery
  └─ react.doc scans bundle_tests/ directory
       ├─ Lists available test files
       ├─ Extracts test names
       └─ Builds test catalog

          ↓

React.doc Agent Execution
  ├─ User: "Test bundle X"
  ├─ react.doc searches knowledge for test catalog
  ├─ react.doc selects relevant tests
  ├─ Runs: pytest bundle_tests/test_X.py -v
  ├─ Parses output
  └─ Reports results to user
```

## Test Files Location

```
kdcube_ai_app/
  apps/
    chat/
      sdk/
        bundle_tests/
          __init__.py
          conftest.py                    ← Shared fixtures

          test_initialization.py         ← Category: Init
          test_configuration.py          ← Category: Config
          test_graph_construction.py     ← Category: Graph
          test_bundlestate.py            ← Category: State
          test_error_handling.py         ← Category: Errors
          test_event_streaming.py        ← Category: Events
          test_accounting.py             ← Category: Accounting
          test_storage.py                ← Category: Storage
          test_model_routing.py          ← Category: Models
          test_execution_flow.py         ← Category: Integration

          fixtures/
            __init__.py
            mock_config.py               ← Mock Config
            mock_services.py             ← Mock services
            bundle_state.py              ← BundleState factory
```

## How react.doc Runs Tests

### User Request
```
"Test the openrouter-data bundle"
```

### react.doc Agent Flow

**Step 1: Search for test information**
```
search_knowledge(query="bundle tests openrouter-data pytest")

Returns docs that reference testing and test categories
```

**Step 2: Understand test structure**
```
From knowledge docs, learns:
- Tests are in: kdcube_ai_app/apps/chat/sdk/bundle_tests/
- Test categories: initialization, configuration, graph, errors, accounting, etc.
- How to run: pytest <test_file> -v
```

**Step 3: Run relevant tests**
```bash
cd kdcube_ai_app/
pytest apps/chat/sdk/bundle_tests/test_initialization.py -v
pytest apps/chat/sdk/bundle_tests/test_configuration.py -v
pytest apps/chat/sdk/bundle_tests/test_error_handling.py -v
# ... etc
```

**Step 4: Collect results**
```
test_initialization.py::test_entrypoint_init_with_valid_config PASSED
test_initialization.py::test_configuration_property PASSED
test_configuration.py::test_config_merge PASSED
test_error_handling.py::test_report_error PASSED
test_accounting.py::test_usage_tracked PASSED
...
```

**Step 5: Report to user**
```
Bundle: openrouter-data
Status: ✓ ALL TESTS PASSED (15/15)

✓ Initialization (3 tests)
✓ Configuration (2 tests)
✓ Graph Construction (3 tests)
✓ Error Handling (4 tests)
✓ Accounting (3 tests)

Recommendation: Bundle ready for use
```

## How react.doc Knows to Run Tests

react.doc learns about tests from **knowledge space documentation** that describes:
1. Where tests are located (SDK path)
2. How to run them (pytest command)
3. What categories exist
4. What each test verifies

This documentation is in `/docs/sdk/bundle-tests-README.md` and `/docs/sdk/tests/bundle-testing-system-README.md`.

React.doc reads these docs, understands the structure, then runs the actual pytest files.

## Execution Flow Summary

```
User Request
  ↓
react.doc reads: /docs/sdk/bundle-tests-README.md
              + /docs/sdk/tests/bundle-testing-system-README.md
  ↓
Understands:
  - Tests in: kdcube_ai_app/apps/chat/sdk/bundle_tests/
  - How to run: pytest <file> -v
  - Categories: initialization, config, graph, errors, etc.
  ↓
Executes:
  pytest apps/chat/sdk/bundle_tests/test_*.py -v --tb=short
  ↓
Parses output:
  PASSED: 15/15 tests
  ✓ Categories passing
  ✗ Categories failing
  ↓
Reports to user:
  "Bundle X: PASSED (15/15 tests)"
```

## File Structure

```
docs/sdk/
  └─ tests/
      └─ bundle-testing-system-README.md (this file - DOCUMENTATION ONLY)

kdcube_ai_app/apps/chat/sdk/
  └─ bundle_tests/
      ├─ __init__.py
      ├─ conftest.py
      ├─ test_initialization.py (ACTUAL TESTS)
      ├─ test_configuration.py
      ├─ test_graph_construction.py
      ├─ test_bundlestate.py
      ├─ test_error_handling.py
      ├─ test_event_streaming.py
      ├─ test_accounting.py
      ├─ test_storage.py
      ├─ test_model_routing.py
      └─ test_execution_flow.py
```

## Summary

- **Tests**: Real Python pytest files in SDK codebase
- **Documentation**: Describes what tests verify and how to run
- **react.doc**: Reads docs, understands structure, runs pytest, reports results
- **No changes needed**: react.doc uses existing knowledge access + subprocess execution
- **Simple and maintainable**: Tests are just pytest, not embedded in docs