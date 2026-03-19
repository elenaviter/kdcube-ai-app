---
title: "Custom Tools Testing System"
summary: "Auto-discovery and execution of pytest custom tool tests by react.doc agent."
tags: ["sdk", "testing", "tools", "custom tools", "architecture", "react.doc"]
keywords: ["custom tools", "tools subsystem", "pytest", "test discovery", "test execution"]
see_also:
  - ks:docs/sdk/tools/custom-tools-README.md
  - ks:docs/sdk/tools/tool-subsystem-README.md
  - ks:docs/sdk/tests/bundle-testing-system-README.md
---

# Custom Tools Testing System

Actual pytest tests for custom tools stored in the SDK codebase. React.doc agent auto-discovers and runs them.

## Architecture

```
Pytest Test Files (actual Python code)
  └─ kdcube_ai_app/apps/chat/sdk/bundle_tests/
       ├─ test_custom_tools_registration.py
       ├─ test_custom_tools_execution.py
       ├─ test_custom_tools_storage.py
       ├─ fixtures/custom_tools.py
       └─ ...

          ↓

Test Registry/Discovery
  └─ react.doc scans bundle_tests/ directory
       ├─ Lists available test files
       ├─ Extracts test names
       └─ Builds test catalog

          ↓

React.doc Agent Execution
  ├─ User: "Test custom tools in bundle X"
  ├─ react.doc searches knowledge for test catalog
  ├─ react.doc selects relevant tests
  ├─ Runs: pytest bundle_tests/test_custom_tools_*.py -v
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
          conftest.py                         ← Shared fixtures

          test_custom_tools_registration.py   ← Category: Tool Registration
          test_custom_tools_execution.py      ← Category: Tool Execution
          test_custom_tools_storage.py        ← Category: Tool Storage
          test_custom_tools_integration.py    ← Category: Integration

          fixtures/
            __init__.py
            custom_tools.py                   ← Tool Fixtures
            mock_storages.py                  ← Storage Mocks
```

## How react.doc Runs Custom Tool Tests

### User Request
```
"Test the custom tools in my bundle"
```

### react.doc Agent Flow

**Step 1: Search for test information**
```
search_knowledge(query="custom tools testing pytest system")

Returns docs that reference tool testing and test structure
```

**Step 2: Understand test structure**
```
From knowledge docs, learns:
- Tests are in: kdcube_ai_app/apps/chat/sdk/bundle_tests/
- Test categories: registration, execution, storage, integration
- How to run: pytest test_custom_tools_*.py -v
- Fixture structure: custom_tools.py, mock_storages.py
```

**Step 3: Run relevant tests**
```bash
cd kdcube_ai_app/
pytest apps/chat/sdk/bundle_tests/test_custom_tools_registration.py -v
pytest apps/chat/sdk/bundle_tests/test_custom_tools_execution.py -v
pytest apps/chat/sdk/bundle_tests/test_custom_tools_storage.py -v
pytest apps/chat/sdk/bundle_tests/test_custom_tools_integration.py -v
```

**Step 4: Collect results**
```
test_custom_tools_registration.py::test_tool_registers PASSED
test_custom_tools_registration.py::test_tool_in_descriptor PASSED
test_custom_tools_execution.py::test_tool_executes PASSED
test_custom_tools_execution.py::test_tool_error_handling PASSED
test_custom_tools_storage.py::test_tool_reads_from_s3 PASSED
test_custom_tools_integration.py::test_tool_in_langgraph PASSED
...
```

**Step 5: Report to user**
```
Custom Tools: openrouter-data-bundle
Status: ✓ ALL TESTS PASSED (12/12)

✓ Registration (2 tests)
✓ Execution (3 tests)
✓ Storage (4 tests)
✓ Integration (3 tests)

Recommendation: Custom tools ready to use
```

## How react.doc Knows to Run Tool Tests

react.doc learns about custom tool tests from **knowledge space documentation** that describes:
1. Where tests are located (SDK path)
2. How to run them (pytest command)
3. What categories exist
4. What each test verifies

This documentation is in `/docs/sdk/tools/custom-tools-README.md` and `/docs/sdk/tests/custom-tools-testing-system-README.md`.

React.doc reads these docs, understands the structure, then runs the actual pytest files.

## Execution Flow Summary

```
User Request
  ↓
react.doc reads: /docs/sdk/tools/custom-tools-README.md
              + /docs/sdk/tests/custom-tools-testing-system-README.md
  ↓
Understands:
  - Tests in: kdcube_ai_app/apps/chat/sdk/bundle_tests/
  - How to run: pytest test_custom_tools_*.py -v
  - Categories: registration, execution, storage, integration
  ↓
Executes:
  pytest apps/chat/sdk/bundle_tests/test_custom_tools_*.py -v --tb=short
  ↓
Parses output:
  PASSED: 12/12 tests
  ✓ Categories passing
  ✗ Categories failing
  ↓
Reports to user:
  "Custom tools X: PASSED (12/12 tests)"
```

## File Structure

```
docs/sdk/
  ├─ tools/
  │   └─ custom-tools-README.md               (checklist & patterns)
  └─ tests/
      └─ custom-tools-testing-system-README.md (this file - DOCUMENTATION ONLY)

kdcube_ai_app/apps/chat/sdk/
  └─ bundle_tests/
      ├─ __init__.py
      ├─ conftest.py
      ├─ test_custom_tools_registration.py    (ACTUAL TESTS)
      ├─ test_custom_tools_execution.py
      ├─ test_custom_tools_storage.py
      ├─ test_custom_tools_integration.py
      └─ fixtures/
          ├─ __init__.py
          ├─ custom_tools.py
          └─ mock_storages.py
```

## Test Categories

### Registration Tests
Verify tool registers with Tools Subsystem.
- [ ] Tool in descriptor
- [ ] Tool accessible via bundle.get_tool()
- [ ] Tool ID format correct

### Execution Tests
Verify tool executes correctly.
- [ ] Tool executes with valid inputs
- [ ] Tool returns expected output
- [ ] Error handling works

### Storage Tests
Verify tool storage integration (S3, local FS, Redis).
- [ ] Tool accesses cloud storage
- [ ] Tool uses local FS cache
- [ ] Tool reads from Redis

### Integration Tests
Verify tool works in LangGraph.
- [ ] Tool callable from nodes
- [ ] Multiple tools don't conflict
- [ ] Tool dependencies resolve

## Summary

- **Tests**: Real Python pytest files in SDK codebase
- **Documentation**: Describes what tests verify and how to run
- **react.doc**: Reads docs, understands structure, runs pytest, reports results
- **No changes needed**: react.doc uses existing knowledge access + subprocess execution
- **Simple and maintainable**: Tests are just pytest, not embedded in docs