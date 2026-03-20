---
title: "React.doc Bundle Testing Guide"
summary: "Instructions for react.doc agent to run and validate bundle tests."
tags: ["sdk", "testing", "react.doc", "instructions", "pytest", "automation"]
keywords: ["how to test bundles", "bundle validation", "pytest commands", "react.doc automation", "test execution"]
see_also:
  - ks:docs/sdk/tests/bundle-testing-methodology-README.md
  - ks:docs/sdk/tests/bundle-tests-README.md
---

# React.doc Bundle Testing Guide

Instructions for react.doc agent on how to test KDCube bundles.

## Quick Start

When user asks to test a bundle, react.doc should:

### Step 1: Ask which bundle to test
```
react.doc: "Which bundle would you like me to test?"
  - react.doc
  - react
  - openrouter-data
  - eco
```

### Step 2: Run pytest with bundle ID
Use the `exec_tools.run_command()` tool to execute:

```bash
pytest /path/to/kdcube_ai_app/apps/chat/sdk/bundle_tests/test_initialization.py --bundle-id=<bundle_name> -v
```

Where `<bundle_name>` is one of:
- `react.doc`
- `react`
- `openrouter-data`
- `eco`
- Any other registered bundle ID

### Step 3: Parse and report results

Look for lines like:
```
test_initialization.py::test_bundle_initializes PASSED
test_initialization.py::test_langgraph_compiles PASSED
test_initialization.py::test_configuration_property PASSED
test_initialization.py::test_bundle_handles_none_redis PASSED
test_initialization.py::test_event_filter PASSED

======================== 5 passed in 0.42s ========================
```

Report to user:
```
Bundle: react.doc
Status: ✓ INITIALIZATION TESTS PASSED (5/5)

✓ test_bundle_initializes
✓ test_langgraph_compiles
✓ test_configuration_property
✓ test_bundle_handles_none_redis
✓ test_event_filter

Recommendation: Bundle initialization looks good
```

## Test File Locations

Tests are located at:
```
kdcube_ai_app/apps/chat/sdk/bundle_tests/
  ├── __init__.py
  ├── conftest.py         ← pytest configuration
  └── test_initialization.py ← initialization tests
```

## How Tests Work

The conftest.py automatically:
1. Accepts `--bundle-id` parameter from command line
2. Discovers bundle from system BundleStore
3. Initializes bundle with mock dependencies
4. Provides initialized bundle to tests

Tests then verify:
- Bundle initializes correctly
- LangGraph compiles
- Configuration loads
- Error handling works
- Event filters work (if provided)

## Commands for React.doc

### Test single bundle initialization
```bash
pytest kdcube_ai_app/apps/chat/sdk/bundle_tests/test_initialization.py --bundle-id=react.doc -v
```

### Test with verbose output
```bash
pytest kdcube_ai_app/apps/chat/sdk/bundle_tests/test_initialization.py --bundle-id=react -v --tb=short
```

### Discover available tests (dry run)
```bash
pytest kdcube_ai_app/apps/chat/sdk/bundle_tests/ --co -q --bundle-id=react.doc
```

## User Requests and Actions

When user says:
```
"Test the react.doc bundle"
```

React.doc should:
1. Search knowledge for "bundle testing"
2. Find this guide and methodology docs
3. Extract command: `pytest ... --bundle-id=react.doc -v`
4. Use `exec_tools.run_command()` to execute
5. Parse results and report

## Test Results Interpretation

### All passed
```
======================== 5 passed in 0.42s ========================
```
→ Report: "✓ All initialization tests passed"

### Some failed
```
FAILED test_initialization.py::test_bundle_initializes
```
→ Report: "✗ Test failed: bundle could not be initialized"

### Bundle not found
```
pytest.skip('Bundle 'unknown-bundle' not found. Available: ...')
```
→ Report: "Bundle not found. Available bundles: react.doc, react, openrouter-data, eco"

## Available Bundles

React.doc can test any registered bundle. Common ones:
- `react.doc` - Documentation/knowledge search agent
- `react` - Main reasoning agent
- `openrouter-data` - Data processing via OpenRouter
- `eco` - Economics/billing related
- Any custom bundle registered with @agentic_workflow decorator

## Error Handling

If tests fail, react.doc should:
1. Check if bundle ID is correct
2. Verify bundle exists: `pytest ... --bundle-id=<id> -v`
3. Report which specific test failed
4. Suggest next steps (check bundle config, dependencies, etc.)

## Summary for React.doc

To test a bundle:
1. Ask user which bundle
2. Run: `pytest sdk/bundle_tests/test_initialization.py --bundle-id=<bundle> -v`
3. Parse: Count PASSED/FAILED
4. Report: Show results and recommendation