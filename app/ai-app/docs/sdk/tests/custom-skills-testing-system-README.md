---
title: "Custom Skills Testing System"
summary: "Auto-discovery and execution of pytest custom skill tests by react.doc agent."
tags: ["sdk", "testing", "skills", "custom skills", "architecture", "react.doc"]
keywords: ["custom skills", "skills subsystem", "pytest", "test discovery", "test execution"]
see_also:
  - ks:docs/sdk/skills/custom-skills-README.md
  - ks:docs/sdk/skills/skills-README.md
  - ks:docs/sdk/tests/bundle-testing-system-README.md
---

# Custom Skills Testing System

Actual pytest tests for custom skills stored in the SDK codebase. React.doc agent auto-discovers and runs them.

## Architecture

```
Pytest Test Files (actual Python code)
  └─ kdcube_ai_app/apps/chat/sdk/bundle_tests/
       ├─ test_custom_skills_registration.py
       ├─ test_custom_skills_manifest.py
       ├─ test_custom_skills_visibility.py
       ├─ test_custom_skills_execution.py
       ├─ fixtures/custom_skills.py
       └─ ...

          ↓

Test Registry/Discovery
  └─ react.doc scans bundle_tests/ directory
       ├─ Lists available test files
       ├─ Extracts test names
       └─ Builds test catalog

          ↓

React.doc Agent Execution
  ├─ User: "Test custom skills in bundle X"
  ├─ react.doc searches knowledge for test catalog
  ├─ react.doc selects relevant tests
  ├─ Runs: pytest bundle_tests/test_custom_skills_*.py -v
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

          test_custom_skills_registration.py  ← Category: Skill Registration
          test_custom_skills_manifest.py      ← Category: Manifest Validation
          test_custom_skills_visibility.py    ← Category: Visibility Control
          test_custom_skills_execution.py     ← Category: Skill Execution

          fixtures/
            __init__.py
            custom_skills.py                  ← Skill Fixtures
```

## How react.doc Runs Custom Skill Tests

### User Request
```
"Test the custom skills in my bundle"
```

### react.doc Agent Flow

**Step 1: Search for test information**
```
search_knowledge(query="custom skills testing pytest system")

Returns docs that reference skill testing and test structure
```

**Step 2: Understand test structure**
```
From knowledge docs, learns:
- Tests are in: kdcube_ai_app/apps/chat/sdk/bundle_tests/
- Test categories: registration, manifest, visibility, execution
- How to run: pytest test_custom_skills_*.py -v
- Fixture structure: custom_skills.py
```

**Step 3: Run relevant tests**
```bash
cd kdcube_ai_app/
pytest apps/chat/sdk/bundle_tests/test_custom_skills_registration.py -v
pytest apps/chat/sdk/bundle_tests/test_custom_skills_manifest.py -v
pytest apps/chat/sdk/bundle_tests/test_custom_skills_visibility.py -v
pytest apps/chat/sdk/bundle_tests/test_custom_skills_execution.py -v
```

**Step 4: Collect results**
```
test_custom_skills_registration.py::test_skill_registers PASSED
test_custom_skills_registration.py::test_skill_in_descriptor PASSED
test_custom_skills_manifest.py::test_manifest_valid PASSED
test_custom_skills_manifest.py::test_manifest_frontmatter PASSED
test_custom_skills_visibility.py::test_skill_visibility PASSED
test_custom_skills_execution.py::test_skill_executes PASSED
...
```

**Step 5: Report to user**
```
Custom Skills: openrouter-data-bundle
Status: ✓ ALL TESTS PASSED (10/10)

✓ Registration (2 tests)
✓ Manifest (2 tests)
✓ Visibility (3 tests)
✓ Execution (3 tests)

Recommendation: Custom skills ready to use
```

## How react.doc Knows to Run Skill Tests

react.doc learns about custom skill tests from **knowledge space documentation** that describes:
1. Where tests are located (SDK path)
2. How to run them (pytest command)
3. What categories exist
4. What each test verifies

This documentation is in `/docs/sdk/skills/custom-skills-README.md` and `/docs/sdk/tests/custom-skills-testing-system-README.md`.

React.doc reads these docs, understands the structure, then runs the actual pytest files.

## Execution Flow Summary

```
User Request
  ↓
react.doc reads: /docs/sdk/skills/custom-skills-README.md
              + /docs/sdk/tests/custom-skills-testing-system-README.md
  ↓
Understands:
  - Tests in: kdcube_ai_app/apps/chat/sdk/bundle_tests/
  - How to run: pytest test_custom_skills_*.py -v
  - Categories: registration, manifest, visibility, execution
  ↓
Executes:
  pytest apps/chat/sdk/bundle_tests/test_custom_skills_*.py -v --tb=short
  ↓
Parses output:
  PASSED: 10/10 tests
  ✓ Categories passing
  ✗ Categories failing
  ↓
Reports to user:
  "Custom skills X: PASSED (10/10 tests)"
```

## File Structure

```
docs/sdk/
  ├─ skills/
  │   └─ custom-skills-README.md              (checklist & patterns)
  └─ tests/
      └─ custom-skills-testing-system-README.md (this file - DOCUMENTATION ONLY)

kdcube_ai_app/apps/chat/sdk/
  └─ bundle_tests/
      ├─ __init__.py
      ├─ conftest.py
      ├─ test_custom_skills_registration.py   (ACTUAL TESTS)
      ├─ test_custom_skills_manifest.py
      ├─ test_custom_skills_visibility.py
      ├─ test_custom_skills_execution.py
      └─ fixtures/
          ├─ __init__.py
          └─ custom_skills.py
```

## Test Categories

### Registration Tests
Verify skill registers with Skills Subsystem.
- [ ] Skill in descriptor
- [ ] Skill accessible via bundle.get_skill()
- [ ] Skill ID format correct

### Manifest Tests
Verify SKILL.md and metadata are valid.
- [ ] SKILL.md frontmatter valid
- [ ] tools.yaml valid (if present)
- [ ] sources.yaml valid (if present)

### Visibility Tests
Verify AGENTS_CONFIG controls skill visibility.
- [ ] Skill visible to enabled agents
- [ ] Skill hidden from disabled agents
- [ ] Wildcard patterns work

### Execution Tests
Verify skill instruction injected correctly.
- [ ] Skill instruction in prompt
- [ ] Skill callable from graph
- [ ] Events flow correctly

## Summary

- **Tests**: Real Python pytest files in SDK codebase
- **Documentation**: Describes what tests verify and how to run
- **react.doc**: Reads docs, understands structure, runs pytest, reports results
- **No changes needed**: react.doc uses existing knowledge access + subprocess execution
- **Simple and maintainable**: Tests are just pytest, not embedded in docs