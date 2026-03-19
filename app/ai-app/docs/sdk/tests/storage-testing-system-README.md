---
title: "Bundle Storage Testing System"
summary: "Auto-discovery and execution of pytest bundle storage tests by react.doc agent."
tags: ["sdk", "testing", "storage", "s3", "redis", "cache", "architecture", "react.doc"]
keywords: ["storage testing", "s3", "redis", "local filesystem", "pytest", "test discovery"]
see_also:
  - ks:docs/sdk/bundle/bundle-storage-cache-README.md
  - ks:docs/sdk/storage/cache-README.md
  - ks:docs/sdk/tests/bundle-testing-system-README.md
---

# Bundle Storage Testing System

Actual pytest tests for bundle storage (Cloud Storage, Local FS, Redis Cache) stored in the SDK codebase. React.doc agent auto-discovers and runs them.

## Architecture

```
Pytest Test Files (actual Python code)
  └─ kdcube_ai_app/apps/chat/sdk/bundle_tests/
       ├─ test_storage_cloud.py
       ├─ test_storage_local_fs.py
       ├─ test_storage_redis.py
       ├─ test_storage_integration.py
       ├─ fixtures/mock_storages.py
       └─ ...

          ↓

Test Registry/Discovery
  └─ react.doc scans bundle_tests/ directory
       ├─ Lists available test files
       ├─ Extracts test names
       └─ Builds test catalog

          ↓

React.doc Agent Execution
  ├─ User: "Test bundle storage"
  ├─ react.doc searches knowledge for test catalog
  ├─ react.doc selects relevant tests
  ├─ Runs: pytest bundle_tests/test_storage_*.py -v
  ├─ Parses output
  └─ Reports results to user
```

## Storage Types

### Cloud Storage (S3)
- **Configuration**: `CB_BUNDLE_STORAGE_URL`
- **Lifetime**: Persistent across restarts
- **Use**: Long-term data, documents, knowledge bases

### Local Filesystem
- **Configuration**: `BUNDLE_STORAGE_ROOT`
- **Lifetime**: Ephemeral (lost on restart)
- **Use**: Runtime caches, temporary files

### Redis Cache
- **Configuration**: `REDIS_URL`, `KV_CACHE_TTL_SECONDS`
- **Lifetime**: Subject to eviction policy
- **Use**: Config overrides, session state

## Test Files Location

```
kdcube_ai_app/
  apps/
    chat/
      sdk/
        bundle_tests/
          __init__.py
          conftest.py                         ← Shared fixtures

          test_storage_cloud.py               ← Category: Cloud Storage (S3)
          test_storage_local_fs.py            ← Category: Local Filesystem
          test_storage_redis.py               ← Category: Redis Cache
          test_storage_integration.py         ← Category: Storage Integration

          fixtures/
            __init__.py
            mock_storages.py                  ← Storage Mocks (S3, Local FS, Redis)
```

## How react.doc Runs Storage Tests

### User Request
```
"Test bundle storage"
```

### react.doc Agent Flow

**Step 1: Search for test information**
```
search_knowledge(query="bundle storage testing pytest")

Returns docs that reference storage testing and test structure
```

**Step 2: Understand test structure**
```
From knowledge docs, learns:
- Tests are in: kdcube_ai_app/apps/chat/sdk/bundle_tests/
- Test categories: cloud storage, local fs, redis, integration
- How to run: pytest test_storage_*.py -v
- Storage types: S3, Local FS, Redis Cache
```

**Step 3: Run relevant tests**
```bash
cd kdcube_ai_app/
pytest apps/chat/sdk/bundle_tests/test_storage_cloud.py -v
pytest apps/chat/sdk/bundle_tests/test_storage_local_fs.py -v
pytest apps/chat/sdk/bundle_tests/test_storage_redis.py -v
pytest apps/chat/sdk/bundle_tests/test_storage_integration.py -v
```

**Step 4: Collect results**
```
test_storage_cloud.py::test_read_from_s3 PASSED
test_storage_cloud.py::test_write_to_s3 PASSED
test_storage_local_fs.py::test_read_from_local_fs PASSED
test_storage_local_fs.py::test_cleanup_temp_files PASSED
test_storage_redis.py::test_read_from_redis PASSED
test_storage_redis.py::test_redis_ttl PASSED
test_storage_redis.py::test_namespace_isolation PASSED
test_storage_integration.py::test_multi_storage_flow PASSED
...
```

**Step 5: Report to user**
```
Bundle Storage
Status: ✓ ALL TESTS PASSED (14/14)

✓ Cloud Storage (4 tests)
✓ Local Filesystem (3 tests)
✓ Redis Cache (4 tests)
✓ Integration (3 tests)

Recommendation: Storage ready for use
```

## How react.doc Knows to Run Storage Tests

react.doc learns about storage tests from **knowledge space documentation** that describes:
1. Where tests are located (SDK path)
2. How to run them (pytest command)
3. What storage types are tested
4. What each test verifies

This documentation is in `/docs/sdk/bundle/bundle-storage-cache-README.md` and `/docs/sdk/tests/storage-testing-system-README.md`.

React.doc reads these docs, understands the structure, then runs the actual pytest files.

## Execution Flow Summary

```
User Request
  ↓
react.doc reads: /docs/sdk/bundle/bundle-storage-cache-README.md
              + /docs/sdk/storage/cache-README.md
              + /docs/sdk/tests/storage-testing-system-README.md
  ↓
Understands:
  - Tests in: kdcube_ai_app/apps/chat/sdk/bundle_tests/
  - How to run: pytest test_storage_*.py -v
  - Storage types: Cloud (S3), Local FS, Redis Cache
  - Configurations: CB_BUNDLE_STORAGE_URL, BUNDLE_STORAGE_ROOT, REDIS_URL
  ↓
Executes:
  pytest apps/chat/sdk/bundle_tests/test_storage_*.py -v --tb=short
  ↓
Parses output:
  PASSED: 14/14 tests
  ✓ Cloud Storage: OK
  ✓ Local FS: OK
  ✓ Redis Cache: OK
  ✓ Integration: OK
  ↓
Reports to user:
  "Bundle storage: PASSED (14/14 tests)"
```

## File Structure

```
docs/sdk/
  ├─ bundle/
  │   └─ bundle-storage-cache-README.md       (storage usage & API)
  ├─ storage/
  │   ├─ cache-README.md                      (Redis cache API)
  │   └─ sdk-store-README.md                  (storage layout)
  └─ tests/
      └─ storage-testing-system-README.md     (this file - DOCUMENTATION ONLY)

kdcube_ai_app/apps/chat/sdk/
  └─ bundle_tests/
      ├─ __init__.py
      ├─ conftest.py
      ├─ test_storage_cloud.py                (ACTUAL TESTS)
      ├─ test_storage_local_fs.py
      ├─ test_storage_redis.py
      ├─ test_storage_integration.py
      └─ fixtures/
          ├─ __init__.py
          └─ mock_storages.py
```

## Test Categories

### Cloud Storage Tests
Verify S3/cloud storage integration.
- [ ] Read operations (files, paths, permissions)
- [ ] Write operations (path isolation, overwrites)
- [ ] Error handling (unavailable, credentials, not found)

### Local Filesystem Tests
Verify local FS storage integration.
- [ ] Read/write operations
- [ ] Temp file cleanup
- [ ] Ephemeral nature (lost on restart)
- [ ] Fallback when S3 unavailable

### Redis Cache Tests
Verify Redis cache integration.
- [ ] Read/write operations
- [ ] TTL and expiration
- [ ] Namespace isolation (tenant/project/bundle)
- [ ] Graceful fallback

### Integration Tests
Verify multi-storage workflows.
- [ ] Storage fallback chain (Redis → Local FS → Cloud)
- [ ] Context isolation (tenant/project/bundle)
- [ ] Error recovery
- [ ] Concurrent access

## Summary

- **Tests**: Real Python pytest files in SDK codebase
- **Documentation**: Describes what tests verify and how to run
- **react.doc**: Reads docs, understands structure, runs pytest, reports results
- **No changes needed**: react.doc uses existing knowledge access + subprocess execution
- **Simple and maintainable**: Tests are just pytest, not embedded in docs