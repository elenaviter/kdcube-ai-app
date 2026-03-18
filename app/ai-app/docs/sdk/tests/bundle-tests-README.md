---
title: "Bundle Tests"
summary: "Checklist of tests to verify a bundle works correctly and integrates with the platform."
tags: ["sdk", "bundle", "testing", "qa"]
keywords: ["bundle tests", "verification", "integration", "error handling", "accounting"]
see_also:
  - ks:docs/sdk/bundle/bundle-dev-README.md
  - ks:docs/sdk/bundle/bundle-platform-properties-README.md
---

# Bundle Tests Checklist

Write these tests to verify your bundle works correctly.

## 1. Initialization Tests

Test that the bundle starts up correctly.

- [ ] Bundle initializes with valid config, redis, and comm_context
- [ ] LangGraph compiles after initialization
- [ ] `configuration` property returns dict with `role_models`
- [ ] Bundle handles None redis (falls back to defaults)
- [ ] Event filter initialized (if provided)

## 2. Configuration Tests

Test that bundle settings work correctly.

- [ ] Default `role_models` applied from code
- [ ] External config overrides respected
- [ ] `bundle_prop("role_models.solver.model")` returns correct model
- [ ] Missing config paths return None (no KeyError)
- [ ] `refresh_bundle_props()` merges Redis overrides
- [ ] Redis overrides take precedence over defaults

## 3. Graph Tests

Test that the LangGraph works.

- [ ] `_build_graph()` returns compiled StateGraph
- [ ] Graph can be invoked with valid BundleState
- [ ] Graph produces output with `final_answer`
- [ ] Graph produces `followups` list
- [ ] All nodes connected (no orphans)
- [ ] Execution completes in < 30 seconds

## 4. BundleState Tests

Test that request/response state is handled correctly.

- [ ] All required fields preserved (request_id, tenant, project, user)
- [ ] `final_answer` populated after execution
- [ ] `followups` populated after execution
- [ ] `error_message` set if error occurs
- [ ] `attachments` field handled correctly
- [ ] State doesn't leak between requests

## 5. Error Handling Tests

Test that errors are caught and reported properly.

- [ ] Node exceptions caught (not propagated)
- [ ] `error_message` set in state when error occurs
- [ ] `chat.error` event emitted
- [ ] Error message is user-friendly (no stack trace)
- [ ] `EconomicsLimitException` NOT caught (re-raised)
- [ ] Multiple errors don't cascade (first one reported)
- [ ] Error doesn't crash system

## 6. Event Streaming Tests

Test that events flow correctly to client.

- [ ] First event has `status="running"` or `status="started"`
- [ ] Last event has `status="done"` or `status="error"`
- [ ] Events include model name and metadata
- [ ] Event filter applied before emit (if provided)
- [ ] No events emitted after done/error
- [ ] Events in logical order (start → processing → done)

## 7. Accounting Tests

Test that cost tracking works.

- [ ] LLM calls wrapped in accounting context
- [ ] Done event includes `usage` dict
- [ ] `usage` contains `prompt_tokens`, `completion_tokens`, `total_tokens`
- [ ] Multiple LLM calls tracked separately
- [ ] Budget pre-check enforced (if using economics)
- [ ] Over-budget requests rejected with `EconomicsLimitException`

## 8. Storage & Props Tests

Test that local storage and config work.

- [ ] `bundle_storage_root()` returns correct path
- [ ] Path includes tenant/project/bundle_id
- [ ] `on_bundle_load()` creates storage directories (if implemented)
- [ ] Knowledge space files accessible (if using ks: paths)
- [ ] Redis unavailable → graceful fallback to defaults

## 9. Model Routing Tests

Test that model selection works.

- [ ] Default model used if no override
- [ ] Config override respected
- [ ] Redis override respected (takes precedence)
- [ ] Model slug passed to LLM correctly
- [ ] Switching models works (Claude → OpenRouter, etc.)

## 10. Integration Flow Test

Test complete request → response flow.

- [ ] `run(ChatTaskPayload)` returns dict with `final_answer`
- [ ] `execute_core(state, thread_id, params)` returns state with answer
- [ ] Sequential requests work (no state leakage)
- [ ] Multiple concurrent requests don't interfere
- [ ] Response time acceptable (< 30s for simple queries)

## 11. Agent Request/Response Test

Test that requests reach agent and responses are returned correctly.

- [ ] Bundle receives user message (ChatTaskPayload)
- [ ] Message reaches entrypoint.run()
- [ ] LangGraph executes completely
- [ ] final_answer written to state
- [ ] Response sent back to user via SSE
- [ ] Client receives complete message
- [ ] No message loss or corruption
- [ ] Multi-turn conversation preserves context
- [ ] Error messages reach client
- [ ] Timeout handled gracefully (> 30s)