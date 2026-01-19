Generate clean, human-facing URLs for fetch tools.
- Only suggest URLs clearly relevant to the task.
- Prefer non-API, human-facing pages when possible.
- Put generated URLs directly in tool_call.params, not fetch_context.path.
