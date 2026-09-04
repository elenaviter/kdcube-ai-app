#!/usr/bin/env python3
"""Run the common Agent Harness acceptance scenario through Claude Code."""

from demo import main_for
from runtime_client import AgentTarget


if __name__ == "__main__":
    main_for(
        AgentTarget(
            adapter="claude",
            bundle_id="harness-claude-demo@1-0",
            agent_id="claude",
            needs_exec_image=False,
            description="the hosted Claude Code adapter",
            required_operations=("WebSearch", "WebFetch"),
        )
    )
