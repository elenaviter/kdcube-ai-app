#!/usr/bin/env python3
"""Run the common Agent Harness acceptance scenario through native ReAct."""

from kdcube_ai_app.apps.chat.sdk.examples.tests.hosted_agent_harness.demo import main_for
from kdcube_ai_app.apps.chat.sdk.examples.tests.hosted_agent_harness.runtime_client import AgentTarget


if __name__ == "__main__":
    main_for(
        AgentTarget(
            adapter="native",
            bundle_id="workspace@2026-03-31-13-36",
            agent_id="main",
            needs_exec_image=True,
            description="the native KDCube ReAct adapter",
            required_operations=("execute_code_python", "web_search", "web_fetch"),
        )
    )
