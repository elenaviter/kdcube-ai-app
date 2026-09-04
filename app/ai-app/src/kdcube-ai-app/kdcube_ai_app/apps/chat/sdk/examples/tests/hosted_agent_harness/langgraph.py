#!/usr/bin/env python3
"""Run the common Agent Harness acceptance scenario through LangGraph."""

from kdcube_ai_app.apps.chat.sdk.examples.tests.hosted_agent_harness.demo import main_for
from kdcube_ai_app.apps.chat.sdk.examples.tests.hosted_agent_harness.runtime_client import AgentTarget


if __name__ == "__main__":
    main_for(
        AgentTarget(
            adapter="langgraph",
            bundle_id="ported-langgraph-agents@2026-07-13",
            agent_id="lg-react",
            needs_exec_image=True,
            description="the hosted LangGraph create_agent adapter",
            required_operations=("run_python", "web_search", "web_fetch"),
        )
    )
