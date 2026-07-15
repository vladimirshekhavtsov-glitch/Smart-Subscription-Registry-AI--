"""Console logger for the agent's reasoning loop.

The task requires that every Thought / Action / Observation step be visible
in the console. Thought/Action come from LangChain's own callback hooks
(``on_agent_action`` / ``on_agent_finish``) — this is the extension point
the framework provides for this, rather than hand-rolling the ReAct loop.

The Observation line is intentionally printed elsewhere, directly inside the
tool functions in ``app/tools.py``: with some tool-calling model/provider
combinations, ``on_tool_end`` was observed to silently never fire (even
though ``on_agent_action``/``on_agent_finish`` fire reliably), which would
silently violate the "every step visible" requirement. Printing at the tool
call site is not dependent on that callback wiring at all.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler


def _format_tool_input(tool_input: Any) -> str:
    if isinstance(tool_input, (dict, list)):
        return json.dumps(tool_input, ensure_ascii=False)
    return str(tool_input)


class ReActConsoleLogger(BaseCallbackHandler):
    """Prints a classic Thought / Action / Action Input / Observation trace."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        thought = (action.log or "").strip()
        # `action.log` from a tool-calling agent already includes the
        # model's own text plus a generated "Invoking: `tool` with `{...}`"
        # note. We print it as the "Thought" and then the structured
        # Action/Action Input for readability.
        if thought:
            print(f"Thought: {thought}")
        print(f"Action: {action.tool}")
        print(f"Action Input: {_format_tool_input(action.tool_input)}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        # Safety net only: our tools catch their own errors and return a
        # JSON {"error": ...} payload rather than raising, so this should
        # not normally trigger — but if some unexpected exception does
        # escape a tool, we still want it visible in the trace.
        print(f"Observation: [tool raised an unexpected error] {error}\n")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        thought = (finish.log or "").strip()
        if thought:
            print(f"Thought: {thought}")
        output = finish.return_values.get("output", "")
        print(f"Final Answer: {output}\n")
