from __future__ import annotations

import os
from datetime import date, timedelta

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .logging_callback import ReActConsoleLogger
from .tools import get_tools

# Default per-provider model. Overridable via GROQ_MODEL / OPENAI_MODEL env vars.
_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}


def _build_llm(provider: str, model_name: str | None, temperature: float) -> BaseChatModel:
    """Instantiate the chat model for the requested provider.
    """
    if provider == "groq":
        from langchain_groq import ChatGroq

        resolved_model = model_name or os.getenv("GROQ_MODEL", _DEFAULT_MODELS["groq"])
        return ChatGroq(model=resolved_model, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        resolved_model = model_name or os.getenv("OPENAI_MODEL", _DEFAULT_MODELS["openai"])
        return ChatOpenAI(model=resolved_model, temperature=temperature)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Supported values: 'groq', 'openai'."
    )

SYSTEM_PROMPT = """\
You are the AI core of "Smart Subscription Registry" — an assistant that helps a \
user understand their recurring payments and subscriptions.

Today's date is {today} ({today_weekday}).

The following date ranges are already computed for you — use these exact \
boundaries, do NOT compute date ranges yourself, your own date arithmetic is \
not reliable enough for this:
- "This week" (current Monday-to-Sunday calendar week) = {week_start} through {week_end}, inclusive.
- "The next 30 days" = {today} through {next_30_days_end}, inclusive.
- "This month" = {month_start} through {month_end}, inclusive.

You have access to tools that let you read the user's financial obligations \
and convert amounts between currencies. Follow these rules strictly:

1. Always call `get_obligations` before answering any question about spending, \
   dates, categories, or statuses. Never invent obligations that weren't \
   returned by the tool. For any question about spending amounts, totals, or \
   "which category/obligation costs the most" (i.e. anything about ongoing \
   cost), only include obligations with status "active" — a "paused" or \
   "cancelled" obligation is not currently costing the user money and must \
   be excluded from sums and comparisons by default. Only include non-active \
   obligations if the user explicitly asks about paused/cancelled items, or \
   explicitly asks about a specific status.
2. Obligations are stored in their original currency. Before summing or \
   comparing amounts across records, convert every amount into a single \
   target currency using `convert_currency` (default to RUB unless the user \
   asks for a different currency). IMPORTANT: if an obligation is ALREADY in \
   the target currency, do not call `convert_currency` for it — but you MUST \
   still include its original amount in any running total or comparison. \
   Never silently drop an obligation from a sum just because it didn't \
   require a conversion call. When grouping obligations by category, use \
   each obligation's `category` field value exactly as returned — do not \
   guess, rename, or merge it into a different category. Before giving your \
   final answer for any sum-by-category or sum-by-date-range question, list \
   out, for EVERY obligation you are including, its title, amount in the \
   target currency, and category/date — so it's possible to verify none \
   were skipped or mis-grouped.
3. When filtering by date, use the pre-computed date ranges given above — \
   never invent your own start/end dates. You MUST check every single \
   obligation returned by `get_obligations` one by one — do not eyeball the \
   list. In your reasoning, explicitly go record by record: state each \
   obligation's title and `next_payment_date`, say whether it falls inside \
   the relevant range above (a simple string/date comparison against the \
   given boundaries, not free-form date math), then only after checking \
   every record move to your conclusion. Skipping a record without checking \
   it is not acceptable, even if the list is long.
4. If a tool call returns a JSON object containing an "error" key, or a \
   currency conversion cannot be completed, do NOT guess, estimate, or make \
   up a number. Clearly tell the user in your final answer that you could \
   not complete the calculation and briefly why.
5. Before every tool call, briefly state (one sentence) what you are about \
   to do and why — this is your "Thought" and will be shown in the console \
   trace.
6. Show your reasoning briefly, then give a clear, concise final answer with \
   the concrete numbers/currency used. Answer in the same language the user \
   asked in.
7. Never fabricate data you don't have. If the available tools genuinely \
   can't answer the question, say so explicitly instead of guessing.
"""


def _compute_date_context(today: date | None = None) -> dict[str, str]:
    """Pre-compute every date boundary the system prompt needs in Python.
    """
    import calendar

    today = today or date.today()
    week_start = today - timedelta(days=today.weekday())  # Monday
    week_end = week_start + timedelta(days=6)  # Sunday
    next_30_days_end = today + timedelta(days=30)
    month_start = today.replace(day=1)
    last_day = calendar.monthrange(today.year, today.month)[1]
    month_end = today.replace(day=last_day)

    return {
        "today": today.isoformat(),
        "today_weekday": today.strftime("%A"),
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "next_30_days_end": next_30_days_end.isoformat(),
        "month_start": month_start.isoformat(),
        "month_end": month_end.isoformat(),
    }


def build_agent_executor(
    provider: str | None = None,
    model_name: str | None = None,
    temperature: float = 0,
    verbose_raw_langchain: bool = False,
) -> AgentExecutor:
    """Construct the AgentExecutor used to answer user questions.
    """
    resolved_provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    llm = _build_llm(resolved_provider, model_name, temperature)

    tools = get_tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    ).partial(**_compute_date_context())

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[ReActConsoleLogger()],
        verbose=verbose_raw_langchain,
        handle_parsing_errors=True,
        max_iterations=8,
    )
