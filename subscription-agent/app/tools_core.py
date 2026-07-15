"""Pure business logic behind the agent's two tools.

This module intentionally has **no dependency on LangChain**. Keeping the
business logic framework-agnostic makes it trivial to unit-test (see
``tests/test_tools_core.py``) and means the logic can be reused unchanged if
the agent framework is ever swapped out (e.g. for LlamaIndex).

The LangChain ``StructuredTool`` wrappers that expose these functions to the
agent live in ``app/tools.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import requests

DATA_PATH = Path(__file__).parent / "data" / "obligations.json"
# frankfurter.app (the old domain referenced in the task brief) was retired.
# The current API lives at frankfurter.dev. Its v1/latest endpoint no longer
# supports from/to/amount filtering (it just returns the full EUR-based rate
# table), so we use the v2 single-pair endpoint instead and multiply by
# `amount` ourselves — this matches frankfurter's own docs, which state there
# is intentionally no server-side "conversion" endpoint anymore.
FRANKFURTER_RATE_URL = "https://api.frankfurter.dev/v2/rate/{base}/{quote}"
REQUEST_TIMEOUT_SECONDS = 10


def load_obligations(data_path: Path = DATA_PATH) -> list[dict[str, Any]]:
    """Load the raw list of obligation records from the local JSON fixture."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_obligations(
    status: Optional[str] = None,
    category: Optional[str] = None,
    data_path: Path = DATA_PATH,
) -> list[dict[str, Any]]:
    """Return the user's financial obligations, optionally filtered.

    Args:
        status: Case-insensitive exact match on the ``status`` field
            (e.g. "active", "paused", "cancelled"). ``None`` means no filter.
        category: Case-insensitive exact match on the ``category`` field
            (e.g. "subscription", "utility", "insurance"). ``None`` means no
            filter.
        data_path: Override for where the fixture lives (used by tests).

    Returns:
        A list of obligation record dicts matching the given filters.
    """
    records = load_obligations(data_path)

    if status is not None:
        records = [r for r in records if r["status"].lower() == status.lower()]

    if category is not None:
        records = [r for r in records if r["category"].lower() == category.lower()]

    return records


class CurrencyConversionError(Exception):
    """Raised when the exchange-rate API cannot be reached or returns
    something we can't parse. Callers that talk to the LLM should NOT let
    this propagate as an uncaught exception into the agent's Final Answer;
    it should be surfaced as an explicit, honest error the agent can relay
    to the user (see ``app/tools.py``)."""


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    session: Optional[requests.Session] = None,
) -> float:
    """Convert ``amount`` from one currency to another using frankfurter.dev.

    Args:
        amount: The amount to convert.
        from_currency: 3-letter ISO currency code to convert from.
        to_currency: 3-letter ISO currency code to convert to.
        session: Optional ``requests.Session`` (injected in tests to avoid
            real network calls).

    Returns:
        The converted amount as a float.

    Raises:
        CurrencyConversionError: if the currencies are invalid, the API is
            unreachable, the API returns a non-2xx response, or the response
            body doesn't contain the expected rate.
    """
    from_currency = (from_currency or "").upper().strip()
    to_currency = (to_currency or "").upper().strip()

    if not from_currency or not to_currency:
        raise CurrencyConversionError("from_currency and to_currency are required")

    # frankfurter has no route for identical base/quote currencies.
    if from_currency == to_currency:
        return float(amount)

    http = session or requests
    url = FRANKFURTER_RATE_URL.format(base=from_currency, quote=to_currency)
    try:
        response = http.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        rate = payload["rate"]
    except requests.exceptions.RequestException as exc:
        raise CurrencyConversionError(
            f"Could not reach the currency exchange API ({from_currency}->{to_currency}): {exc}"
        ) from exc
    except (KeyError, ValueError, TypeError) as exc:
        raise CurrencyConversionError(
            f"Unexpected response from the currency exchange API for "
            f"{from_currency}->{to_currency}: {exc}"
        ) from exc

    return float(amount) * float(rate)
