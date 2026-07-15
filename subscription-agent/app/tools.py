from __future__ import annotations

import json
from typing import Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from . import tools_core


def _print_observation(result: str) -> str:
    """Print the Thought/Action/Observation trace's Observation line and
    hand the result straight through.
    """
    print(f"Observation: {result}\n", flush=True)
    return result


class GetObligationsInput(BaseModel):
    status: Optional[str] = Field(
        default=None,
        description="Filter by status: 'active', 'paused', or 'cancelled'. Omit for no filter.",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Filter by category, e.g. 'subscription', 'utility', 'insurance', "
            "'membership', 'loan'. Omit for no filter."
        ),
    )


def _get_obligations(status: Optional[str] = None, category: Optional[str] = None) -> str:
    records = tools_core.get_obligations(status=status, category=category)
    return _print_observation(json.dumps(records, ensure_ascii=False))


class ConvertCurrencyInput(BaseModel):
    amount: float = Field(description="The numeric amount to convert.")
    from_currency: str = Field(description="3-letter source currency code, e.g. 'USD'.")
    to_currency: str = Field(description="3-letter target currency code, e.g. 'RUB'.")


def _convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    try:
        converted = tools_core.convert_currency(amount, from_currency, to_currency)
    except tools_core.CurrencyConversionError as exc:
        return _print_observation(json.dumps({"error": str(exc)}, ensure_ascii=False))

    return _print_observation(
        json.dumps(
            {
                "amount": amount,
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
                "converted_amount": converted,
            },
            ensure_ascii=False,
        )
    )


def get_tools() -> list[StructuredTool]:
    """Build the two agent tools."""
    return [
        StructuredTool.from_function(
            func=_get_obligations,
            name="get_obligations",
            description=(
                "Return the user's financial obligations (subscriptions, bills, "
                "loans, etc.) as a JSON array. Each record has fields: id, title, "
                "amount, currency, category, next_payment_date (YYYY-MM-DD), status. "
                "Optionally filter by status and/or category. Always call this "
                "before answering any question about spending, dates, or categories."
            ),
            args_schema=GetObligationsInput,
        ),
        StructuredTool.from_function(
            func=_convert_currency,
            name="convert_currency",
            description=(
                "Convert an amount from one currency to another using live "
                "exchange rates (frankfurter.dev). Returns JSON with "
                "'converted_amount', or a JSON object with an 'error' key if the "
                "conversion could not be performed — in that case do NOT guess a "
                "number, tell the user the conversion failed."
            ),
            args_schema=ConvertCurrencyInput,
        ),
    ]
