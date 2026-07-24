from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import requests

DATA_PATH = Path(__file__).parent / "data" / "obligations.json"

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

    records = load_obligations(data_path)

    if status is not None:
        records = [r for r in records if r["status"].lower() == status.lower()]

    if category is not None:
        records = [r for r in records if r["category"].lower() == category.lower()]

    return records


class CurrencyConversionError(Exception):

def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    session: Optional[requests.Session] = None,
) -> float:

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
