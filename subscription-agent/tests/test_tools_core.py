"""Unit tests for the pure business logic in app/tools_core.py.

Written with the stdlib `unittest` module (no LangChain import needed here),
so these tests do not depend on any LLM/agent framework being installed —
but the test classes are still auto-discovered and run fine by `pytest`,
which is what the README recommends for day-to-day use.

Run with either:
    pytest
    python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import requests

from app import tools_core

FIXTURE_RECORDS = [
    {
        "id": "1",
        "title": "Netflix",
        "amount": 9.99,
        "currency": "USD",
        "category": "subscription",
        "next_payment_date": "2026-07-09",
        "status": "active",
    },
    {
        "id": "2",
        "title": "Disney+",
        "amount": 8.99,
        "currency": "EUR",
        "category": "subscription",
        "next_payment_date": "2026-07-11",
        "status": "paused",
    },
    {
        "id": "3",
        "title": "Аренда VPS",
        "amount": 12.00,
        "currency": "USD",
        "category": "utility",
        "next_payment_date": "2026-08-01",
        "status": "active",
    },
]


class GetObligationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_path = Path(self._tmpdir.name) / "obligations.json"
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(FIXTURE_RECORDS, f)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_no_filters_returns_all_records(self):
        result = tools_core.get_obligations(data_path=self.data_path)
        self.assertEqual(len(result), 3)

    def test_filter_by_status_is_case_insensitive(self):
        result = tools_core.get_obligations(status="ACTIVE", data_path=self.data_path)
        titles = {r["title"] for r in result}
        self.assertEqual(titles, {"Netflix", "Аренда VPS"})

    def test_filter_by_category(self):
        result = tools_core.get_obligations(category="subscription", data_path=self.data_path)
        titles = {r["title"] for r in result}
        self.assertEqual(titles, {"Netflix", "Disney+"})

    def test_filter_by_status_and_category_combined(self):
        result = tools_core.get_obligations(
            status="active", category="subscription", data_path=self.data_path
        )
        self.assertEqual([r["title"] for r in result], ["Netflix"])

    def test_no_match_returns_empty_list(self):
        result = tools_core.get_obligations(status="cancelled", data_path=self.data_path)
        self.assertEqual(result, [])

    def test_real_fixture_has_enough_diverse_records(self):
        """Sanity-check the actual shipped fixture meets the task's spec
        (10-15 records, multiple currencies and categories)."""
        result = tools_core.get_obligations()
        self.assertGreaterEqual(len(result), 10)
        self.assertLessEqual(len(result), 15)
        currencies = {r["currency"] for r in result}
        categories = {r["category"] for r in result}
        self.assertGreaterEqual(len(currencies), 2)
        self.assertGreaterEqual(len(categories), 2)


def _fake_response(json_body: dict, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} error"
        )
    else:
        response.raise_for_status.return_value = None
    return response


class ConvertCurrencyTests(unittest.TestCase):
    def test_same_currency_short_circuits_without_calling_api(self):
        session = MagicMock()
        result = tools_core.convert_currency(100, "USD", "usd", session=session)
        self.assertEqual(result, 100.0)
        session.get.assert_not_called()

    def test_successful_conversion_returns_converted_amount(self):
        session = MagicMock()
        session.get.return_value = _fake_response(
            {"date": "2026-07-08", "base": "USD", "quote": "RUB", "rate": 89.45}
        )
        result = tools_core.convert_currency(10, "USD", "RUB", session=session)
        self.assertAlmostEqual(result, 894.5)
        session.get.assert_called_once()
        (called_url,), _kwargs = session.get.call_args
        self.assertIn("/v2/rate/USD/RUB", called_url)

    def test_api_http_error_raises_conversion_error_not_silent_guess(self):
        session = MagicMock()
        session.get.return_value = _fake_response({}, status_code=503)
        with self.assertRaises(tools_core.CurrencyConversionError):
            tools_core.convert_currency(10, "USD", "RUB", session=session)

    def test_network_failure_raises_conversion_error(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError("no network")
        with self.assertRaises(tools_core.CurrencyConversionError):
            tools_core.convert_currency(10, "USD", "RUB", session=session)

    def test_malformed_response_raises_conversion_error(self):
        session = MagicMock()
        # Missing the "rate" field entirely.
        session.get.return_value = _fake_response(
            {"date": "2026-07-08", "base": "USD", "quote": "RUB"}
        )
        with self.assertRaises(tools_core.CurrencyConversionError):
            tools_core.convert_currency(10, "USD", "RUB", session=session)


if __name__ == "__main__":
    unittest.main()
