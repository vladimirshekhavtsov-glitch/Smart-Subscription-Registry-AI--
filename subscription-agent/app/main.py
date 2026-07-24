from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


_REQUIRED_KEY_BY_PROVIDER = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _check_api_key() -> None:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    required_key = _REQUIRED_KEY_BY_PROVIDER.get(provider)

    if required_key is None:
        print(
            f"ERROR: unknown LLM_PROVIDER '{provider}'. Supported values: "
            f"{', '.join(_REQUIRED_KEY_BY_PROVIDER)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.getenv(required_key):
        print(
            f"ERROR: LLM_PROVIDER is '{provider}' but {required_key} is not set. "
            "Copy .env.example to .env and fill in your key (see README for details).",
            file=sys.stderr,
        )
        sys.exit(1)


def run_question(executor, question: str) -> None:
    print(f"\nQuestion: {question}")
    print("-" * 60)
    result = executor.invoke({"input": question})
    print("=" * 60)
    print(f"FINAL ANSWER: {result['output']}")
    print("=" * 60)


def main() -> None:
    load_dotenv()
    _check_api_key()

    from .agent import build_agent_executor

    executor = build_agent_executor()

    cli_question = " ".join(sys.argv[1:]).strip()
    env_question = os.getenv("AGENT_QUESTION", "").strip()
    question = cli_question or env_question

    if question:
        run_question(executor, question)
        return

    print("Smart Subscription Registry — интерактивный режим.")
    print("Задайте вопрос о ваших подписках и платежах (или 'exit' для выхода).\n")
    while True:
        try:
            question = input("Ваш вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "выход"}:
            break
        run_question(executor, question)


if __name__ == "__main__":
    main()
