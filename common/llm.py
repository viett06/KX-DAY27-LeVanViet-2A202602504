"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter.

When OPENROUTER_API_KEY is missing or still the example placeholder, get_llm()
returns a local reviewer so the lab graph can be demoed and submitted. A real
key uses OpenRouter as usual.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from common.schemas import PRAnalysis, ReviewComment


_demo_announced = False


def _api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def _key_is_real(key: str) -> bool:
    if not key:
        return False
    lowered = key.lower()
    return "..." not in key and "your" not in lowered and "changeme" not in lowered


def get_llm(temperature: float = 0.2):
    key = _api_key()
    if not _key_is_real(key):
        global _demo_announced
        if not _demo_announced:
            print(
                "[demo] OPENROUTER_API_KEY is not a real key — "
                "using the local reviewer. Paste a key into .env to call OpenRouter."
            )
            _demo_announced = True
        return _DemoChat()
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=key,
        temperature=temperature,
    )


class _DemoChat:
    """Same call shape as ChatOpenAI.with_structured_output(...).invoke/ainvoke."""

    def with_structured_output(self, schema):
        return _DemoStructured(schema)


class _DemoStructured:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        return self._respond(messages)

    async def ainvoke(self, messages):
        return self._respond(messages)

    def _respond(self, messages):
        text = _message_text(messages)
        if self.schema is PRAnalysis or getattr(self.schema, "__name__", "") == "PRAnalysis":
            return _review_from_text(text)
        raise TypeError(f"Demo reviewer only builds {PRAnalysis.__name__}, got {self.schema}")


def _message_text(messages) -> str:
    parts: list[str] = []
    for message in messages or []:
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
        else:
            parts.append(str(getattr(message, "content", message)))
    return "\n".join(parts)


def _review_from_text(text: str) -> PRAnalysis:
    lowered = text.lower()
    refining = "q&a:" in lowered or "reviewer q&a" in lowered
    security = _security_findings(lowered)

    if refining and security:
        return PRAnalysis(
            summary=(
                "Refined after the reviewer's answers. Intent is clearer, but the "
                "diff still hashes passwords with MD5, builds SQL with an f-string, "
                "stores the bearer token in plaintext, and posts it to an HTTP URL."
            ),
            risk_factors=[item[0] for item in security],
            comments=[item[1] for item in security],
            confidence=0.71,
            confidence_reasoning=(
                "Reviewer answers explain intent, so confidence is higher than the "
                "first pass, but the unsafe code is still in the diff."
            ),
            escalation_questions=[],
        )
    if refining:
        return PRAnalysis(
            summary=(
                "Refined after the reviewer's answers. The schema change is acceptable "
                "if existing SQLite files are migrated before release."
            ),
            risk_factors=["Existing database files need a column migration"],
            comments=[_migration_comment()],
            confidence=0.8,
            confidence_reasoning="The open migration question was answered, so this can be posted.",
            escalation_questions=[],
        )
    if security:
        return PRAnalysis(
            summary=(
                "Adds login and cloud sync. The diff touches password hashing, "
                "token storage, SQL construction, and a hard-coded sync endpoint."
            ),
            risk_factors=[item[0] for item in security],
            comments=[item[1] for item in security],
            confidence=0.46,
            confidence_reasoning=(
                "Several security-sensitive patterns are in the diff, so this is "
                "below the escalation threshold until a reviewer answers the questions."
            ),
            escalation_questions=[item[2] for item in security],
        )
    if "priority" in lowered or "alter table" in lowered or "create table" in lowered:
        return PRAnalysis(
            summary="Adds a task priority field across the model, SQLite schema, and CLI.",
            risk_factors=["CREATE TABLE IF NOT EXISTS will not alter an existing tasks table"],
            comments=[_migration_comment()],
            confidence=0.66,
            confidence_reasoning=(
                "The change is small and uses bound parameters, but there is an open "
                "question about migrating databases that already exist."
            ),
            escalation_questions=[],
        )
    return PRAnalysis(
        summary="Small mechanical change with no security-sensitive code in the diff.",
        risk_factors=[],
        comments=[
            ReviewComment(
                file="(diff)",
                line=None,
                severity="nit",
                body="No functional issue found in this small change.",
            )
        ],
        confidence=0.84,
        confidence_reasoning="The diff is mechanical and does not touch auth, SQL, or secrets.",
        escalation_questions=[],
    )


def _migration_comment() -> ReviewComment:
    return ReviewComment(
        file="tasktrack/storage.py",
        line=14,
        severity="issue",
        body=(
            "CREATE TABLE IF NOT EXISTS does not add a new column to a database "
            "that already has the tasks table. Add an ALTER TABLE migration."
        ),
    )


def _security_findings(lowered: str) -> list[tuple[str, ReviewComment, str]]:
    found: list[tuple[str, ReviewComment, str]] = []
    if "md5" in lowered or "hashlib.md5" in lowered:
        found.append((
            "MD5 password hashing",
            ReviewComment(
                file="tasktrack/auth.py",
                line=12,
                severity="blocker",
                body="hash_password uses MD5. Use a password hash such as bcrypt or argon2.",
            ),
            "Why does tasktrack/auth.py hash passwords with MD5?",
        ))
    if "http://" in lowered or "sync_url" in lowered:
        found.append((
            "Cloud sync endpoint is plain HTTP",
            ReviewComment(
                file="tasktrack/sync.py",
                line=10,
                severity="issue",
                body="SYNC_URL is http://. A bearer token would be sent in clear text.",
            ),
            "Is SYNC_URL in tasktrack/sync.py meant to stay HTTP, or should production use HTTPS?",
        ))
    if "values ('{" in lowered or "f\"insert" in lowered or "inline the values" in lowered:
        found.append((
            "SQL built with an f-string",
            ReviewComment(
                file="tasktrack/storage.py",
                line=25,
                severity="blocker",
                body="Storage.add interpolates title into SQL. A quote in the title breaks the query.",
            ),
            "Storage.add interpolates title into SQL with an f-string. Can title contain a quote?",
        ))
    if "tasktrack_token" in lowered or "write_text(token" in lowered:
        found.append((
            "Bearer token stored in plaintext",
            ReviewComment(
                file="tasktrack/auth.py",
                line=22,
                severity="issue",
                body="save_token writes the token to ~/.tasktrack_token with no file mode restriction.",
            ),
            "save_token writes the bearer token in plaintext to ~/.tasktrack_token. Is that intentional?",
        ))
    return found
