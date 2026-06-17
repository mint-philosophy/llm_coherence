"""OpenRouter spend monitoring used by model experiment runners."""

from __future__ import annotations

import sys

from llm_coherence.runtime.api_keys import load_api_key

OPENROUTER_AUTH_URL = "https://openrouter.ai/api/v1/auth/key"
DEFAULT_THRESHOLDS = [0.50, 0.65, 0.75, 0.85, 0.90, 0.95]


async def check_openrouter_usage() -> dict | None:
    """Query OpenRouter for current usage and limit."""
    api_key = load_api_key("openrouter")
    if not api_key:
        print(
            "  [budget] No OpenRouter API key found - budget monitoring disabled.",
            file=sys.stderr,
        )
        return None
    try:
        import httpx
    except ImportError:
        print(
            "  [budget] httpx is not installed - budget monitoring disabled.",
            file=sys.stderr,
        )
        return None
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                OPENROUTER_AUTH_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                usage = data.get("usage_monthly", data.get("usage", 0))
                return {"usage": usage, "limit": data.get("limit")}
            print(
                f"  [budget] OpenRouter API returned status {resp.status_code}; check skipped.",
                file=sys.stderr,
            )
    except Exception as exc:
        print(
            f"  [budget] Budget check failed ({type(exc).__name__}: {exc}); continuing.",
            file=sys.stderr,
        )
    return None


class BudgetMonitor:
    """Tracks OpenRouter spend and warns at configured thresholds."""

    def __init__(
        self,
        check_interval: int = 5,
        thresholds: list[float] | None = None,
        auto_stop_at: float = 0.95,
    ):
        self.check_interval = check_interval
        self.thresholds = sorted(thresholds or DEFAULT_THRESHOLDS)
        self.auto_stop_at = auto_stop_at
        self._tasks_since_check = 0
        self._alerted: set[float] = set()
        self._check_failures = 0
        self.should_stop = False
        self.last_usage: float | None = None
        self.last_limit: float | None = None

    async def on_task_completed(self) -> None:
        self._tasks_since_check += 1
        if self._tasks_since_check < self.check_interval:
            return
        self._tasks_since_check = 0
        await self._check()

    async def force_check(self) -> None:
        self._tasks_since_check = 0
        await self._check()

    async def _check(self) -> None:
        info = await check_openrouter_usage()
        if info is None:
            self._check_failures += 1
            if self._check_failures >= 3:
                print(
                    f"\n  [budget] {self._check_failures} consecutive budget checks failed. "
                    "Spend monitoring is not active.",
                    file=sys.stderr,
                )
            return

        self._check_failures = 0
        if info["limit"] is None:
            if self.last_limit is None:
                print(
                    "  [budget] No spend limit set on this key - budget auto-stop is disabled.",
                    file=sys.stderr,
                )
            return

        usage = info["usage"]
        limit = info["limit"]
        self.last_usage = usage
        self.last_limit = limit
        pct = usage / limit if limit > 0 else 0

        for threshold in self.thresholds:
            if pct >= threshold and threshold not in self._alerted:
                self._alerted.add(threshold)
                print(f"\n{'=' * 60}")
                print(
                    f"  BUDGET WARNING: {pct:.1%} of limit used "
                    f"(${usage:.2f} / ${limit:.2f})"
                )
                print(f"{'=' * 60}\n")

        if pct >= self.auto_stop_at and not self.should_stop:
            self.should_stop = True
            print("  Approaching limit - stopping after current tasks finish.")
            print("  Progress is checkpointed. Resume with --resume after limit increase.\n")

    def summary(self) -> str:
        if self.last_usage is not None and self.last_limit is not None:
            pct = self.last_usage / self.last_limit if self.last_limit > 0 else 0
            return f"${self.last_usage:.2f} / ${self.last_limit:.2f} ({pct:.1%})"
        return "unknown (no check performed)"

