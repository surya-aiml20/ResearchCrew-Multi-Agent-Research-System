"""CrewAI agent definitions + a resilient single-task runner.

crewai is imported lazily so the rest of the package (and the test-suite) works without it.
"""
from __future__ import annotations

import logging
import time
from typing import Protocol

from .config import ConfigError, Settings

log = logging.getLogger(__name__)


class AgentRunner(Protocol):
    def run(self, role: str, description: str, expected_output: str) -> str: ...


class AgentError(RuntimeError):
    pass


# GPT-OSS models are *reasoning* models: hidden reasoning tokens count against max_tokens, so budgets
# are generous (otherwise the visible answer can come back empty).
AGENT_SPECS: dict[str, dict] = {
    "planner": dict(
        role="Research Planner",
        goal="Break complex questions into precise, searchable sub-questions.",
        backstory="A veteran research lead who scopes projects so nothing important is missed "
                  "and nothing is duplicated.",
        tier="fast", temperature=0.3, max_tokens=3000, effort="low"),
    "researcher": dict(
        role="Evidence-Grounded Researcher",
        goal="Answer sub-questions strictly from the supplied evidence, with precise citations.",
        backstory="A meticulous analyst who refuses to state anything she cannot trace to a source.",
        tier="smart", temperature=0.1, max_tokens=4000, effort="low"),
    "critic": dict(
        role="Research Critic",
        goal="Find evidence gaps, weak support and contradictions, and propose targeted follow-up searches.",
        backstory="A sceptical peer reviewer known for catching unsupported claims.",
        tier="smart", temperature=0.0, max_tokens=4000, effort="medium"),
    "writer": dict(
        role="Report Writer",
        goal="Turn verified findings into a clear, honest, well-structured report.",
        backstory="A senior technical editor who writes for busy decision-makers.",
        tier="smart", temperature=0.3, max_tokens=7000, effort="low"),
    "router": dict(
        role="Conversation Router",
        goal="Decide whether a chat message needs new research or can be answered from the conversation.",
        backstory="A careful dispatcher who rewrites follow-ups into standalone questions.",
        tier="fast", temperature=0.0, max_tokens=2000, effort="low"),
    "chat": dict(
        role="Research Assistant",
        goal="Answer follow-up questions about earlier reports accurately and concisely.",
        backstory="A helpful analyst who only relies on what the earlier research actually established.",
        tier="smart", temperature=0.3, max_tokens=4000, effort="low"),
}


class CrewAgentRunner:
    """Runs each pipeline step as a one-task CrewAI crew, with retry + backoff
    (Groq rate limits are the usual cause of transient failures)."""

    def __init__(self, settings: Settings, retries: int = 4) -> None:
        if not settings.groq_api_key:
            raise ConfigError("GROQ_API_KEY is not set. Add it to your .env file and restart the app.")
        from .compat import patch_litellm
        patch_litellm()   # strips CrewAI's `cache_breakpoint` key that Groq rejects
        self.settings, self.retries = settings, retries
        self._agents: dict = {}

    def _agent(self, key: str):
        if key not in self._agents:
            from crewai import LLM, Agent

            spec = AGENT_SPECS[key]
            model = self.settings.model_fast if spec["tier"] == "fast" else self.settings.model_smart
            extra = {"reasoning_effort": spec["effort"]} if "gpt-oss" in model else {}
            llm = LLM(model=model, api_key=self.settings.groq_api_key,
                      temperature=spec["temperature"], max_tokens=spec["max_tokens"], **extra)
            self._agents[key] = Agent(role=spec["role"], goal=spec["goal"], backstory=spec["backstory"],
                                      llm=llm, allow_delegation=False, verbose=False)
        return self._agents[key]

    def run(self, role: str, description: str, expected_output: str) -> str:
        from crewai import Crew, Process, Task

        agent = self._agent(role)
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                task = Task(description=description, expected_output=expected_output, agent=agent)
                crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
                out = crew.kickoff()
                text = str(getattr(out, "raw", out)).strip()
                if text:
                    return text
                raise AgentError("empty response (increase max_tokens or lower reasoning effort)")
            except Exception as exc:  # noqa: BLE001
                last = exc
                msg = str(exc).lower()
                if "model_decommissioned" in msg or "decommissioned" in msg or "does not exist" in msg:
                    raise AgentError(f"Groq model unavailable: {exc}. Check MODEL_SMART / MODEL_FAST "
                                     f"in .env (use groq/openai/gpt-oss-120b and groq/openai/gpt-oss-20b).") from exc
                if "401" in msg or "invalid api key" in msg:
                    raise AgentError("Groq rejected the API key (401). Check GROQ_API_KEY in .env.") from exc
                rate = "rate limit" in msg or "429" in msg or "rate_limit" in msg
                wait = 20 * (attempt + 1) if rate else 3 * (2 ** attempt)
                log.warning("%s attempt %d/%d failed: %s (retry in %ds)", role, attempt + 1,
                            self.retries, str(exc)[:300], wait)
                if attempt < self.retries - 1:
                    time.sleep(wait)
        raise AgentError(f"{role} failed after {self.retries} attempts: {last}")