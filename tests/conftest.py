import pytest

from researchcrew.config import Settings
from researchcrew.models import Evidence


class FakeRetriever:
    """Returns distinct evidence per query so de-duplication does not hide behaviour."""
    def __init__(self, empty_for: tuple[str, ...] = ()):
        self.calls: list[str] = []
        self.empty_for = empty_for

    def retrieve(self, query):
        self.calls.append(query)
        if any(tok in query for tok in self.empty_for):
            return []
        return [Evidence("web", f"Page about {query}", f"https://example.com/{abs(hash(query)) % 9999}",
                         f"Detailed passage about {query} with facts.", 0.9),
                Evidence("pdf", "paper.pdf", f"paper.pdf, p.{len(query) % 7 + 1}",
                         f"Paper passage on {query}.", 0.8)]


class ScriptedRunner:
    """Replaces the LLM: answers by role from a queue of scripted responses."""
    def __init__(self, scripts: dict[str, list[str]]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.log: list[tuple[str, str]] = []

    def run(self, role, description, expected_output):
        self.log.append((role, description))
        queue = self.scripts[role]
        return queue.pop(0) if len(queue) > 1 else queue[0]


@pytest.fixture
def settings():
    return Settings(max_iterations=2, max_subquestions=3, sufficiency_score=8.0)
