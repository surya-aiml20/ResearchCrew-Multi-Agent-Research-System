import json

from conftest import FakeRetriever, ScriptedRunner
from researchcrew.pipeline import ResearchPipeline

PLAN = json.dumps({"sub_questions": [
    {"id": 1, "question": "What is RAG?", "search_queries": ["rag overview"]},
    {"id": 2, "question": "What is fine tuning?", "search_queries": ["fine tuning overview"]},
]})
RESEARCH = "ANSWER: RAG retrieves passages [E1] and fine-tuning changes weights [E2, E99].\nCONFIDENCE: medium\nUNRESOLVED: costs"
CRITIC_BAD = json.dumps({"score": 5, "verdict": "insufficient", "summary": "Cost data missing.",
                         "gaps": [{"sub_question_id": 2, "issue": "no cost data",
                                   "follow_up_queries": ["fine tuning cost comparison"]}]})
CRITIC_OK = json.dumps({"score": 9, "verdict": "sufficient", "summary": "Good.", "gaps": []})
WRITER = "# Report\n## Executive Summary\nRAG is flexible [E1]. Fine-tuning is costly [E2] [E77].\n## Sources\n1. fake"


def build(settings, critic, retriever=None, plan=PLAN):
    runner = ScriptedRunner({"planner": [plan], "researcher": [RESEARCH], "critic": critic, "writer": [WRITER]})
    events = []
    pipe = ResearchPipeline(runner, retriever or FakeRetriever(), settings, lambda s, m: events.append((s, m)))
    return pipe, runner, events


def test_critic_gap_triggers_targeted_reresearch(settings):
    pipe, runner, events = build(settings, [CRITIC_BAD, CRITIC_OK])
    res = pipe.run("RAG vs fine-tuning?")
    assert res.stats["iterations"] == 1 and len(res.reviews) == 2
    assert res.reviews[0].gaps[0].follow_up_queries == ["fine tuning cost comparison"]
    assert pipe.retriever.calls[-1] == "fine tuning cost comparison"
    # sub-question 2 researched twice (initial + revision), sub-question 1 once
    assert [f.revision for f in res.findings] == [0, 1]
    assert sum(r == "researcher" for r, _ in runner.log) == 3
    assert any(stage == "critique" for stage, _ in events)


def test_stops_immediately_when_critic_satisfied(settings):
    pipe, _, _ = build(settings, [CRITIC_OK])
    res = pipe.run("q")
    assert res.stats["iterations"] == 0 and len(res.reviews) == 1


def test_loop_is_bounded_by_max_iterations(settings):
    pipe, _, _ = build(settings, [CRITIC_BAD])   # critic never satisfied
    res = pipe.run("q", max_iterations=1)
    assert res.stats["iterations"] == 1 and len(res.reviews) == 2


def test_hallucinated_citations_never_reach_final_report(settings):
    pipe, _, _ = build(settings, [CRITIC_OK])
    res = pipe.run("q")
    assert "E99" not in res.report and "E77" not in res.report and "fake" not in res.report
    assert res.stats["unique_sources"] >= 1 and "## Sources" in res.report
    assert all(f.evidence_ids for f in res.findings)


def test_unparseable_plan_falls_back_to_single_question(settings):
    pipe, _, _ = build(settings, [CRITIC_OK], plan="I refuse to output JSON")
    res = pipe.run("my question")
    assert len(res.plan) == 1 and res.plan[0].question == "my question"


def test_no_evidence_raises_automatic_gap_and_stops_without_progress(settings):
    pipe, _, _ = build(settings, [CRITIC_OK], retriever=FakeRetriever(empty_for=("fine",)))
    res = pipe.run("q")
    first = res.reviews[0]
    assert first.verdict == "insufficient" and any(g.auto for g in first.gaps)
    assert len(res.reviews) == 1          # re-search found nothing new -> loop ends, no infinite retry
    assert "No evidence" in res.findings[1].answer
