"""Tests for CallerSimulator: deterministic fact-reveal and turn structure."""

import asyncio

from hinglish_bench.personas import PERSONAS
from hinglish_bench.providers.base import ChatMessage, ChatRequest, Role
from hinglish_bench.providers.mock import MockProvider
from hinglish_bench.schemas import HiddenFact, Scenario, Turn
from hinglish_bench.simulator import CallerSimulator, _keyword_matches

# ------------------------------------------------------------------ #
# _keyword_matches — fast keyword reveal logic                         #
# ------------------------------------------------------------------ #


def _fact(key: str, value: str, reveal_if: list[str]) -> HiddenFact:
    return HiddenFact(key=key, value=value, reveal_if=reveal_if)


def test_reveal_when_phrase_present() -> None:
    fact = _fact("order_id", "ORD-1", ["order id", "order number"])
    assert _keyword_matches(fact, "could you share your order id please?") is True


def test_reveal_case_insensitive() -> None:
    fact = _fact("order_id", "ORD-1", ["order id"])
    assert _keyword_matches(fact, "YOUR ORDER ID?") is True


def test_no_reveal_when_phrase_absent() -> None:
    fact = _fact("order_id", "ORD-1", ["order id"])
    assert _keyword_matches(fact, "how can I help you today?") is False


def test_empty_reveal_if_never_reveals() -> None:
    fact = _fact("secret", "xyz", [])
    assert _keyword_matches(fact, "tell me your secret order id number") is False


# ------------------------------------------------------------------ #
# CallerSimulator — integration with MockProvider                      #
# ------------------------------------------------------------------ #


def _make_scenario() -> Scenario:
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / "scenarios" / "refund_torn_kurta.yaml"
    return Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_next_turn_returns_caller_turn() -> None:
    scenario = _make_scenario()
    persona = PERSONAS["english"]
    mock = MockProvider(lambda req: "I need a refund please.")
    role = Role(name="caller", provider=mock, model="mock")
    sim = CallerSimulator(scenario, persona, role, run_index=0)

    turn, pt, ct = asyncio.run(sim.next_turn([]))
    assert turn.speaker == "caller"
    assert turn.text == "I need a refund please."
    assert pt > 0
    assert ct > 0


def test_fact_revealed_after_agent_asks() -> None:
    scenario = _make_scenario()
    persona = PERSONAS["english"]

    # Capture only caller-simulation calls (messages[0].role == "system").
    caller_prompts: list[str] = []

    def responder(req: ChatRequest) -> str:
        if req.messages[0].role == "system":
            caller_prompts.append(req.messages[0].content)
        return "My order ID is ORD-88412."

    mock = MockProvider(responder)
    role = Role(name="caller", provider=mock, model="mock")
    sim = CallerSimulator(scenario, persona, role)

    # First turn: no agent message yet — nothing revealed.
    asyncio.run(sim.next_turn([]))
    assert "order_id: ORD-88412" not in caller_prompts[0]

    # Second turn: agent asked for "order id" — keyword matches order_id.
    agent_turn = Turn(speaker="agent", text="Could you share your order id please?")
    asyncio.run(sim.next_turn([agent_turn]))
    # The caller system prompt should now contain the revealed fact.
    assert "order_id: ORD-88412" in caller_prompts[1]


def test_already_revealed_fact_is_not_re_triggered() -> None:
    scenario = _make_scenario()
    persona = PERSONAS["english"]
    mock = MockProvider(lambda req: "ok")
    role = Role(name="caller", provider=mock, model="mock")
    sim = CallerSimulator(scenario, persona, role)

    # Trigger reveal once.
    agent_ask = Turn(speaker="agent", text="Please share your order number.")
    asyncio.run(sim.next_turn([agent_ask]))
    count_before = len(sim._revealed)

    # Trigger again — count must not increase.
    asyncio.run(sim.next_turn([agent_ask, Turn(speaker="caller", text="ORD-88412"), agent_ask]))
    assert len(sim._revealed) == count_before


def test_system_prompt_contains_persona_text() -> None:
    scenario = _make_scenario()
    persona = PERSONAS["hinglish"]
    captured: list[ChatMessage] = []

    def responder(req: ChatRequest) -> str:
        captured.extend(req.messages)
        return "x"

    role = Role(name="caller", provider=MockProvider(responder), model="mock")
    sim = CallerSimulator(scenario, persona, role)
    asyncio.run(sim.next_turn([]))
    system_content = captured[0].content
    # Persona system prompt text should be embedded.
    assert "code-switch" in system_content.lower() or "hinglish" in system_content.lower()


def test_classifier_fallback_reveals_fact_when_keywords_miss() -> None:
    """When keyword matching fails but the classifier returns YES, the fact is revealed."""
    scenario = _make_scenario()
    persona = PERSONAS["english"]
    call_log: list[ChatRequest] = []

    def responder(req: ChatRequest) -> str:
        call_log.append(req)
        # Caller turn: just acknowledge
        if req.messages[0].role == "system":
            return "Got it."
        # Classifier call: single user message asking YES/NO → answer YES
        return "YES"

    role = Role(name="caller", provider=MockProvider(responder), model="mock")
    sim = CallerSimulator(scenario, persona, role)

    # Agent turn uses a Hinglish phrase that has NO keyword match
    hinglish_ask = Turn(speaker="agent", text="aapka order number kya hai?")
    asyncio.run(sim.next_turn([hinglish_ask]))

    # The fact must be revealed despite no keyword match
    assert "order_id" in sim._revealed


def test_classifier_not_called_for_keyword_matched_fact() -> None:
    """When keyword matching succeeds for a fact, the classifier is skipped for that fact.

    The scenario has 2 facts (order_id, upi_id). "order id" keyword matches order_id,
    so no classifier call for it. upi_id has no matching keyword, so one classifier call
    fires for it — returning "NO" (responder returns "ok", not "YES"), leaving upi_id
    unrevealed. Total classifier calls: 1 (only for upi_id).
    """
    scenario = _make_scenario()
    persona = PERSONAS["english"]
    call_log: list[ChatRequest] = []

    def responder(req: ChatRequest) -> str:
        call_log.append(req)
        return "ok"  # classifier returns non-YES → upi_id stays unrevealed

    role = Role(name="caller", provider=MockProvider(responder), model="mock")
    sim = CallerSimulator(scenario, persona, role)

    english_ask = Turn(speaker="agent", text="Could you please share your order id?")
    asyncio.run(sim.next_turn([english_ask]))

    caller_calls = [r for r in call_log if r.messages[0].role == "system"]
    classifier_calls = [r for r in call_log if r.messages[0].role != "system"]

    # One caller turn; one classifier call for upi_id (no keyword matched "order id")
    assert len(caller_calls) == 1
    assert len(classifier_calls) == 1
    # order_id was revealed (keyword matched); upi_id was not (classifier returned "ok")
    assert sim._revealed.get("order_id") == "ORD-88412"
    assert "upi_id" not in sim._revealed
