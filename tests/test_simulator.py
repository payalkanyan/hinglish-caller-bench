"""Tests for CallerSimulator: deterministic fact-reveal and turn structure."""

import asyncio

from hinglish_bench.personas import PERSONAS
from hinglish_bench.providers.base import ChatMessage, ChatRequest, Role
from hinglish_bench.providers.mock import MockProvider
from hinglish_bench.schemas import HiddenFact, Scenario, Turn
from hinglish_bench.simulator import CallerSimulator, _agent_asked_for

# ------------------------------------------------------------------ #
# _agent_asked_for — pure reveal logic                                 #
# ------------------------------------------------------------------ #


def _fact(key: str, value: str, reveal_if: list[str]) -> HiddenFact:
    return HiddenFact(key=key, value=value, reveal_if=reveal_if)


def test_reveal_when_phrase_present() -> None:
    fact = _fact("order_id", "ORD-1", ["order id", "order number"])
    assert _agent_asked_for(fact, "could you share your order id please?") is True


def test_reveal_case_insensitive() -> None:
    fact = _fact("order_id", "ORD-1", ["order id"])
    assert _agent_asked_for(fact, "YOUR ORDER ID?") is True  # caller text is pre-lowercased


def test_no_reveal_when_phrase_absent() -> None:
    fact = _fact("order_id", "ORD-1", ["order id"])
    assert _agent_asked_for(fact, "how can I help you today?") is False


def test_empty_reveal_if_never_reveals() -> None:
    fact = _fact("secret", "xyz", [])
    assert _agent_asked_for(fact, "tell me your secret order id number") is False


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

    # The mock captures the system prompt so we can inspect revealed facts.
    captured: list[str] = []

    def responder(req: ChatRequest) -> str:
        captured.append(req.messages[0].content)
        return "My order ID is ORD-88412."

    mock = MockProvider(responder)
    role = Role(name="caller", provider=mock, model="mock")
    sim = CallerSimulator(scenario, persona, role)

    # First turn: no agent message yet — nothing revealed.
    asyncio.run(sim.next_turn([]))
    assert "order_id: ORD-88412" not in captured[0]

    # Second turn: agent asked for "order id".
    agent_turn = Turn(speaker="agent", text="Could you share your order id please?")
    asyncio.run(sim.next_turn([agent_turn]))
    # Now the system prompt should contain the revealed fact.
    assert "order_id: ORD-88412" in captured[1]


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
