"""Caller simulator: an LLM playing the caller, with deterministic fact-reveal logic."""

from hinglish_bench.providers.base import ChatMessage, Role
from hinglish_bench.schemas import HiddenFact, Persona, Scenario, Turn


class CallerSimulator:
    """Simulates one caller for one (scenario, persona, run) triple.

    Control flow is fully deterministic:
    - Which hidden facts to reveal is decided by substring matching, not an LLM call.
    - The call ends when the runner detects success, escalation, or max_turns.
    - The LLM is only responsible for phrasing — it never controls the flow.
    """

    def __init__(
        self,
        scenario: Scenario,
        persona: Persona,
        role: Role,
        run_index: int = 0,
    ) -> None:
        self._scenario = scenario
        self._persona = persona
        self._role = role
        self._run_index = run_index
        # key → value for facts that have been revealed so far
        self._revealed: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def next_turn(self, turns: list[Turn]) -> tuple[Turn, int, int]:
        """Return the caller's next turn plus (prompt_tokens, completion_tokens).

        Call this AFTER appending the agent's most-recent turn to `turns`.
        """
        # Step 1: deterministic fact reveal based on the last agent message.
        if turns and turns[-1].speaker == "agent":
            self._update_revealed(turns[-1].text)

        # Step 2: LLM phrases the message in persona.
        messages = self._build_messages(turns)
        resp = await self._role.chat(messages, temperature=0.7, run_index=self._run_index)
        turn = Turn(speaker="caller", text=resp.text)
        return turn, resp.prompt_tokens, resp.completion_tokens

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _update_revealed(self, agent_text: str) -> None:
        for fact in self._scenario.hidden_facts:
            if fact.key not in self._revealed and _agent_asked_for(fact, agent_text):
                self._revealed[fact.key] = fact.value

    def _build_system(self) -> str:
        revealed_lines = "\n".join(f"  - {k}: {v}" for k, v in self._revealed.items())
        unrevealed_keys = [
            f.key for f in self._scenario.hidden_facts if f.key not in self._revealed
        ]
        parts = [
            self._persona.system_prompt.strip(),
            "",
            "## Your goal",
            self._scenario.caller_goal.strip(),
            "",
            "## Facts you may share with the agent",
            revealed_lines or "  (none revealed yet — wait for the agent to ask)",
            "",
            "## Facts NOT yet revealed — share ONLY when the agent explicitly asks for them",
            (", ".join(unrevealed_keys) if unrevealed_keys else "(all facts revealed)"),
            "",
            "## Hard rules",
            "- Never say you are an AI, a simulation, or a test.",
            "- Reveal a fact ONLY if the agent asks for it and it appears in the 'may share' list.",
            "- Stay in your persona's language and style throughout.",
        ]
        if self._persona.example_lines:
            parts += ["", "## Example lines in your style"]
            parts += [f"  - {line}" for line in self._persona.example_lines]
        return "\n".join(parts)

    def _build_messages(self, turns: list[Turn]) -> list[ChatMessage]:
        messages: list[ChatMessage] = [ChatMessage(role="system", content=self._build_system())]
        for turn in turns:
            # Caller's POV: agent messages are "user", caller's own turns are "assistant".
            role = "assistant" if turn.speaker == "caller" else "user"
            messages.append(ChatMessage(role=role, content=turn.text))
        return messages


def _agent_asked_for(fact: HiddenFact, text: str) -> bool:
    """True if any of the fact's reveal_if phrases appear in the agent text (case-insensitive)."""
    lower = text.lower()
    return any(phrase in lower for phrase in fact.reveal_if)
