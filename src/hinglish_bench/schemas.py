"""Pydantic models for scenarios, personas, and recorded runs."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Domain = Literal["refund", "emi_reminder", "delivery"]
PersonaId = Literal["english", "hindi_devanagari", "hindi_roman", "hinglish", "switcher"]
Trait = Literal["impatient", "typos", "vague"]


class HiddenFact(BaseModel):
    """Something the caller knows but reveals only when the agent asks for it.

    `reveal_if` is a list of lowercase substrings. If any of them appears in the
    agent's last message (case-insensitive), the fact is revealed to the caller.
    This is deterministic and needs no extra LLM call.
    """

    key: str = Field(pattern=r"^[a-z_]+$", description="Fixed fact key, e.g. order_id")
    value: str
    reveal_if: list[str] = Field(default_factory=list)


class ExpectedToolCall(BaseModel):
    """A tool call the agent must make. `args` is a subset match against the real call."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ForbiddenRule(BaseModel):
    """More than `max_count` calls to `name` is a failure, e.g. refunding twice."""

    name: str
    max_count: int = 0


class Scenario(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    domain: Domain
    caller_goal: str
    hidden_facts: list[HiddenFact] = Field(default_factory=list)
    # Order of this list is the order the grader checks.
    expected_tool_calls: list[ExpectedToolCall]
    # (a, b): the first call to `a` must come before the first call to `b`.
    precedence: list[tuple[str, str]] = Field(default_factory=list)
    forbidden: list[ForbiddenRule] = Field(default_factory=list)
    # Partial match against the mock DB. Each top-level key is a table. A list value
    # passes if every item in it matches some row of the table.
    success_state: dict[str, Any]
    max_turns: int = Field(default=12, ge=1, le=40)

    @model_validator(mode="after")
    def _unique_fact_keys(self) -> "Scenario":
        keys = [f.key for f in self.hidden_facts]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate hidden_facts keys in {self.id}")
        return self


class Persona(BaseModel):
    id: PersonaId
    system_prompt: str
    traits: list[Trait] = Field(default_factory=list)
    example_lines: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    speaker: Literal["caller", "agent"]
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class RunRecord(BaseModel):
    """One line of results.jsonl. Infra errors are kept apart from model failures."""

    scenario_id: str
    persona_id: PersonaId
    model: str
    run_index: int = Field(ge=0)
    turns: list[Turn] = Field(default_factory=list)
    final_db: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    infra_error: str | None = None
    # Set by the runner; Stage 3 graders never need to re-derive this.
    end_reason: Literal["success", "escalated", "max_turns", "infra_error"] | None = None
