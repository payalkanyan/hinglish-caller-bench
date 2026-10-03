from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from hinglish_bench.schemas import Persona, Scenario

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((SCENARIO_DIR / name).read_text(encoding="utf-8"))
    return data


def test_sample_scenarios_load() -> None:
    paths = sorted(SCENARIO_DIR.glob("*.yaml"))
    assert len(paths) >= 2
    for path in paths:
        Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_unknown_domain_rejected() -> None:
    data = _load("refund_torn_kurta.yaml")
    data["domain"] = "loans"
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_duplicate_fact_keys_rejected() -> None:
    data = _load("refund_torn_kurta.yaml")
    data["hidden_facts"].append({"key": "order_id", "value": "ORD-1"})
    with pytest.raises(ValidationError, match="duplicate"):
        Scenario.model_validate(data)


def test_max_turns_bounds() -> None:
    data = _load("emi_due_date_shift.yaml")
    data["max_turns"] = 0
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_reveal_if_is_optional() -> None:
    """HiddenFact without reveal_if should default to empty list (no reveal)."""
    data = _load("refund_torn_kurta.yaml")
    for fact in data.get("hidden_facts", []):
        fact.pop("reveal_if", None)
    scenario = Scenario.model_validate(data)
    for fact in scenario.hidden_facts:
        assert fact.reveal_if == []


def test_reveal_if_loaded_from_yaml() -> None:
    """The updated YAML files should have reveal_if populated."""
    scenario = Scenario.model_validate(_load("refund_torn_kurta.yaml"))
    fact = next(f for f in scenario.hidden_facts if f.key == "order_id")
    assert len(fact.reveal_if) > 0
    assert "order id" in fact.reveal_if


def test_persona_id_restricted() -> None:
    with pytest.raises(ValidationError):
        Persona.model_validate({"id": "tamil", "system_prompt": "x"})
