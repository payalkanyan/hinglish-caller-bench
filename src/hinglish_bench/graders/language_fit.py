"""Language-fit grader: script detection + LLM judge (1–5) + export/agreement helpers."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

from hinglish_bench.providers.base import ChatMessage, Role
from hinglish_bench.schemas import RunRecord

_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "language_fit_judge.txt"

# Devanagari block: U+0900–U+097F
_DEVA_LO = 0x0900
_DEVA_HI = 0x097F
# Basic Latin A–Z / a–z + extended Latin
_LATIN_RANGES = [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)]


def detect_script(text: str) -> Literal["devanagari", "latin", "mixed", "empty"]:
    """Classify the dominant script of `text`.

    Counts Devanagari code points vs Latin code points.
    Returns "devanagari" if Devanagari > 66 % of total script chars,
    "latin" if Latin > 66 %, "mixed" if both are present below that threshold,
    and "empty" if no script characters are found.
    """
    deva = 0
    latin = 0
    for ch in text:
        cp = ord(ch)
        if _DEVA_LO <= cp <= _DEVA_HI:
            deva += 1
        elif any(lo <= cp <= hi for lo, hi in _LATIN_RANGES):
            latin += 1
    total = deva + latin
    if total == 0:
        return "empty"
    if deva / total > 0.66:
        return "devanagari"
    if latin / total > 0.66:
        return "latin"
    return "mixed"


class _JudgeOutput(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


@dataclass
class LanguageFitResult:
    score: int
    reason: str
    script: str
    cached: bool


async def grade_language_fit(
    record: RunRecord,
    persona_id: str,
    judge_role: Role,
) -> LanguageFitResult:
    """Score how well the agent accommodates the caller's language (1–5).

    Passes the full interleaved transcript to the judge; validates score with Pydantic.
    """
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    transcript_lines = [
        f"{i + 1}. [{t.speaker.upper()}] {t.text}" for i, t in enumerate(record.turns)
    ]
    transcript = "\n".join(transcript_lines)

    caller_text = " ".join(t.text for t in record.turns if t.speaker == "caller")
    script = detect_script(caller_text)

    prompt = template.format(persona_id=persona_id, transcript=transcript)
    messages = [ChatMessage(role="user", content=prompt)]
    resp = await judge_role.chat(messages, temperature=0.0)

    data = _JudgeOutput.model_validate_json(resp.text)
    return LanguageFitResult(
        score=data.score, reason=data.reason, script=script, cached=resp.cached
    )


# ------------------------------------------------------------------ #
# Export for hand-labelling                                            #
# ------------------------------------------------------------------ #


def _build_transcript(record: RunRecord, max_chars: int = 2000) -> str:
    lines = [f"[{t.speaker.upper()}] {t.text}" for t in record.turns]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + " …(trimmed)"
    return text


def export_for_labelling(
    records: list[RunRecord],
    output_path: Path,
    n: int = 30,
) -> None:
    """Sample ≤ n records and write a CSV for human scoring.

    Columns: scenario_id, persona_id, run_index, transcript, judge_score, human_score.
    judge_score and human_score are left blank for the human to fill.
    """
    sample = random.sample(records, min(n, len(records)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scenario_id",
                "persona_id",
                "run_index",
                "transcript",
                "judge_score",
                "human_score",
            ],
        )
        writer.writeheader()
        for r in sample:
            writer.writerow(
                {
                    "scenario_id": r.scenario_id,
                    "persona_id": r.persona_id,
                    "run_index": r.run_index,
                    "transcript": _build_transcript(r),
                    "judge_score": "",
                    "human_score": "",
                }
            )


# ------------------------------------------------------------------ #
# Judge–human agreement                                                #
# ------------------------------------------------------------------ #


def compute_agreement(judge: list[float], human: list[float]) -> dict[str, Any]:
    """Compute agreement metrics between paired judge and human scores.

    Raises ValueError if either list is empty or the lists have different lengths.
    Returns:
        mae: mean absolute error
        within_1: fraction of pairs where |judge - human| <= 1
        pearson_r: Pearson correlation, or None if either series has zero variance
    """
    if not judge or not human:
        raise ValueError("score lists must not be empty")
    if len(judge) != len(human):
        raise ValueError(
            f"judge ({len(judge)}) and human ({len(human)}) lists must have equal length"
        )
    n = len(judge)
    diffs = [abs(j - h) for j, h in zip(judge, human, strict=True)]
    mae = sum(diffs) / n
    within_1 = sum(1 for d in diffs if d <= 1) / n

    mean_j = sum(judge) / n
    mean_h = sum(human) / n
    cov = sum((j - mean_j) * (h - mean_h) for j, h in zip(judge, human, strict=True))
    var_j = sum((j - mean_j) ** 2 for j in judge)
    var_h = sum((h - mean_h) ** 2 for h in human)
    if var_j == 0 or var_h == 0:
        pearson_r = None
    else:
        import math

        pearson_r = cov / math.sqrt(var_j * var_h)

    return {"mae": mae, "within_1": within_1, "pearson_r": pearson_r}
