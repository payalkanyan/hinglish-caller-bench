# hinglish-caller-bench

**Evaluate AI customer-service agents with simulated Indian callers speaking Hindi, Hinglish, and English.**

You plug in your agent. The harness runs hundreds of realistic customer calls — in 5 language styles — and tells you exactly where it fails: task completion, tool compliance, and language fit. With confidence intervals and the worst transcripts to read.

> v0.1 · delivery + EMI domains · reference-agent baseline

---

## How it works

1. **Install and clone** the repo to get the scenario library
2. **Plug in your agent** — as a Python class or (v0.2) an HTTP endpoint
3. **The harness runs the calls** — simulated callers in 5 language styles work through scenarios with your agent, turn by turn
4. **Every conversation is saved** — transcripts, tool calls, final database state
5. **Get three scores** — task completion, tool correctness, and language fit — broken down by caller type, with 95% confidence intervals and the worst failures to read

---

## Baseline results (v0.1)

6 scenarios × 5 personas × 3 runs = **90 conversations**, delivery + EMI domains.
Caller: Groq `qwen/qwen3.8-27b` · Agent: Gemini `gemini-3.5-flash-lite`

| Persona | Domain | Success% | Tool% | pass@3 |
|---|---|---|---|---|
| english | delivery | 89% | 44% | 0.67 |
| english | emi_reminder | 100% | 33% | 1.00 |
| hindi_devanagari | delivery | 56% | 33% | 0.33 |
| hindi_devanagari | emi_reminder | 100% | 89% | 1.00 |
| hindi_roman | delivery | 78% | 56% | 0.33 |
| hindi_roman | emi_reminder | 100% | 78% | 1.00 |
| hinglish | delivery | 89% | 0% | 0.67 |
| hinglish | emi_reminder | 100% | 44% | 1.00 |
| switcher | delivery | 67% | 22% | 0.33 |
| switcher | emi_reminder | 100% | 22% | 1.00 |

**Key finding:** Task success (88%) consistently outpaces tool correctness (42%). The agent
satisfies callers without following the required tool-call sequence — a process compliance
gap that CSAT scores cannot see.

---

## Install

```bash
git clone https://github.com/payalkanyan/hinglish-caller-bench
cd hinglish-caller-bench
pip install .
# or: pip install hinglish-caller-bench  (then git clone for the scenarios/)
```

```bash
# optional: bar charts
pip install "hinglish-caller-bench[plot]"
```

---

## Quick start

```bash
# Smoke test — no API key, < 10 s
hcb demo

# Run the benchmark (free-tier: split caller and agent across Groq + Gemini)
export GROQ_API_KEY=gsk_...
export GEMINI_API_KEY=AIza...
hcb run --caller-provider groq --agent-provider gemini \
        --domains delivery,emi_reminder --n-scenarios 6 --runs 3 --max-turns 8

# Report with language-fit judge (samples 30 records to save quota)
hcb report results/ --language-fit --lf-sample 30

# Charts
hcb plot results/
```

---

## Plugging in your own agent

The reference agent in `src/hinglish_bench/examples/reference_agent.py` is a working example.
To test your own agent, implement the `AgentUnderTest` interface:

```python
from hinglish_bench.agent import AgentUnderTest, AgentTurn, ToolDef
from hinglish_bench.schemas import Turn

class MyAgent(AgentUnderTest):
    async def respond(self, turns: list[Turn], tools: list[ToolDef]) -> AgentTurn:
        # call your agent here — HTTP, SDK, whatever
        ...
```

Then pass it to `run_batch` directly in Python:

```python
from hinglish_bench.runner import run_batch
from hinglish_bench.scenarios import load_scenarios_dir
from hinglish_bench.personas import PERSONAS

scenarios = load_scenarios_dir("scenarios/")
records = await run_batch(
    scenarios=scenarios,
    personas=list(PERSONAS.values()),
    caller_role=caller_role,          # your provider config
    agent_factory=lambda: (MyAgent(), lambda: {}),
    tools=your_tool_defs,
    runs=3,
    results_path="results/runs.jsonl",
)
```

---

## v0.1 scope and known limits

**What v0.1 covers:**
- 30 scenarios across 3 domains (delivery, EMI reminder, refund)
- 5 caller personas (English, Hindi Devanagari, Hindi Roman, Hinglish, code-switcher)
- 3 graders: task completion, tool correctness, language fit
- Free-tier friendly: checkpointing, split-provider mode, LLM-judge sampling

**Current limitation — tool/scenario coupling:**
The scenarios assume a specific mock tool set (`lookup_order`, `reschedule_emi`, etc.) and a mock database. A real company's agent has its own tools and backend, so it can't run these scenarios without adaptation.

**Roadmap for v0.2:**
- `--agent-url` flag: point the harness at any HTTP endpoint, no Python needed
- Tool adapter: map your tool names to scenario tools with a small config file
- Custom scenario YAML: write scenarios for your own tools while reusing the personas, simulator, and graders

---

## Commands

### `hcb run`

| Option | Default | Description |
|---|---|---|
| `--domains` | all | Filter: `delivery,emi_reminder,refund` |
| `--n-scenarios` | `10` | Scenarios (stratified sample) |
| `--all-scenarios` | off | Use full filtered set |
| `--personas` | `all` | Comma-separated or `all` |
| `--runs` | `3` | Runs per (scenario, persona) |
| `--provider` | `gemini` | Provider for both roles |
| `--caller-provider` | — | Override provider for caller only |
| `--agent-provider` | — | Override provider for agent only |
| `--max-turns` | from YAML | Override turn cap |
| `--concurrency` | `4` | Parallel conversations |
| `--dry-run` | off | Print plan, no API calls |

Checkpoints after every conversation. Re-run the same command to resume; infra errors are retried automatically.

### `hcb report`

| Option | Default | Description |
|---|---|---|
| `--k` | `3` | k for pass@k |
| `--language-fit` | off | Run LLM language-fit judge |
| `--lf-sample N` | all | Score only N sampled records |
| `--provider` | `gemini` | Provider for judge |

Writes `summary.json` (pass rates, Wilson 95% CIs, pass@k) and `report.md` (table + 3 worst transcripts).

### `hcb plot`

Reads `summary.json`, writes `plots/success_rate.png`, `plots/tool_rate.png`, `plots/pass_at_k.png`.

---

## Metrics

| Metric | What it measures |
|---|---|
| **Task success** | Did the agent reach the scenario's required end state? |
| **Tool correctness** | Required tools called, correct args, right order, no forbidden calls |
| **pass@k** | P(at least one of k runs succeeds) — unbiased estimator |
| **Language fit** | LLM judge 1–5: did the agent match the caller's language register? *(experimental)* |

---

## License

MIT
