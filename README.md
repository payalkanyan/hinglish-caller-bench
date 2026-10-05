# hinglish-caller-bench

Stress-test AI customer-service agents with simulated Indian callers speaking Hinglish,
Hindi (Devanagari and Roman), English, and code-switching styles.

## Baseline results (v0.1)

6 scenarios × 5 personas × 3 runs = **90 conversations**, delivery + EMI domains.
Caller simulator: Groq `qwen/qwen3.8-27b`. Reference agent: Gemini `gemini-3.5-flash-lite`.

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

**Key finding:** Task success (88% overall) consistently outpaces tool correctness (42% overall).
The agent satisfies callers without following the required tool call sequence — a process
compliance gap that would be invisible to CSAT scores alone.

---

## Install

```bash
# End users
pip install hinglish-caller-bench

# Contributors / development
git clone https://github.com/your-org/hinglish-caller-bench
cd hinglish-caller-bench
uv sync --dev
uv run hcb demo        # smoke-test; no API key needed
```

To generate charts, also install the optional `plot` extra:

```bash
pip install "hinglish-caller-bench[plot]"
```

---

## Quick start

```bash
# 1. Smoke test with MockProvider (< 10 s, no API key)
uv run hcb demo

# 2. Run the benchmark — split providers to stay within free-tier quotas
export GROQ_API_KEY=gsk_...
export GEMINI_API_KEY=AIza...
uv run hcb run \
  --caller-provider groq --agent-provider gemini \
  --domains delivery,emi_reminder \
  --n-scenarios 6 --personas all --runs 3 --max-turns 8

# 3. Generate summary and report (language-fit judge on a 30-record sample)
uv run hcb report results/ --language-fit --lf-sample 30

# 4. Generate bar charts (requires matplotlib)
uv run hcb plot results/
```

---

## Commands

### `hcb demo`

Runs 2 scenarios × 2 personas against a scripted `MockProvider`. No API keys, no network,
completes in under 10 seconds. Prints a Rich table with end reason and token counts.

### `hcb run [OPTIONS]`

Runs the reference-agent baseline. Supports split-provider mode to spread load across
free-tier quotas (caller on Groq, agent on Gemini, or any combination).

| Option | Default | Description |
|---|---|---|
| `--scenarios-dir` | `scenarios/` | Directory of scenario YAML files |
| `--results-dir` | `results/` | Directory to write `runs.jsonl` |
| `--n-scenarios` | `10` | Number of scenarios (stratified sample) |
| `--all-scenarios` | off | Use all scenarios in the filtered set |
| `--domains` | all | Comma-separated domain filter: `delivery,emi_reminder,refund` |
| `--personas` | `all` | Comma-separated persona IDs or `all` |
| `--runs` | `3` | Runs per (scenario, persona) combination |
| `--provider` | `gemini` | Provider for both roles: `groq \| gemini \| openrouter \| ollama` |
| `--caller-provider` | — | Override provider for caller simulator only |
| `--agent-provider` | — | Override provider for agent only |
| `--caller-model` | provider default | Model ID for the caller simulator |
| `--agent-model` | provider default | Model ID for the reference agent |
| `--max-turns` | from scenario YAML | Override max turns per conversation |
| `--concurrency` | `4` | Max simultaneous conversations |
| `--dry-run` | off | Print run plan and exit (no API calls) |

Runs checkpoint to `runs.jsonl` after each conversation. Re-running the same command
resumes from where it left off; infra errors (rate limits, timeouts) are automatically
retried on the next run.

### `hcb report [RESULTS_DIR]`

Reads `runs.jsonl` and writes:
- `results/summary.json` — pass rates, Wilson 95% CIs, pass@k per persona × domain
- `results/report.md` — Markdown table + 3 worst-transcript examples

| Option | Default | Description |
|---|---|---|
| `--k` | `3` | k for pass@k estimator |
| `--language-fit` | off | Run LLM language-fit judge (requires provider API key) |
| `--lf-sample N` | all | Score only N randomly sampled records (saves quota) |
| `--provider` | `gemini` | Provider for the language-fit judge |
| `--judge-model` | provider default | Model for the language-fit judge |

### `hcb plot [RESULTS_DIR]`

Reads `results/summary.json` and writes PNG charts to `results/plots/`. Requires
`pip install "hinglish-caller-bench[plot]"`.

Charts produced:
- `success_rate.png` — task success rate per persona × domain with 95% Wilson CI
- `tool_rate.png` — tool correctness rate per persona × domain
- `pass_at_k.png` — pass@k estimator per persona × domain (skipped if no data)

---

## Benchmark design

### Corpus

- **30 scenarios** across 3 domains: 10 refund, 10 emi_reminder, 10 delivery
- **5 personas:** `english`, `hindi_devanagari`, `hindi_roman`, `hinglish`, `switcher`
- **6 mock tools:** `lookup_order`, `initiate_refund`, `check_emi_status`, `reschedule_emi`,
  `track_delivery`, `escalate_to_human`

Full experiment: 30 × 5 × 3 = **450 conversations**. Use `--n-scenarios`, `--domains`,
`--personas`, and `--runs` to run a representative subset.

### Metrics

| Metric | What it measures |
|---|---|
| **Task success rate** | Did the agent reach the scenario's `success_state`? |
| **Tool correctness rate** | Required tools called with correct args, right order, no forbidden extras |
| **pass@k** | Unbiased estimator: P(at least one of k runs succeeds) |
| **Language-fit score** | LLM judge (1–5): how well the agent matched the caller's language register |

### Grading

Each `RunRecord` (raw transcript + DB snapshot) is graded independently after the run.
Grades are never stored in the record; `hcb report` computes them from `runs.jsonl`.
Infrastructure errors (rate limits, provider timeouts) are excluded from all rates and
reported separately so they never inflate failure counts.

---

## Output files

| File | Written by | Contents |
|---|---|---|
| `results/runs.jsonl` | `hcb run` | One JSON line per conversation (`RunRecord`) |
| `results/summary.json` | `hcb report` | Aggregated pass rates, CIs, pass@k |
| `results/report.md` | `hcb report` | Markdown report with worst transcripts |
| `results/plots/*.png` | `hcb plot` | Bar charts per persona × domain |

---

## Extending

**Add a scenario:** drop a YAML file in `scenarios/`. The schema is in
`src/hinglish_bench/schemas.py` (`Scenario`). Scenario `id` must match the filename
(without `.yaml`). Run `hcb run --dry-run` to validate the corpus.

**Swap the agent:** implement `AgentUnderTest` (see `src/hinglish_bench/agent.py`) and
pass it to `run_batch` in `src/hinglish_bench/runner.py`. The reference agent in
`src/hinglish_bench/examples/reference_agent.py` is a complete example.

---

## License

MIT
