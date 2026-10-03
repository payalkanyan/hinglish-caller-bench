# hinglish-caller-bench

Stress-test AI customer-service agents with simulated Indian callers speaking Hinglish,
Hindi (Devanagari and Roman), English, and code-switching styles.

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

# 2. Run the benchmark (requires GROQ_API_KEY)
export GROQ_API_KEY=gsk_...
uv run hcb run --n-scenarios 10 --personas english,hinglish --runs 3

# 3. Generate summary and report
uv run hcb report results/

# 4. Generate bar charts (requires matplotlib)
uv run hcb plot results/
```

---

## Commands

### `hcb demo`

Runs 2 scenarios × 2 personas against a scripted `MockProvider`. No API keys, no network,
completes in under 10 seconds. Prints a Rich table with end reason and token counts.

### `hcb run [OPTIONS]`

Runs the reference-agent baseline against real Groq models. Requires `GROQ_API_KEY`.

| Option | Default | Description |
|---|---|---|
| `--scenarios-dir` | `scenarios/` | Directory of scenario YAML files |
| `--results-dir` | `results/` | Directory to write `runs.jsonl` |
| `--n-scenarios` | `10` | Number of scenarios (stratified sample) |
| `--all-scenarios` | off | Use all 30 scenarios |
| `--personas` | `all` | Comma-separated persona IDs or `all` |
| `--runs` | `3` | Runs per (scenario, persona) combination |
| `--caller-model` | `llama-3.3-70b-versatile` | Groq model for the caller simulator |
| `--agent-model` | `llama-3.3-70b-versatile` | Groq model for the reference agent |
| `--concurrency` | `4` | Max simultaneous conversations |
| `--dry-run` | off | Print run plan and exit (no API calls) |

### `hcb report [RESULTS_DIR]`

Reads `runs.jsonl` and writes:
- `results/summary.json` — pass rates, Wilson CIs, pass@k per persona × domain
- `results/report.md` — human-readable Markdown with worst transcripts

```bash
uv run hcb report results/ --scenarios-dir scenarios/ --k 3
```

### `hcb plot [RESULTS_DIR]`

Reads `results/summary.json` and writes PNG charts to `results/plots/`. Requires the
`plot` extra (`pip install "hinglish-caller-bench[plot]"`).

Charts produced:
- `success_rate.png` — task success rate per persona × domain with 95% Wilson CI
- `tool_rate.png` — tool correctness rate per persona × domain
- `pass_at_k.png` — pass@k estimator per persona × domain (skipped if no data)

```bash
uv run hcb plot results/
uv run hcb plot results/ --output-dir my_charts/
```

---

## Benchmark design

### Corpus vs experiment

**Corpus (fixed):**
- 30 scenarios across 3 domains: 10 refund, 10 emi_reminder, 10 delivery
- 5 personas: `english`, `hindi_devanagari`, `hindi_roman`, `hinglish`, `switcher`
- 6 mock tools: `lookup_order`, `initiate_refund`, `check_emi_status`, `reschedule_emi`,
  `track_delivery`, `escalate_to_human`

**Full experiment:**
- 30 scenarios × 5 personas × 3 runs = **450 conversations**
- `--n-scenarios`, `--personas`, and `--runs` can reduce this (e.g. 10 × 2 × 1 = 20)

### Metrics

- **Task success rate** — did the agent reach the scenario's `success_state`?
- **Tool correctness rate** — did required tools get called with correct arguments,
  in the right order, without forbidden extras?
- **pass@k** — unbiased estimator: probability that at least one of k runs succeeds
- **Language-fit score** — LLM judge (1–5) of how well the agent matched the caller's
  language register (requires a judge model; off by default)

### Grading

Each `RunRecord` (raw transcript + DB snapshot) is graded independently after the run.
Grades are never stored in the record itself; `hcb report` computes them from `runs.jsonl`.

---

## Output files

| File | Written by | Contents |
|---|---|---|
| `results/runs.jsonl` | `hcb run` | One JSON line per conversation (RunRecord) |
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
