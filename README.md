# hinglish-caller-bench

**Evaluate AI customer-service agents with simulated Indian callers speaking Hindi, Hinglish, and English.**

You plug in your agent. The harness runs hundreds of realistic customer calls — in 5 language styles — and tells you exactly where it fails: task completion, tool compliance, and language fit. With confidence intervals and the worst transcripts to read.

> v0.2 · `--agent-url` · harness-executed tool calls · `hcb init`

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
pip install hinglish-caller-bench
# optional: bar charts
pip install "hinglish-caller-bench[plot]"
```

---

## Quick start

```bash
# 1. Scaffold a project with bundled scenarios
hcb init myproject/
cd myproject/
cp .env.example .env  # fill in your API keys

# 2. Smoke test — no API key, < 10 s
hcb demo

# 3. Evaluate your agent endpoint
export GROQ_API_KEY=gsk_...
hcb run --agent-url http://localhost:8000/chat \
        --caller-provider groq \
        --domains delivery,emi_reminder --n-scenarios 6 --runs 3

# 4. Or run the built-in baseline (Groq caller + Gemini agent)
export GROQ_API_KEY=gsk_...
export GEMINI_API_KEY=AIza...
hcb run --caller-provider groq --agent-provider gemini \
        --domains delivery,emi_reminder --n-scenarios 6 --runs 3 --max-turns 8

# 5. Report and charts
hcb report results/ --language-fit --lf-sample 30
hcb plot results/
```

---

## Testing your own agent

Point the harness at any HTTP endpoint with `--agent-url`. **No Python changes needed.**

### Protocol

Your endpoint receives:

```json
POST /chat
{
  "messages": [
    {"role": "system", "content": "You are a customer service agent..."},
    {"role": "user", "content": "Bhai mera order deliver nahi hua"},
    ...
  ],
  "tools": [
    {"name": "lookup_order", "description": "...", "parameters": {...}},
    ...
  ]
}
```

And returns:

```json
{
  "text": "Let me check your order.",
  "tool_calls": [{"name": "lookup_order", "args": {"order_id": "ORD-123"}}],
  "prompt_tokens": 120,
  "completion_tokens": 40
}
```

**The harness executes the tool calls** against its MockDB and feeds the results back in a loop — so `task_completion` and `tool_correctness` remain fully gradeable without you managing any state.

### Minimal Flask example

```python
from flask import Flask, request, jsonify
import openai

app = Flask(__name__)
client = openai.OpenAI()

@app.post("/chat")
def chat():
    body = request.json
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=body["messages"],
        tools=[{"type": "function", "function": t} for t in body["tools"]],
    )
    msg = response.choices[0].message
    tool_calls = [
        {"name": tc.function.name, "args": json.loads(tc.function.arguments)}
        for tc in (msg.tool_calls or [])
    ]
    return jsonify({"text": msg.content or "", "tool_calls": tool_calls})
```

Then: `hcb run --agent-url http://localhost:5000/chat --n-scenarios 6 --runs 3`

---

## Scope and known limits

**What's covered:**
- 30 scenarios across 3 domains (delivery, EMI reminder, refund)
- 5 caller personas (English, Hindi Devanagari, Hindi Roman, Hinglish, code-switcher)
- 3 graders: task completion, tool correctness, language fit
- Free-tier friendly: checkpointing, split-provider mode, LLM-judge sampling
- `--agent-url`: evaluate any HTTP agent endpoint without Python changes

**Current limitation — tool/scenario coupling:**
The scenarios assume specific mock tool names (`lookup_order`, `reschedule_emi`, etc.). If your agent uses different tool names, you'll need a thin adapter (a 6-line rename, or a flag coming in a future version).

**Roadmap for v0.3:**
- `--adapter PATH`: YAML config to map your tool names to scenario tool names
- Custom scenario YAML: write scenarios for your own tools while reusing personas and graders
- Language-fit judge calibration (currently scores too high across the board)

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
| `--agent-url` | — | HTTP endpoint to evaluate (replaces built-in agent) |
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

### `hcb init`

```bash
hcb init [output_dir]
```

Scaffolds a new project with bundled scenarios, `.env.example`, and `adapter_example.yaml`. Run this after `pip install hinglish-caller-bench`.

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
