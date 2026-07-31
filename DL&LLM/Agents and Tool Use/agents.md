# Agents and Tool Use

A language model on its own is a text predictor: give it a prompt, get back text. An **agent** wraps that model in a loop that lets it *act* — call tools, observe the results, and decide what to do next — until a goal is met. This tutorial explains the core agentic design patterns and then builds a working agent, with tool use and a reasoning loop, entirely from scratch: no agent framework, just a served model, a tool registry, and a parse/act/observe loop.

The companion script `agents.py` runs four stages, each demonstrating one pattern, using a single instruction-tuned model (served with vLLM) to play every role.

---

## 1. What Makes a Program an "Agent"

A plain LLM call is a straight line: prompt in, text out. An agent adds a **loop** and **the ability to affect the world** (through tools), plus somewhere to keep **state** (memory):

```
        ┌───────────────────────────────────────────────┐
        │                                               │
        ▼                                               │
   ┌─────────┐   reason    ┌──────────┐   act     ┌──────────┐
   │  MODEL  │ ────────>   │ DECISION │ ────────> │  TOOL    │
   │ (brain) │             │ tool? or │           │ (world)  │
   └─────────┘             │  done?   │           └────┬─────┘
        ▲                  └──────────┘                │
        │                       observe                │
        └──────────────────────────────────────────────┘
              (feed the result back into the model)
```

Four design patterns recur in almost every agent:

| Pattern | What it means | Where in the script |
|---------|---------------|---------------------|
| **Planning** | Break a goal into ordered steps before acting | Stage 4 (Planner) |
| **Tool use** | Call external functions for facts/computation the model shouldn't guess | Stages 1–4 |
| **Memory** | Carry state across steps (and across sessions) | Stage 2 (scratchpad) |
| **Reflection** | Critique and revise the model's own output | Stage 3 |

---

## 2. Tool Use and Function Calling

LLMs are unreliable at arithmetic, cannot look up facts they never saw, and have no access to your systems. **Tools** fix this: you expose a set of functions, describe their interfaces, and let the model *request* a call by emitting a structured object instead of prose.

```
   User question
        │
        ▼
   ┌─────────┐     {"tool": "calculator",            ┌──────────────┐
   │  MODEL  │ ──>  "args": {"expression":    ──>    │ execute_tool │ ──> result
   └─────────┘       "234 * (56 + 78)"}}             └──────────────┘
```

A **tool interface** is three things: a **name**, a **description** (so the model knows *when* to use it), and a typed **argument schema** (so the model knows *how* to call it). In `agents.py` the registry looks like:

```python
TOOLS = {
  "calculator":   {"fn": ..., "description": "...", "args": {"expression": "string"}},
  "search":       {"fn": ..., "description": "...", "args": {"query": "string"}},
  "unit_convert": {"fn": ..., "description": "...", "args": {"value","from_unit","to_unit"}},
}
```

The interface is rendered into the system prompt, the model replies with JSON, we parse it, run the real Python function, and hand back the result. (Commercial APIs formalise exactly this as "function calling" / "tool calling"; the mechanism is identical — a schema in, a structured call out.)

**Why structured, not free text?** A JSON call is *parseable and executable*. Free-form "I would compute 234 × 134" is neither. Structure is what turns a chatbot into a system component.

The tools here are pure-Python and offline — a safe arithmetic evaluator (never `eval`), a keyword lookup over a small knowledge base, and a unit converter — so every run is deterministic and needs no network.

---

## 3. The Reasoning Loop (ReAct)

A single tool call rarely solves a real task. The **ReAct** pattern (Reason + Act) interleaves thinking and acting in a loop:

```
   Thought  → "I need the Earth–Moon distance"
   Action   → search("earth moon distance")
   Observation ← "384400 kilometres"
   Thought  → "convert to metres, then divide by the speed of light"
   Action   → calculator("384400000 / 299792458")
   Observation ← "1.282..."
   Thought  → "that is the answer"
   Final    → "About 1.28 seconds."
```

Each turn the model emits **one** JSON object — either a tool call or a `final_answer`. We execute the tool, append the result as an `Observation:`, and loop. The running list of messages **is the agent's working memory**: everything it has thought, done, and seen so far is in context, which is how it chains steps together. A `max_steps` budget stops runaway loops.

> **Memory has two timescales.** *Working memory* is the scratchpad within one task (the message list). *Long-term memory* persists facts across tasks — a database, vector store, or key-value store the agent can write to and read from with dedicated tools. This tutorial focuses on working memory; the same loop extends to long-term memory by adding `remember` / `recall` tools.

---

## 4. Multi-Step Reasoning and Error Recovery

Tools fail: bad arguments, missing data, malformed calls. A brittle agent crashes; a robust one **treats the error as just another observation** and adapts.

```
   Action   → unit_convert(5, "miles", "kilometer")
   Observation ← ERROR: unknown unit 'kilometer'; supported: m, km, mi, ft
   Thought  → "I used the wrong unit names, retry with 'mi' and 'km'"
   Action   → unit_convert(5, "mi", "km")
   Observation ← "8.04672"
```

The script catches every tool exception and feeds the message back as `ERROR: ...`. Because the error text is *informative* (it lists the valid units), the model can self-correct on the next step. This is the single cheapest robustness win in agent design: **make your tool errors readable, and let the model read them.**

### Reflection (self-critique)

The model can also improve its *own* output. The **reflection** pattern runs the answer back through the model as a critic — "list concrete flaws" — and then revises using that critique. It catches omissions and vagueness a single forward pass misses, at the cost of extra calls. Stage 3 shows the draft, the critique, and the improved answer.

---

## 5. Multi-Agent Architectures

One agent with one context can get overloaded on a complex task. **Multi-agent** designs split the work across specialised roles, each with its own prompt (and, optionally, its own tools):

```
                    ┌───────────┐
        task ─────> │  PLANNER  │  decompose into sub-tasks
                    └─────┬─────┘
             ┌────────────┼────────────┐
             ▼            ▼            ▼
        ┌────────┐   ┌────────┐   ┌────────┐
        │worker 1│   │worker 2│   │worker 3│   each solves one sub-task
        └───┬────┘   └───┬────┘   └───┬────┘   with the ReAct loop + tools
            └────────────┼────────────┘
                   ┌─────▼──────┐
                   │SYNTHESIZER │  combine into one final answer
                   └────────────┘
```

This **orchestrator–worker** shape is the most common, but the family is broad:

| Architecture | Idea |
|--------------|------|
| **Orchestrator–worker** | A planner delegates sub-tasks to workers, a synthesizer merges (Stage 4) |
| **Debate / critic** | Two agents argue or one critiques another to surface errors |
| **Pipeline** | Fixed roles in sequence (e.g. researcher → writer → editor) |
| **Blackboard** | Agents share a common memory they all read and write |

The trade-off: more agents mean more robustness and parallelism, but also more model calls (cost/latency) and more places for miscommunication. Reach for multi-agent when a single context genuinely can't hold the task — not by default.

---

## Tutorial

The script builds all of the above from scratch and runs four stages end to end.

### Requirements

This tutorial serves a local open model with **vLLM** and expects a **GPU at run time** (like the other DL&LLM tutorials). The default model is `Qwen/Qwen3-8B`; override with `MODEL_NAME`.

```bash
pip install -r requirements.txt
```

### Run

```bash
python agents.py                 # all four stages
python agents.py --stage 1       # tool interfaces & function calling
python agents.py --stage 2       # the ReAct reasoning loop + memory
python agents.py --stage 3       # error recovery & reflection
python agents.py --stage 4       # multi-agent planner/worker/synthesizer

# smaller/faster model, or multi-GPU:
MODEL_NAME=Qwen/Qwen3-4B python agents.py
TENSOR_PARALLEL=2 python agents.py
```

Each stage prints its full reasoning trace (Thought / Action / Observation) to the console, and the complete run is saved to `outputs/agent_traces.json`.

### Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_NAME` | `Qwen/Qwen3-8B` | Any vLLM-servable instruct model |
| `MAX_MODEL_LEN` | `4096` | Context window |
| `TENSOR_PARALLEL` | `1` | GPUs for tensor parallelism |
| `TEMPERATURE` | `0.2` | Low → more deterministic tool use |
| `MAX_STEPS` | `6` | Reasoning-loop step budget |
| `SEED` | `0` | Sampling seed |

> **Note:** small models occasionally emit malformed JSON. The parser is
> best-effort (it strips ``<think>`` blocks and code fences and extracts the
> first `{...}`), and a malformed step becomes an `ERROR` observation the agent
> can recover from — but a larger model gives cleaner traces.
