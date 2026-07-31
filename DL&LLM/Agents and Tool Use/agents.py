"""
agents.py
=========
Build an LLM agent with tool use and a reasoning loop FROM SCRATCH — no agent
framework, just a served model, a tool registry, and a parse/act/observe loop.

    Stage 1  Tool interfaces     define typed tools + let the model emit a
                                 structured (JSON) call, then execute it
    Stage 2  Reasoning loop      a ReAct loop (Thought → Action → Observation)
                                 with a working-memory scratchpad
    Stage 3  Reflect & recover   feed tool errors back for self-correction, then
                                 self-critique and improve the answer
    Stage 4  Multi-agent         a Planner decomposes a task, Workers solve each
                                 sub-task with tools, a Synthesizer combines them

A single instruction-tuned model, served with vLLM, plays every role (planner,
worker, critic). The tools are pure-Python and offline, so the whole tutorial is
deterministic and needs no network.

Run:
    python agents.py                 # all four stages
    python agents.py --stage 2       # just the reasoning loop
    MODEL_NAME=Qwen/Qwen3-4B python agents.py
"""

import os
import re
import ast
import json
import argparse
import operator

# Shared Hugging Face cache (matches the lab server). Set before HF imports.
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/export/projects/nlp/.cache"))

from vllm import LLM, SamplingParams


# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", "1"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "6"))
SEED = int(os.environ.get("SEED", "0"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")


# =============================================================================
# MODEL HELPERS
# =============================================================================
def build_model():
    print(f"[setup] loading {MODEL_NAME} with vLLM ...")
    return LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TENSOR_PARALLEL,
        trust_remote_code=True,
    )


def render_messages(tokenizer, messages):
    """Apply the model's chat template to a list of {role, content} dicts."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_json_object(text):
    """Best-effort extraction of a single JSON object from model output."""
    text = strip_think(text)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0), strict=False)
    except json.JSONDecodeError:
        return {}


def llm_generate(llm, tokenizer, messages, max_tokens=512):
    sampling = SamplingParams(temperature=TEMPERATURE, top_p=0.9,
                              max_tokens=max_tokens, seed=SEED)
    prompt = render_messages(tokenizer, messages)
    out = llm.generate([prompt], sampling, use_tqdm=False)
    return strip_think(out[0].outputs[0].text)


# =============================================================================
# TOOLS  — each is a plain Python function; a schema describes its interface
# =============================================================================
# A tiny offline knowledge base so "search" is deterministic and needs no net.
KNOWLEDGE_BASE = {
    "speed of light": "The speed of light is 299792458 metres per second.",
    "earth moon distance": "The average Earth–Moon distance is 384400 kilometres.",
    "population of france": "The population of France is about 68 million (2024).",
    "population of germany": "The population of Germany is about 83 million (2024).",
    "eiffel tower height": "The Eiffel Tower is 330 metres tall.",
    "speed of sound": "The speed of sound in air is about 343 metres per second.",
}

# Allowed operators for the safe calculator (no eval of arbitrary code).
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def tool_calculator(expression: str):
    """Evaluate an arithmetic expression like '384400000 / 299792458'."""
    try:
        tree = ast.parse(str(expression), mode="eval")
        return str(_safe_eval(tree.body))
    except Exception:
        raise ValueError(
            f"could not evaluate {expression!r}; pass a pure arithmetic "
            "expression using numbers and + - * / ** %")


def tool_search(query: str):
    """Look a fact up in the knowledge base by keyword overlap."""
    q = str(query).lower()
    best, best_score = None, 0
    for key, value in KNOWLEDGE_BASE.items():
        score = sum(1 for w in key.split() if w in q)
        if score > best_score:
            best, best_score = value, score
    if best is None:
        raise KeyError(f"no knowledge-base entry matches {query!r}")
    return best


_UNITS_TO_METRES = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048}


def tool_unit_convert(value: float, from_unit: str, to_unit: str):
    """Convert a length between units. Supported: m, km, mi, ft."""
    fu, tu = str(from_unit).lower(), str(to_unit).lower()
    for u in (fu, tu):
        if u not in _UNITS_TO_METRES:
            raise ValueError(
                f"unknown unit {u!r}; supported units are: "
                f"{', '.join(_UNITS_TO_METRES)}")
    metres = float(value) * _UNITS_TO_METRES[fu]
    return str(metres / _UNITS_TO_METRES[tu])


# Registry: name -> function + human/LLM-readable interface description.
TOOLS = {
    "calculator": {
        "fn": tool_calculator,
        "description": "Evaluate an arithmetic expression and return the number.",
        "args": {"expression": "string, e.g. '384400000 / 299792458'"},
    },
    "search": {
        "fn": tool_search,
        "description": "Look up a fact (distances, populations, constants).",
        "args": {"query": "string, e.g. 'speed of light'"},
    },
    "unit_convert": {
        "fn": tool_unit_convert,
        "description": "Convert a length between units m, km, mi, ft.",
        "args": {"value": "number", "from_unit": "m|km|mi|ft",
                 "to_unit": "m|km|mi|ft"},
    },
}


def render_tool_specs(tools):
    lines = []
    for name, spec in tools.items():
        args = ", ".join(f"{k} ({v})" for k, v in spec["args"].items())
        lines.append(f'  - {name}: {spec["description"]}  args: {{{args}}}')
    return "\n".join(lines)


def execute_tool(tools, name, args):
    """Run a tool by name; raises a clear error the agent can read and recover."""
    if name not in tools:
        raise KeyError(f"unknown tool {name!r}; available: {', '.join(tools)}")
    return tools[name]["fn"](**args)


# =============================================================================
# THE REASONING LOOP  (ReAct: reason → act → observe, repeat)
# =============================================================================
REACT_SYSTEM = """You are a careful reasoning agent that solves tasks using tools.

At EACH step reply with ONE JSON object and nothing else, in one of two forms:

  to call a tool:   {{"thought": "<why>", "tool": "<name>", "args": {{...}}}}
  to finish:        {{"thought": "<why>", "final_answer": "<answer>"}}

Available tools:
{tools}

Rules:
- Use tools for any calculation or fact lookup; never guess numbers.
- After each tool call you will receive an "Observation:". Use it.
- If an Observation starts with ERROR, read it and correct your next call.
- When you have enough information, return final_answer. Be concise.

Example:
  User: What is 12 * (3 + 4)?
  {{"thought": "compute with the calculator", "tool": "calculator", "args": {{"expression": "12 * (3 + 4)"}}}}
  Observation: 84
  {{"thought": "I have the result", "final_answer": "84"}}"""


def run_agent(llm, tokenizer, task, tools=TOOLS, max_steps=MAX_STEPS, verbose=True):
    """One ReAct episode. The `messages` list IS the working memory."""
    system = REACT_SYSTEM.format(tools=render_tool_specs(tools))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Task: {task}"}]
    trace = []
    for step in range(1, max_steps + 1):
        raw = llm_generate(llm, tokenizer, messages, max_tokens=384)
        obj = parse_json_object(raw)
        thought = obj.get("thought", "")

        if "final_answer" in obj:
            if verbose:
                print(f"  step {step} · THOUGHT: {thought}")
                print(f"           · FINAL: {obj['final_answer']}")
            trace.append({"step": step, "thought": thought,
                          "final_answer": obj["final_answer"]})
            return obj["final_answer"], trace

        name, args = obj.get("tool"), obj.get("args", {}) or {}
        if not name:
            observation = ("ERROR: malformed step. Reply with ONE JSON object "
                           "containing either 'tool'+'args' or 'final_answer'.")
        else:
            try:
                observation = execute_tool(tools, name, args)
            except Exception as e:                       # <-- error recovery
                observation = f"ERROR: {type(e).__name__}: {e}"

        if verbose:
            print(f"  step {step} · THOUGHT: {thought}")
            print(f"           · ACTION: {name}({json.dumps(args)})")
            print(f"           · OBSERVATION: {observation}")
        trace.append({"step": step, "thought": thought, "tool": name,
                      "args": args, "observation": str(observation)})

        # Extend working memory with what we just did and saw.
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "(no final answer — step budget exhausted)", trace


# =============================================================================
# REFLECTION  (self-critique → revise)
# =============================================================================
def reflect_and_improve(llm, tokenizer, task, draft):
    critique = llm_generate(llm, tokenizer, [
        {"role": "system", "content": "You are a strict reviewer."},
        {"role": "user", "content":
            f"Task:\n{task}\n\nDraft answer:\n{draft}\n\n"
            "List concrete flaws (missing points, errors, vagueness) as short "
            "bullets. If the draft is already excellent, reply exactly 'OK'."},
    ], max_tokens=256)
    if critique.strip().upper().startswith("OK"):
        return draft, critique, False
    improved = llm_generate(llm, tokenizer, [
        {"role": "system", "content": "You improve answers using a critique."},
        {"role": "user", "content":
            f"Task:\n{task}\n\nDraft:\n{draft}\n\nCritique:\n{critique}\n\n"
            "Write the improved answer only."},
    ], max_tokens=400)
    return improved, critique, True


# =============================================================================
# MULTI-AGENT  (Planner → Workers → Synthesizer)
# =============================================================================
def plan_subtasks(llm, tokenizer, task):
    raw = llm_generate(llm, tokenizer, [
        {"role": "system", "content":
            "You are a planner. Break the task into 2-4 independent, concrete "
            'sub-tasks. Reply ONLY as JSON: {"subtasks": ["...", "..."]}.'},
        {"role": "user", "content": task},
    ], max_tokens=256)
    obj = parse_json_object(raw)
    subs = obj.get("subtasks") or []
    return [s for s in subs if isinstance(s, str)][:4]


def synthesize(llm, tokenizer, task, results):
    joined = "\n".join(f"- {q}\n  → {a}" for q, a in results)
    return llm_generate(llm, tokenizer, [
        {"role": "system", "content":
            "You combine sub-task results into one clear final answer."},
        {"role": "user", "content":
            f"Original task:\n{task}\n\nSub-task results:\n{joined}\n\n"
            "Write the final answer."},
    ], max_tokens=400)


def multi_agent_solve(llm, tokenizer, task):
    print("  [planner] decomposing the task ...")
    subtasks = plan_subtasks(llm, tokenizer, task)
    if not subtasks:
        subtasks = [task]
    for i, s in enumerate(subtasks, 1):
        print(f"    plan {i}. {s}")
    results = []
    for i, s in enumerate(subtasks, 1):
        print(f"\n  [worker {i}] solving: {s}")
        ans, _ = run_agent(llm, tokenizer, s, max_steps=4, verbose=True)
        results.append((s, ans))
    print("\n  [synthesizer] combining worker results ...")
    final = synthesize(llm, tokenizer, task, results)
    return final, subtasks, results


# =============================================================================
# STAGES
# =============================================================================
def stage1_tool_interfaces(llm, tokenizer):
    print("\n" + "=" * 70)
    print("  STAGE 1 — TOOL INTERFACES & FUNCTION CALLING")
    print("=" * 70)
    print("\nTools exposed to the model:\n" + render_tool_specs(TOOLS))
    print("\nWe ask a single-tool question and let the model emit a STRUCTURED")
    print("call, which we parse and execute ourselves.\n")

    question = "What is 234 * (56 + 78)?"
    print(f"Question: {question}\n")
    messages = [
        {"role": "system", "content":
            "Reply with ONE JSON object calling a tool: "
            '{"tool": "<name>", "args": {...}}.\n'
            "Tools:\n" + render_tool_specs(TOOLS)},
        {"role": "user", "content": question},
    ]
    raw = llm_generate(llm, tokenizer, messages, max_tokens=200)
    obj = parse_json_object(raw)
    print(f"  model emitted: {json.dumps(obj)}")
    try:
        result = execute_tool(TOOLS, obj.get("tool"), obj.get("args", {}) or {})
        print(f"  executed → {result}")
    except Exception as e:
        print(f"  execution error: {e}")
    return {"question": question, "call": obj}


def stage2_reasoning_loop(llm, tokenizer):
    print("\n" + "=" * 70)
    print("  STAGE 2 — REASONING LOOP (ReAct) + WORKING MEMORY")
    print("=" * 70)
    task = ("How many seconds does light take to travel from the Earth to the "
            "Moon? Look up the distance and the speed of light, then compute it.")
    print(f"\nTask: {task}\n")
    answer, trace = run_agent(llm, tokenizer, task)
    print(f"\n  ► answer: {answer}")
    return {"task": task, "trace": trace, "answer": answer}


def stage3_reflect_recover(llm, tokenizer):
    print("\n" + "=" * 70)
    print("  STAGE 3 — ERROR RECOVERY & REFLECTION")
    print("=" * 70)

    print("\n(a) Error recovery — a wrong unit name makes the tool raise; the")
    print("    agent reads the ERROR observation and retries with a valid unit.\n")
    task = ("Convert 5 miles to kilometres using the unit_convert tool, then "
            "state the result.")
    print(f"Task: {task}\n")
    answer, trace = run_agent(llm, tokenizer, task)
    recovered = any(str(s.get("observation", "")).startswith("ERROR") for s in trace)
    print(f"\n  ► answer: {answer}")
    print(f"  (an ERROR observation occurred and was recovered from: {recovered})")

    print("\n(b) Reflection — draft an answer, self-critique, then improve.\n")
    q = ("List three risks of deploying an autonomous LLM agent, each with one "
         "mitigation.")
    draft = llm_generate(llm, tokenizer, [
        {"role": "user", "content": q}], max_tokens=300)
    print("  draft:\n" + "\n".join("    " + l for l in draft.splitlines()))
    improved, critique, changed = reflect_and_improve(llm, tokenizer, q, draft)
    print("\n  critique:\n" + "\n".join("    " + l for l in critique.splitlines()))
    if changed:
        print("\n  improved:\n" + "\n".join("    " + l for l in improved.splitlines()))
    else:
        print("\n  (reviewer judged the draft already good)")
    return {"recovery_task": task, "recovered": recovered,
            "reflection_changed": changed}


def stage4_multi_agent(llm, tokenizer):
    print("\n" + "=" * 70)
    print("  STAGE 4 — MULTI-AGENT (Planner → Workers → Synthesizer)")
    print("=" * 70)
    task = ("Compare the populations of France and Germany: report each, their "
            "combined total, and the percentage France makes up of that total.")
    print(f"\nTask: {task}\n")
    final, subtasks, results = multi_agent_solve(llm, tokenizer, task)
    print(f"\n  ► final answer:\n{final}")
    return {"task": task, "subtasks": subtasks,
            "results": [{"subtask": q, "answer": a} for q, a in results],
            "final": final}


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Agents & Tool Use tutorial")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4],
                        help="Run a single stage (default: all).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    llm = build_model()
    tokenizer = llm.get_tokenizer()

    stages = {
        1: stage1_tool_interfaces,
        2: stage2_reasoning_loop,
        3: stage3_reflect_recover,
        4: stage4_multi_agent,
    }
    to_run = [args.stage] if args.stage else [1, 2, 3, 4]

    results = {}
    for s in to_run:
        results[f"stage{s}"] = stages[s](llm, tokenizer)

    out_path = os.path.join(OUTPUT_DIR, "agent_traces.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "stages": results}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nSaved traces to {out_path}")


if __name__ == "__main__":
    main()
