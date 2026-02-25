# /model-test

Test the local LLM's tool-calling accuracy against the behavior test suite.

**Usage:** `/model-test [--model <model-name>] [--suite <suite-name>]`

**Defaults:** model = whatever is running on localhost:11434, suite = all

## Prerequisites

Verify a model is running:
```bash
curl -s http://localhost:11434/api/tags
```

If not running, instruct user:
```
No model detected at localhost:11434. To start:
  ollama serve                           # Start Ollama
  ollama run qwen2.5:32b-instruct       # Load the dev model
```

## Test Suites

### Suite 1: Tool Selection (100 prompts)

**File:** `tests/model-behavior/tool_selection.test.ts`

Tests whether the model selects the correct tool(s) for a given prompt. Each test case defines:
- `prompt`: what the user says
- `expectedTools`: the tool(s) the model should call
- `expectedParamKeys` (optional): verify the model provides the right parameter keys

**Scoring:** Exact match on tool names. Partial credit if the model selects a superset (extra tools but includes the expected ones).

### Suite 2: Multi-Step Chains (50 scenarios)

**File:** `tests/model-behavior/multi_step_chains.test.ts`

Tests whether the model can plan and execute multi-step workflows. Each test:
- Provides a complex user request requiring 3+ tool calls
- Verifies the model produces the correct sequence of calls
- Checks that output from one tool is correctly fed as input to the next

**Scoring:** Chain completion rate (did all steps execute?) and order accuracy (were tools called in a valid order?).

### Suite 3: Edge Cases (30 scenarios)

**File:** `tests/model-behavior/edge_cases.test.ts`

Tests model behavior on tricky inputs:
- Ambiguous requests ("organize my stuff")
- Missing files (model should report error, not hallucinate)
- Permission denied scenarios
- Requests outside the tool set ("write me Python code" → should decline)
- Multiple valid interpretations (should ask for clarification)

**Scoring:** Appropriate response rate (did the model behave correctly?).

## Steps

1. Check model availability (see Prerequisites).
2. Run the specified suite(s):
   ```bash
   npx vitest run tests/model-behavior/tool_selection.test.ts
   npx vitest run tests/model-behavior/multi_step_chains.test.ts
   npx vitest run tests/model-behavior/edge_cases.test.ts
   ```
3. Store results in `tests/model-behavior/.results/<timestamp>.json`:
   ```json
   {
     "timestamp": "2026-02-12T10:30:00Z",
     "model": "qwen2.5:32b-instruct",
     "results": {
       "tool_selection": { "total": 100, "passed": 92, "accuracy": 0.92 },
       "multi_step_chains": { "total": 50, "completed": 41, "rate": 0.82 },
       "edge_cases": { "total": 30, "appropriate": 27, "rate": 0.90 }
     }
   }
   ```
4. Compare with previous results (if any) and flag regressions:
   - Tool selection accuracy dropped > 2% → ⚠️ regression
   - Chain completion rate dropped > 5% → ❌ significant regression
5. Report summary:

```
=== Model Behavior Test Results ===
Model: qwen2.5:32b-instruct

Tool Selection:    92/100 (92.0%)  [target: >90%] ✅
Multi-Step Chains: 41/50  (82.0%)  [target: >80%] ✅
Edge Cases:        27/30  (90.0%)  [target: >85%] ✅

vs. Previous Run (2026-02-10):
  Tool Selection:    +1.0% ✅
  Multi-Step Chains: -2.0% (within tolerance)
  Edge Cases:        +3.3% ✅
```
