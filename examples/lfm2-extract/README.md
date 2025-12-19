# Evaluate and fine-tune LFM2-Extract on semi-medical data

## Test various models without structured output generation

```
uv run example-raw-generation.py \
    --model-id LiquidAI/LFM2-1.2B-Extract \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v1
# wrong -> output does not adhere to JSON schema

uv run example-raw-generation.py \
    --model-id LiquidAI/LFM2-1.2B-Extract \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# wrong -> output does not adhere to JSON schema

uv run example-raw-generation.py \
    --model-id LiquidAI/LFM2-700M \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v1
# works!

uv run example-raw-generation.py \
    --model-id LiquidAI/LFM2-700M \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# wrong -> output does not adhere to JSON schema
```

## Test various model with structured output generation

```
uv run example-structured-generation.py \
    --model-id LiquidAI/LFM2-1.2B-Extract \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v1
# wrong! misses medical condition information in the output

uv run example-structured-generation.py \
    --model-id LiquidAI/LFM2-1.2B-Extract \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# wrong! dosage=Dosage(text='20 mg twice daily,'))

uv run example-structured-generation.py \
    --model-id LiquidAI/LFM2-700M \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v1
# wrong! misses medical condition information

uv run example-structured-generation.py \
    --model-id LiquidAI/LFM2-700M \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# works!
```

## Test things work with the GGUF checkpoints and llama.cpp

```
uv run example-with-llama-cpp.py \
    --model-id LiquidAI/LFM2-700M-GGUF \
    --model-file LFM2-700M-Q4_0.gguf \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# wrong! hallucinated medication `simvastatin`

uv run example-with-llama-cpp.py \
    --model-id LiquidAI/LFM2-700M-GGUF \
    --model-file LFM2-700M-Q8_0.gguf \
    --user-prompt "I have high cholesterol and take atorvastatin 20 mg once daily" \
    --system-prompt-version v2
# works! { "entities": [ { "category": "MEDICAL_CONDITION", "text": "high cholesterol" }, { "category": "MEDICATION", "text": "atorvastatin 20 mg once daily" , "dosage": { "text": "20 mg" } } ]}
```

### Attention!
Make sure you install the llama.cpp build that is optimized for your backend. For example,
for my Macbook this is the install command.

```
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

To find the right command for your platform [see these instructions](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends).


## Example inputs

```
Input: I have diabetes and take metformin 500 mg twice a day.
Output: [ { "text": "diabetes", "category": "MEDICAL_CONDITION" }, { "text": "metformin", "category": "MEDICATION", "dosage": { "text": "500 mg twice a day" } } ]

Input: My blood pressure was 120/80.
Output: [ { "text": "blood pressure", "category": "MEASUREMENT", "value": { "text": "120/80" } } ]

Input: "I have high cholesterol and take atorvastatin 20 mg once daily."
Output: [{"text": "high cholesterol", "category": "MEDICAL_CONDITION"}, {"text": "atorvastatin", "category": "MEDICATION", "dosage": {"text": "20 mg once daily"}}]
```

## Systemm prompt

```
Return data as a JSON object with the following schema:

The output should be an array of objects. Each object represents an extracted medical entity and must contain:

1. "text" (string, required): The extracted medical term, condition name, medication name, or measurement name
2. "category" (string, required): One of "MEDICAL_CONDITION", "MEDICATION", or "MEASUREMENT"
3. Additional fields based on category:
   - If category is "MEDICATION": include "dosage" object with "text" field containing dosage information
   - If category is "MEASUREMENT": include "value" object with "text" field containing the measurement value and units
   - If category is "MEDICAL_CONDITION": no additional fields required

Schema structure:
[
  {
    "text": "string",
    "category": "MEDICAL_CONDITION" | "MEDICATION" | "MEASUREMENT",
    "dosage": {  // only for MEDICATION
      "text": "string"
    },
    "value": {  // only for MEASUREMENT
      "text": "string"
    }
  }
]

Examples:
Input: "I have diabetes and take metformin 500 mg twice a day."
Output: [{"text": "diabetes", "category": "MEDICAL_CONDITION"}, {"text": "metformin", "category": "MEDICATION", "dosage": {"text": "500 mg twice a day"}}]

Input: "My blood pressure was 120/80."
Output: [{"text": "blood pressure", "category": "MEASUREMENT", "value": {"text": "120/80"}}]
```

## TODOs

- [x] Vibe check different models
    - [x] LFM2-1.2B-Extract
    - [x] LFM2-700M
- [x] Add structured generation
- [ ] Evaluate on 50 samples
- [ ] Generate a training/eval dataset for fine-tuning
- [ ] Fine-tune