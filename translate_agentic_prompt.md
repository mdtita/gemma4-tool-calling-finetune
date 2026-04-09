# Translation Prompt for Existing Agentic Data

Use this prompt with any model (local or API) to translate a subset of your existing
`gemma4_e4b_agentic_dataset.jsonl` into Arabic for the tool-calling training layer.

## How to Use

1. **Extract a subset** (~3-5K conversations) from your existing dataset:
   ```bash
   head -5000 gemma4_e4b_agentic_dataset.jsonl > agentic_subset_to_translate.jsonl
   ```

2. **Feed each conversation** to your chosen model with the prompt below.

3. **Save results** as `arabic_translated_agentic.jsonl` in the same directory.
   The `build_arabic_dataset.py` script will automatically pick it up.

## Translation Rules

- Translate **user messages** and **system prompts** to Arabic (MSA فصحى)
- Keep **tool/function call JSON** in English (function names, parameter names, JSON keys)
- Keep **code blocks** in English
- Translate **natural language descriptions** within tool calls to Arabic
- Keep **file paths**, **URLs**, and **variable names** in English

## Prompt

```
You are a professional Arabic translator specializing in technical content.
Translate the following multi-turn conversation into Arabic (Modern Standard Arabic / فصحى).

RULES:
1. Translate all user messages and system messages to Arabic (MSA).
2. For assistant messages:
   - Translate natural language text to Arabic
   - Keep ALL code blocks, JSON structures, function names, and parameter names in English
   - Keep file paths, URLs, and variable names in English
   - Translate explanatory text around code to Arabic
3. For tool/function messages:
   - Keep the JSON structure and function names in English
   - Translate any natural language descriptions to Arabic
4. Maintain the exact same message structure (roles, order, count)
5. Output ONLY the translated conversation in the same JSON format

INPUT CONVERSATION:
{paste the "messages" array here}

OUTPUT (translated JSON, same structure):
```

## Example

**Input:**
```json
[
  {"role": "system", "content": "You are a helpful coding assistant with access to tools."},
  {"role": "user", "content": "Fix the typo in src/utils.py where 'retrun' should be 'return'."},
  {"role": "assistant", "content": "I'll read the file first to find the typo.\n\n```json\n{\"tool\": \"read\", \"path\": \"src/utils.py\"}\n```"}
]
```

**Expected Output:**
```json
[
  {"role": "system", "content": "أنت مساعد برمجة مفيد ولديك إمكانية الوصول إلى أدوات."},
  {"role": "user", "content": "أصلح الخطأ الإملائي في src/utils.py حيث 'retrun' يجب أن تكون 'return'."},
  {"role": "assistant", "content": "سأقرأ الملف أولاً لأجد الخطأ الإملائي.\n\n```json\n{\"tool\": \"read\", \"path\": \"src/utils.py\"}\n```"}
]
```

## Batch Processing Script

If you want to automate this with a local model (e.g., via llama-server), here's a helper:

```python
#!/usr/bin/env python3
"""Batch-translate agentic conversations to Arabic using a local LLM API."""
import json
import requests
import sys

API_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "your-model-name"
INPUT_FILE = "agentic_subset_to_translate.jsonl"
OUTPUT_FILE = "arabic_translated_agentic.jsonl"

SYSTEM_PROMPT = """You are a professional Arabic translator. Translate the conversation to Arabic (MSA).
RULES: Keep code, JSON, function names, file paths in English. Translate natural language to Arabic.
Output ONLY the translated JSON array of messages, no explanation."""

with open(INPUT_FILE) as f_in, open(OUTPUT_FILE, "w") as f_out:
    for i, line in enumerate(f_in):
        row = json.loads(line)
        messages_str = json.dumps(row["messages"], ensure_ascii=False, indent=2)

        resp = requests.post(API_URL, json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate this conversation:\\n{messages_str}"}
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        })

        try:
            translated = json.loads(resp.json()["choices"][0]["message"]["content"])
            f_out.write(json.dumps({"messages": translated}, ensure_ascii=False) + "\\n")
            print(f"  [{i+1}] OK")
        except Exception as e:
            print(f"  [{i+1}] SKIP: {e}")

print(f"Done! Translated conversations saved to {OUTPUT_FILE}")
```
