"""
Arabic Agentic Dataset Builder
==============================
Downloads, normalizes, and merges multiple Arabic-focused datasets into a
single ShareGPT-format JSONL file for Gemma 4 fine-tuning.

Layers:
  1. Arabic Conversation & Instruction (40%)  — CIDAR, Evol-Instruct, ShareGPT, Aya
  2. Arabic Tool Calling (20%)                — Arabic_Function_Calling, AISA-AR
  3. Translation AR↔EN (20%)                  — OPUS-100, FLORES
  4. Arabic Knowledge & QA (20%)              — Aya (remaining), Wikipedia QA

Output: arabic_agentic_dataset.jsonl

Usage:
  python build_arabic_dataset.py
  python build_arabic_dataset.py --dry-run          # Preview counts, don't write
  python build_arabic_dataset.py --max-total 10000   # Limit total samples
"""

import json
import os
import random
import argparse
import ast
import hashlib
from collections import defaultdict

# ── Arabic System Prompt ──────────────────────────────────────────────────
ARABIC_SYSTEM_PROMPT = (
    "أنت مساعد ذكي ثنائي اللغة (عربي/إنجليزي). يمكنك:\n"
    "- إجراء محادثات باللغة العربية الفصحى والعامية المصرية\n"
    "- ترجمة النصوص بين العربية والإنجليزية\n"
    "- البحث عن المعلومات وجمعها وتلخيصها في تقارير عربية\n"
    "- استخدام الأدوات المتاحة لتنفيذ المهام\n\n"
    "عند استخدام الأدوات، اكتب استدعاء الأداة بصيغة JSON المحددة.\n"
    "أجب دائماً باللغة التي يستخدمها المستخدم ما لم يُطلب خلاف ذلك."
)

OUTPUT_FILE = "arabic_agentic_dataset.jsonl"
SEED = 3407


# ── Helpers ─────────────────────────────────────────────────────────────

def content_hash(messages: list) -> str:
    """Deterministic hash for deduplication."""
    text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()


def has_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return any('\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' for c in text)


def inject_system_prompt(messages: list) -> list:
    """Inject Arabic system prompt if conversation lacks one."""
    if messages and messages[0].get("role") == "system":
        return messages  # Already has system message
    return [{"role": "system", "content": ARABIC_SYSTEM_PROMPT}] + messages


def normalize_roles(messages: list) -> list:
    """Normalize roles for Gemma 4 compatibility."""
    ROLE_MAP = {
        "human": "user",
        "gpt": "assistant",
        "model": "assistant",
        "function_call": "assistant",
        "observation": "tool",
        "function": "tool",
        "developer": "system",     # AISA-Framework uses 'developer' for system
    }
    normalized = []
    for msg in messages:
        role = ROLE_MAP.get(msg.get("role", ""), msg.get("role", "user"))
        content = msg.get("content", "") or ""
        if isinstance(content, list):
            # Handle content arrays (multimodal format)
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts)

        # Handle tool_calls embedded in assistant messages (AISA format)
        tool_calls = msg.get("tool_calls", None)
        if tool_calls and role == "assistant":
            tc_parts = []
            for tc in tool_calls:
                func = tc.get("function", {})
                fname = func.get("name", "unknown")
                fargs = func.get("arguments", {})
                tc_parts.append(json.dumps(
                    {"tool": fname, "arguments": fargs},
                    ensure_ascii=False,
                ))
            tc_text = "\n".join(tc_parts)
            if content:
                content = f"{content}\n\n{tc_text}"
            else:
                content = tc_text

        content = content.strip() if isinstance(content, str) else ""
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def validate_conversation(messages: list) -> bool:
    """Check conversation is valid for training."""
    if len(messages) < 2:
        return False
    # Must have at least one user and one assistant message
    roles = {m["role"] for m in messages}
    if "user" not in roles or "assistant" not in roles:
        return False
    # Content must not be trivially short
    total_content = sum(len(m.get("content", "")) for m in messages)
    if total_content < 50:
        return False
    return True


# ── Dataset Loaders ─────────────────────────────────────────────────────

def load_cidar(max_samples: int = 10000) -> list:
    """arbml/CIDAR — Gold standard Arabic instruction dataset."""
    print("  Loading arbml/CIDAR...")
    try:
        from datasets import load_dataset
        ds = load_dataset("arbml/CIDAR", split="train")
        samples = []
        for row in ds:
            instruction = row.get("instruction", "") or ""
            output = row.get("output", "") or ""
            input_text = row.get("input", "") or ""
            if input_text:
                instruction = f"{instruction}\n\n{input_text}"
            if not instruction.strip() or not output.strip():
                continue
            messages = [
                {"role": "user", "content": instruction.strip()},
                {"role": "assistant", "content": output.strip()},
            ]
            samples.append({"messages": inject_system_prompt(messages)})
            if len(samples) >= max_samples:
                break
        print(f"    → {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"    ⚠️ Failed: {e}")
        return []


def load_evol_instruct_arabic(max_samples: int = 8000) -> list:
    """FreedomIntelligence/Evol-Instruct-Arabic — Evolved complexity instructions."""
    print("  Loading FreedomIntelligence/Evol-Instruct-Arabic...")
    try:
        from datasets import load_dataset
        ds = load_dataset("FreedomIntelligence/Evol-Instruct-Arabic", split="train")
        # Shuffle and subsample
        indices = list(range(len(ds)))
        random.shuffle(indices)
        samples = []
        for idx in indices:
            row = ds[idx]
            # Try different column names
            instruction = row.get("instruction", "") or row.get("input", "") or ""
            output = row.get("output", "") or row.get("response", "") or ""
            if not instruction.strip() or not output.strip():
                continue
            messages = [
                {"role": "user", "content": instruction.strip()},
                {"role": "assistant", "content": output.strip()},
            ]
            samples.append({"messages": inject_system_prompt(messages)})
            if len(samples) >= max_samples:
                break
        print(f"    → {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"    ⚠️ Failed: {e}")
        return []


def load_arabic_sharegpt(max_samples: int = 5000) -> list:
    """Omartificial-Intelligence-Space/Arabic-ShareGPT — Multi-turn Arabic chats."""
    print("  Loading Omartificial-Intelligence-Space/Arabic-ShareGPT...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Omartificial-Intelligence-Space/Arabic-ShareGPT", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        samples = []
        for idx in indices:
            row = ds[idx]
            # ShareGPT format: "conversations" field
            convos = row.get("conversations", []) or row.get("messages", [])
            if not convos:
                continue
            messages = normalize_roles(convos)
            if not validate_conversation(messages):
                continue
            samples.append({"messages": inject_system_prompt(messages)})
            if len(samples) >= max_samples:
                break
        print(f"    → {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"    ⚠️ Failed: {e}")
        return []


def load_aya_arabic(max_samples: int = 6000) -> list:
    """CohereForAI/aya_dataset — Arabic subset, human-curated multi-task."""
    print("  Loading CohereForAI/aya_dataset (Arabic)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("CohereForAI/aya_dataset", split="train")
        samples = []
        for row in ds:
            lang = row.get("language", "") or row.get("language_code", "")
            if lang not in ("ara", "ar", "Arabic", "arabic"):
                continue
            inputs = row.get("inputs", "") or ""
            targets = row.get("targets", "") or ""
            if not inputs.strip() or not targets.strip():
                continue
            messages = [
                {"role": "user", "content": inputs.strip()},
                {"role": "assistant", "content": targets.strip()},
            ]
            samples.append({"messages": inject_system_prompt(messages)})
            if len(samples) >= max_samples:
                break
        print(f"    → {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"    ⚠️ Failed: {e}")
        return []


def load_arabic_function_calling(max_samples: int = 10000) -> list:
    """HeshamHaroon/Arabic_Function_Calling — Arabic tool calling dataset.
    
    Schema: query_ar, query_en, function_name, arguments, dialect, domain, requires_function
    Dialects: MSA (14K), Egyptian (10K), Gulf (11K), Levantine (6K), Maghrebi (2K)
    We keep MSA + Egyptian only.
    """
    print("  Loading HeshamHaroon/Arabic_Function_Calling...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HeshamHaroon/Arabic_Function_Calling", split="train")
        samples = []
        dialect_counts = defaultdict(int)
        for row in ds:
            # Filter: MSA + Egyptian only
            dialect = row.get("dialect", "")
            if dialect not in ("MSA", "Egyptian"):
                continue
            dialect_counts[dialect] += 1

            query_ar = (row.get("query_ar", "") or "").strip()
            func_name = (row.get("function_name", "") or "").strip()
            arguments = row.get("arguments", "") or ""
            requires_func = row.get("requires_function", True)

            if not query_ar:
                continue

            # Build assistant response as a tool call
            if requires_func and func_name:
                # Parse arguments if string
                if isinstance(arguments, str):
                    try:
                        args_obj = json.loads(arguments)
                    except json.JSONDecodeError:
                        args_obj = {"raw": arguments}
                else:
                    args_obj = arguments or {}

                tool_call = json.dumps(
                    {"tool": func_name, "arguments": args_obj},
                    ensure_ascii=False, indent=2,
                )
                assistant_content = f"سأستخدم الأداة المناسبة لتنفيذ طلبك.\n\n```json\n{tool_call}\n```"
            else:
                # No function needed — answer directly
                assistant_content = "لا يلزم استخدام أداة لهذا الطلب. كيف يمكنني مساعدتك بطريقة أخرى؟"

            messages = [
                {"role": "user", "content": query_ar},
                {"role": "assistant", "content": assistant_content},
            ]
            samples.append({"messages": inject_system_prompt(messages)})
            if len(samples) >= max_samples:
                break

        print(f"    → {len(samples)} samples (" +
              ", ".join(f"{d}: {c}" for d, c in sorted(dialect_counts.items())) + ")")
        return samples
    except Exception as e:
        print(f"    ⚠️ Failed: {e}")
        return []


def load_aisa_function_call(max_samples: int = 10000) -> list:
    """AISA-Framework/AISA-AR-FunctionCall — Arabic agentic function calling.
    
    Schema: messages (with developer/user/assistant roles + tool_calls),
            tools, dialect, domain, requires_function, tool_called
    The messages use 'developer' role (→ system) and embed tool_calls
    as sub-objects in assistant messages with content=None.
    We keep MSA + Egyptian only.
    """
    print("  Loading AISA-Framework/AISA-AR-FunctionCall...")
    try:
        from datasets import load_dataset
        ds = load_dataset("AISA-Framework/AISA-AR-FunctionCall", split="train")
        samples = []
        dialect_counts = defaultdict(int)
        for row in ds:
            # Filter: MSA + Egyptian only
            dialect = row.get("dialect", "")
            if dialect not in ("MSA", "Egyptian"):
                continue
            dialect_counts[dialect] += 1

            raw_messages = row.get("messages", [])
            if not raw_messages:
                continue

            # Handle messages stored as string (some HF datasets serialize as Python repr)
            if isinstance(raw_messages, str):
                try:
                    raw_messages = json.loads(raw_messages)
                except json.JSONDecodeError:
                    try:
                        raw_messages = ast.literal_eval(raw_messages)
                    except (ValueError, SyntaxError):
                        continue

            # Normalize roles (developer→system, handle tool_calls)
            messages = normalize_roles(raw_messages)
            if not validate_conversation(messages):
                continue

            samples.append({"messages": inject_system_prompt(messages)})
            if len(samples) >= max_samples:
                break

        print(f"    → {len(samples)} samples (" +
              ", ".join(f"{d}: {c}" for d, c in sorted(dialect_counts.items())) + ")")
        return samples
    except Exception as e:
        print(f"    ⚠️ Failed: {e}")
        return []


def load_opus_translation(max_samples: int = 8000) -> list:
    """Helsinki-NLP/opus-100 ar-en — Parallel translation corpus as instructions."""
    print("  Loading Helsinki-NLP/opus-100 (ar-en)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Helsinki-NLP/opus-100", "ar-en", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)

        samples = []
        # 50/50 split: AR→EN and EN→AR
        ar_to_en_count = 0
        en_to_ar_count = 0
        half = max_samples // 2

        for idx in indices:
            row = ds[idx]
            translation = row.get("translation", {})
            ar_text = translation.get("ar", "").strip()
            en_text = translation.get("en", "").strip()
            if not ar_text or not en_text:
                continue
            # Skip very short pairs
            if len(ar_text) < 20 or len(en_text) < 20:
                continue

            if ar_to_en_count < half:
                messages = [
                    {"role": "user", "content": f"ترجم النص التالي إلى الإنجليزية:\n\n{ar_text}"},
                    {"role": "assistant", "content": en_text},
                ]
                samples.append({"messages": inject_system_prompt(messages)})
                ar_to_en_count += 1
            elif en_to_ar_count < half:
                messages = [
                    {"role": "user", "content": f"Translate the following text to Arabic:\n\n{en_text}"},
                    {"role": "assistant", "content": ar_text},
                ]
                samples.append({"messages": inject_system_prompt(messages)})
                en_to_ar_count += 1

            if len(samples) >= max_samples:
                break

        print(f"    → {len(samples)} samples ({ar_to_en_count} AR→EN, {en_to_ar_count} EN→AR)")
        return samples
    except Exception as e:
        print(f"    ⚠️ Failed: {e}")
        return []


def load_flores_translation(max_samples: int = 2000) -> list:
    """facebook/flores — Expert-translated benchmark quality pairs."""
    print("  Loading facebook/flores (ar)...")
    try:
        from datasets import load_dataset
        # FLORES-200 uses language codes like "ara_Arab" and "eng_Latn"
        ds = load_dataset("facebook/flores", "all", split="devtest")
        samples = []
        for row in ds:
            ar_text = row.get("sentence_ara_Arab", "").strip()
            en_text = row.get("sentence_eng_Latn", "").strip()
            if not ar_text or not en_text:
                continue

            # AR→EN
            messages_ar_en = [
                {"role": "user", "content": f"ترجم إلى الإنجليزية:\n{ar_text}"},
                {"role": "assistant", "content": en_text},
            ]
            samples.append({"messages": inject_system_prompt(messages_ar_en)})

            # EN→AR
            messages_en_ar = [
                {"role": "user", "content": f"Translate to Arabic:\n{en_text}"},
                {"role": "assistant", "content": ar_text},
            ]
            samples.append({"messages": inject_system_prompt(messages_en_ar)})

            if len(samples) >= max_samples:
                break

        print(f"    → {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"    ⚠️ Failed: {e}")
        return []


def load_user_translated(filepath: str = "arabic_translated_agentic.jsonl") -> list:
    """Load user-translated agentic data if available."""
    print(f"  Looking for user-translated data: {filepath}...")
    if not os.path.isfile(filepath):
        print(f"    → Not found (skipping — create with translate_agentic_prompt.md)")
        return []
    try:
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                messages = row.get("messages", [])
                messages = normalize_roles(messages)
                if validate_conversation(messages):
                    samples.append({"messages": inject_system_prompt(messages)})
        print(f"    → {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"    ⚠️ Failed: {e}")
        return []


# ── Main Pipeline ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Arabic agentic training dataset")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without writing")
    parser.add_argument("--max-total", type=int, default=50000, help="Max total samples")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output JSONL path")
    args = parser.parse_args()

    random.seed(SEED)

    print("=" * 60)
    print("Arabic Agentic Dataset Builder")
    print("=" * 60)

    # ── Layer 1: Conversation & Instruction (40%) ─────────────────
    print("\n📚 Layer 1: Arabic Conversation & Instruction Following")
    layer1 = []
    layer1.extend(load_cidar())
    layer1.extend(load_evol_instruct_arabic())
    layer1.extend(load_arabic_sharegpt())
    layer1.extend(load_aya_arabic())
    print(f"  Layer 1 total: {len(layer1)}")

    # ── Layer 2: Tool Calling (20%) ───────────────────────────────
    print("\n🔧 Layer 2: Arabic Tool Calling & Function Calling")
    layer2 = []
    layer2.extend(load_arabic_function_calling())
    layer2.extend(load_aisa_function_call())
    layer2.extend(load_user_translated())
    print(f"  Layer 2 total: {len(layer2)}")

    # ── Layer 3: Translation AR↔EN (20%) ──────────────────────────
    print("\n🌐 Layer 3: Translation (AR ↔ EN)")
    layer3 = []
    layer3.extend(load_opus_translation())
    layer3.extend(load_flores_translation())
    print(f"  Layer 3 total: {len(layer3)}")

    # ── Layer 4: Knowledge & QA (20%) ─────────────────────────────
    # Aya data already loaded in Layer 1; Layer 4 supplements with
    # whatever additional QA datasets are available
    print("\n📖 Layer 4: Arabic Knowledge & QA")
    layer4 = []  # Additional QA datasets can be added here
    print(f"  Layer 4 total: {len(layer4)} (supplemented by Aya in Layer 1)")

    # ── Merge & Balance ───────────────────────────────────────────
    print("\n⚖️  Merging and balancing...")
    all_samples = layer1 + layer2 + layer3 + layer4

    # Deduplicate
    seen_hashes = set()
    unique_samples = []
    dupes = 0
    for sample in all_samples:
        h = content_hash(sample["messages"])
        if h in seen_hashes:
            dupes += 1
            continue
        seen_hashes.add(h)
        unique_samples.append(sample)

    print(f"  Removed {dupes} duplicates")
    print(f"  Unique samples: {len(unique_samples)}")

    # Cap at max_total
    if len(unique_samples) > args.max_total:
        random.shuffle(unique_samples)
        unique_samples = unique_samples[:args.max_total]
        print(f"  Capped to {args.max_total} samples")

    # Final shuffle
    random.shuffle(unique_samples)

    # ── Stats ─────────────────────────────────────────────────────
    print("\n📊 Final Dataset Statistics:")
    print(f"  Total samples: {len(unique_samples)}")

    # Count Arabic content
    arabic_count = sum(1 for s in unique_samples
                       if any(has_arabic(m.get("content", "")) for m in s["messages"]))
    print(f"  Contains Arabic: {arabic_count} ({arabic_count/max(len(unique_samples),1)*100:.0f}%)")

    # Average conversation length
    avg_msgs = sum(len(s["messages"]) for s in unique_samples) / max(len(unique_samples), 1)
    print(f"  Avg messages/conversation: {avg_msgs:.1f}")

    # Role distribution
    role_counts = defaultdict(int)
    for s in unique_samples:
        for m in s["messages"]:
            role_counts[m["role"]] += 1
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count} messages")

    if args.dry_run:
        print("\n🔍 DRY RUN — no file written.")
        print("  Run without --dry-run to write the dataset.")
        return

    # ── Write ─────────────────────────────────────────────────────
    print(f"\n💾 Writing to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in unique_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Done! {args.output} ({file_size:.1f} MB)")
    print(f"\n✅ Dataset ready for training.")


if __name__ == "__main__":
    main()
