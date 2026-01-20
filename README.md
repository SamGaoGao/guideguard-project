[![Concept Prototype](https://img.shields.io/badge/status-concept_prototype-orange?logo=idea)](https://github.com/samgaogao/guideguard)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

# GuideGuard (GG)  
> **Guide the generation. Guard the structure.**

A lightweight, non-invasive engine for LLM decoding — **zero training**, works out-of-the-box with **Llama, Qwen, GLM, Mistral, Gemma, Phi**, and any HuggingFace-compatible autoregressive model via `LogitsProcessor`.

Created by Gao Gao (高高) • MIT Licensed • Free for commercial use

---

## 💡 Why GuideGuard?

Large language models often waste massive compute exploring tokens that **sound plausible in isolation but break sentence structure** — even when grammar makes the next word’s role obvious.

GuideGuard cuts that waste at the source with two real-time mechanisms:

### 🧭 Stage 1: Guide — Focus Early, Save Compute  
When the grammatical role of the next word is already clear from context,  
there’s no need to sample from all 100k+ tokens.

- After **“The”**, we expect a **noun** — not a pure verb like *“explode”*.  
  ✅ “The explosion happened.”  
  ❌ “The explode…”  

- After Chinese passive marker **“被”**, we expect a **verb phrase** — not a bare noun like *“苹果”*.  
  ✅ “他被批评了。”  
  ❌ “他被苹果。”

In these high-certainty moments, GuideGuard **shrinks the candidate pool** to only tokens matching the expected grammatical role.

→ **Fewer wasted samples, lower latency, less cost.**

### 🛡️ Stage 2: Guard — Self-Correct as You Go  
After each token is generated, GG checks it against basic syntactic rules.

If a clear violation is detected — e.g., *“The explode”* or *“被 桌子”* —  
GG discards the token and asks the model to resample (up to 5 times).  
If no valid alternative exists, it safely falls back to the original output.

→ **Catches errors the moment they happen, not after hundreds of tokens.**

Because it’s built on **universal grammatical roles** (NOUN, VERB, PASSIVE, etc.),  
GuideGuard is **language-agnostic** — extendable to English, Chinese, or any language.

> **Technical Note**: Language agnosticism requires providing:
> - A UPOS mapper (`token_id → Universal POS tag`) for your tokenizer
> - Language-specific meta-rules (`(upos_{t-2}, upos_{t-1}) → expected_upos_t`)

> ✨ **Value**: Lower compute cost + more robust output — with **zero training**.

---

## ⚠️ This Is a Concept Prototype

While the core logic has been validated for correctness (state safety, batch processing, constraint activation), this release is **strictly a research prototype**:

- ❌ **Not production-ready**: You must implement key linguistic components yourself.
- ✅ **But logically sound**: No shared state, proper MCU detection, working fuses.
- 💡 **Designed for collaboration**: A minimal, correct seed for community extension.

We prioritize **clarity over completeness** so you can understand, verify, and build upon the idea.

---

## 🧩 What You Must Implement

To use GuideGuard, provide three simple components:

1. **`upos_mapper`**: `token_id → UPOS tag` (e.g., `"NOUN"`, `"VERB"`)  
   → Map your tokenizer’s IDs to universal POS tags.

2. **`upos_to_token_ids`**: `UPOS → Set[token_id]`  
   → Reverse lookup: which tokens belong to each grammatical class?

3. **`meta_rules`**: `{(upos_{t-2}, upos_{t-1}): expected_upos_t}`  
   → Your language-specific syntactic rules (e.g., `("NOUN", "VERB") → "NOUN"`).

These can be static dictionaries, rule-based systems, or even small classifiers — the interface is open.

---

## 🧪 Demo

`demo.py` provides a **self-contained, runnable example** that validates the core logic:
- Shows how invalid tokens are blocked
- Demonstrates constraint activation after MCU detection
- Verifies batch safety and fuse behavior

Run it to see the “Guide & Guard” effect in action.

---

## 🔒 Safety & Design Principles

- **No shared state**: Each sequence in a batch is processed independently.
- **Basic fuses**: Prevent pathological over-constraint (e.g., too few allowed tokens).
- **HuggingFace native**: Implements `transformers.LogitsProcessor` — drop-in compatible.
- **Fail-safe**: Falls back to original logits if constraints are too restrictive.

---

## ⚡ Performance Note

This prototype prioritizes **conceptual clarity over speed**. For production use, consider:
- Caching UPOS mappings
- Vectorizing batch operations
- Precomputing allowed token sets per rule

---

## 📚 How to Cite

If you use GuideGuard in academic or technical work, please cite it as software:

> **APA Format**: Gao, S. (2026). *GuideGuard: Delayed Syntactic Constraint via Minimal Complete Units* [Computer software]. https://github.com/samgaogao/guideguard

---

## 🌐 Vision

GuideGuard is more than code — it's a provocation:  
*What if LLMs could be both more creative and more efficient, simply by respecting the grammar they already know?*

---

## ©️ License

MIT License.  
Concept and implementation by Gao Gao (高高).  
Built for open research and collaboration — free for commercial and non-commercial use.
