# GuideGuard (GG)

**Guide the generation. Guard the structure.**

- **Guide** → **saves compute** by shrinking the candidate pool when grammar is clear  
- **Guard** → **reduces structural errors** by correcting violations as they happen  

A lightweight, non-invasive engine for LLM decoding — zero training, Works out-of-the-box with Llama, Qwen, GLM, Mistral, Gemma, Phi, and any HuggingFace-compatible autoregressive model via  LogitsProcessor .

Created by [Gao Gao (高高)](https://github.com/samgaogao) • MIT Licensed • Free for commercial use

---

###  How It Works

####  Stage 1: **Guide** — Focus Early, Save Compute

When the grammatical role of the next word is already clear from context,  
there’s no need to sample from all 100k+ tokens.

For example:
- After `"The"`, we expect a **noun** — not a pure verb like `"explode"`.
  -  Correct: `"The explosion happened."`
  -  Invalid: `"The explode..."` (*explode* is almost exclusively a verb and has no standard noun usage)
- After the Chinese passive marker `"被"` (used to form passive voice), we expect a **verb phrase** — not a bare noun like `"苹果"`.

In these high-certainty moments, GuideGuard **immediately shrinks the candidate pool** to only tokens matching the expected grammatical role.

→ Fewer wasted samples, lower latency, less cost.

#### ️ Stage 2: **Guard** — Self-Correct as You Go

After each token is generated, GG checks it against basic syntactic rules.

If a clear violation is detected — e.g., `"The explode"` or `"被 桌子"` —  
GG discards the token and asks the model to resample (up to 5 times).  
If no valid alternative exists, it safely falls back to the original output.

→ Catches errors **the moment they happen**, not after hundreds of tokens.

---

###  Why It Matters

LLMs often waste compute exploring words that **sound plausible alone** but **break sentence structure** — even when the grammar is obvious.

GuideGuard cuts that waste at the source:
- **Lower compute cost**: By focusing the search early in high-certainty contexts  
- **More robust output**: By enforcing basic grammar in real time  

And because it’s built on universal grammatical roles (noun, verb, passive marker, etc.),  
it’s **not limited to English or Chinese** — you can extend it to any language.

---

###  Quick Start

```bash
git clone https://github.com/samgaogao/guideguard.git

