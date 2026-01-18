# GuideGuard (GG)

> **Guide the generation. Guard the structure.**  
> A lightweight, non-invasive engine that applies dynamic syntactic constraints during LLM decoding — reducing structure-induced errors and saving compute, with zero domain knowledge required.

Created by [Gao Gao (高高)](https://github.com/samgaogao) • MIT Licensed • Free for commercial use

—

## ✨ Why GuideGuard?

Most hallucinations aren’t lies — they‘re **structural drifts**:  
- For instance, the Chinese character  "被" usually indicates a passive voice, meaning the subject is acted upon rather than acting. "被" followed by a noun (should be verb, or noun turned into verb, or verb with adverbial modification)  
- “The” followed by a verb (should be noun)  
- Missing arguments, broken dependencies  

GuideGuard doesn’t claim to eliminate all hallucinations.  
It reduces the risk of **structure-induced errors** by keeping generation aligned with universal grammar patterns.

And because it prunes the token search space early, it also **saves compute** — fewer wasted tokens, lower latency, less cost.

Immediately after the generation of a new token, GuideGuard checks it with universal grammar patterns and erase apparently inappropriate token,forcing the LLM generate another replacing token(up to five times).

GG acts only when probability is above threshold (80%,for instance).

—

## 🚀 Features

- ✅ **Non-invasive**: Plug into any HuggingFace model via `LogitsProcessor`
- ✅ **Zero training**: Works out-of-the-box with Llama, Qwen, GLM, GPT, etc.
- ✅ **Multi-language**: Built-in support for Chinese & English
- ✅ **Smart fallback**: ”Eraser“ prevents dead ends
- ✅ **MIT licensed**: Use freely in commercial products

—

## 🛠️ Quick Start

```bash
pip install guideguard  # (coming soon) OR
git clone https://github.com/samgaogao/guideguard.git
