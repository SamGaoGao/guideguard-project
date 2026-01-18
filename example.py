# Copyright (c) 2026 Gao Gao. MIT License.
"""
Example usage of GuideGuard with HuggingFace transformers.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from guideguard import GuideGuardLogitsProcessor

print("=== GuideGuard v0.1.1 Examples ===\n")

# Example 1: Chinese - enforce verb after "被"
print("1. Chinese passive voice:")
tokenizer_zh = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
model_zh = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

gg_zh = GuideGuardLogitsProcessor(tokenizer_zh, language="zh")

prompt = "他被"
inputs = tokenizer_zh(prompt, return_tensors="pt")

outputs = model_zh.generate(
    **inputs,
    max_new_tokens=8,
    logits_processor=[gg_zh],
    pad_token_id=tokenizer_zh.eos_token_id
)

print("With GuideGuard:", tokenizer_zh.decode(outputs[0], skip_special_tokens=True))

# Example 2: English - noun after "the"
print("\n2. English article constraint:")
tokenizer_en = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model_en = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

gg_en = GuideGuardLogitsProcessor(tokenizer_en, language="en")

prompt_en = "The"
inputs_en = tokenizer_en(prompt_en, return_tensors="pt")

outputs_en = model_en.generate(
    **inputs_en,
    max_new_tokens=6,
    logits_processor=[gg_en],
    pad_token_id=tokenizer_en.eos_token_id
)

print("With GuideGuard:", tokenizer_en.decode(outputs_en[0], skip_special_tokens=True))

print("\n✅ Examples complete. Try your own prompts!")
