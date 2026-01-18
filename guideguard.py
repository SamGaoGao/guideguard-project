# Copyright (c) 2026 Gao Gao (高高). Licensed under the MIT License.
"""
GuideGuard: Non-invasive syntactic constraint for LLM decoding.
- Guards against structure-induced errors by masking invalid tokens at logits level.
- Guides generation toward universal grammar patterns (e.g., "被" → verb in Chinese).
- Includes fallback "eraser" to avoid dead ends.

Author: Gao Gao (高高)
Project: GuideGuard (GG)
"""

from typing import List, Optional, Dict, Any
import torch
from transformers import LogitsProcessor
import spacy

# Lazy load spaCy models
_SPACY_MODELS = {}

def _get_spacy_model(lang: str):
    if lang not in _SPACY_MODELS:
        try:
            if lang == "zh":
                _SPACY_MODELS[lang] = spacy.load("zh_core_web_sm")
            elif lang == "en":
                _SPACY_MODELS[lang] = spacy.load("en_core_web_sm")
            else:
                raise ValueError(f"Unsupported language: {lang}")
        except OSError:
            raise RuntimeError(
                f"spaCy model for '{lang}' not found. Install with:\n"
                f"python -m spacy download {'zh_core_web_sm' if lang == 'zh' else 'en_core_web_sm'}"
            )
    return _SPACY_MODELS[lang]

class GuideGuardLogitsProcessor(LogitsProcessor):
    """
    Applies dynamic syntactic constraints during LLM decoding.
    
    Only activates when context shows high certainty of next POS tag.
    Falls back to unconstrained generation if no valid tokens remain.
    """
    
    def __init__(
        self,
        tokenizer,
        language: str = "zh",  # "zh" or "en"
        confidence_threshold: float = 0.85,
        enable_eraser: bool = True
    ):
        self.tokenizer = tokenizer
        self.nlp = _get_spacy_model(language)
        self.confidence_threshold = confidence_threshold
        self.enable_eraser = enable_eraser
        
        # Enhanced syntactic patterns (v0.1.1)
        self.syntactic_patterns = {
            "zh": {
                "NP": [("的", "NOUN"), ("一个", "NOUN"), ("这种", "NOUN"), ("那个", "NOUN")],
                "VP": [("被", "VERB"), ("正在", "VERB"), ("已经", "VERB"), ("开始", "VERB")],
                "AP": [("很", "ADJ"), ("非常", "ADJ"), ("特别", "ADJ"), ("极其", "ADJ")]
            },
            "en": {
                "NP": [("the", "NOUN"), ("a", "NOUN"), ("an", "NOUN"), ("this", "NOUN")],
                "VP": [("will", "VERB"), ("can", "VERB"), ("is", "VERB"), ("has", "VERB")],
                "PP": [("in", "NOUN"), ("on", "NOUN"), ("at", "NOUN"), ("with", "NOUN")]
            }
        }
    
    def _predict_next_pos(self, input_ids: torch.LongTensor) -> Optional[str]:
        """Enhanced heuristic with pattern matching."""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[-3:].tolist())
        text = self.tokenizer.convert_tokens_to_string(tokens).strip().lower()
        
        if not text:
            return None
        
        lang = self.nlp.meta["lang"]
        patterns = self.syntactic_patterns.get(lang, {})
        
        # Check all patterns
        for category, rules in patterns.items():
            for trigger, expected_pos in rules:
                if text.endswith(trigger):
                    return expected_pos
        
        return None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        next_pos = self._predict_next_pos(input_ids)
        
        if next_pos is None:
            return scores  # No constraint applied
        
        # Get top-k candidate tokens and check their POS distribution
        topk_vals, topk_indices = torch.topk(scores, k=20, dim=-1)
        allowed_by_candidates = set()
        
        pos_count = {}
        for token_id in topk_indices[0][:10].tolist():  # inspect top 10
            token_str = self.tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(token_str, str):
                clean_token = token_str.replace("▁", "").replace("##", "")
                if clean_token.strip():
                    try:
                        doc = self.nlp(clean_token)
                        if doc and doc[0].pos_ == next_pos:
                            allowed_by_candidates.add(token_id)
                            pos_count[next_pos] = pos_count.get(next_pos, 0) + 1
                    except Exception:
                        pass
        
        # If at least 3 top candidates match the expected POS, enforce constraint
        if pos_count.get(next_pos, 0) >= 3:
            mask = torch.full_like(scores, -float("inf"))
            allowed_list = list(allowed_by_candidates)
            if allowed_list:
                mask[:, allowed_list] = scores[:, allowed_list]
                
                # Eraser fallback
                if self.enable_eraser:
                    original_max = torch.max(scores)
                    constrained_max = torch.max(mask)
                    if constrained_max < original_max * 0.1:
                        return scores
                
                return mask
        
        return scores
