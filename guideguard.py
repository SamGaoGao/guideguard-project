# Copyright (c) 2026 Gao Gao (高高). Licensed under the MIT License.
"""
GuideGuard: Non-invasive syntactic constraint for LLM decoding.
- Guides Generation,Guards Structure.

Author: Gao Gao (高高)
Project: GuideGuard (GG)

GuideGuard: Linguistically-Awakened LLM Output Control

A research prototype that applies gentle, context-aware grammatical constraints 
to LLM generation using part-of-speech (POS) rules.

How it works:
1. Waits until a meaningful phrase fragment is formed (e.g., "The dog")
2. Predicts the expected next word class (e.g., VERB)
3. Only then, restricts the output vocabulary to valid candidates,and erase highly dubious tokens
4. Steps back if constraints are too strict or model is highly confident

Philosophy: Guide, don't force. Guard, don't block.

Use this to:
- Save compute, and Reduce syntactic errors in generated text
- Explore linguistically-informed decoding
- Build safer, more coherent LLM pipelines

Note: This is a minimal, readable implementation for research and education.
For production, expect to optimize performance and extend rule coverage.
"""

from typing import Dict, Set, Tuple, Callable
import torch
from transformers import LogitsProcessor

class MCU_GuardLogitsProcessor(LogitsProcessor):
    """
    Apply POS-based constraints ONLY after MCU detection.

    Args:
        upos_mapper: token_id -> UPOS tag (e.g., "NOUN", "VERB")
        upos_to_token_ids: UPOS -> Set of valid token IDs
        meta_rules: {(prev_upos, curr_upos): next_expected_upos}
        min_tokens_after_punct: Minimum tokens after punctuation to activate
    """

    def __init__(
        self,
        upos_mapper: Callable[[int], str],
        upos_to_token_ids: Dict[str, Set[int]],
        meta_rules: Dict[Tuple[str, str], str],
        min_tokens_after_punct: int = 5
    ):
        self.upos_mapper = upos_mapper
        self.upos_to_token_ids = upos_to_token_ids
        self.meta_rules = meta_rules
        self.min_tokens_after_punct = min_tokens_after_punct

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, seq_len = input_ids.shape
        vocab_size = scores.shape[1]

        # If sequence length is less than 2, return original scores directly
        if seq_len < 2:
            return scores

        # Create a mask initialized to -inf (block all)
        final_mask = torch.full_like(scores, -float("inf"))

        for batch_idx in range(batch_size):
            # Reconstruct full UPOS sequence for this batch item (no shared state!)
            upos_sequence = []
            tokens_since_punct = 0

            for i in range(seq_len):
                token_id = input_ids[batch_idx, i].item()
                upos = self.upos_mapper(token_id)
                upos_sequence.append(upos)

                # Count tokens since last punctuation (for activation threshold)
                if upos == "PUNCT":
                    tokens_since_punct = 0
                else:
                    tokens_since_punct += 1

            # Check if we should activate constraints
            should_activate = (
                tokens_since_punct >= self.min_tokens_after_punct and
                seq_len >= 2
            )

            if should_activate:
                # Get last two UPOS tags
                upos_t_minus_2 = upos_sequence[-2]
                upos_t_minus_1 = upos_sequence[-1]

                # Look up expected next UPOS
                expected_upos = self.meta_rules.get((upos_t_minus_2, upos_t_minus_1))

                if expected_upos is not None:
                    allowed_ids = self.upos_to_token_ids.get(expected_upos, set())

                    # 🔥 Basic Fuses (Safety Nets)
                    fuse_triggered = False

                    # Fuse 1: Too few allowed tokens (relative to vocab size)
                    if len(allowed_ids) < max(1, vocab_size * 0.001):
                        fuse_triggered = True

                    # Fuse 2: Model is extremely confident in a disallowed token
                    top_token = scores[batch_idx].argmax().item()
                    if not fuse_triggered and top_token not in allowed_ids and scores[batch_idx, top_token] > 5.0:
                        fuse_triggered = True  # 5.0 is example threshold; adjust per model

                    if not fuse_triggered and allowed_ids:
                        # Apply constraint: allow only valid tokens
                        final_mask[batch_idx, list(allowed_ids)] = scores[batch_idx, list(allowed_ids)]
                        continue

            # If no constraint applied (or fuse triggered), keep original scores
            final_mask[batch_idx] = scores[batch_idx]

        return final_mask
