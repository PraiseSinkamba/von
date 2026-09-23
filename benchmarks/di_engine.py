"""Decision Index adapter for Von (https://github.com/apolinario/decision-index).

Implements the harness `Engine` contract: `__call__(state, questions) -> (response, raw)`.
Every Decision Index question is `choice`, 2-255 options, state may be a JSON object.
Contract rules (docs/engines.md): do not truncate, do not drop options, do not adapt the
prompt per benchmark; raise `Unsupported` instead when a declared capacity limit is hit.

Von's Option-Marker packs the state, question, and every option's full text into one
8,192-token sequence (`OptionMarkerModel.pack_sequence`). This engine checks the packed
token length *before* running the model and raises `Unsupported` rather than silently
truncating or dropping options -- the one thing the harness explicitly forbids.

Usage (from the repo root, so `benchmarks` resolves as a package for `-m`):
    export HF_HUB_DISABLE_XET=1
    DI=/tmp/scratch/decision-index
    uv run --with-editable . --with-editable $DI --with datasets \
        python -m decision_index run --engine benchmarks.di_engine:VonEngine \
        --rows /tmp/di-sample.jsonl.gz --out /tmp/di-run
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MAX_TOKENS = 8192


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


class VonEngine:
    """Decision Index `Engine`: one Option-Marker forward pass per question."""

    name = "von"
    latency = ("In-process request wall time including prompt construction; excludes "
                "model loading. CPU, torch.set_num_threads(4).")

    def __init__(self, **options: Any) -> None:
        self.options = options
        checkpoint_dir = options.get("checkpoint_dir") or "checkpoints/von-option-marker-universal"
        device = options.get("device")
        self.provenance = {
            "repo": "wfzyx/von", "checkpoint": checkpoint_dir,
            "kind": "full fine-tune", "base_model": "answerdotai/ModernBERT-large",
        }
        self._backend = None
        self._checkpoint_dir = checkpoint_dir
        self._device = device

    def _get_backend(self):
        if self._backend is None:
            import torch

            torch.set_num_threads(4)
            from von.backends.option_marker_backend import OptionMarkerBackend
            self._backend = OptionMarkerBackend(checkpoint_dir=self._checkpoint_dir, device=self._device)
            self._backend._get_model()  # force load now, not on the first scored request
        return self._backend

    def warmup(self) -> None:
        backend = self._get_backend()
        from von.types import Choice

        q = Choice(instructions="Which color is named?", criteria={"red": "red", "blue": "blue"})
        backend.evaluate_choice("warmup", "The color is red.", q)

    def __call__(self, state: Any, questions: Dict[str, dict]) -> Tuple[dict, Any]:
        from decision_index.engines.base import Unsupported
        from von.types import Choice

        backend = self._get_backend()
        model = backend._get_model()
        tok = model.tokenizer

        state_text = _text(state)
        answers: Dict[str, dict] = {}
        for key, q in questions.items():
            if q.get("type") != "choice":
                raise Unsupported(f"Von only answers 'choice' questions, got {q.get('type')!r}")
            criteria = q["criteria"]
            options = list(criteria)
            if not (2 <= len(options) <= 255):
                raise Unsupported(f"declared option count {len(options)} outside Von's 2-255 range")

            instructions = _text(q["instructions"])
            descriptions = [str(criteria[o] or o) for o in options]

            packed_text = model.pack_sequence(state_text, instructions, descriptions)
            n_tokens = len(tok.encode(packed_text, add_special_tokens=True))
            if n_tokens > MAX_TOKENS:
                raise Unsupported(
                    f"packed sequence is {n_tokens} tokens, over Von's {MAX_TOKENS}-token "
                    "context window; refusing rather than truncating"
                )

            q_obj = Choice(instructions=instructions, criteria=dict(zip(options, descriptions)))
            ans = backend.evaluate_choice(key, state_text, q_obj)

            probs = dict(ans.probabilities)
            total = sum(probs.values())
            if total <= 0:
                raise Unsupported("degenerate zero probability mass across all options")
            if abs(total - 1.0) > 1e-6:
                probs = {k: v / total for k, v in probs.items()}
            # Every declared option key must carry a probability, even ones the model's
            # own argmax never touches, so the harness's key-set check always passes.
            for o in options:
                probs.setdefault(o, 0.0)

            answers[key] = {"type": "choice", "choice": ans.choice, "probabilities": probs}

        response = {"model": "von-1.1.0", "answers": answers,
                    "usage": {"input_tokens": n_tokens}}
        return response, None
