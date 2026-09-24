"""Verifies OptionMarkerModel's independent_options mode is truly order-invariant.

This is the core mathematical guarantee behind the cross-option-attention-blocking
architecture change: permuting which option occupies which slot in the packed
sequence must not change any option's computed logit. See
src/von/models/option_marker.py build_independent_option_masks /
build_option_invariant_position_ids for the design rationale.
"""

import itertools
import os

import pytest
import torch

from von.models.option_marker import OptionMarkerModel

CHECKPOINT = "checkpoints/von-option-marker-universal"

# This checks the encoder's own mask/position-id construction against a real
# checkpoint's config (max_position_embeddings, sliding_window, etc.), so it
# needs local weights rather than the base_model_id's hub-download fallback
# used elsewhere (that would fetch published weights of unknown attention
# mode). checkpoints/ is gitignored and absent in CI -- skip cleanly there
# instead of turning a missing local checkpoint into a false CI failure.
pytestmark = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT),
    reason=f"local checkpoint {CHECKPOINT!r} not present (gitignored; not fetched in CI)",
)

@pytest.fixture(scope="module")
def model():
    m = OptionMarkerModel(base_model_id=CHECKPOINT)
    m.eval()
    return m


def _score(model, state, question, options, independent_options):
    packed = model.pack_sequence(state, question, options)
    enc = model.tokenizer(packed, return_tensors="pt")
    positions = (enc["input_ids"][0] == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()
    with torch.no_grad():
        logits = model(
            enc["input_ids"], enc["attention_mask"], [positions],
            independent_options=independent_options,
        )[0]
    return logits


def test_two_option_reversal_is_invariant(model):
    state = "The invoice was submitted three days after the policy deadline of 30 days."
    question = "Is this invoice compliant?"
    options = ["Yes, it is compliant.", "No, it is not compliant."]

    fwd = _score(model, state, question, options, independent_options=True)
    rev = _score(model, state, question, list(reversed(options)), independent_options=True)

    assert torch.allclose(fwd, rev.flip(0), atol=1e-4)


def test_three_option_all_permutations_are_invariant(model):
    state = "The shipment weighed 42kg against a 40kg limit for the express tier."
    question = "What is the verdict?"
    options = ["Compliant, within policy.", "Non-compliant, exceeds threshold.", "Unclear, needs review."]

    restored_per_perm = []
    for perm in itertools.permutations(range(3)):
        opts_p = [options[i] for i in perm]
        logits = _score(model, state, question, opts_p, independent_options=True)
        restored = [None] * 3
        for slot, orig_idx in enumerate(perm):
            restored[orig_idx] = logits[slot].item()
        restored_per_perm.append(restored)

    baseline = restored_per_perm[0]
    for restored in restored_per_perm[1:]:
        for a, b in zip(baseline, restored):
            assert abs(a - b) < 1e-3


def test_default_mode_is_not_forced_invariant(model):
    """Sanity check the test itself: the ORIGINAL full-cross-attention mode is
    NOT expected to be order invariant, confirming independent_options is doing
    real work rather than the invariance being trivially true either way."""
    state = "The invoice was submitted three days after the policy deadline of 30 days."
    question = "Is this invoice compliant?"
    options = ["Yes, it is compliant.", "No, it is not compliant."]

    fwd = _score(model, state, question, options, independent_options=False)
    rev = _score(model, state, question, list(reversed(options)), independent_options=False)

    assert not torch.allclose(fwd, rev.flip(0), atol=1e-4)


def test_batching_with_padding_matches_unbatched(model):
    state = "The invoice was submitted three days after the policy deadline of 30 days."
    question = "Is this invoice compliant?"
    short_opts = ["Yes, it is compliant.", "No, it is not compliant."]
    long_opts = [
        "Compliant, fully within the policy window and no further action needed.",
        "Not compliant, exceeds the deadline threshold and requires escalation.",
    ]

    packed_short = model.pack_sequence(state, question, short_opts)
    packed_long = model.pack_sequence(state, question, long_opts)
    enc = model.tokenizer([packed_short, packed_long], return_tensors="pt", padding=True)
    pos_batch = [
        (row == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        for row in enc["input_ids"]
    ]
    with torch.no_grad():
        batched = model(
            enc["input_ids"], enc["attention_mask"], pos_batch, independent_options=True,
        )

    unbatched = _score(model, state, question, short_opts, independent_options=True)
    assert torch.allclose(batched[0], unbatched, atol=1e-4)
