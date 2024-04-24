"""Utility functions for compiler passes."""

from typing import Dict

import tvm
from tvm import IRModule, relax, te, tir
from tvm.relax.frontend import nn
from tvm.script import tir as T


def get_vocab_size(mod: IRModule) -> tir.PrimExpr:
    """Get the vocabulary size from the model."""
    # Prefill method exists in base models.
    # Prefill_to_last_hidden method exists in base model and speculative small models
    if "prefill" in mod:
        vocab_size = mod["prefill"].ret_struct_info.fields[0].shape[-1]
    else:
        assert (
            "prefill_to_last_hidden_states" in mod
        ), "Everay model should either has 'prefill' or 'prefill_to_last_hidden_states' method"
        vocab_size = mod["prefill_to_last_hidden_states"].ret_struct_info.fields[0].shape[-1]

    return vocab_size
