import numpy as np
import pytest
import tvm
import tvm.testing

from mlc_llm.op.batch_spec_verify import batch_spec_verify


draft_probs = np.load("draft_probs.npy")
draft_tokens = np.load("draft_tokens.npy")
model_probs = np.load("model_probs.npy")
token_tree_first_child = np.load("token_tree_first_child.npy")
token_tree_next_sibling = np.load("token_tree_next_sibling.npy")
uniform_samples = np.load("uniform_samples.npy")
token_tree_parent_ptr = np.load("token_tree_parent_ptr.npy")

print('draft_probs')
print(draft_probs)
print('draft_tokens')
print(draft_tokens)
print('model_probs')
print(model_probs)
print('token_tree_first_child')
print(token_tree_first_child)
print('token_tree_next_sibling')
print(token_tree_next_sibling)
print('uniform_samples')
print(uniform_samples)
print('token_tree_parent_ptr')
print(token_tree_parent_ptr)

dev = tvm.cuda(0)
draft_probs_tvm = tvm.nd.array(draft_probs, dev)
draft_tokens_tvm = tvm.nd.array(draft_tokens, dev)
model_probs_tvm = tvm.nd.array(model_probs, dev)
token_tree_first_child_tvm = tvm.nd.array(token_tree_first_child, dev)
token_tree_next_sibling_tvm = tvm.nd.array(token_tree_next_sibling, dev)
uniform_samples_tvm = tvm.nd.array(uniform_samples, dev)
token_tree_parent_ptr_tvm = tvm.nd.array(token_tree_parent_ptr, dev)


kernel = batch_spec_verify(draft_probs.shape[-1])

mod = tvm.build(kernel, target="cuda")
mod(
    draft_probs_tvm,
    draft_tokens_tvm,
    model_probs_tvm,
    token_tree_first_child_tvm,
    token_tree_next_sibling_tvm,
    uniform_samples_tvm,
    token_tree_parent_ptr_tvm,
)

print(model_probs_tvm.numpy())
