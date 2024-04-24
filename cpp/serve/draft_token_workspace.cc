/*!
 * Copyright (c) 2024 by Contributors
 * \file serve/draft_token_manager.cc
 */

#include "draft_token_workspace.h"

#include "model.h"

namespace mlc {
namespace llm {
namespace serve {
DraftTokenWorkspaceManagerObj::DraftTokenWorkspaceManagerObj(int max_num_tokens, int vocab_size,
                                                             int hidden_size,
                                                             DLDataType hidden_states_dtype,
                                                             Device device, const FunctionTable& ft)
    : gather_hidden_states_func_(ft.gather_hidden_states_func_),
      scatter_hidden_states_func_(ft.scatter_hidden_states_func_),
      gather_probs_func_(ft.gather_probs_func_),
      scatter_probs_func_(ft.scatter_probs_func_),
      max_num_tokens_(max_num_tokens),
      vocab_size_(vocab_size),
      hidden_size_(hidden_size),
      device_(device) {
  probs_device_ = NDArray::Empty({max_num_tokens, vocab_size}, DataType::Float(32), device);
  hidden_states_device_ =
      NDArray::Empty({max_num_tokens, hidden_size}, hidden_states_dtype, device);
  free_slots_.resize(max_num_tokens);
  std::iota(free_slots_.begin(), free_slots_.end(), 0);

  slots_device_ = NDArray::Empty({max_num_tokens}, DataType::Int(32), device);
  src_indices_device_ = NDArray::Empty({max_num_tokens}, DataType::Int(32), device);
}

void DraftTokenWorkspaceManagerObj::AllocateSlots(int num_slots, std::vector<int>* result) {
  ICHECK_LE(num_slots, free_slots_.size());
  result->assign(free_slots_.begin(), free_slots_.begin() + num_slots);
  std::vector<int> allocated(free_slots_.begin(), free_slots_.begin() + num_slots);
  free_slots_.resize(free_slots_.size() - num_slots);
}

void DraftTokenWorkspaceManagerObj::CopyInProbs(const NDArray& probs,
                                                const std::vector<int>& slots) {
  ICHECK_EQ(probs->shape[0], slots.size());
  NDArray slots_device =
      slots_device_.CreateView({static_cast<int64_t>(slots.size())}, DataType::Int(32));
  slots_device.CopyFromBytes(slots.data(), slots.size() * sizeof(int));
  scatter_probs_func_(probs, slots_device, probs_device_);
}

void DraftTokenWorkspaceManagerObj::CopyInHiddenStates(const NDArray& hidden_states,
                                                       const std::vector<int>& slots) {
  ICHECK_EQ(hidden_states->shape[0], slots.size());
  NDArray slots_device =
      slots_device_.CreateView({static_cast<int64_t>(slots.size())}, DataType::Int(32));
  slots_device.CopyFromBytes(slots.data(), slots.size() * sizeof(int));
  scatter_hidden_states_func_(hidden_states, slots_device, hidden_states_device_);
}

void DraftTokenWorkspaceManagerObj::GatherProbs(const std::vector<int>& slots, NDArray* dst) {
  ICHECK_EQ((*dst)->shape[0], slots.size());
  NDArray slots_device =
      slots_device_.CreateView({static_cast<int64_t>(slots.size())}, DataType::Int(32));
  slots_device.CopyFromBytes(slots.data(), slots.size() * sizeof(int));
  gather_probs_func_(probs_device_, slots_device, *dst);
}

void DraftTokenWorkspaceManagerObj::GatherHiddenStates(const std::vector<int>& slots,
                                                       NDArray* dst) {
  ICHECK_EQ((*dst)->shape[0], slots.size());
  NDArray slots_device =
      slots_device_.CreateView({static_cast<int64_t>(slots.size())}, DataType::Int(32));
  slots_device.CopyFromBytes(slots.data(), slots.size() * sizeof(int));
  gather_hidden_states_func_(hidden_states_device_, slots_device, *dst);
}

void DraftTokenWorkspaceManagerObj::FreeSlots(const std::vector<int>& slots) {
  std::copy(slots.begin(), slots.end(), std::back_inserter(free_slots_));
}

void DraftTokenWorkspaceManagerObj::AllocWorkspace(ModelWorkspace* workspace) {
  workspace->draft_probs_on_device =
      NDArray::Empty({max_num_tokens_, vocab_size_}, DataType::Float(32), device_);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc