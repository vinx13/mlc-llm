/*!
 *  Copyright (c) 2024 by Contributors
 * \file serve/draft_token_manager.h
 */

#ifndef MLC_LLM_SERVE_DRAFT_TOKEN_MANAGER_H_
#define MLC_LLM_SERVE_DRAFT_TOKEN_MANAGER_H_
#include <tvm/runtime/device_api.h>

#include <numeric>
#include <optional>
#include <vector>

#include "data.h"
#include "function_table.h"
namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

struct DraftToken {
  SampleResult sample_result;
  NDArray probs;
  int hidden_states_slot_id;
  int prob_slot_id;
};

struct ModelWorkspace;
class DraftTokenWorkspaceManagerObj : public Object {
 public:
  DraftTokenWorkspaceManagerObj(int max_num_tokens, int vocab_size, int hidden_size,
                                DLDataType hidden_states_dtype, Device device,
                                const FunctionTable& ft);

  void AllocateSlots(int num_slots, std::vector<int>* result);

  void CopyInProbs(const NDArray& probs, const std::vector<int>& slots);

  void CopyInHiddenStates(const NDArray& hidden_states, const std::vector<int>& slots);

  void GatherProbs(const std::vector<int>& slots, NDArray* dst);

  void GatherHiddenStates(const std::vector<int>& slots, NDArray* dst);

  void FreeSlots(const std::vector<int>& slots);
  static constexpr const char* _type_key = "mlc.serve.DraftTokenWorkspaceManager";

  void AllocWorkspace(ModelWorkspace* workspace);

  NDArray probs_device_;
  NDArray hidden_states_device_;

  PackedFunc gather_probs_func_;
  PackedFunc gather_hidden_states_func_;
  PackedFunc scatter_hidden_states_func_;
  PackedFunc scatter_probs_func_;

  NDArray slots_device_;
  NDArray src_indices_device_;
  std::vector<int> free_slots_;
  int max_num_tokens_;
  int vocab_size_;
  int hidden_size_;
  DLDevice device_;
};

class DraftTokenWorkspaceManager : public ObjectRef {
 public:
  DraftTokenWorkspaceManager(int num_slots, int vocab_size, int hidden_size,
                             DLDataType hidden_states_dtype, Device device,
                             const FunctionTable& ft) {
    data_ = make_object<DraftTokenWorkspaceManagerObj>(num_slots, vocab_size, hidden_size,
                                                       hidden_states_dtype, device, ft);
  }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(DraftTokenWorkspaceManager, ObjectRef,
                                        DraftTokenWorkspaceManagerObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_DRAFT_TOKEN_MANAGER_H_