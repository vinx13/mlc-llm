import tvm
from tvm import relax, te, tir
from tvm.relax.testing import nn
from tvm.script import relax as R, tir as T
from tvm import relax
from tvm.relax.frontend.nn import Tensor
from .llama import MixtralConfig, Linear


def get_scatter_func(dtype):
    @T.prim_func
    def scatter_func(
        x_handle: T.handle,
        indices_handle: T.handle,
        out_handle: T.handle,
    ) -> None:
        total_rows = T.int64()
        hidden_size = T.int64()
        x = T.match_buffer(x_handle, (total_rows, hidden_size), dtype)
        indices = T.match_buffer(indices_handle, (total_rows,), "int32")
        out = T.match_buffer(out_handle, (total_rows, hidden_size), dtype)
        T.func_attr({"global_symbol": "scatter", "tir.noalias": True})
        for i in range(total_rows):
            for j in range(hidden_size):
                with T.block("scatter"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    out[indices[vi], vj] = x[vi, vj]

    return scatter_func


def get_top2_func(dtype, index_dtype):
    assert index_dtype == "int32"

    @T.prim_func
    def top2_func(
        x_handle: T.handle,
        out_handle: T.handle,
        out_index_handle: T.handle,
    ) -> None:
        total_rows = T.int64()
        num_experts = T.int64()
        x = T.match_buffer(x_handle, (total_rows, num_experts), dtype)
        out = T.match_buffer(out_handle, (total_rows, 2), dtype)
        out_index = T.match_buffer(out_index_handle, (total_rows, 2), index_dtype)
        local_top_k = T.alloc_buffer((2,), dtype=dtype, scope="local")
        local_top_k_index = T.alloc_buffer((2,), dtype=index_dtype, scope="local")
        T.func_attr({"global_symbol": "top2", "tir.noalias": True, "tir.is_scheduled": True})
        for io in T.thread_binding(0, T.ceildiv(total_rows, T.int64(1024)), "blockIdx.x"):
            for ii in T.thread_binding(0, T.min(total_rows, T.int64(1024)), "threadIdx.x"):
                if io * T.int64(1024) + ii < total_rows:
                    local_top_k[0] = T.min_value(dtype)
                    local_top_k_index[0] = 0
                for k in range(num_experts):
                    if x[io * T.int64(1024) + ii, k] > local_top_k[0]:
                        local_top_k[1] = local_top_k[0]
                        local_top_k_index[1] = local_top_k_index[0]
                        local_top_k[0] = x[io * T.int64(1024) + ii, k]
                        local_top_k_index[0] = k
                    elif x[io * T.int64(1024) + ii, k] > local_top_k[1]:
                        local_top_k[1] = x[io * T.int64(1024) + ii, k]
                        local_top_k_index[1] = k

                for k in T.unroll(2):
                    out[io * T.int64(1024) + ii, k] = local_top_k[k]
                    out_index[io * T.int64(1024) + ii, k] = local_top_k_index[k]

    return top2_func


def emit_tir_funcs(bb: relax.BlockBuilder, config: MixtralConfig):
    bb.add_func(get_scatter_func(config.dtype), "scatter")
    bb.add_func(get_top2_func(config.dtype, "int32"), "top2")


class MoELinear(nn.Module):
    def __init__(self, config: MixtralConfig, num_experts, in_features, out_features, bias=False):
        assert not bias, "bias not supported"
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_scheme = config.quantization_scheme

        if config.quantization_scheme.name == "q0f16":
            # weight is row major
            self.weight = nn.Parameter(
                (num_experts, in_features, out_features),
                dtype="float16",
            )
        elif config.quantization_scheme.name == "q4f16_ft":
            assert out_features % 8 == 0
            self.weight = nn.Parameter(
                (num_experts, in_features, out_features // 2),
                dtype="int8",
            )
            self.scales = nn.Parameter(
                (num_experts, out_features),
                dtype="float16",
            )
        else:
            assert False, "unsupported quantization scheme"

    def forward(self, x, rows_before):
        assert len(x.struct_info.shape) == 2
        total_rows = x.struct_info.shape[0]
        if self.quantization_scheme.name == "q0f16":
            return nn.emit(
                relax.call_dps_packed(
                    "cutlass.moe_gemm_f16f16",
                    [
                        x,
                        self.weight,
                        rows_before,
                        total_rows,
                        self.out_features,  # gemm_n
                        self.in_features,  # gemm_k
                        self.num_experts,
                    ],
                    out_sinfo=relax.TensorStructInfo(
                        (total_rows, self.out_features),
                        x.struct_info.dtype,
                    ),
                )
            )
        else:
            return nn.emit(
                relax.call_dps_packed(
                    "cutlass.moe_gemm_s4f16",
                    [
                        x,
                        self.weight,
                        self.scales,
                        rows_before,
                        total_rows,
                        self.out_features,  # gemm_n
                        self.in_features,  # gemm_k
                        self.num_experts,
                    ],
                    out_sinfo=relax.TensorStructInfo(
                        (total_rows, self.out_features),
                        x.struct_info.dtype,
                    ),
                )
            )


class MoEMLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.combine_matmul = config.combine_matmul

        self.num_shards = config.num_shards
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size // self.num_shards

        self.down_proj = MoELinear(
            config, self.num_experts, intermediate_size, hidden_size, bias=False
        )
        if config.combine_matmul:
            self.gate_up_combined_proj = MoELinear(
                config,
                self.num_experts,
                hidden_size,
                2 * intermediate_size,
                bias=False,
            )
            # FIXME: rename to 'gate_up_proj' that's consistent with llama. using this name for now to avoid conflicting pname str replacing rules
            # TODO: check sharding is correct, note that the weight is row major
            self.gate_up_combined_proj.weight.shard_dim = 2
            self.gate_up_combined_proj.weight.shard_strategy = "moe_shard_gate_up"
            self.down_proj.weight.shard_dim = 1
            self.down_proj.weight.shard_strategy = "moe_shard_mlp_k"
        else:
            self.gate_proj = MoELinear(
                config, self.num_experts, config.hidden_size, config.intermediate_size, bias=False
            )
            self.up_proj = MoELinear(
                config, self.num_experts, config.hidden_size, config.intermediate_size, bias=False
            )

    def forward(self, hidden_states: relax.Expr, rows_before: relax.Expr):
        # TODO: disco
        if self.combine_matmul:
            gate_up_results = nn.emit(
                relax.op.split(
                    self.gate_up_combined_proj(hidden_states, rows_before),
                    indices_or_sections=2,
                    axis=-1,
                )
            )
            gate_result = relax.TupleGetItem(gate_up_results, 0)
            up_result = relax.TupleGetItem(gate_up_results, 1)
        else:
            gate_result = self.gate_proj(hidden_states, rows_before)
            up_result = self.up_proj(hidden_states, rows_before)
        result = self.down_proj(nn.emit(relax.op.nn.silu(gate_result) * up_result), rows_before)
        return result


class MoE(nn.Module):
    def __init__(self, config: MixtralConfig):
        self.experts = MoEMLP(config)
        self.num_shards = config.num_shards
        self.gate = Linear(
            in_features=config.hidden_size,
            out_features=config.num_local_experts,
            bias=False,
            dtype=config.dtype,
        )
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_local_experts

    def topk(self, x, is_ascend, index_dtype, k=-1):
        if not is_ascend and k == 2:
            # fast path
            total_rows = x.struct_info.shape[0]
            result = nn.emit(
                relax.call_dps_packed(
                    "top2",
                    [x],
                    out_sinfo=[
                        relax.TensorStructInfo([total_rows, k], x.struct_info.dtype),
                        relax.TensorStructInfo([total_rows, k], index_dtype),
                    ],
                )
            )
            return relax.TupleGetItem(result, 0), relax.TupleGetItem(result, 1)

        # topk along axis -1
        result = nn.emit(
            relax.call_dps_packed(
                "tvm.contrib.thrust.sort_dps",
                [x, is_ascend],
                out_sinfo=[
                    x.struct_info,
                    relax.TensorStructInfo(x.struct_info.shape, index_dtype),
                ],
            )
        )
        sorted_x = relax.TupleGetItem(result, 0)
        indices = relax.TupleGetItem(result, 1)
        if k != -1:
            ndim = len(x.struct_info.shape)
            beg = [0] * ndim
            end = [x.struct_info.shape[i] for i in range(ndim - 1)] + [k]
            axes = list(range(ndim))
            sorted_x = nn.emit(
                relax.op.strided_slice(sorted_x, axes, beg, end, assume_inbound=True)
            )
            indices = nn.emit(relax.op.strided_slice(indices, axes, beg, end, assume_inbound=True))
        return sorted_x, indices

    def compute_rows_before(self, sorted_expert_ids):
        return nn.emit(
            relax.call_dps_packed(
                "moe_compute_rows_before",
                [sorted_expert_ids],
                out_sinfo=relax.TensorStructInfo([self.num_experts], "int64"),
            )
        )

    def scatter(self, linear_out, indices):
        return nn.emit(
            relax.call_dps_packed(
                "scatter",
                [linear_out, indices],
                out_sinfo=linear_out.struct_info,
            )
        )

    def get_token_indices(self, indices):
        def te_compute(x):
            return tvm.te.compute(
                x.shape,
                lambda *idx: tvm.tir.indexdiv(
                    x(*idx), tvm.runtime.const(self.num_experts_per_tok, dtype="int32")
                ).astype("int32"),
            )

        return nn.emit_te(te_compute, indices)

    def topk_mask(self, indices):
        from functools import reduce

        def te_topk_mask_op(topk_indices):
            ntokens = topk_indices.shape[0]
            assert topk_indices.shape[1] == self.num_experts_per_tok
            return te.compute(
                (ntokens, self.num_experts),
                lambda i, j: tir.expr.Select(
                    reduce(
                        lambda a, b: tir.Or(a, b),
                        [topk_indices[i, k] == j for k in range(self.num_experts_per_tok)],
                    ),
                    true_value=tir.const(1, "int32"),
                    false_value=tir.const(0, "int32"),
                ),
            )

        return nn.emit_te(te_topk_mask_op, indices)

    def get_indices(
        self, cumsum_colwise_flattened: relax.Expr, expert_indices: relax.Expr
    ) -> relax.Expr:
        from tvm import relax
        from tvm.script import tir as T

        @T.prim_func
        def get_flattened_expert_indices_scheduled(
            var_cumsum_colwise_flattened: T.handle,
            var_expert_indices: T.handle,
            var_flattened_expert_indices: T.handle,
        ):
            T.func_attr({"tir.is_scheduled": 1})
            batch_size = T.SizeVar("batch_size", "int32")
            cumsum_flattened_length = T.SizeVar("cumsum_flattened_length", "int32")

            cumsum_colwise_flattened = T.match_buffer(
                var_cumsum_colwise_flattened, shape=[cumsum_flattened_length], dtype="int32"
            )
            expert_indices = T.match_buffer(
                var_expert_indices, shape=[batch_size, self.num_experts_per_tok], dtype="int32"
            )
            flattened_expert_indices = T.match_buffer(
                var_flattened_expert_indices,
                shape=[batch_size * self.num_experts_per_tok],
                dtype="int32",
            )

            for io in T.thread_binding(
                0, T.ceildiv(cumsum_flattened_length, T.int32(1024)), "blockIdx.x"
            ):
                for ii in T.thread_binding(
                    0, T.min(cumsum_flattened_length, T.int32(1024)), "threadIdx.x"
                ):
                    with T.block("get_indices"):
                        vi = T.axis.spatial(cumsum_flattened_length, io * T.int32(1024) + ii)
                        T.where(io * T.int32(1024) + ii < cumsum_flattened_length)
                        T.reads(
                            cumsum_colwise_flattened[vi - 1 : vi - 1 + 2], expert_indices[:, 0:2]
                        )
                        T.writes(flattened_expert_indices[:])
                        expert_idx = T.alloc_buffer(shape=(), dtype="int32", scope="local")
                        if (
                            vi == 0 and cumsum_colwise_flattened[vi] > 0
                        ) or cumsum_colwise_flattened[vi] != cumsum_colwise_flattened[vi - 1]:
                            idx: T.SizeVar("idx", "int32") = cumsum_colwise_flattened[vi] - 1
                            instance_id: T.SizeVar("instance_id", "int32") = T.truncmod(
                                vi, batch_size
                            )
                            expert_id: T.SizeVar("expert_id", "int32") = T.truncdiv(vi, batch_size)
                            for j in T.serial(0, self.num_experts_per_tok):
                                with T.block("select_expert"):
                                    vj = T.axis.spatial(self.num_experts_per_tok, j)
                                    vinstance_id = T.axis.spatial(batch_size, instance_id)
                                    vexpert_id = T.axis.spatial(
                                        T.truncdiv(cumsum_flattened_length, batch_size), expert_id
                                    )
                                    if expert_indices[vinstance_id, vj] == vexpert_id:
                                        expert_idx[()] = vj
                            flattened_expert_indices[idx] = (
                                instance_id * self.num_experts_per_tok + expert_idx[()]
                            )

        bb = relax.BlockBuilder.current()
        gvar = bb.add_func(get_flattened_expert_indices_scheduled, "get_flattened_expert_indices")
        return bb.emit(
            relax.call_tir(
                gvar,
                [cumsum_colwise_flattened, expert_indices],
                out_sinfo=relax.TensorStructInfo(
                    [expert_indices.struct_info.shape[0] * self.num_experts_per_tok], "int32"
                ),
            )
        )

    def cumsum(self, data: relax.Expr) -> relax.Expr:
        return nn.emit(
            relax.call_dps_packed(
                "tvm.contrib.thrust.sum_scan",
                [data],
                out_sinfo=data.struct_info,
            )
        )

    # def get_indptr(self, cumsum_colwise_flattened: relax.Expr) -> relax.Expr:
    #     from tvm import relax
    #     from tvm.script import tir as T

    #     @T.prim_func
    #     def get_expert_instance_indptr(
    #         var_cumsum_colwise_flattened: T.handle,
    #         var_expert_instance_indptr: T.handle,
    #         batch_size: T.int32,
    #     ):
    #         cumsum_colwise_flattened = T.match_buffer(
    #             var_cumsum_colwise_flattened, shape=[batch_size * self.num_experts], dtype="int32"
    #         )
    #         expert_instance_indptr = T.match_buffer(
    #             var_expert_instance_indptr, shape=[self.num_experts], dtype="int64"
    #         )

    #         for expert_id in T.serial(0, self.num_experts):
    #             with T.block("indptr"):
    #                 vexpert_id = T.axis.spatial(self.num_experts, expert_id)
    #                 expert_instance_indptr[vexpert_id] = cumsum_colwise_flattened[
    #                     vexpert_id * batch_size - 1
    #                 ]

    #     bb = relax.BlockBuilder.current()
    #     gvar = bb.add_func(get_expert_instance_indptr, "get_expert_instance_indptr")
    #     return bb.emit(
    #         relax.call_tir(
    #             gvar,
    #             [cumsum_colwise_flattened],
    #             out_sinfo=relax.TensorStructInfo([self.num_experts], "int64"),
    #             tir_vars=[cumsum_colwise_flattened.struct_info.shape[0] // self.num_experts],
    #         )
    #     )
    def get_indptr(self, cumsum_colwise_flattened: relax.Expr) -> relax.Expr:
        indptr = nn.emit(
            relax.op.strided_slice(
                cumsum_colwise_flattened,
                axes=[0],
                begin=[
                    cumsum_colwise_flattened.struct_info.shape[0] // self.num_experts
                    - 1
                ],
                end=[cumsum_colwise_flattened.struct_info.shape[0]],
                strides=[self.num_experts],
                assume_inbound=True,
            )
        )
        return nn.emit(relax.op.astype(indptr, "int64"))

    def forward(self, hidden_states):
        hidden_states_shape = hidden_states.struct_info.shape
        hidden_size = hidden_states_shape[-1]
        # reshape to 2D
        hidden_states = nn.emit(relax.op.reshape(hidden_states, (-1, hidden_size)))

        gate = self.gate(hidden_states)
        scores = nn.emit(relax.op.nn.softmax(gate, axis=-1))

        expert_weights, expert_indices = self.topk(
            scores, is_ascend=False, k=self.num_experts_per_tok, index_dtype="int32"
        )  # (num_tokens, top_k), (num_tokens, top_k)
        expert_weights = nn.emit(expert_weights / R.sum(expert_weights, axis=-1, keepdims=True))

        expert_mask = self.topk_mask(expert_indices)
        mask_T_flattened = nn.emit(relax.op.flatten(relax.op.permute_dims(expert_mask)))

        cumsum_colwise_flattened = self.cumsum(mask_T_flattened)
        flattened_indices = self.get_indices(cumsum_colwise_flattened, expert_indices)
        indptr = self.get_indptr(cumsum_colwise_flattened)
        # indptr = nn.emit(relax.op.strided_slice(indptr, axes=[0], begin=[1], assume_inbound=True))

        # flattened_indices = nn.emit(relax.op.flatten(expert_indices))
        # sorted_expert_ids, indices = self.topk(
        #     flattened_indices, is_ascend=True, index_dtype="int32"
        # )

        # rows_before = self.compute_rows_before(sorted_expert_ids)
        token_indices = self.get_token_indices(flattened_indices)
        gathered_x = nn.emit(relax.op.take(hidden_states, token_indices, axis=0))
        # linear_out = self.experts(gathered_x, rows_before)
        # unpermuted = self.scatter(linear_out, indices)
        linear_out = self.experts(gathered_x, indptr)
        unpermuted = self.scatter(linear_out, flattened_indices)

        unflattened = nn.emit(
            relax.op.reshape(unpermuted, (-1, self.num_experts_per_tok, hidden_size))
        )
        expert_weights = nn.emit(
            relax.op.reshape(expert_weights, (-1, self.num_experts_per_tok, 1))
        )
        weighted_sum = nn.emit(relax.op.sum(unflattened * expert_weights, axis=1))

        # reshape back to 3D
        weighted_sum = nn.emit(relax.op.reshape(weighted_sum, hidden_states_shape))
        return weighted_sum
