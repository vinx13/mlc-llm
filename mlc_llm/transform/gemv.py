import tvm
from tvm.script  import tir as T
from tvm.tir.stmt_functor import post_order_visit
from tvm import meta_schedule as ms
from tvm.topi.utils import get_const_int

def get_gemv(func):
    """
    Check if the function is a GEMV computation
    """
    gemv_block = None

    def fvisit(stmt):
        if isinstance(stmt, tvm.tir.Block):
            if stmt.init is not None:
                nonlocal gemv_block
                gemv_block = stmt
    # print(func.body)
    post_order_visit(func.body, fvisit)
    if gemv_block is not None:
        if not (len(gemv_block.reads) == 2 and len(gemv_block.writes) == 1):
            return None
        iters = gemv_block.iter_vars
        found_spatial_loop = False
        found_reduction_loop = False
        for iter_var in iters:
            if iter_var.iter_type == tvm.tir.IterVar.DataPar:
                if isinstance(iter_var.dom.extent, tvm.tir.IntImm) and get_const_int(iter_var.dom.extent) == 1:
                    continue
                else:
                    if not found_spatial_loop:
                        found_spatial_loop = True
                    else:
                        return None
            elif iter_var.iter_type == tvm.tir.IterVar.CommReduce:
                if not found_reduction_loop:
                    found_reduction_loop = True
                else:
                    return None
            else:
                return None
        return gemv_block
    return None


@T.prim_func
def fused_decode1_matmul7(lv8: T.Buffer((T.int64(824), T.int64(4096)), "uint16"), lv9: T.Buffer((T.int64(103), T.int64(4096)), "float16"), lv1583: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv8[v_i // T.int64(5), v_j], lv9[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", lv8[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv9[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1583[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1583[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]


@T.prim_func
def fused_decode6_fused_matmul9_add3_int3_int16_fp16_before(lv1623: T.Buffer((T.int64(2208), T.int64(4096)), "uint16"), lv1624: T.Buffer((T.int64(276), T.int64(4096)), "float16"), lv167: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv165: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1623[v_i // T.int64(5), v_j], lv1624[v_i // T.int64(40), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(T.Cast("uint32", lv1623[v_i // T.int64(5), v_j]), T.Cast("uint32", v_i % T.int64(5)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * lv1624[v_i // T.int64(40), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv167[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv167[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv165[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv165[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]


@T.prim_func
def fused_decode6_fused_matmul9_add3_int3_int16_fp16_after(lv1158: T.Buffer((T.int64(2208), T.int64(4096)), "uint16"), lv60: T.Buffer((T.int64(276), T.int64(4096)), "float16"), lv6: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv4: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": 1})
    # with T.block("root"):
    var_decode_intermediate_local = T.alloc_buffer((T.int64(11040), T.int64(4096)), scope="local", dtype="float16")
    var_scale_intermediate_local = T.alloc_buffer((T.int64(276), T.int64(4096)), scope="local", dtype="float16")
    var_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), scope="local", dtype="float16")
    lv6_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11040)), scope="shared", dtype="float16")
    for i0_i1_i2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
        for i2_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
            for i2_2 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax0, ax1_ax2_fused_0 in T.grid(T.int64(1), T.int64(44)):
                    for ax1_ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("lv2749_shared"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v2 = T.axis.spatial(T.int64(11040), ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1)
                            T.reads(lv6[v0, v1, v2])
                            T.writes(lv6_shared[v0, v1, v2])
                            T.where(ax1_ax2_fused_0 * T.int64(256) + ax1_ax2_fused_1 < T.int64(11040))
                            T.block_attr({"buffer_dim_align": [[0, 1, 32, 8]]})
                            lv6_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(11008), lv6[v0, v1, v2], T.float16(0))
                with T.block("matmul_init"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                    T.reads()
                    T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                    var_matmul_intermediate_local[v_i0, v_i1, v_i2] = T.float32(0)
                for k_0_0 in range(T.int64(138)):
                    for ax0_0 in T.unroll(T.int64(80)):
                        for ax1 in range(T.int64(1)):
                            with T.block("decode"):
                                v_j = T.axis.spatial(T.int64(11040), k_0_0 * T.int64(80) + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv1158[v_j // T.int64(5), v_i])
                                T.writes(var_decode_intermediate_local[v_j, v_i])
                                var_decode_intermediate_local[v_j, v_i] = T.Cast("float16", T.Cast("int16", T.bitwise_and(T.shift_right(T.Cast("uint16", lv1158[v_j // T.int64(5), v_i]), T.Cast("uint16", v_j % T.int64(5)) * T.uint16(3)), T.uint16(7))) - T.int16(3))
                    for ax0_0 in T.unroll(T.int64(2)):
                        for ax1 in range(T.int64(1)):
                            with T.block("scale"):
                                v_j = T.axis.spatial(T.int64(276), k_0_0 * T.int64(2) + ax0_0)
                                v_i = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax1)
                                T.reads(lv60[v_j, v_i])
                                T.writes(var_scale_intermediate_local[v_j, v_i])
                                var_scale_intermediate_local[v_j, v_i] = lv60[v_j, v_i]
                    for k_0_2_k_1_fused in range(T.int64(80)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_1 * T.int64(256) + i2_2)
                            v_k = T.axis.reduce(T.int64(11040), k_0_0 * T.int64(80) + k_0_2_k_1_fused)
                            T.reads(var_matmul_intermediate_local[v_i0, v_i1, v_i2], lv6_shared[v_i0, v_i1, v_k], var_decode_intermediate_local[v_k, v_i2], var_scale_intermediate_local[v_k // T.int64(40), v_i2])
                            T.writes(var_matmul_intermediate_local[v_i0, v_i1, v_i2])
                            var_matmul_intermediate_local[v_i0, v_i1, v_i2] = var_matmul_intermediate_local[v_i0, v_i1, v_i2] + lv6_shared[v_i0, v_i1, v_k] * var_decode_intermediate_local[v_k, v_i2] * var_scale_intermediate_local[v_k // T.int64(40), v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    with T.block("var_matmul_intermediate_local"):
                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                        v2 = T.axis.spatial(T.int64(4096), i0_i1_i2_0_fused * T.int64(256) + i2_2 + ax2)
                        T.reads(lv4[v0, v1, v2], var_matmul_intermediate_local[v0, v1, v2])
                        T.writes(p_output0_intermediate[v0, v1, v2])
                        p_output0_intermediate[v0, v1, v2] = lv4[v0, v1, v2] + var_matmul_intermediate_local[v0, v1, v2]


def schedule(mod, sch, gemv):
    gemv = sch.get_block(gemv.name_hint)
    loops = sch.get_loops(gemv)
    fused = sch.fuse(*loops[:-1])
    bx, tx = sch.split(fused, factors=[None, 256])
    k0, k1 = sch.split(loops[-1], factors=[None, 32])
    sch.bind(bx, 'blockIdx.x')
    sch.bind(tx, 'threadIdx.x')
    X_shared= sch.cache_read(gemv, read_buffer_index=0, storage_scope="shared")
    Y_local = sch.cache_read(gemv, read_buffer_index=1, storage_scope="local")
    gemv_local = sch.cache_write(gemv, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(gemv_local, tx)
    sch.compute_at(X_shared, bx, preserve_unit_loops=True)
    sch.compute_at(Y_local, k0, preserve_unit_loops=True)

    consumers = sch.get_consumers(gemv_local)
    while len(consumers) > 0:
        for c in consumers:
            if tvm.tir.schedule.analysis.is_output_block(sch, c):
                sch.reverse_compute_inline(c)
            else:
                sch.compute_inline(c)
        consumers = sch.get_consumers(gemv_local)

    for block in [X_shared, Y_local]:
        producers = sch.get_producers(block)
        while len(producers) > 0:
            for p in producers:
                sch.compute_inline(p)
            producers = sch.get_producers(block)


    fused = sch.get_loops(X_shared)[-2:]
    fused = sch.fuse(*fused)
    _, ttx = sch.split(fused, factors=[None, 256])
    sch.bind(ttx, 'threadIdx.x')
    sch.decompose_reduction(gemv, k0)
    return sch.mod["main"]


def apply_default_gemv_schedule(mod):
    gemv = get_gemv(mod)
    if gemv is None:
        return None
    sch = tvm.tir.Schedule(mod)
    return schedule(mod, sch, gemv)


def main():
    mod = fused_decode6_fused_matmul9_add3_int3_int16_fp16_before
    new_mod = apply_default_gemv_schedule(mod)
    print(new_mod)
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target:
        tvm.build(fused_decode6_fused_matmul9_add3_int3_int16_fp16_after)
        tvm.build(new_mod)


# pylint: disable=missing-docstring
import tvm
from tvm import IRModule


@tvm.transform.module_pass(opt_level=0, name="ApplyDefaultGEMVSchedule")
class ApplyDefaultGEMVSchedule:  # pylint: disable=too-few-public-methods
    def __init__(self):
        pass

    def transform_module(
        self,
        mod: IRModule,
        ctx: tvm.transform.PassContext,
    ) -> IRModule:
        for gv in mod.functions:
            func = mod[gv]
            if not isinstance(func, tvm.tir.PrimFunc):
                continue
            scheduled_func = apply_default_gemv_schedule(func)
            if scheduled_func is not None:
                mod[gv] = scheduled_func
                print("- Apply default gemv schedule to:", gv.name_hint)

        return mod


if __name__ == "__main__":
    main()
