""" Quantization techniques for FP8 """

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

from tvm import DataType, DataTypeCode, IRModule
from tvm import dlight as dl
from tvm import relax, te, tir, topi
from tvm import nd
from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm.script import tir as T
from tvm.target import Target

from mlc_llm.loader import QuantizeMapping
from mlc_llm.nn import MixtralExperts

from . import per_tensor_quantization as ptq
from .utils import apply_sharding

from ..op import cutlass, extern,moe_matmul, ft_gemm
import numpy as np


def quantize(
    x: nn.Tensor, quantize_dtype: str, kind="fp8-max", name: str = "quantize", **kwargs
) -> Tuple[nn.Tensor, ...]:
    """
    Quantizes the input tensor to a specified lower-precision datatype using different quantization schemes.

    This function supports quantization schemes such as 'fp8-max', where each element in the tensor is scaled and
    quantized to a target datatype that uses fewer bits than the original datatype. The fp8-max range scheme
    scales the tensor values based on the maximum value in the tensor to utilize the full range of the target datatype.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor to be quantized.

    quantize_dtype : DataType
        The target datatype for quantization.

    kind : str, optional
        The kind of quantization scheme to use.

    name : str, optional
        A name hint for the operation.

    **kwargs : dict
        Additional keyword arguments for quantization parameters. For 'fp8-max', 'max_int_value' must be provided,
        which defines the maximum integer value that can be represented in the target datatype.

    Returns
    -------
    result : Tuple[nn.Tensor, ...]
        A list of tensors from the qunatization,
        Usually the quantized tensor, and parameter tensors like scale and zero point

    """
    if kind == "fp8-max":
        # quant: Tensor(dtype="e4m3_float8") = (x / scale); scale: float16 = max(x) / fp8_max_int_value);
        assert (
            "max_int_value" in kwargs
        ), "'max_int_value' must be provided when using fp8-max quantization"

        def fused_compute_scale_and_quantize(
            tensor: te.Tensor,
            max_abs: te.Tensor,
            axis: int,
            out_shape: Optional[List[tir.PrimExpr]] = None,
        ):
            max_int = tir.const(kwargs["max_int_value"], x.dtype)
            min_scaling_factor = tir.const(1.0 / (kwargs["max_int_value"] * 512.0), x.dtype)

            scale = te.compute(
                (1,),
                lambda *idx: te.max(
                    max_abs(*idx).astype(x.dtype) / max_int,
                    min_scaling_factor,
                ),
                name="scale",
            )
            scaled_act = te.compute(
                shape=tensor.shape,
                fcompute=lambda *idx: tir.Cast(
                    quantize_dtype,
                    tensor(*idx) / scale[0],
                ),
            )

            return scaled_act, scale

        max_abs = nn.op.extern(
            "tvm.contrib.cuda.reduce_max_abs",
            [x],
            nn.Tensor.placeholder((1,), x.dtype),
        )

        quant, scale = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
            lambda tensor, max_tensor: fused_compute_scale_and_quantize(  # pylint: disable=protected-access
                tensor,
                max_tensor,
                axis=None,
                out_shape=x.shape,
            ),
            name_hint="quantize_act",
            args=[x, max_abs],
        )
        return quant, scale
    else:
        raise ValueError("Unknown quantization kind")


def dequantize(
    quant: nn.Tensor,
    scale: nn.Tensor,
    zero: nn.Tensor = None,
    kind="fp8-max",
    name="dequantize",
    **kwargs,
) -> nn.Tensor:
    """
    Dequantizes the input tensor from a specified lower-precision datatype back to a higher-precision datatype.

    This function supports dequantization schemes such as 'fp8-max', where each element in the quantized tensor
    is converted back to a higher-precision format using the provided scale. The 'fp8-max' scheme specifically
    reverses the scaling applied during quantization, without utilizing a zero-point adjustment.

    Parameters
    ----------
    quant : nn.Tensor
        The quantized tensor to be dequantized.

    scale : nn.Tensor
        The scale used during quantization.
        original higher-precision format.

    zero : nn.Tensor, optional
        The zero-point used during quantization.

    kind : str, optional
        The kind of dequantization scheme to use.

    name : str, optional
        A name hint for the operation.

    **kwargs : dict
        Additional keyword arguments for dequantization parameters.

    Returns
    -------
    nn.Tensor
        The dequantized tensor.

    """
    if kind == "fp8-max":
        # dequant: Tensor(dtype="float16") = (quant * scale); scale precompute by quantization
        assert zero == None, "FP8 max range quantization does not utilzie a zero point"
        return quant * scale
    else:
        raise ValueError("Unknown quantization kind")


class FP8PerTensorQuantizeMixtralExperts(
    ptq.PerTensorQuantizeMixtralExperts
):  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        num_local_experts,
        in_features,
        out_features,
        config: ptq.PerTensorQuantize,
        runtime: str = "cast",
        tensor_parallel_shards=1,
    ):  # pylint: disable=too-many-arguments
        super().__init__(num_local_experts, in_features, out_features, config)
        self.runtime = runtime
        self.tensor_parallel_shards = tensor_parallel_shards

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts",
        config: ptq.PerTensorQuantize,
        runtime: str = "cast",
    ) -> "FP8PerTensorQuantizeMixtralExperts":
        """
        Converts a non-quantized MixtralExperts to a per-tensor quantized MixtralExperts.

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        weight_config : GroupQuantize
            The group quantization weight_config.

        Returns
        -------
        ret : MixtralExpertsFP8
            The per-tensor quantized MixtralExperts.
        """
        quantized_mistral_experts = FP8PerTensorQuantizeMixtralExperts(
            num_local_experts=src.num_local_experts,
            in_features=src.in_features,
            out_features=src.out_features,
            config=config,
            runtime=runtime,
            tensor_parallel_shards=src.tensor_parallel_shards,
        )

        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, f"{shard.name}_q_weight", quantized_mistral_experts.q_weight)
            # scale doesn't need to be sharded since it's the same for all shards

        return quantized_mistral_experts

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        w = self.q_weight
        if indptr.ndim == 2:
            assert indptr.shape[0] == 1
            return moe_matmul.dequantize_float8_gemv(x, w, self.q_scale, indptr, self.config.weight_dtype)
            # # w = self.config.dequantize_float8(w, self.q_scale, self.config.weight_dtype)
            # w = nn.tensor_expr_op(self.config.dequantize_float8, "dequantize", [w, self.q_scale, self.config.weight_dtype])
            # return moe_matmul.gemv(x, w, indptr)

        x = nn.op.astype(x, dtype=self.config.activation_dtype)
        scale = self.q_scale.astype("float32") if self.q_scale is not None else nn.wrap_nested(relax.Constant(nd.array(np.array([1.0]).astype("float32"))), "scale")
        assert extern.get_store().cutlass_group_gemm, "cutlass FP8 group gemm is not available"
        return cutlass.group_gemm(
            x, w, indptr, scale, self.config.weight_dtype, self.config.model_dtype
        )
