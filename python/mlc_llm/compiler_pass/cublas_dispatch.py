"""A compiler pass that dispatches patterns to CUBLAS."""

from typing import Literal

import tvm
import tvm.relax.backend.contrib.cublas as _cublas
from tvm import IRModule, relax
from tvm.relax.backend import get_patterns_with_prefix


@tvm.transform.module_pass(opt_level=0, name="CublasDispatch")
class CublasDispatch:  # pylint: disable=too-few-public-methods,broad-exception-raised
    """A compiler pass that dispatches patterns to CUBLAS."""

    def __init__(self, backend: Literal["cublas", "hipblas"] = "cublas"):
        self.backend = backend

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        has_backend = tvm.get_global_func(f"relax.ext.{self.backend}", True)
        if not has_backend:
            raise Exception(f"{self.backend} is not enabled.")

        patterns = get_patterns_with_prefix(self.backend)

        model_names = [
            gv.name_hint for gv, func in mod.functions.items() if isinstance(func, relax.Function)
        ]
        # exclude single batch decode
        model_names = [name for name in model_names if "batch" in name or "decode" not in name]
        mod = tvm.transform.Sequential(
            [
                relax.transform.FuseOpsByPattern(
                    patterns,
                    bind_constants=False,
                    annotate_codegen=True,
                    entry_functions=model_names,
                ),
                relax.transform.RunCodegen({}, entry_functions=model_names),
            ]
        )(mod)
        return mod
