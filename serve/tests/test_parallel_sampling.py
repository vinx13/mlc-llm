import torch

import argparse
import json
import random
import os
from pathlib import Path

from mlc_serve.engine import (
    Request,
    ChatMessage,
    DebugOptions,
    SamplingParams,
    StoppingCriteria,
    get_engine_config,
)
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule
from mlc_serve.logging_utils import configure_logging


def _test(args: argparse.Namespace):
    engine_config = get_engine_config(
        {
            "use_staging_engine": args.use_staging_engine,
            "max_num_sequences": args.max_num_sequences,
            "max_input_len": args.max_input_len,
            "min_decode_steps": args.min_decode_steps,
            "max_decode_steps": args.max_decode_steps,
        }
    )

    if args.use_staging_engine:
        engine = StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(args.model_artifact_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": args.model_artifact_path,
                "engine_config": engine_config,
            },
        )
        engine.start()
    else:
        engine = SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path=args.model_artifact_path,
                engine_config=engine_config,
            )
        )

    sampling_params_random = SamplingParams(
        temperature=1.0,
        top_p=1.0,
    )

    if args.long_prompt:
        with open("serve/tests/data/long_prompts.json", "r") as f:
            prompts = json.load(f)["prompts"]
    else:
        prompts = [
            "The president of the United States is a powerful man. But he can also be",
            "The future of AI is full of promise. But we need to carefully",
        ]

    num_sequences = 3

    for i, prompt in enumerate(prompts):
        engine.add(
            [
                Request(
                    request_id=str(i),
                    messages=[ChatMessage(role="user", content=prompt)],
                    sampling_params=sampling_params_random,
                    stopping_criteria=StoppingCriteria(
                        max_tokens=args.max_output_len, stop_sequences=None
                    ),
                    debug_options=DebugOptions(prompt=prompt),
                    num_sequences=num_sequences,
                )
            ]
        )

    generated = [["" for _ in range(num_sequences)] for _ in range(len(prompts))]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == num_sequences
            for i, seq in enumerate(res.sequences):
                if not seq.is_finished:
                    generated[int(res.request_id)][i] += seq.delta

    if args.long_prompt:
        for g in generated:
            for i, seq in enumerate(g):
                print(f"Generated {i}-th sample = '{seq}'")
                print("")
            print("")
    else:
        for p, g in zip(prompts, generated):
            print(f"Prompt = '{p}'")
            for i, seq in enumerate(g):
                print(f"Generated {i}-th sample = '{seq}'")
            print("")

    if args.use_staging_engine:
        engine.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="dist")
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-num-sequences", type=int, default=8)
    parser.add_argument("--max-output-len", type=int, default=30)
    parser.add_argument("--long-prompt", action="store_true")
    parser.add_argument("--use-staging-engine", action="store_true")
    parser.add_argument("--min-decode-steps", type=int, default=12)
    parser.add_argument("--max-decode-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.model_artifact_path = Path(os.path.join(args.artifact_path, args.local_id))

    if not os.path.exists(args.model_artifact_path):
        raise Exception(f"Invalid local id: {args.local_id}")

    if args.long_prompt:
        args.max_input_len = 10000
        args.max_num_sequences = 5

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    configure_logging(enable_json_logs=False, log_level="INFO")

    _test(args)