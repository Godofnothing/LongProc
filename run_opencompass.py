import os
import argparse
import random

from longproc.longproc_data import load_longproc_data
from tqdm import tqdm

import torch
import torch.nn as nn
from opencompass.models import LlamaShadowKV, LlamaYaShadowKV, Qwen3ShadowKV, HuggingFacewithChatTemplate


def str2bool(v: str) -> bool:
    """
    Convert a string to a boolean.

    Accepts (case‑insensitive):
        - true, t, yes, y, 1  → True
        - false, f, no, n, 0 → False

    Anything else raises an ArgumentTypeError, which argparse will turn
    into a nice CLI error message.
    """
    if isinstance(v, bool):
        # In case it is already a bool (e.g. when called directly)
        return v

    val = v.lower()
    if val in ('yes', 'true', 't', 'y', '1'):
        return True
    if val in ('no', 'false', 'f', 'n', '0'):
        return False

    raise argparse.ArgumentTypeError(
        f"Boolean value expected for '--use-higgs-quantization'. "
        f"Got '{v}'. Use true/false (case‑insensitive)."
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="html_to_tsv_0.5k")
    parser.add_argument("--path", type=str, default="./data", help="Path to data")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p")
    parser.add_argument("--model_class", type=str, default="huggingface", help="Model class (huggingface, llama_shadowkv, qwen3_shadowkv)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model")
    parser.add_argument("--max_seq_len", type=int, default=131072, help="Max sequence length")
    parser.add_argument("--sparse_budget", type=float, default=None, help="Sparse budget for ShadowKV")
    parser.add_argument("--local_budget", type=int, default=32)
    parser.add_argument("--outlier_budget", type=int, default=384)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=160)
    # Boolean flag – off by default, on when the flag is present.
    parser.add_argument(
        "--use-higgs-quantization",
        dest="use_higgs_quantization",
        type=str2bool,
        default=False,
        help="Enable Higgs quantization (default: %(default)s)."
    )
    # Integer options – each gets a default value that matches the
    # defaults you supplied in the code you posted.
    parser.add_argument(
        "--higgs-hadamard-groupsize",
        dest="higgs_hadamard_groupsize",
        type=int,
        default=128,
        help="Group size for the Hadamard transform (default: %(default)s)."
    )
    parser.add_argument(
        "--higgs-edenn-d",
        dest="higgs_edenn_d",
        type=int,
        default=16,
        help="Depth `d` for the EDE (Embedding‑Decomposition‑Embedding) network "
             "(default: %(default)s)."
    )
    parser.add_argument(
        "--higgs-edenn-n",
        dest="higgs_edenn_n",
        type=int,
        default=256,
        help="Width `n` for the EDE network (default: %(default)s)."
    )
    parser.add_argument(
        "--higgs-channel-size",
        dest="higgs_channel_size",
        type=int,
        default=1024,
        help="Number of channels used in the Higgs pipeline (default: %(default)s)."
    )
    parser.add_argument(
        "--higgs-chunk-size",
        dest="higgs_chunk_size",
        type=int,
        default=64,
        help="Chunk size for processing Higgs data (default: %(default)s)."
    )

    parser.add_argument("--test_loading", action="store_true", help="Test loading data")


    return parser.parse_args()

def test_loading_all():
    def test_loading(dataset):
        data, eval_func = load_longproc_data(dataset, "./data")
        print(f"Dataset: {dataset}")
        print(f"N samples: {len(data)}")
        print(f"Eval func: {eval_func}")
        print(f"Max input chars: {max([len(d['input_prompt']) for d in data])}")
        print(f"Max output chars: {max([len(d['reference_output']) for d in data])}")
    [test_loading(d) for d in ["path_traversal_0.5k", "path_traversal_2k", "path_traversal_8k"]]

    [test_loading(d) for d in ["html_to_tsv_0.5k", "html_to_tsv_2k", "html_to_tsv_8k"]]

    [test_loading(d) for d in ["pseudo_to_code_0.5k", "pseudo_to_code_2k",]]

    [test_loading(d) for d in ["travel_planning_2k", "travel_planning_8k"]]

    [test_loading(d) for d in ["tom_tracking_0.5k", "tom_tracking_2k", "tom_tracking_8k"]]

    [test_loading(d) for d in ["countdown_0.5k", "countdown_2k", "countdown_8k"]]


def build_model(args):
    if args.model_class == "huggingface":
        return HuggingFacewithChatTemplate(
            args.model_path,
            max_seq_len=args.max_seq_len
        )
    elif args.model_class == "llama_shadowkv":
        return LlamaShadowKV(
            args.model_path,
            sparse_budget=args.sparse_budget,
            local_budget=args.local_budget,
            outlier_budget=args.outlier_budget,
            chunk_size=args.chunk_size,
            rank=args.rank,
            max_length=args.max_seq_len,
            use_higgs_quantization=args.use_higgs_quantization,
            higgs_hadamard_groupsize=args.higgs_hadamard_groupsize,
            higgs_edenn_d=args.higgs_edenn_d,
            higgs_edenn_n=args.higgs_edenn_n,
            higgs_channel_size=args.higgs_channel_size,
            higgs_chunk_size=args.higgs_chunk_size,
        )
    elif args.model_class == "qwen3_shadowkv":
        return Qwen3ShadowKV(
            args.model_path,
            sparse_budget=args.sparse_budget,
            local_budget=args.local_budget,
            outlier_budget=args.outlier_budget,
            chunk_size=args.chunk_size,
            rank=args.rank,
            max_length=args.max_seq_len,
            use_higgs_quantization=args.use_higgs_quantization,
            higgs_hadamard_groupsize=args.higgs_hadamard_groupsize,
            higgs_edenn_d=args.higgs_edenn_d,
            higgs_edenn_n=args.higgs_edenn_n,
            higgs_channel_size=args.higgs_channel_size,
            higgs_chunk_size=args.higgs_chunk_size,
        )
    

def query_opencompass(model: nn.Module, user_prompt: str, max_tokens: int) -> str:
    generated_text = model.generate([user_prompt], max_out_len=max_tokens)[0]

    return generated_text


def main():
    args = _parse_args()

    if args.test_loading:
        test_loading_all()
        return

    random.seed(args.seed)

    # allows some buffer to accomdate variations in token usage for different tokenizers
    if args.max_tokens is None:
        if "0.5k" in args.dataset:
            args.max_tokens = 1024
        elif "2k" in args.dataset:
            args.max_tokens = 3072
        elif "8k" in args.dataset:
            args.max_tokens = 9216


    dataset, eval_func = load_longproc_data(args.dataset, args.path)
    random.shuffle(dataset)
    if args.n_samples is not None:
        dataset = dataset[:args.n_samples]

    # Load model
    model = build_model(args)

    eval_metrics = []
    num_inspect = 3
    for i, d in tqdm(list(enumerate(dataset[:args.n_samples]))):
        if i < num_inspect:
            print(f"Sample {i+1}/{args.n_samples}")
            print(f"Prompt: {d['input_prompt']}")
            print(f"Reference: {d['reference_output']}")

        prediction = query_opencompass(model, d["input_prompt"], args.max_tokens)

        metrics, additional_info = eval_func(prediction, d)
        if i < num_inspect:
            print(f"Prediction: {prediction}")
            print(f"Metrics: {metrics}")
            print(f"Additional info: {additional_info}")
        eval_metrics.append(metrics)

    for k, v in metrics.items():
        print(f"{k}: {sum([m[k] for m in eval_metrics])/len(eval_metrics)}")

if __name__ == '__main__':
    main()
