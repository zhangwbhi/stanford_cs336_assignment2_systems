# bench_flash_attention.py
import argparse
import math
import itertools
import torch
import pandas as pd
import triton
import triton.testing
from triton.testing import do_bench
from pathlib import Path

from flash_attention import TritonFlashAttention2

def pytorch_attention(Q, K, V, causal: bool, scale: float):
    S = torch.einsum("b q d, b k d-> b q k", Q, K) * scale
    if causal:
        nq, nk = S.shape[1], S.shape[2]
        q_idx = torch.arange(nq, device=S.device)[:, None]
        k_idx = torch.arange(nk, device=S.device)[None, :]
        mask = q_idx < k_idx
        S = S.masked_fill(mask.unsqueeze(0), float("-inf"))
    P = torch.softmax(S, dim=-1)
    O = torch.einsum("b q k, b k d-> b q d", P, V)
    return O


def make_random_inputs(seq_len, d_model, dtype, device):
    torch.manual_seed(1234)
    Q = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    return Q, K, V


# ------------------
# Benchmark wrappers
# ------------------

def bench_forward_triton(Q, K, V, is_causal):
    def fn():
        TritonFlashAttention2.apply(Q, K, V, is_causal)
        torch.cuda.synchronize()
    return do_bench(fn, warmup=10, rep=50)


def bench_fwd_bwd_triton(Q, K, V, is_causal):
    def fn():
        O = TritonFlashAttention2.apply(Q, K, V, is_causal)
        grad_out = torch.randn_like(O)
        O.backward(grad_out)
        torch.cuda.synchronize()
    return do_bench(fn, warmup=10, rep=40)


def bench_forward_pytorch(Q, K, V, is_causal, scale):
    def fn():
        pytorch_attention(Q, K, V, is_causal, scale)
        torch.cuda.synchronize()
    return do_bench(fn, warmup=10, rep=50)


def bench_fwd_bwd_pytorch(Q, K, V, is_causal, scale):
    def fn():
        O = pytorch_attention(Q, K, V, is_causal, scale)
        grad_out = torch.randn_like(O)
        O.backward(grad_out)
        torch.cuda.synchronize()
    return do_bench(fn, warmup=10, rep=40)


# ------------------
# Main benchmark loop
# ------------------

def run_benchmarks(device_str="cuda:0", out_csv="results.csv"):
    device = torch.device(device_str)

    seq_lengths = [2 ** e for e in range(7, 17)]  # 128 .. 65536
    d_models = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    is_causal = True

    results = []
    total = len(seq_lengths) * len(d_models) * len(dtypes)

    for i, (seq_len, d_model, dtype) in enumerate(itertools.product(seq_lengths, d_models, dtypes), 1):
        scale = 1.0 / math.sqrt(d_model)
        print(f"\n[{i}/{total}] seq_len={seq_len}, d_model={d_model}, dtype={dtype}")

        try:
            Q, K, V = make_random_inputs(seq_len, d_model, dtype, device)
        except RuntimeError as e:
            print(f"Skipping config due to error: {e}")
            continue

        torch.cuda.empty_cache()

        # PyTorch
        try:
            Qp, Kp, Vp = [x.detach().clone().requires_grad_(True) for x in (Q, K, V)]
            t_fwd_pt = bench_forward_pytorch(Qp, Kp, Vp, is_causal, scale)

            Qp, Kp, Vp = [x.detach().clone().requires_grad_(True) for x in (Q, K, V)]
            t_fb_pt = bench_fwd_bwd_pytorch(Qp, Kp, Vp, is_causal, scale)

            t_bwd_pt = max(t_fb_pt - t_fwd_pt, 0.0)  # backward only
        except RuntimeError as e:
            print(f"PyTorch run failed: {e}")
            t_fwd_pt = t_bwd_pt = t_fb_pt = float("nan")

        # Triton
        try:
            Qt, Kt, Vt = [x.detach().clone().requires_grad_(True) for x in (Q, K, V)]
            t_fwd_tr = bench_forward_triton(Qt, Kt, Vt, is_causal)

            Qt, Kt, Vt = [x.detach().clone().requires_grad_(True) for x in (Q, K, V)]
            t_fb_tr = bench_fwd_bwd_triton(Qt, Kt, Vt, is_causal)

            t_bwd_tr = max(t_fb_tr - t_fwd_tr, 0.0)
        except RuntimeError as e:
            print(f"Triton run failed: {e}")
            t_fwd_tr = t_bwd_tr = t_fb_tr = float("nan")

        results.append({
            "device": device_str,
            "impl": "pytorch",
            "dtype": str(dtype).split(".")[-1],
            "seq_len": seq_len,
            "d_model": d_model,
            "fwd_ms": t_fwd_pt,
            "bwd_ms": t_bwd_pt,
            "fwd_bwd_ms": t_fb_pt,
        })
        results.append({
            "device": device_str,
            "impl": "triton",
            "dtype": str(dtype).split(".")[-1],
            "seq_len": seq_len,
            "d_model": d_model,
            "fwd_ms": t_fwd_tr,
            "bwd_ms": t_bwd_tr,
            "fwd_bwd_ms": t_fb_tr,
        })

    df = pd.DataFrame(results)
    df = df[["device", "impl", "dtype", "seq_len", "d_model", "fwd_ms", "bwd_ms", "fwd_bwd_ms"]]

    print("\n=== Benchmark summary ===")
    print(df.to_markdown(index=False))

    res_dir = Path(__file__).parent / "../results"
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(res_dir / args.output, "w") as f:
        f.write(df.to_markdown(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--out", type=str, default="results.csv", help="CSV output filename")
    args = parser.parse_args()
    run_benchmarks(device_str=args.device, out_csv=args.out)