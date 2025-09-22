import torch
import torch.nn as nn
from einops import einsum
from pathlib import Path
import argparse
import yaml
import itertools
import pandas as pd
import torch._dynamo
torch._dynamo.config.cache_size_limit = 32
from cs336_systems.benchmark_end2end import benchmark_model_end_to_end

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x -= torch.max(x, dim=i, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=i, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.size(-1)
    scaled_dot_prod = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") / d_k ** 0.5
    if mask is not None:
        scaled_dot_prod = scaled_dot_prod.masked_fill(~mask, - torch.inf)
    attention_weights = softmax(scaled_dot_prod, -1)
    return einsum(attention_weights, V, "... seq_q seq_k, ... seq_k d_V -> ... seq_q d_V")


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self._initialize_weights()

    def _initialize_weights(self):
        std = torch.sqrt(torch.tensor(2 / (self.in_features + self.out_features)))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3.0 * std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "b s d_in, d_out d_in -> b s d_out")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)

        self.out_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        mask = torch.tril(torch.ones(S, S)).bool().to(Q.device)

        attn_weighted_vals = scaled_dot_product_attention(Q, K, V, mask)
        return self.out_proj(attn_weighted_vals)


def main():
    parser = argparse.ArgumentParser(description="Benchmarking Torch Attention")
    parser.add_argument("--output", type=str, help="file to save result")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="use torch.compile"
    )

    args = parser.parse_args()
    res_dir = Path(__file__).parent / "../results"
    res_dir.mkdir(parents=True, exist_ok=True)

    results = []

    model_dims = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]


    for d_model, seq_len in itertools.product(model_dims, seq_lens):
        print(f"benchmarking model: d_model={d_model}, seq_len={seq_len}")
        # Initialize model
        model = MultiHeadSelfAttention(
            d_model
        ).to("cuda")

        if args.compile:
            model = torch.compile(model)

        # Create random batch
        x = torch.randn(8, seq_len, d_model, device="cuda")
        try:
            fwd_mean, fwd_std, bwd_mean, bwd_std, mem_before_bwd_mean, mem_before_bwd_std, mem_after_bwd_mean, mem_after_bwd_std =benchmark_model_end_to_end(model, x, False, 5, 100, "cuda", False)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA OOM encountered. Clearing cache...")
                fwd_mean, fwd_std, bwd_mean, bwd_std, mem_before_bwd_mean, mem_before_bwd_std, mem_after_bwd_mean, mem_after_bwd_std = None, None, None, None, None, None, None, None
                torch.cuda.empty_cache()   # free up cached memory
            else:
                raise e

        del model, x
        torch.cuda.empty_cache()

        results.append({
            "d_model": d_model,
            "seq_len": seq_len,
            "Mean time fwd (s)": round(fwd_mean, 6),
            "Std fwd (s)": round(fwd_std, 6),
            "Mean time bwd (s)": round(bwd_mean, 6),
            "Std bwd (s)": round(bwd_std, 6),
            "Memory before bwd (GB)": mem_before_bwd_mean,
            "Memory after bwd (GB)": mem_after_bwd_mean,
        })

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    with open(res_dir / args.output, "w") as f:
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    main()