import timeit
import torch
import torch.nn as nn
import yaml
from statistics import mean, stdev
from typing import Tuple
from pathlib import Path
import yaml
import pandas as pd
import sys
import argparse
from contextlib import nullcontext
import itertools

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

def get_random_batch(
        batch_size: int,
        context_length: int,
        vocab_size: int,
        device: str
) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device)


def benchmark_model_end_to_end(
        model: nn.Module,
        x: torch.Tensor,
        forward_only: bool,
        warmup_steps: int,
        repeats: int,
        device: torch.device | str,
        autocast: bool,
) -> Tuple[float, float, float, float]:
    """
        Benchmark LLM model forward (and backward) pass end-to-end.
        Args:
            model: LLM model
            x: input batch (randomly generated)
            forward_only: if True, benchmark only the forward pass
            warmup_steps: num of warmup steps
            repeats: num of repeats of benchmarking
            device: cpu or gpu
        Returns:
            Tuple of (forward_time mean, forward_time std, backward_time mean, backward_time std)
    """
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else nullcontext()

    if not forward_only:
        optimizer = AdamW(model.parameters(), lr=1e-5)

    # Warm up
    print(f"Warmming up for {warmup_steps} times!")


    with ctx:
        for _ in range(warmup_steps):
            y_ = model(x)
            if not forward_only:
                optimizer.zero_grad()
                loss = cross_entropy(y_, x)
                loss.backward()
                optimizer.step()
            if device == torch.device("cuda") or device == "cuda":
                torch.cuda.synchronize()

    # Benchmarking
    print(f"Benchmarking for {repeats} times!")
    forward_times = []
    backward_times =[]

    with ctx:
        for _ in range(repeats):
            start_time = timeit.default_timer()
            y_ = model(x)
            if device == torch.device("cuda") or device == "cuda":
                torch.cuda.synchronize()
            forward_times.append(timeit.default_timer() - start_time)

            if not forward_only:
                start_time = timeit.default_timer()
                optimizer.zero_grad()
                loss = cross_entropy(y_, x)
                loss.backward()
                optimizer.step()
                if device == torch.device("cuda") or device == "cuda":
                    torch.cuda.synchronize()
                backward_times.append(timeit.default_timer() - start_time)

    return mean(forward_times), stdev(forward_times), mean(backward_times) if not forward_only else None, stdev(backward_times) if not forward_only else None





def main():
    parser = argparse.ArgumentParser(description="Benchmarking LLM")
    parser.add_argument("--output", type=str, help="file to save result")
    args = parser.parse_args()
    res_dir = Path(__file__).parent / "../results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_dir = Path(__file__).parent / "../configs"
    model_config_file = config_dir / "models.yaml"
    results = []


    with open(model_config_file, "r") as f:
        model_configs = yaml.safe_load(f)

    shared_params = model_configs["shared_parameters"]

    for model_name, autocast in itertools.product(model_configs["models"], [True, False]):
        config = model_configs["models"][model_name]
        print(f"Benchmarking model: {model_name}")

        # Initialize model
        model = BasicsTransformerLM(
            vocab_size=shared_params["vocab_size"],
            context_length=shared_params["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=shared_params["rope_theta"],
        ).to(shared_params["device"])

        # Create random batch
        x = get_random_batch(
            shared_params["batch_size"],
            shared_params["context_length"],
            shared_params["vocab_size"],
            shared_params["device"]
        )

        fwd_mean, fwd_std, bwd_mean, bwd_std =benchmark_model_end_to_end(model, x, shared_params["forward_only"], shared_params["warmup_steps"], shared_params["repeats"], shared_params["device"], autocast)
        del model, x
        torch.cuda.empty_cache()

        results.append({
            "model": model_name,
            "d_model": config["d_model"],
            "d_ff": config["d_ff"],
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "Context Length": shared_params["context_length"],
            "Mean time fwd (s)": round(fwd_mean, 6),
            "Std fwd (s)": round(fwd_std, 6),
            "Mean time bwd (s)": round(bwd_mean, 6) if not shared_params["forward_only"] else None,
            "Std bwd (s)": round(bwd_std, 6) if not shared_params["forward_only"] else None,
            "Warmup Steps": shared_params["warmup_steps"],
            "autocast": autocast
        })

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    with open(res_dir / args.output, "w") as f:
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    main()