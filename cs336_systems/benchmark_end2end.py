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

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr

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
        device: torch.device | str
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
    if not forward_only:
        optimizer = AdamW(model.parameters(), lr=1e-5)

    # Warm up
    print(f"Warmming up for {warmup_steps} times!")

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
    # Load configuration
    config_dir = Path(__file__).parent / "../configs"
    results = []

    for file in config_dir.glob("*.yaml"):
        model_name = file.stem

        with open(file, "r") as f:
            config = yaml.safe_load(f)


        # Initialize model
        model = BasicsTransformerLM(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=config["rope_theta"],
        ).to(config["device"])

        #model = torch.compile(model)

        # Create random batch
        x = get_random_batch(
            config["batch_size"],
            config["context_length"],
            config["vocab_size"],
            config["device"]
        )

        fwd_mean, fwd_std, bwd_mean, bwd_std =benchmark_model_end_to_end(model, x, config["forward_only"], config["warmup_steps"], config["repeats"], config["device"])
        del model, x
        torch.cuda.empty_cache()


        results.append({
            "model": model_name,
            "d_model": config["d_model"],
            "d_ff": config["d_ff"],
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "Context Length": config["context_length"],
            "Mean time fwd (s)": round(fwd_mean, 6),
            "Std fwd (s)": round(fwd_std, 6),
            "Mean time bwd (s)": round(bwd_mean, 6) if not config["forward_only"] else None,
            "Std bwd (s)": round(bwd_std, 6) if not config["forward_only"] else None,
            "Warmup Steps": config["warmup_steps"],
        })

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()