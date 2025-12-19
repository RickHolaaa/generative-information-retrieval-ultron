import argparse
import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers.models.t5.configuration_t5 import T5Config

from model import T5ForPretrain


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    Set random seed for Python / NumPy / PyTorch, and enable deterministic behavior.
    Returns the seed for convenient logging.
    """
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def get_time() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_args(file_name: str):
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--save_path", type=str, default="./saved_models")
    parser.add_argument("--dataset_path", type=str, default="./data")

    # Representation
    parser.add_argument("--repr", type=str, default="pq", choices=["pq", "hc"])

    # Hierarchical clustering args (only used if --repr hc)
    parser.add_argument("--hc_K1", type=int, default=256)
    parser.add_argument("--hc_K2", type=int, default=256)
    parser.add_argument("--hc_niter", type=int, default=25)

    # model config
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--num_samples", type=str, default="10K", choices=["10K", "100K"]
    )

    if file_name == "train.py":
        parser.add_argument("--batch_size", type=int, default=500)  # 1000 for 100K
        parser.add_argument("--learning_rate", type=float, default=5e-4)
        parser.add_argument("--warmup_ratio", type=float, default=0.1)

    if file_name == "evaluate.py":
        parser.add_argument("--batch_size", type=int, default=100)
        parser.add_argument("--num_beams", type=int, default=20)
        parser.add_argument("--noise_factor", type=float, default=0.0)

        # HC reranking controls
        parser.add_argument("--hc_max_candidates", type=int, default=2000)

    # PQ
    parser.add_argument("--num_subspace", type=int, default=4)  # M
    parser.add_argument("--num_clusters", type=int, default=128)  # K

    args = parser.parse_args()

    # Convenience: for HC, vecid length is 2 and vocab must cover both levels.
    if args.repr == "hc":
        args.num_subspace = 2
        args.num_clusters = max(args.hc_K1, args.hc_K2)

    return args


def save_model(model: T5ForPretrain, save_dir: str, name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), Path(save_dir) / f"{name}.pth")


def save_config(args, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(Path(save_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)


def load_model(args) -> T5ForPretrain:
    from dataset import Sift1mDataset

    config = T5Config(
        is_encoder_decoder=False,
        vocab_size=args.num_clusters,  # output vocab size
        d_model=Sift1mDataset.VECTOR_DIM,  # 128
        d_ff=args.d_ff,
        d_kv=args.d_kv,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
    )
    model = T5ForPretrain(config, args)
    return model


def plot_loss(losses: List[float], save_dir: str = None) -> None:
    plt.title("Learning Curve")
    x = np.arange(len(losses))
    plt.plot(x, losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(Path(save_dir) / "loss.png")
    else:
        plt.show()
    plt.clf()
