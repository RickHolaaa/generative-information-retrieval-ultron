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
    Set random seed for Python / NumPy / PyTorch, and enable deterministic (reproducible) behavior.
    Returns the seed for convenient logging.

    Args:
        seed: The random seed to use
        deterministic: Whether to enable deterministic algorithms and related settings for torch
    """
    import os
    import random

    import numpy as np
    import torch

    # --- Python and NumPy ---
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # Note: Ideally set at the very beginning of the program
    random.seed(seed)
    np.random.seed(seed)

    # --- PyTorch (CPU / CUDA) ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Turn off TF32 to avoid minor differences caused by different GPU/Driver
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    # cuDNN related: benchmark should be turned off in deterministic scenarios
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if deterministic:
        # Enable deterministic algorithms (some ops will throw errors if not supported)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Older PyTorch versions don't have this API, so skip
            pass

        # CuBLAS requires this environment variable for full determinism in some cases
        # Choose one configuration: 16:8 saves memory; change to 4096:8 if errors occur
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        # If you encounter "CUBLAS_WORKSPACE_CONFIG not set" or deterministic errors, use:
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_time() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_args(file_name: str):
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--save_path", type=str, default="./saved_models")
    parser.add_argument("--dataset_path", type=str, default="./data")

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
    
    # Quantizer type: 'pq' for Product Quantization, 'hierarchical' for Hierarchical Clustering
    parser.add_argument(
        "--quantizer", type=str, default="pq", choices=["pq", "hierarchical"],
        help="Quantization method: 'pq' (Product Quantization) or 'hierarchical' (Hierarchical Clustering)"
    )

    if file_name == "train.py":
        parser.add_argument("--batch_size", type=int, default=500)  # 1000 for 100K
        parser.add_argument("--learning_rate", type=float, default=5e-4)
        parser.add_argument("--warmup_ratio", type=float, default=0.1)

    if file_name == "evaluate.py":
        parser.add_argument("--batch_size", type=int, default=100)
        parser.add_argument("--num_beams", type=int, default=20)
        parser.add_argument("--noise_factor", type=float, default=0.0)

    # PQ
    parser.add_argument("--num_subspace", type=int, default=4)  # M
    parser.add_argument("--num_clusters", type=int, default=128)  # K

    return parser.parse_args()


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
        vocab_size=args.num_clusters,
        d_model=Sift1mDataset.VECTOR_DIM,
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
