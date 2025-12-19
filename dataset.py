import json
import os
from math import log2, log10
from pathlib import Path
from typing import List, NamedTuple, Tuple

import faiss
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow_datasets as tfds
import torch
from nanopq import PQ
from torch import LongTensor, Tensor
from torch.utils.data import Dataset


class Feature(NamedTuple):
    x: Tensor
    id: LongTensor
    label: LongTensor
    neighbors: LongTensor
    dist: Tensor


class PQRetriever(PQ):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_neighbors(self, q: np.ndarray, k: int = 100) -> np.ndarray:
        dtable = self.dtable(q)
        dists = dtable.adist(self.codewords)
        return np.argsort(dists)[:k]


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def unpack_pq_codes(packed: np.ndarray, m: int, nbits: int) -> np.ndarray:
    """
    Unpack FAISS PQ packed codes into shape (N, m) ints in [0, 2^nbits-1].

    packed: (N, code_size_bytes) uint8
    returns: (N, m) int64
    """
    if packed.ndim != 2:
        raise ValueError(f"packed codes must be 2D, got shape {packed.shape}")

    N = packed.shape[0]
    K = 1 << nbits
    out = np.zeros((N, m), dtype=np.int64)

    # Interpret packed bytes as bitstream (little-endian within each byte)
    # FAISS packs codes consecutively: code0 (nbits), code1 (nbits), ...
    for i in range(N):
        bitpos = 0
        for j in range(m):
            val = 0
            for b in range(nbits):
                byte_idx = (bitpos + b) // 8
                bit_idx = (bitpos + b) % 8
                bit = (packed[i, byte_idx] >> bit_idx) & 1
                val |= bit << b
            out[i, j] = val
            bitpos += nbits
            if out[i, j] >= K:
                raise ValueError("Unpacked code out of range; check nbits/unpack.")
    return out


class IVFPQRetriever:
    """Inverted File + Product Quantization over residuals (training artifacts only)."""

    def __init__(
        self,
        num_coarse: int,
        num_fine_subspace: int,
        num_fine_clusters: int,
        vec_dim: int = 128,
    ):
        if not _is_power_of_two(num_fine_clusters):
            raise ValueError(
                f"num_fine_clusters must be power of two, got {num_fine_clusters}"
            )

        self.num_coarse = num_coarse
        self.num_fine_subspace = num_fine_subspace
        self.num_fine_clusters = num_fine_clusters
        self.vec_dim = vec_dim

        self.nbits = int(log2(num_fine_clusters))  # 2^nbits == K
        self.coarse_centroids = None
        self.fine_pq = None

    def fit(self, X: np.ndarray) -> None:
        X = X.astype(np.float32)

        # Step 1: Coarse quantization
        print(f"Training coarse quantization with {self.num_coarse} clusters...")
        coarse_kmeans = faiss.Kmeans(
            self.vec_dim, self.num_coarse, niter=20, verbose=True
        )
        coarse_kmeans.train(X)
        self.coarse_centroids = coarse_kmeans.centroids

        # Step 2: Assign to coarse
        coarse_index = faiss.IndexFlatL2(self.vec_dim)
        coarse_index.add(self.coarse_centroids)
        _, coarse_assignments = coarse_index.search(X, 1)
        coarse_assignments = coarse_assignments.flatten()

        # Step 3: Train PQ on residuals
        residuals = X - self.coarse_centroids[coarse_assignments]
        print(
            f"Training fine PQ on residuals: m={self.num_fine_subspace}, "
            f"K={self.num_fine_clusters} (nbits={self.nbits})"
        )
        self.fine_pq = faiss.IndexPQ(self.vec_dim, self.num_fine_subspace, self.nbits)
        self.fine_pq.train(residuals)
        self.fine_pq.add(residuals)

    def encode(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (coarse_ids, packed_fine_codes)."""
        X = X.astype(np.float32)

        coarse_index = faiss.IndexFlatL2(self.vec_dim)
        coarse_index.add(self.coarse_centroids)
        _, coarse_ids = coarse_index.search(X, 1)
        coarse_ids = coarse_ids.flatten()

        residuals = X - self.coarse_centroids[coarse_ids]
        packed_fine_codes = self.fine_pq.sa_encode(residuals)  # packed bytes

        return coarse_ids, packed_fine_codes


class Sift1mDataset(Dataset):
    VECTOR_DIM = 128
    _DATASET_SIZE = {"database": 1_000_000, "test": 10_000}

    def __init__(self, split: str, args, index_path: str) -> None:
        super().__init__()

        self.args = args
        self.dataset_split = split
        self.index_path = Path(index_path)
        self.dataset_path = Path(args.dataset_path) / args.num_samples

        use_ivf = getattr(args, "use_ivf_pq", False)
        if use_ivf:
            self.vecid_len = 1 + args.num_fine_subspace  # coarse + M fine tokens
            vocab_size = args.num_coarse_clusters + (
                args.num_fine_subspace * args.num_fine_clusters
            )
            self.num_digits = len(str(vocab_size - 1))
        else:
            self.vecid_len = args.num_subspace
            self.num_digits = int(log10(args.num_clusters)) + 1

        self.vector_name = "query" if self.dataset_split == "test" else "key"
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._to_features(args.num_subspace, args.num_clusters)

    @property
    def vecids(self) -> Tuple[List[int]]:
        def _to_int_list(s: str) -> List[int]:
            return [
                int(s[i : i + self.num_digits])
                for i in range(0, len(s), self.num_digits)
            ]

        return (_to_int_list(vecid_str) for vecid_str in self.id_table.keys())

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Feature:
        return self.features[index]

    def vecid_to_str(self, vecid: Tensor) -> str:
        if vecid.size(0) != self.vecid_len:
            raise ValueError(
                f"your vecid shape is {vecid.shape}, but expect torch.size([{self.vecid_len}])"
            )
        return "".join(f"{tokid.item():0{self.num_digits}d}" for tokid in vecid)

    def _to_features(self, num_subspace: int, k: int) -> None:
        dataset = {
            "database": torch.load(self.dataset_path / "database.pt"),
            "test": torch.load(
                self.dataset_path / f"test_{self.args.noise_factor:.02f}.pt"
            )
            if self.dataset_split == "test"
            else None,
        }

        use_ivf = getattr(self.args, "use_ivf_pq", False)

        if use_ivf:
            C = self.args.num_coarse_clusters
            M = self.args.num_fine_subspace
            K = self.args.num_fine_clusters

            if not _is_power_of_two(K):
                raise ValueError("num_fine_clusters must be power of two for IVF-PQ.")
            nbits = int(log2(K))

            # If already cached, just load database_code directly.
            db_code_file = self.index_path / "database_code.npy"

            if self.dataset_split == "database":
                ivf_pq = IVFPQRetriever(
                    num_coarse=C,
                    num_fine_subspace=M,
                    num_fine_clusters=K,
                    vec_dim=self.VECTOR_DIM,
                )
                ivf_pq.fit(dataset["database"]["key"].numpy())
                coarse_ids, packed = ivf_pq.encode(dataset["database"]["key"].numpy())

                # Save artifacts
                np.save(
                    self.index_path / "coarse_centroids.npy", ivf_pq.coarse_centroids
                )
                faiss.write_index(
                    ivf_pq.fine_pq, str(self.index_path / "fine_pq.index")
                )
                np.save(self.index_path / "coarse_ids.npy", coarse_ids)
                np.save(self.index_path / "fine_codes_packed.npy", packed)

                fine_codes = unpack_pq_codes(
                    packed.astype(np.uint8), m=M, nbits=nbits
                )  # (N,M)

                # Offset fine tokens into disjoint ranges: C + s*K + code
                offsets = (C + np.arange(M) * K).reshape(1, M)  # (1,M)
                fine_tok = fine_codes + offsets  # (N,M)

                database_code = np.concatenate(
                    [
                        coarse_ids.reshape(-1, 1).astype(np.int64),
                        fine_tok.astype(np.int64),
                    ],
                    axis=1,
                )  # (N, 1+M)

                np.save(db_code_file, database_code)

            else:
                if not db_code_file.exists():
                    raise FileNotFoundError(
                        f"Missing {db_code_file}. Build database split first."
                    )
                database_code = np.load(db_code_file)

            self.codewords = None
            database_code = torch.from_numpy(database_code).long()

        else:
            # Standard PQ path
            pq = PQRetriever(M=num_subspace, Ks=k)
            if self.dataset_split == "database":
                pq.fit(dataset["database"]["key"].numpy())
                np.save(self.index_path / "codebook.npy", pq.codewords)
            else:
                pq.codewords = np.load(self.index_path / "codebook.npy")
                pq.Ds = self.VECTOR_DIM // num_subspace

            database_code = pq.encode(dataset["database"]["key"].numpy())  # (N, M)
            self.codewords = torch.from_numpy(pq.codewords)  # (M, K, dim/M)
            database_code = torch.from_numpy(database_code).long()

        print("generate features...")
        self.features = [
            Feature(
                x=emb.unsqueeze(dim=0),
                id=uid,
                label=database_code[neighbors],
                neighbors=neighbors,
                dist=dist,
            )
            for emb, uid, neighbors, dist in zip(*dataset[self.dataset_split].values())
        ]

        if self.dataset_split == "database":
            self.id_table = {
                self.vecid_to_str(f.label[0]): f.id.item() for f in self.features
            }
            with open(self.index_path / "id_table.json", "w", encoding="utf-8") as f:
                json.dump(self.id_table, f)
        else:
            with open(self.index_path / "id_table.json", "r", encoding="utf-8") as f:
                self.id_table = json.load(f)


def build_dataset(
    dataset_path: str, k: int, num_samples: int = None, noise_factor: float = 0.0
) -> None:
    def _save(
        x: np.ndarray,
        idx: np.ndarray,
        neighbors: np.ndarray,
        dist: np.ndarray,
        path: Path,
        split: str,
    ) -> None:
        if (path / f"{split}.pt").exists():
            return

        vector_name = "key" if split == "database" else "query"
        torch.save(
            {
                vector_name: (
                    torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                ).to(torch.float32),
                "unique_id": torch.from_numpy(idx)
                if isinstance(idx, np.ndarray)
                else idx,
                "neighbors": torch.from_numpy(neighbors)
                if isinstance(neighbors, np.ndarray)
                else neighbors,
                "dist": torch.from_numpy(dist)
                if isinstance(dist, np.ndarray)
                else dist,
            },
            path / f"{split}.pt",
        )

    def _load_dataset(split: str):
        if split == "database" and (dataset_path / "database.pt").exists():
            data = torch.load(dataset_path / "database.pt")
            return data["key"], data["unique_id"]
        data = tfds.load(
            "sift1m", split=split, batch_size=Sift1mDataset._DATASET_SIZE[split]
        )
        data = next(iter(data))
        x = data["embedding"].numpy() / 255.0
        unique_id = data["index"].numpy()
        return x, unique_id

    def _random_sample(x: np.array, n: int):
        perm = np.random.permutation(x.shape[0])
        x = x[perm][:n]
        return x, perm

    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)
    dataset_path /= f"{num_samples // 1000}K"
    dataset_path.mkdir(parents=True, exist_ok=True)

    # training set
    x, unique_id = _load_dataset("database")
    if num_samples is not None and num_samples < x.shape[0]:
        x, _ = _random_sample(x, num_samples)
        unique_id = np.arange(num_samples)

    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    dist, indices = index.search(x, k)
    _save(x, unique_id, indices, dist, dataset_path, "database")

    # testing set
    if num_samples is not None and num_samples // 10 < x.shape[0]:
        q, perm = _random_sample(x, num_samples // 10)
        unique_id = unique_id[perm]
    noise = np.random.uniform(-noise_factor, noise_factor, size=q.shape)
    q = np.clip(q + noise, 0, 1)
    dist, indices = index.search(q, k)
    _save(q, unique_id, indices, dist, dataset_path, f"test_{noise_factor:.02f}")


if __name__ == "__main__":
    from utils import set_seed

    set_seed()
    build_dataset("./data", k=100, num_samples=10_000, noise_factor=0.0)
