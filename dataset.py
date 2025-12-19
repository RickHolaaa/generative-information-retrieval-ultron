import json
import os
from collections import defaultdict
from math import log10
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

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


class HierarchicalKMeansIndexer:
    """
    2-level hierarchical kmeans: code = [y1, y2]
    Level-2 is trained per level-1 cluster.
    """

    def __init__(self, K1: int, K2: int, niter: int = 25, seed: int = 123):
        self.K1, self.K2 = K1, K2
        self.niter = niter
        self.seed = seed
        self.centroids1: np.ndarray | None = None
        self.centroids2: Dict[int, np.ndarray] = {}

    def fit(self, x: np.ndarray):
        D = x.shape[1]

        km1 = faiss.Kmeans(D, self.K1, niter=self.niter, verbose=True, seed=self.seed)
        km1.train(x)
        self.centroids1 = km1.centroids

        index1 = faiss.IndexFlatL2(D)
        index1.add(self.centroids1)
        _, y1 = index1.search(x, 1)
        y1 = y1.reshape(-1)

        for c in range(self.K1):
            xs = x[y1 == c]
            if len(xs) == 0:
                self.centroids2[c] = np.repeat(
                    self.centroids1[c][None, :], self.K2, axis=0
                )
                continue

            km2 = faiss.Kmeans(
                D, self.K2, niter=self.niter, verbose=False, seed=self.seed
            )
            km2.train(xs)
            self.centroids2[c] = km2.centroids

    def encode(self, x: np.ndarray) -> np.ndarray:
        D = x.shape[1]
        index1 = faiss.IndexFlatL2(D)
        index1.add(self.centroids1)
        _, y1 = index1.search(x, 1)
        y1 = y1.reshape(-1)

        y2 = np.zeros_like(y1)
        for c in range(self.K1):
            idx = np.where(y1 == c)[0]
            if len(idx) == 0:
                continue
            index2 = faiss.IndexFlatL2(D)
            index2.add(self.centroids2[c])
            _, sub = index2.search(x[idx], 1)
            y2[idx] = sub.reshape(-1)

        return np.stack([y1, y2], axis=1).astype(np.int64)

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "hc_centroids1.npy", self.centroids1)
        np.savez(
            folder / "hc_centroids2.npz",
            **{str(k): v for k, v in self.centroids2.items()},
        )
        meta = {"K1": self.K1, "K2": self.K2, "niter": self.niter, "seed": self.seed}
        with open(folder / "hc_meta.json", "w") as f:
            json.dump(meta, f)

    def load(self, folder: Path):
        self.centroids1 = np.load(folder / "hc_centroids1.npy")
        z = np.load(folder / "hc_centroids2.npz")
        self.centroids2 = {int(k): z[k] for k in z.files}
        with open(folder / "hc_meta.json", "r") as f:
            meta = json.load(f)
        self.K1, self.K2 = meta["K1"], meta["K2"]
        self.niter, self.seed = meta["niter"], meta["seed"]


class Sift1mDataset(Dataset):
    VECTOR_DIM = 128

    _DATASET_SIZE = {"database": 1_000_000, "test": 10_000}

    def __init__(self, split: str, args, index_path: str) -> None:
        super().__init__()

        self.args = args
        self.dataset_split = split
        self.index_path = Path(index_path)
        self.dataset_path = Path(args.dataset_path) / args.num_samples

        self.vecid_len = args.num_subspace
        self.num_digits = int(log10(args.num_clusters)) + 1
        self.vector_name = "query" if self.dataset_split == "test" else "key"

        self.index_path.mkdir(parents=True, exist_ok=True)

        self.codewords: Tensor | None = (
            None  # PQ codewords (M,K,Ds) OR HC centroids1 (K1,128) as Tensor
        )
        self._to_features()

    @property
    def vecids(self) -> Tuple[List[int]]:
        """
        Yield all valid vecids as lists of ints, derived from id_table keys.
        """

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

    def _build_posting_id_table(self) -> None:
        """
        Build id_table as VecID string -> list of vector ids.
        Safe for both PQ and HC.
        """
        postings = defaultdict(list)
        for f in self.features:
            key = self.vecid_to_str(f.label[0])  # code of the vector itself
            postings[key].append(f.id.item())
        self.id_table = postings

    def _to_features(self) -> None:
        dataset = {
            "database": torch.load(self.dataset_path / "database.pt"),
            "test": torch.load(
                self.dataset_path / f"test_{self.args.noise_factor:.02f}.pt"
            )
            if self.dataset_split == "test"
            else None,
        }

        # ----------------------------
        # Build representation codes
        # ----------------------------
        if self.args.repr == "pq":
            num_subspace = self.args.num_subspace
            k = self.args.num_clusters

            pq = PQRetriever(M=num_subspace, Ks=k)
            if self.dataset_split == "database":
                pq.fit(dataset["database"]["key"].numpy())
                np.save(self.index_path / "codebook.npy", pq.codewords)
            else:
                pq.codewords = np.load(self.index_path / "codebook.npy")
                pq.Ds = self.VECTOR_DIM // num_subspace

            database_code = pq.encode(dataset["database"]["key"].numpy())  # (N, M)
            database_code = torch.from_numpy(database_code).long()

            self.codewords = torch.from_numpy(pq.codewords)  # (M, K, Ds)

            self.vecid_len = num_subspace
            self.num_digits = int(log10(k)) + 1

        elif self.args.repr == "hc":
            K1, K2 = self.args.hc_K1, self.args.hc_K2
            hc = HierarchicalKMeansIndexer(
                K1=K1, K2=K2, niter=self.args.hc_niter, seed=self.args.random_seed
            )

            if self.dataset_split == "database":
                hc.fit(dataset["database"]["key"].numpy())
                hc.save(self.index_path)
            else:
                hc.load(self.index_path)

            database_code = hc.encode(dataset["database"]["key"].numpy())  # (N, 2)
            database_code = torch.from_numpy(database_code).long()

            # We'll store level-1 centroids as "codewords" for embedding y1 during training/eval.
            self.codewords = torch.from_numpy(hc.centroids1).float()  # (K1, 128)

            self.vecid_len = 2
            self.num_digits = int(log10(max(K1, K2))) + 1

        else:
            raise ValueError(f"Unknown repr: {self.args.repr}")

        # ----------------------------
        # Generate features
        # ----------------------------
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

        # ----------------------------
        # Save / load id_table
        # ----------------------------
        if self.dataset_split == "database":
            self._build_posting_id_table()
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

    def _random_sample(x: np.ndarray, n: int):
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
