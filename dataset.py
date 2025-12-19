import argparse
import os
import json
from math import log10
from pathlib import Path
from typing import Sequence, Tuple, List, NamedTuple

import faiss
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
import tensorflow_datasets as tfds
import torch
from nanopq import PQ
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, DataLoader


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


class Sift1mDataset(Dataset):
    VECTOR_DIM = 128

    _DATASET_SIZE = {
        "database": 1_000_000,
        "test": 10_000
    }

    def __init__(self, split: str, args, index_path: str) -> None:
        super().__init__()

        self.args = args
        self.dataset_split = split
        self.index_path = Path(index_path)
        self.dataset_path = Path(args.dataset_path)/args.num_samples
        self.vecid_len = args.num_subspace
        self.num_digits = int(log10(args.num_clusters)) + 1
        self.vector_name = "query" if self.dataset_split == "test" else "key"

        self.index_path.mkdir(parents=True, exist_ok=True)

        self._to_features(args.num_subspace, args.num_clusters)
    
    @property
    def vecids(self) -> Tuple[List[int]]:
        def _to_int_list(s: str) -> List[int]:
            return [int(s[i:i + self.num_digits]) for i in range(0, len(s), self.num_digits)]
        return (_to_int_list(vecid_str) for vecid_str in self.id_table.keys())
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index: int) -> Feature:
        return self.features[index]

    def vecid_to_str(self, vecid: Tensor) -> str:
        if vecid.size(0) != self.vecid_len:
            raise ValueError(f"your vecid shape is {vecid.shape}, but expect torch.size([{self.vecid_len}])")
        return ''.join(f"{tokid.item():0{self.num_digits}d}" for tokid in vecid)
    
    def _to_features(self, num_subspace: int, k: int) -> None:
        dataset = {
            "database": torch.load(self.dataset_path/"database.pt"),
            "test": torch.load(self.dataset_path/f"test_{self.args.noise_factor:.02f}.pt") if self.dataset_split == "test" else None
        }

        # generate vector id
        pq = PQRetriever(M=num_subspace, Ks=k)
        if self.dataset_split == "database":
            pq.fit(dataset["database"]["key"].numpy())
            np.save(self.index_path/"codebook.npy", pq.codewords)
        else:
            pq.codewords = np.load(self.index_path/"codebook.npy")
            pq.Ds = self.VECTOR_DIM // num_subspace
        database_code = pq.encode(dataset["database"]["key"].numpy())   # (num_samples, M)
        self.codewords = torch.from_numpy(pq.codewords)                 # (M, K, vec_dim / M)

        print("generate features...")
        database_code = torch.from_numpy(database_code).long()
        self.features = [
            Feature(x=emb.unsqueeze(dim=0), id=id, label=database_code[neighbors], neighbors=neighbors, dist=dist)
            for emb, id, neighbors, dist in zip(*dataset[self.dataset_split].values())
        ]
        if self.dataset_split == "database":
            self.id_table = {self.vecid_to_str(f.label[0]): f.id.item() for f in self.features}
            with open(self.index_path/"id_table.json", 'w', encoding="utf-8") as f:
                json.dump(self.id_table, f)
        else:
            with open(self.index_path/"id_table.json", 'r', encoding="utf-8") as f:
                self.id_table = json.load(f)


def build_dataset(dataset_path: str, k: int, num_samples: int = None, noise_factor: float = 0.0) -> None:
    def _save(x: np.ndarray, idx: np.ndarray, neighbors: np.ndarray, dist: np.ndarray, path: Path, split: str) -> None:
        if (path/f"{split}.pt").exists():
            return

        vector_name = "key" if split == "database" else "query"
        torch.save({
            vector_name: (torch.from_numpy(x) if isinstance(x, np.ndarray) else x).to(torch.float32),
            "unique_id": torch.from_numpy(idx) if isinstance(idx, np.ndarray) else idx,
            "neighbors": torch.from_numpy(neighbors) if isinstance(neighbors, np.ndarray) else neighbors,
            "dist": torch.from_numpy(dist) if isinstance(dist, np.ndarray) else dist,
        }, path/f"{split}.pt")
    
    def _load_dataset(split: str):
        if split == "database" and (dataset_path/"database.pt").exists():
            data = torch.load(dataset_path/"database.pt")
            return data["key"], data["unique_id"]
        data = tfds.load('sift1m', split=split, batch_size=Sift1mDataset._DATASET_SIZE[split])
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
    dataset_path /= f"{num_samples//1000}K"
    dataset_path.mkdir(parents=True, exist_ok=True)

    # training set
    x, unique_id = _load_dataset("database")
    if num_samples is not None and num_samples < x.shape[0]:
        x, _ = _random_sample(x, num_samples)
        unique_id = np.arange(num_samples)

    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    dist, indices = index.search(x, k)
    dataset_path: Path
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

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_path", type=str, default="./data/10K")
    # parser.add_argument("--num_subspace", type=int, default=4)
    # parser.add_argument("--num_clusters", type=int, default=128)
    # args = parser.parse_args()

    # folder = "test"
    # dataloader = DataLoader(Sift1mDataset(split="test", args=args, index_path=Path("./saved_models")/folder), batch_size=100, shuffle=True, pin_memory=True)
    # print(len(dataloader.dataset))

    # batch: Feature
    # for batch in dataloader:
    #     print(batch.x.shape, batch.id.shape, batch.label.shape, batch.neighbors, batch.dist.shape)
    #     # print(type(batch.x))
    #     # print(type(batch.id))
    #     # print(batch.x.shape)    # (batch_size, 128)
    #     # print(batch.x.shape)    # (batch_size,)
    #     # print(batch.id)
    #     break
