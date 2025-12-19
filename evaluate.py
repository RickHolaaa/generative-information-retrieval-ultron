from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Feature, Sift1mDataset
from model import T5ForPretrain
from trie import Trie
from utils import get_args, load_model, set_seed


@dataclass
class Metric:
    ks: Sequence[int] = (1, 10, 20)
    total: int = 0
    corr_num: Dict[int, int] = None

    def __post_init__(self) -> None:
        self.corr_num = {k: 0 for k in self.ks}

    def __str__(self) -> str:
        return ", ".join(f"R@{k}: {r:.3f}" for k, r in zip(self.ks, self.recall))

    @property
    def recall(self) -> Tuple[float]:
        return tuple(self.corr_num[k] / self.total for k in self.ks)

    def update(self, preds: Tensor, labels: Tensor) -> None:
        self.total += len(labels)

        for ranked, label in zip(preds, labels):
            for k in self.ks:
                if label in ranked[:k]:
                    self.corr_num[k] += 1


@torch.no_grad()
def constrained_beam_search(
    model: T5ForPretrain,
    codebooks: Tensor,
    q: Tensor,
    args,
    prefix_allowed_token_fn: Callable[[List[int]], List[int]],
) -> LongTensor:
    def _sort_beams(
        beams: List[Tuple[LongTensor, Tensor]],
    ) -> Tuple[LongTensor, List[Tuple[LongTensor, Tensor]]]:
        scores = torch.stack([s for _, s in beams])  # (num_beams, batch_size)
        tokids = torch.stack(
            [v for v, _ in beams]
        )  # (num_beams, batch_size, tokid_len)
        idx = torch.argsort(scores, dim=0, descending=True)

        idx_exp = idx.unsqueeze(2).expand(tokids.shape)
        sorted_tokids = torch.gather(
            tokids, 0, idx_exp
        )  # (num_beams, batch_size, tokid_len)
        sorted_scores = torch.gather(scores, 0, idx)  # (num_beams, batch_size)

        num_beams = len(tokids)
        sorted_beams: List[Tuple[LongTensor, Tensor]] = [
            (sorted_tokids[i], sorted_scores[i]) for i in range(num_beams)
        ]

        return sorted_tokids, sorted_beams

    # (tokids: List[LongTensor], score: Tensor)
    beams: List[Tuple[LongTensor, Tensor]] = [
        (
            torch.empty((args.batch_size, 0), device=q.device).long(),
            torch.zeros(args.batch_size, device=q.device),
        )
    ]
    vecid_len = args.num_subspace
    for i in range(vecid_len):
        seqs: List[Tensor] = [q for _ in beams]
        tokids: LongTensor  # (batch_size, tokid_len)
        for j, (tokids, _) in enumerate(beams):
            if tokids.size(1) < 1:
                continue
            centroids = codebooks[i, tokids]  # (batch_size, tokid_len, vec_dim / M)
            centroids = model.output_proj(centroids)  # (batch_size, tokid_len, vec_dim)
            seqs[j] = torch.cat([seqs[j], centroids], dim=1)
        seqs = torch.cat(seqs, dim=0)  # (num_beams * batch_size, tokid_len, vec_dim)

        outputs = model(decoder_inputs_embeds=seqs)
        logits = outputs.logits[:, -1, :].view(
            len(beams), args.batch_size, args.num_clusters
        )

        # apply constraints
        for i, (tokids, _) in enumerate(beams):
            for j in range(args.batch_size):
                allowed = prefix_allowed_token_fn([0] + tokids[j].tolist())
                mask = (
                    torch.ones(args.num_clusters)
                    .bool()
                    .to(logits.device, non_blocking=True)
                )
                mask[allowed] = 0
                logits[i, j, mask] += float("-inf")
        logprobs = logits.log_softmax(dim=-1)

        new_beams: List[Tuple[LongTensor, Tensor]] = []
        for j, (tokids, score) in enumerate(beams):
            # (k, batch_size)
            topk_logp, topk_idx = torch.topk(
                logprobs[j, ...].T, k=min(args.num_clusters, args.num_beams), dim=0
            )
            for lp, idx in zip(topk_logp, topk_idx):
                new_tokids = torch.cat([tokids, idx.unsqueeze(1)], dim=1)
                new_beams.append((new_tokids, score + lp))
        _, new_beams = _sort_beams(new_beams)
        beams = new_beams[: args.num_beams]

    sorted_vecids, _ = _sort_beams(beams)

    return sorted_vecids[: args.num_beams].transpose(0, 1)


def main(args):
    device = "cuda:0"
    # Include quantizer type in folder name
    path = Path(args.save_path) / f"{args.num_samples}_{args.quantizer}"
    model = load_model(args).to(device)
    ckpt = f"epoch{args.epochs}"
    model.load_state_dict(torch.load(path / f"{ckpt}.pth"))
    print(f"#params: {model.num_parameters(only_trainable=True) / 1e6:.0f}M")

    dataset = Sift1mDataset(split="test", args=args, index_path=path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)
    codebooks = dataset.codewords.to(device)

    vecid: List[int]
    vecid_trie = Trie([[0] + vecid for vecid in dataset.vecids])
    prefix_allowed_token_fn = lambda x: vecid_trie.get(x)

    # for t in vecid_trie:
    #     print(t)

    # print(prefix_allowed_token_fn([0, 4]))

    metric = Metric()

    model.eval()
    pbar = tqdm(dataloader, desc=f"[Evaluating {ckpt}]")
    batch: Feature  # just a type hint
    for i, batch in enumerate(pbar):
        q = batch.x.to(device, non_blocking=True)
        vecid = constrained_beam_search(
            model, codebooks, q, args, prefix_allowed_token_fn=prefix_allowed_token_fn
        )
        pred_id = (
            torch.tensor(
                [
                    dataset.id_table[dataset.vecid_to_str(x)]
                    for batchv in vecid
                    for x in batchv
                ]
            )
            .view(args.batch_size, args.num_beams)
            .long()
        )
        metric.update(pred_id, batch.neighbors[:, 0])
        pbar.set_postfix_str(str(metric))


if __name__ == "__main__":
    args = get_args(Path(__file__).name)
    set_seed(args.random_seed)
    main(args)
