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
        """
        preds:  (B, >=maxK) ranked ids
        labels: (B,) ground-truth nearest neighbor id
        """
        self.total += len(labels)
        for ranked, label in zip(preds, labels):
            for k in self.ks:
                if label.item() in ranked[:k].tolist():
                    self.corr_num[k] += 1


@torch.no_grad()
def constrained_beam_search_pq(
    model: T5ForPretrain,
    codebooks: Tensor,
    q: Tensor,
    args,
    prefix_allowed_token_fn: Callable[[List[int]], List[int]],
) -> LongTensor:
    """
    Original PQ constrained beam search.
    Returns: (B, num_beams, M)
    """

    def _sort_beams(beams: List[Tuple[LongTensor, Tensor]]):
        scores = torch.stack([s for _, s in beams])  # (num_beams, B)
        tokids = torch.stack([v for v, _ in beams])  # (num_beams, B, L)
        idx = torch.argsort(scores, dim=0, descending=True)
        idx_exp = idx.unsqueeze(2).expand(tokids.shape)
        sorted_tokids = torch.gather(tokids, 0, idx_exp)
        sorted_scores = torch.gather(scores, 0, idx)
        sorted_beams = [
            (sorted_tokids[i], sorted_scores[i]) for i in range(len(sorted_tokids))
        ]
        return sorted_tokids, sorted_beams

    beams: List[Tuple[LongTensor, Tensor]] = [
        (
            torch.empty((args.batch_size, 0), device=q.device).long(),
            torch.zeros(args.batch_size, device=q.device),
        )
    ]

    vecid_len = args.num_subspace
    for step in range(vecid_len):
        seqs: List[Tensor] = [q for _ in beams]

        for j, (tokids, _) in enumerate(beams):
            if tokids.size(1) < 1:
                continue
            centroids = codebooks[step, tokids]  # (B, L, Ds)
            centroids = model.output_proj(centroids)  # (B, L, 128)
            seqs[j] = torch.cat([seqs[j], centroids], dim=1)

        seqs = torch.cat(seqs, dim=0)  # (num_beams*B, L, 128)

        outputs = model(decoder_inputs_embeds=seqs)
        logits = outputs.logits[:, -1, :].view(
            len(beams), args.batch_size, args.num_clusters
        )

        # apply trie constraints
        for bi, (tokids, _) in enumerate(beams):
            for b in range(args.batch_size):
                allowed = prefix_allowed_token_fn([0] + tokids[b].tolist())
                mask = torch.ones(
                    args.num_clusters, device=logits.device, dtype=torch.bool
                )
                mask[allowed] = 0
                logits[bi, b, mask] += float("-inf")

        logprobs = logits.log_softmax(dim=-1)

        new_beams: List[Tuple[LongTensor, Tensor]] = []
        for bi, (tokids, score) in enumerate(beams):
            topk_logp, topk_idx = torch.topk(
                logprobs[bi, ...].T, k=min(args.num_clusters, args.num_beams), dim=0
            )
            for lp, idx in zip(topk_logp, topk_idx):
                new_tokids = torch.cat([tokids, idx.unsqueeze(1)], dim=1)
                new_beams.append((new_tokids, score + lp))

        _, new_beams = _sort_beams(new_beams)
        beams = new_beams[: args.num_beams]

    sorted_vecids, _ = _sort_beams(beams)
    return sorted_vecids[: args.num_beams].transpose(0, 1)  # (B, num_beams, M)


@torch.no_grad()
def constrained_beam_search_hc(
    model: T5ForPretrain,
    centroids1: Tensor,
    q: Tensor,
    args,
    prefix_allowed_token_fn: Callable[[List[int]], List[int]],
) -> LongTensor:
    """
    HC beam search for 2 tokens: [y1, y2].
    Conditioning:
      - step0: from q
      - step1: from q + projected centroid1[y1]
    Returns: (B, num_beams, 2)
    """

    def _sort_beams(beams: List[Tuple[LongTensor, Tensor]]):
        scores = torch.stack([s for _, s in beams])  # (num_beams, B)
        tokids = torch.stack([v for v, _ in beams])  # (num_beams, B, L)
        idx = torch.argsort(scores, dim=0, descending=True)
        idx_exp = idx.unsqueeze(2).expand(tokids.shape)
        sorted_tokids = torch.gather(tokids, 0, idx_exp)
        sorted_scores = torch.gather(scores, 0, idx)
        sorted_beams = [
            (sorted_tokids[i], sorted_scores[i]) for i in range(len(sorted_tokids))
        ]
        return sorted_tokids, sorted_beams

    beams: List[Tuple[LongTensor, Tensor]] = [
        (
            torch.empty((args.batch_size, 0), device=q.device).long(),
            torch.zeros(args.batch_size, device=q.device),
        )
    ]

    # step 0
    outputs = model(decoder_inputs_embeds=q)
    logits0 = outputs.logits[:, -1, :].view(1, args.batch_size, args.num_clusters)

    # constraints for empty prefix
    for b in range(args.batch_size):
        allowed = prefix_allowed_token_fn([0])
        mask = torch.ones(args.num_clusters, device=logits0.device, dtype=torch.bool)
        mask[allowed] = 0
        logits0[0, b, mask] += float("-inf")

    logp0 = logits0.log_softmax(dim=-1)[0]  # (B, V)

    topk_logp, topk_idx = torch.topk(
        logp0, k=min(args.num_clusters, args.num_beams), dim=-1
    )  # (B, nb)

    new_beams = []
    for k in range(topk_idx.size(1)):
        y1 = topk_idx[:, k].unsqueeze(1)  # (B,1)
        score = topk_logp[:, k]  # (B,)
        new_beams.append((y1, score))

    _, new_beams = _sort_beams(new_beams)
    beams = new_beams[: args.num_beams]

    # step 1
    extended = []
    for tokids, score in beams:
        y1 = tokids[:, 0]
        c1 = centroids1.index_select(0, y1).to(q)  # (B,128)
        c1 = model.output_proj(c1).unsqueeze(1)  # (B,1,128)
        seq = torch.cat([q, c1], dim=1)  # (B,2,128)

        out = model(decoder_inputs_embeds=seq)
        logits1 = out.logits[:, -1, :]  # (B,V)

        for b in range(args.batch_size):
            allowed = prefix_allowed_token_fn([0] + tokids[b].tolist())
            mask = torch.ones(
                args.num_clusters, device=logits1.device, dtype=torch.bool
            )
            mask[allowed] = 0
            logits1[b, mask] += float("-inf")

        logp1 = logits1.log_softmax(dim=-1)

        topk_lp, topk_id = torch.topk(
            logp1, k=min(args.num_clusters, args.num_beams), dim=-1
        )  # (B, nb)
        for k in range(topk_id.size(1)):
            y2 = topk_id[:, k].unsqueeze(1)
            new_tok = torch.cat([tokids, y2], dim=1)  # (B,2)
            extended.append((new_tok, score + topk_lp[:, k]))

    _, extended = _sort_beams(extended)
    beams = extended[: args.num_beams]

    tokids = torch.stack([t for t, _ in beams])  # (nb, B, 2)
    return tokids.transpose(0, 1)  # (B, nb, 2)


def rank_candidates_by_l2(
    q_vec: Tensor, cand_ids: List[int], db_vecs: Tensor, topk: int
) -> List[int]:
    if len(cand_ids) == 0:
        return []
    ids = torch.tensor(cand_ids, device=q_vec.device, dtype=torch.long)
    vecs = db_vecs.index_select(0, ids)  # (C,128)
    d = ((vecs - q_vec) ** 2).sum(dim=1)  # (C,)
    order = torch.argsort(d)[:topk]
    return ids[order].tolist()


def main(args):
    device = "cuda:0"
    path = Path(args.save_path) / args.num_samples

    model = load_model(args).to(device)
    ckpt = f"epoch{args.epochs}"
    model.load_state_dict(torch.load(path / f"{ckpt}.pth"))
    print(f"#params: {model.num_parameters(only_trainable=True) / 1e6:.0f}M")

    dataset = Sift1mDataset(split="test", args=args, index_path=path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    # Trie constraints
    vecid_trie = Trie([[0] + vecid for vecid in dataset.vecids])
    prefix_allowed_token_fn = lambda x: vecid_trie.get(x)

    # Load db vectors for HC reranking (and it doesn't hurt PQ either)
    db = torch.load(dataset.dataset_path / "database.pt")
    db_vecs = db["key"].to(device, non_blocking=True)

    # Representation tables
    if args.repr == "pq":
        codebooks = dataset.codewords.to(device)  # (M,K,Ds)
    else:
        centroids1 = dataset.codewords.to(device)  # (K1,128)

    metric = Metric()
    model.eval()

    pbar = tqdm(dataloader, desc=f"[Evaluating {ckpt}]")
    batch: Feature
    for batch in pbar:
        q = batch.x.to(device, non_blocking=True)  # (B,1,128)

        if args.repr == "pq":
            vecid = constrained_beam_search_pq(
                model, codebooks, q, args, prefix_allowed_token_fn
            )
            # For PQ, treat each generated VecID as a posting list too:
            # we'll take the first id in the posting list (PQ tends to be near-unique),
            # OR fall back to -1.
            pred_rows = []
            topK = max(metric.ks)
            for b in range(args.batch_size):
                ranked = []
                for code in vecid[b]:  # each beam => vecid
                    key = dataset.vecid_to_str(code)
                    lst = dataset.id_table.get(key, [])
                    if isinstance(lst, list) and len(lst) > 0:
                        ranked.append(lst[0])
                    elif isinstance(lst, int):
                        ranked.append(lst)
                    if len(ranked) >= topK:
                        break
                if len(ranked) < topK:
                    ranked += [-1] * (topK - len(ranked))
                pred_rows.append(ranked)
            pred_id = torch.tensor(pred_rows, device=device, dtype=torch.long)

        else:
            vecid = constrained_beam_search_hc(
                model, centroids1, q, args, prefix_allowed_token_fn
            )

            # Expand posting lists + rerank by true L2 distance to query
            topK = max(metric.ks)
            pred_rows = []
            for b in range(args.batch_size):
                beams = vecid[b]  # (num_beams,2)
                cand = []
                seen = set()
                for code in beams:
                    key = dataset.vecid_to_str(code)
                    for cid in dataset.id_table.get(key, []):
                        if cid not in seen:
                            seen.add(cid)
                            cand.append(cid)
                    if len(cand) >= args.hc_max_candidates:
                        break

                ranked = rank_candidates_by_l2(q[b, 0], cand, db_vecs, topK)
                if len(ranked) < topK:
                    ranked += [-1] * (topK - len(ranked))
                pred_rows.append(ranked)

            pred_id = torch.tensor(pred_rows, device=device, dtype=torch.long)

        metric.update(pred_id, batch.neighbors[:, 0].to(device))
        pbar.set_postfix_str(str(metric))


if __name__ == "__main__":
    args = get_args(Path(__file__).name)
    set_seed(args.random_seed)
    main(args)
