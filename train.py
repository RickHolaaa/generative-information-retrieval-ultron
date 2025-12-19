from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler
from transformers.modeling_outputs import Seq2SeqLMOutput

from dataset import Feature, Sift1mDataset
from model import T5ForPretrain
from utils import get_args, load_model, plot_loss, save_config, save_model, set_seed


def compute_loss_pq(
    model: T5ForPretrain, codewords: Tensor, x: Tensor, labels: Tensor, args
) -> Tensor:
    """
    PQ teacher-forcing loss.
    codewords: (M, K, Ds)
    x:         (B, 1, 128)  (starts as query embedding)
    labels:    (B, num_neighbors, M)
    """
    loss = 0.0
    vecid_len = labels.size(2)

    for i in range(vecid_len):
        outputs: Seq2SeqLMOutput = model(decoder_inputs_embeds=x)
        next_logits = outputs.logits[:, -1, :]  # (B, K)

        tokid = labels[:, 0, i]  # (B,)
        indexing_loss = model.loss_fct(next_logits, tokid)
        loss += indexing_loss

        centroid = torch.index_select(codewords[i], dim=0, index=tokid)  # (B, Ds)
        centroid = model.output_proj(centroid.to(x)).unsqueeze(dim=1)  # (B, 1, 128)
        x = torch.cat([x, centroid], dim=1)  # (B, 1+i, 128)

    return loss / vecid_len


def compute_loss_hc(
    model: T5ForPretrain, centroids1: Tensor, x: Tensor, labels: Tensor, args
) -> Tensor:
    """
    Hierarchical clustering loss for 2-level code [y1, y2].
    We embed y1 by appending projected level-1 centroid.
    centroids1: (K1, 128)
    x:          (B, 1, 128)
    labels:     (B, num_neighbors, 2)
    """
    # step 0: predict y1 from q
    out = model(decoder_inputs_embeds=x)
    logits = out.logits[:, -1, :]  # (B, V)
    y1 = labels[:, 0, 0]
    loss1 = model.loss_fct(logits, y1)

    # append centroid1[y1]
    c1 = centroids1.index_select(0, y1).to(x)  # (B, 128)
    c1 = model.output_proj(c1).unsqueeze(1)  # (B, 1, 128)
    x2 = torch.cat([x, c1], dim=1)

    # step 1: predict y2 conditioned on (q, centroid1)
    out = model(decoder_inputs_embeds=x2)
    logits = out.logits[:, -1, :]
    y2 = labels[:, 0, 1]
    loss2 = model.loss_fct(logits, y2)

    return (loss1 + loss2) / 2


def main(args):
    device = "cuda:0"
    model = load_model(args).to(device)

    folder = args.num_samples
    path = Path(args.save_path) / folder
    save_config(args, path)

    dataset = Sift1mDataset(split="database", args=args, index_path=path)

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        generator=g,
    )

    # representation-specific "codewords"
    rep_table = dataset.codewords.to(device)

    t_total = int(len(dataloader.dataset) * args.epochs // args.batch_size)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * t_total),
        num_training_steps=t_total,
    )

    print(folder.center(30, "-"))
    losses = []
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"[{epoch + 1:2d}/{args.epochs}]")
        batch: Feature
        for batch in pbar:
            x = batch.x.to(device, non_blocking=True)  # (B,1,128)
            labels = batch.label.to(device, non_blocking=True)  # (B,100,M) or (B,100,2)

            if args.repr == "pq":
                loss = compute_loss_pq(model, rep_table, x, labels, args)
            else:
                loss = compute_loss_hc(model, rep_table, x, labels, args)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix_str(f"loss: {loss:.4f}")

        if (epoch + 1) % 100 == 0:
            save_model(model, path, f"epoch{epoch + 1}")
            plot_loss(losses, save_dir=path)


if __name__ == "__main__":
    args = get_args(Path(__file__).name)
    set_seed(args.random_seed)
    main(args)
