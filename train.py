from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from transformers.modeling_outputs import Seq2SeqLMOutput

from dataset import Sift1mDataset, Feature
from model import T5ForPretrain
from utils import load_model, save_model, save_config, get_args, get_time, set_seed, plot_loss


def cross_entropy(logits: Tensor, soft_targets: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    return loss.mean()  # average over batch


def compute_loss(model: T5ForPretrain, codewords: Tensor, x: Tensor, labels: Tensor, lam=0.2, num_negs=20) -> Tensor:
    '''
        codewords:  (M, K, vec_dim / M)
        x:          (batch_size, seq_len, vec_dim)
        label:      (batch_size, num_neighbors, vecid_len)
    '''
    loss = 0
    vecid_len = labels.size(2)
    for i in range(vecid_len):
        outputs: Seq2SeqLMOutput = model(decoder_inputs_embeds=x)
        next_logits = outputs.logits[:, -1, :]

        # indexing loss
        tokid = labels[:, 0, i]
        indexing_loss = model.loss_fct(next_logits, tokid)

        # retrieval loss
        L = min(num_negs, labels.size(1) - 1)
        if L > 0:
            negs = labels[:, 1 : 1 + L, i]  # (B, L)

            pos_score = next_logits.gather(1, tokid.unsqueeze(1))  # (B,1)
            neg_score = next_logits.gather(1, negs)  # (B,L)
            scores = torch.cat([pos_score, neg_score], dim=1)  # (B,1+L)

            targets = torch.zeros(
                scores.size(0), dtype=torch.long, device=scores.device
            )
            retr_loss = F.cross_entropy(scores, targets)
        else:
            retr_loss = 0.0

        loss = loss + indexing_loss + lam * retr_loss
        
        centroid = torch.index_select(codewords[i], dim=0, index=tokid)         # (batch_size, vec_dim / M)
        centroid = model.output_proj(centroid.to(x)).unsqueeze(dim=1)           # (batch_size, 1, vec_dim)
        x = torch.cat([x, centroid], dim=1)                                     # (batch_size, vecid_len, vec_dim)
    return loss / vecid_len


def main(args):
    device = "cuda:0"
    model = load_model(args).to(device)
    # folder = get_time()
    folder = args.num_samples
    path = Path(args.save_path)/folder
    save_config(args, path)

    dataset = Sift1mDataset(split="database", args=args, index_path=path)
    g = torch.Generator()
    g.manual_seed(args.random_seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, generator=g)
    codewords = dataset.codewords.to(device)

    t_total = int(len(dataloader.dataset) * args.epochs // args.batch_size)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=args.warmup_ratio * t_total, num_training_steps=t_total)

    print(folder.center(30, '-'))
    losses = []
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"[{epoch + 1:2d}/{args.epochs}]")
        batch: Feature  # just a type hint
        for batch in pbar:
            # print(batch.x.shape, batch.label.shape, batch.label[0])
            inputs = (batch.x.to(device, non_blocking=True), batch.label.to(device, non_blocking=True))
            loss = compute_loss(model, codewords, *inputs, lam=3, num_negs=50)
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
