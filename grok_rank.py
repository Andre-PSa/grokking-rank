"""Reproduce grokking on modular addition and track the rank of the weights.

Task: predict (a + b) mod p from the pair (a, b), training on a fraction of
all p*p pairs and testing on the rest.

Model: a small MLP. Each input number gets a learned embedding (rows of E),
the two embeddings are concatenated, passed through one ReLU hidden layer
(weights W1) and a linear output layer over the p classes.

Training: by default, full-batch AdamW with strong weight decay, the standard
recipe that produces grokking. With --optimizer sgd, it runs plain full-batch
gradient descent on  loss + (weight_decay / 2) * ||theta||^2  instead (PyTorch's
SGD weight_decay adds weight_decay * theta to the gradient), which is the
setting the L2 theory describes. Plain GD usually needs a much larger
learning rate than AdamW (try e.g. --lr 0.5) and may grok more slowly.

Logged every `--log-every` steps: train/test accuracy and loss, plus two
"effective rank" measures for E, W1, W2 and the two factorised products
W1_a E^T and W1_b E^T, where W1 = [W1_a, W1_b] splits W1 into the halves that
act on the embedding of a and of b. Each product maps a number (one-hot)
directly to the hidden layer, so it is a linear factorisation U V^T. (W2 W1 is
not tracked: the ReLU between them means their product is not a map the
network computes.) The two rank measures are:
  - entropy rank (Roy & Vetterli): exp(entropy of normalised singular values)
  - stable rank: ||W||_F^2 / ||W||_op^2

Outputs: a CSV of the logs and a figure with accuracy and effective rank.

Example:
    python grok_rank.py                      # default run
    python grok_rank.py --init-scale 3       # larger init (expected: longer plateau)
    python grok_rank.py --weight-decay 0.3   # weaker decay (expected: longer plateau)
    python grok_rank.py --optimizer sgd --lr 0.5 --steps 50000   # plain GD + L2
"""

import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def entropy_rank(W: torch.Tensor) -> float:
    s = torch.linalg.svdvals(W)
    p = s / s.sum()
    p = p[p > 0]
    return math.exp(-(p * p.log()).sum().item())


def stable_rank(W: torch.Tensor) -> float:
    s = torch.linalg.svdvals(W)
    return (s.pow(2).sum() / s[0].pow(2)).item()


class MLP(nn.Module):
    def __init__(self, p, d_embed, d_hidden, init_scale, scale_biases):
        super().__init__()
        self.d_embed = d_embed
        self.E = nn.Embedding(p, d_embed)
        self.W1 = nn.Linear(2 * d_embed, d_hidden)
        self.W2 = nn.Linear(d_hidden, p)
        # Scale the initial weight matrices (E, W1, W2) by init_scale; the
        # biases are scaled too only with --scale-biases.
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith("bias") and not scale_biases:
                    continue
                param.mul_(init_scale)

    def forward(self, ab):
        x = torch.cat([self.E(ab[:, 0]), self.E(ab[:, 1])], dim=1)
        return self.W2(torch.relu(self.W1(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train-frac", type=float, default=0.4)
    ap.add_argument("--d-embed", type=int, default=128)
    ap.add_argument("--d-hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--init-scale", type=float, default=1.0)
    ap.add_argument("--scale-biases", action="store_true")
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="grok_rank")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    p = args.p

    # All p*p pairs, randomly split into train and test.
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    pairs = torch.stack([a.flatten(), b.flatten()], dim=1)
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    perm = torch.randperm(len(pairs))
    n_train = int(args.train_frac * len(pairs))
    tr, te = perm[:n_train], perm[n_train:]

    model = MLP(p, args.d_embed, args.d_hidden, args.init_scale, args.scale_biases)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        # No momentum, full batch: exactly gradient descent on
        # loss + (weight_decay / 2) * ||theta||^2.
        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    rows = []
    for step in range(args.steps + 1):
        logits = model(pairs[tr])
        loss = loss_fn(logits, labels[tr])
        if step % args.log_every == 0:
            with torch.no_grad():
                train_acc = (logits.argmax(1) == labels[tr]).float().mean().item()
                te_logits = model(pairs[te])
                test_loss = loss_fn(te_logits, labels[te]).item()
                test_acc = (te_logits.argmax(1) == labels[te]).float().mean().item()
                E, W1, W2 = model.E.weight, model.W1.weight, model.W2.weight
                d = model.d_embed
                mats = {"E": E, "W1": W1, "W2": W2,
                        "W1aE": W1[:, :d] @ E.T,   # hidden <- number a
                        "W1bE": W1[:, d:] @ E.T}   # hidden <- number b
                row = dict(step=step, train_loss=loss.item(), test_loss=test_loss,
                           train_acc=train_acc, test_acc=test_acc)
                for k, M in mats.items():
                    row[f"{k}_entropy_rank"] = entropy_rank(M)
                    row[f"{k}_stable_rank"] = stable_rank(M)
                rows.append(row)
            if step % (10 * args.log_every) == 0:
                print(f"step {step:6d}  train acc {train_acc:.3f}  test acc {test_acc:.3f}"
                      f"  rank(E) {row['E_entropy_rank']:.1f}  rank(W1) {row['W1_entropy_rank']:.1f}")
        if step == args.steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    steps = [r["step"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(steps, [r["train_acc"] for r in rows], label="train")
    ax1.plot(steps, [r["test_acc"] for r in rows], label="test")
    ax1.set_ylabel("accuracy")
    ax1.legend()
    ax1.set_title(f"Modular addition, p={p}, train frac={args.train_frac}, "
                  f"{args.optimizer}, wd={args.weight_decay}, init x{args.init_scale}")
    ax2.plot(steps, [r["E_entropy_rank"] for r in rows], label="embedding E")
    ax2.plot(steps, [r["W1_entropy_rank"] for r in rows], label="hidden layer W1")
    ax2.plot(steps, [r["W2_entropy_rank"] for r in rows], label="output layer W2")
    ax2.plot(steps, [r["W1aE_entropy_rank"] for r in rows], label="product W1_a E^T")
    ax2.set_ylabel("effective rank (entropy)")
    ax2.set_xlabel("training step")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.set_xscale("symlog", linthresh=100)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    print(f"saved {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()  