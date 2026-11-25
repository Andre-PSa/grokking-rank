"""Aggregate many grok_rank.py runs: summary statistics and the proposal figure.

Usage:  python plot_seeds.py runs        (folder with seed*.csv files)
Writes: prelim.pdf, prelim.png and prints per-seed and summary numbers.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = sys.argv[1] if len(sys.argv) > 1 else "runs"
files = sorted(glob.glob(os.path.join(folder, "seed*.csv")))
runs = [pd.read_csv(f) for f in files]
# keep only finished runs (same length as the longest one)
n = max(len(d) for d in runs)
runs = [d for d in runs if len(d) == n]
print(f"{len(runs)} finished runs out of {len(files)} files")

THRESH = 0.95  # "generalised" = test accuracy reaches 95%


def frac_after(d, col, t):
    """Fraction of the total fall (init -> end) that happens after step t."""
    r = d[col]
    at_t = r[d.step == t].iloc[0]
    return (at_t - r.iloc[-1]) / (r.iloc[0] - r.iloc[-1])


rows = []
for f, d in zip(files, runs):
    if (d.test_acc >= THRESH).any():
        t = int(d.step[d.test_acc >= THRESH].iloc[0])
        rows.append(dict(
            run=os.path.basename(f),
            t_train100=int(d.step[d.train_acc >= 0.999].iloc[0]),
            t_test95=t,
            entropy_after=frac_after(d, "W1_entropy_rank", t),
            stable_after=frac_after(d, "W1_stable_rank", t),
        ))
    else:
        rows.append(dict(run=os.path.basename(f), t_test95=None))
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))
ok = tab.dropna()
print(f"\n{len(ok)}/{len(tab)} runs grokked (test acc >= {THRESH})")
for c in ["t_train100", "t_test95", "entropy_after", "stable_after"]:
    v = ok[c].astype(float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    print(f"{c:>14}: mean {v.mean():.3f}  sd {v.std():.3f}  "
          f"middle 95% [{lo:.3f}, {hi:.3f}]")

# figure: mean over seeds, shaded 10th-90th percentile
step = runs[0].step.values
m = step <= 12000
fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax2 = ax.twinx()
series = [
    ("train_acc", "0.45", "--", "train accuracy", ax, False),
    ("test_acc", "k", "-", "test accuracy", ax, False),
    ("W1_entropy_rank", "tab:blue", "-", "entropy rank of $W_1$", ax2, True),
    ("W1_stable_rank", "tab:orange", "-", "stable rank of $W_1$", ax2, True),
]
for col, color, ls, label, a, norm in series:
    A = np.array([(d[col] / d[col].iloc[0] if norm else d[col]).values
                  for d in runs])[:, m]
    a.plot(step[m], A.mean(0), color=color, ls=ls, lw=1.4, label=label)
    lo, hi = np.percentile(A, [10, 90], axis=0)
    a.fill_between(step[m], lo, hi, color=color, alpha=0.18, lw=0)
ax.set_xlabel("training step")
ax.set_ylabel("accuracy")
ax2.set_ylabel("rank (rel. to init)")
ax.set_ylim(0, 1.05)
ax2.set_ylim(0, 1.05)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center", ncol=4,
           frameon=False, bbox_to_anchor=(0.5, 1.0))
for a in (ax, ax2):
    a.spines["top"].set_visible(False)
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig("prelim.pdf")
plt.savefig("prelim.png", dpi=150)
print(f"\nsaved prelim.pdf / prelim.png (mean over {len(runs)} seeds, "
      "shading = 10th-90th percentile)")