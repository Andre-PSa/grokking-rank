# grokking-rank

Trains a small MLP on (a + b) mod 97 and tracks the effective rank of its weights while it groks.

`runs/` has the logs for seeds 0-23 (AdamW, lr 1e-3, weight decay 1.0, 40% of the pairs for training, 20k steps).

```
pip install torch numpy pandas matplotlib
python plot_seeds.py runs                                    # stats and figure
python run_seeds.py --start 0 --end 23 --folder new_runs     # rerun the seeds
```

Test accuracy reaches 95% around step 3800. About 72% of the fall in W1's entropy rank (init to step 20k) happens after that, compared with about 8% for its stable rank.
