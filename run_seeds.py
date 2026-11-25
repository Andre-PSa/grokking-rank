"""Run grok_rank.py for many seeds, a few at a time (works on Windows, macOS, Linux).

Usage:  python run_seeds.py --start 4 --end 99 --jobs 4
Seeds that already have runs/seed<N>.csv are skipped, so you can stop
(Ctrl+C) and restart at any time.
"""
import argparse
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=4)
ap.add_argument("--end", type=int, default=99)      # inclusive
ap.add_argument("--jobs", type=int, default=4)      # runs at the same time
ap.add_argument("--folder", default="runs")
args = ap.parse_args()

os.makedirs(args.folder, exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
todo = [s for s in range(args.start, args.end + 1)
        if not os.path.exists(os.path.join(args.folder, f"seed{s}.csv"))]
print(f"{len(todo)} seeds to run, {args.jobs} at a time")

running = {}
t0 = time.time()
done = 0
try:
    while todo or running:
        while todo and len(running) < args.jobs:
            s = todo.pop(0)
            out = os.path.join(args.folder, f"seed{s}")
            log = open(os.path.join(args.folder, f"log{s}.txt"), "w")
            p = subprocess.Popen(
                [sys.executable, "grok_rank.py", "--seed", str(s), "--out", out],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            running[s] = (p, log)
        for s, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                del running[s]
                done += 1
                status = "ok" if p.returncode == 0 else f"FAILED (see log{s}.txt)"
                mins = (time.time() - t0) / 60
                print(f"seed {s}: {status}  [{done} done, {len(todo)} left, "
                      f"{mins:.0f} min elapsed]")
        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopping: killing running seeds (they will be rerun next time).")
    for p, log in running.values():
        p.kill()
        log.close()
