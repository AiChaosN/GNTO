"""
Run pure BAO (BaoForPostgreSQL, no LIMAO lifelong learning) on the same
workload LIMAO was tested on (``LIMAOLifeLongRLDB/imdb_assorted_3/``, 112 JOB
queries) and write a log compatible with ``0519_extract_l3_endtoend.py``.

Prereqs (user must do these BEFORE invoking this script):
  1. Stop the system PostgreSQL so port 5432 is free:
       sudo systemctl stop postgresql
  2. Start the custom PG 12.5 with pg_bao preloaded:
       /home/AiChaosN/Project/Phd/project/postgresql-12.5/bin/pg_ctl \\
           -D /home/AiChaosN/Project/Phd/project/databases \\
           -l /tmp/pg12.log start

This script then:
  - Spawns the **vanilla BAO** bao_server (not LIMAO's) on port 9381.
  - Connects to localhost:5432 / imdbload.
  - Runs N iterations over the 112 queries: each iteration is
      (a) measure PG default time for each query (bao_select=False, bao_reward=True)
      (b) run with BAO steering (bao_select=True, bao_reward=True)
    plus a model retrain (``baoctl.py --retrain``) after each iteration.
  - Writes log lines compatible with ``0519_extract_l3_endtoend.py``:
      x x <q.sql> <t> PG
      BAO <q.sql> <t>

Usage:
  python experiments/steering/run_pure_bao.py --iterations 4 --output results/PureBao_<ts>/
"""

import os
import sys
import time
import argparse
import subprocess
import signal
from datetime import datetime
from pathlib import Path

import psycopg2

BAO_REPO = "/home/AiChaosN/Project/Workspace/01_Research/BaoForPostgreSQL"
QUERIES_DIR = "/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/imdb_assorted_3"
PG_DSN = "dbname=imdbload user=AiChaosN host=localhost port=5432"
NUM_ARMS = 49  # match LIMAO's setup
STATEMENT_TIMEOUT_MS = 32000  # match the 32-second cap observed in LIMAO logs


def run_query(sql, bao_select=False, bao_reward=False, num_arms=NUM_ARMS,
              timeout_ms=STATEMENT_TIMEOUT_MS, retries=3):
    """Execute one SQL with the requested Bao toggles; return elapsed seconds."""
    start = time.perf_counter()
    last_exc = None
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(PG_DSN)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
            cur.execute(f"SET enable_bao_selection TO {bao_select}")
            cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
            cur.execute(f"SET bao_num_arms TO {num_arms}")
            cur.execute(f"SET statement_timeout TO {timeout_ms}")
            try:
                cur.execute(sql)
                _ = cur.fetchall()
            except psycopg2.errors.QueryCanceled:
                # statement_timeout fired -- pin the time at the cap
                conn.close()
                return timeout_ms / 1000.0
            conn.close()
            return time.perf_counter() - start
        except Exception as e:
            last_exc = e
            time.sleep(1)
    raise RuntimeError(f"run_query failed after {retries} retries: {last_exc}")


def _cuda_env():
    """Subprocess env with libcuda.so reachable under WSL2."""
    env = os.environ.copy()
    extra = "/usr/lib/wsl/lib"
    cur = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = extra + (":" + cur if cur else "")
    return env


def start_bao_server(log_path):
    """Spawn the vanilla BAO bao_server. Returns subprocess.Popen handle."""
    server_dir = os.path.join(BAO_REPO, "bao_server")
    f = open(log_path, "w")
    proc = subprocess.Popen(
        ["python3", "main.py"],
        cwd=server_dir,
        stdout=f, stderr=subprocess.STDOUT,
        env=_cuda_env(),
    )
    # wait until it's actually listening
    for _ in range(20):
        time.sleep(0.5)
        with open(log_path) as r:
            if "Listening on localhost port" in r.read():
                print(f"[server] up (pid {proc.pid}); log -> {log_path}")
                return proc
    proc.terminate()
    raise RuntimeError("bao_server didn't start in 10s")


def baoctl_retrain():
    server_dir = os.path.join(BAO_REPO, "bao_server")
    res = subprocess.run(["python3", "baoctl.py", "--retrain"], cwd=server_dir,
                         capture_output=True, text=True, timeout=600,
                         env=_cuda_env())
    return res.returncode, res.stdout, res.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=4,
                    help="number of (PG warmup + BAO steered) passes (LIMAO ran 4 before stopping)")
    ap.add_argument("--queries-dir", default=QUERIES_DIR)
    ap.add_argument("--retrain-every", type=int, default=0,
                    help="retrain Bao every N queries within the BAO-steered phase. "
                         "0 (default) = retrain once per outer iteration (4 retrains total in 4-iter). "
                         "Set to ~10 to match LIMAO's per-partition retrain cadence (~40 retrains).")
    ap.add_argument("--model-type", choices=["BAO", "GNTO"], default=None,
                    help="If set, write this ModelType into BaoForPostgreSQL/bao_server/bao.cfg "
                         "before spawning the server (restored on exit). When omitted, uses whatever "
                         "is currently in bao.cfg.")
    ap.add_argument("--output", default=None,
                    help="output dir (default: GNTO/results/PureBao_<ts>)")
    args = ap.parse_args()

    gnto_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if args.output is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        tag = f"PureBao_{args.model_type}_{ts}" if args.model_type else f"PureBao_{ts}"
        args.output = os.path.join(gnto_root, "results", tag)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output}")

    # Optionally flip bao.cfg's ModelType, restore at end.
    cfg_path = os.path.join(BAO_REPO, "bao_server", "bao.cfg")
    saved_cfg = None
    if args.model_type:
        with open(cfg_path) as f:
            saved_cfg = f.read()
        new_cfg = []
        for line in saved_cfg.splitlines():
            if line.startswith("ModelType"):
                new_cfg.append(f"ModelType = {args.model_type}")
            else:
                new_cfg.append(line)
        with open(cfg_path, "w") as f:
            f.write("\n".join(new_cfg) + "\n")
        print(f"[cfg] ModelType -> {args.model_type}")

    queries = sorted(Path(args.queries_dir).glob("*.sql"))
    print(f"Loaded {len(queries)} queries from {args.queries_dir}")
    if not queries:
        sys.exit("No .sql files found.")

    # Smoke-check the PG connection before we spawn the server
    try:
        conn = psycopg2.connect(PG_DSN)
        conn.close()
    except Exception as e:
        sys.exit(f"Cannot connect to PG via '{PG_DSN}': {e}\n"
                 f"Did you start PG 12.5 (`pg_ctl -D ../databases start`) and stop the system PG?")

    server_log = os.path.join(args.output, "bao_server.log")
    run_log_path = os.path.join(args.output, "bao_run.log")
    server = start_bao_server(server_log)
    run_log = open(run_log_path, "w", buffering=1)

    try:
        for it in range(args.iterations):
            print(f"\n=== Iteration {it+1}/{args.iterations} ===")

            # Phase A: PG default measurements (also feeds rewards into Bao)
            for qpath in queries:
                sql = qpath.read_text()
                t = run_query(sql, bao_select=False, bao_reward=True)
                line = f"x x {qpath.name} {t} PG"
                print(line); run_log.write(line + "\n")

            # Phase B: retrain Bao on collected rewards
            print("  [retrain] ...")
            rc, out, err = baoctl_retrain()
            print(f"  [retrain] rc={rc}")
            if rc != 0:
                print("  stderr:", err[-500:])

            # Phase C: BAO-steered runs, optionally retraining every N queries.
            retrain_every = args.retrain_every
            for q_idx, qpath in enumerate(queries):
                if retrain_every > 0 and q_idx > 0 and q_idx % retrain_every == 0:
                    print(f"  [retrain] mid-iter at q_idx={q_idx} ...")
                    rc, out, err = baoctl_retrain()
                    print(f"  [retrain] rc={rc}")
                    if rc != 0:
                        print("  stderr:", err[-500:])
                sql = qpath.read_text()
                t = run_query(sql, bao_select=True, bao_reward=True)
                line = f"BAO {qpath.name} {t}"
                print(line); run_log.write(line + "\n")

    finally:
        run_log.close()
        print(f"\n[server] terminating pid {server.pid}")
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        if saved_cfg is not None:
            with open(cfg_path, "w") as f:
                f.write(saved_cfg)
            print("[cfg] restored bao.cfg")
        print(f"\nDONE. log: {run_log_path}")
        print(f"Feed it to: python examples/0519_extract_l3_endtoend.py \\")
        print(f"             --bao-log {run_log_path} \\")
        print(f"             --gnto-log .../gnto_log \\")
        print(f"             --out-dir results/Summary_0519_purebao/")


if __name__ == "__main__":
    main()
