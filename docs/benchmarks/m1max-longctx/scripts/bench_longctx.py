#!/usr/bin/env python3
"""Long-context A/B benchmark of MTPLX on Apple M1 Max (product HTTP path).

Each cell = (arm, context size, KV mode) runs in its own freshly started
`mtplx serve` process, so no RAM/SSD session cache or Metal allocator state is
shared between cells. Per cell the same streamed request is sent N times:

  request 1  cold prefill of the whole prompt (TTFT = prefill)
  request 2+ the prompt hits the RAM prefix cache; decode is re-measured on
             the same process (the generated text must be identical)

An arm is an MTPLX git commit checked out as a detached worktree, run from
source with PYTHONPATH, plus optional env overrides. Raw per-cell artifacts
(server log, /metrics rows, 5 s memory trace) go to --work-dir; a sanitized
summary row per cell is appended to --out (JSONL).

Usage:
  python scripts/bench_longctx.py --plan configs/plan.json \
      --mtplx-repo /path/to/MTPLX --model /path/to/model --python /path/to/python \
      --work-dir work --out results/raw-cells.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

QUESTION = (
    "\n\nThe records above are synthetic telemetry. In about 120 words, describe "
    "how the sensor values and status fields vary across the records, then list "
    "three concrete observations."
)
STATUSES = ("nominal", "degraded", "recovering", "offline", "calibrating")


def build_telemetry_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    """Deterministic synthetic records trimmed to ~target_tokens raw tokens."""

    lines = [
        f"Record {i:06d}: sensor={((i * 7919) % 997) / 10:.1f} "
        f"status={STATUSES[(i * 31) % len(STATUSES)]} "
        f"zone={chr(65 + (i * 13) % 26)}{(i * 17) % 90 + 10}.\n"
        for i in range(target_tokens // 20 + 16)
    ]
    budget = max(1, target_tokens - len(tokenizer.encode(QUESTION).ids))
    lo, hi = 1, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode("".join(lines[:mid])).ids) <= budget:
            lo = mid
        else:
            hi = mid - 1
    text = "".join(lines[:lo]) + QUESTION
    return text, len(tokenizer.encode(text).ids)


def get_json(url: str, timeout: float = 30.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def port_pids(port: int) -> list[int]:
    out = subprocess.run(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.strip().isdigit()]


def free_port(port: int) -> list[int]:
    """`mtplx serve` re-execs the server in its own session; kill by port."""
    killed = []
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGKILL, 20)):
        for pid in port_pids(port):
            try:
                os.kill(pid, sig)
                killed.append(pid)
            except ProcessLookupError:
                pass
        deadline = time.time() + wait
        while time.time() < deadline and port_pids(port):
            time.sleep(1)
    if port_pids(port):
        raise RuntimeError(f"port {port} still busy: {port_pids(port)}")
    return killed


def memory_sample(pid: int | None) -> dict:
    sample: dict = {"t": time.time()}
    try:
        page = 16384
        for line in subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
            for key, name in (("Pages free", "free"), ("Pages wired down", "wired"),
                              ("Pages occupied by compressor", "compressed")):
                if line.startswith(key):
                    sample[f"{name}_bytes"] = int(line.split(":")[1].strip().rstrip(".")) * page
        sample["swapusage"] = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                                             text=True, timeout=5).stdout.strip()
        if pid:
            rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            sample["server_rss_bytes"] = int(rss) * 1024 if rss else None
    except Exception as exc:  # telemetry must never kill the run
        sample["error"] = repr(exc)
    return sample


class MemoryTrace(threading.Thread):
    def __init__(self, path: Path, port: int, interval: float = 5.0):
        super().__init__(daemon=True)
        self.path, self.port, self.interval = path, port, interval
        self.stop_event = threading.Event()

    def run(self) -> None:
        with self.path.open("a") as fh:
            while not self.stop_event.is_set():
                pids = port_pids(self.port)
                fh.write(json.dumps(memory_sample(pids[0] if pids else None)) + "\n")
                fh.flush()
                self.stop_event.wait(self.interval)


def stream_request(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
    started = time.perf_counter()
    first_token_s = None
    parts: list[str] = []
    usage = finish_reason = error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                error = obj.get("error") or error
                usage = obj.get("usage") or usage
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = "".join(delta.get(k) or "" for k in ("content", "reasoning_content", "reasoning"))
                    if piece:
                        if first_token_s is None:
                            first_token_s = time.perf_counter() - started
                        parts.append(piece)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except Exception as exc:
        error = repr(exc)
    text = "".join(parts)
    return {"wall_s": time.perf_counter() - started, "ttft_s": first_token_s, "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "usage": usage,
            "finish_reason": finish_reason, "error": error}


METRIC_KEYS = (
    "prompt_tokens", "cached_tokens", "new_prefill_tokens", "completion_tokens", "ttft_s",
    "prefill_tok_s", "decode_elapsed_s", "decode_tok_s", "request_tok_s", "verify_calls",
    "accepted_drafts", "drafted_tokens", "accepted_by_depth", "peak_memory_bytes",
    "active_memory_bytes", "paged_kv_quant_mode", "finish_reason", "cache_source", "request_enable_thinking",
    "prompt_eval_time_s", "request_elapsed_s", "server_elapsed_s", "effective_max_tokens", "mtp_depth",
    "verify_time_s", "draft_time_s",
)


def summarize_request(req: dict, latest: dict | None) -> dict:
    latest = latest or {}
    row = {k: latest.get(k) for k in METRIC_KEYS}
    cv = latest.get("compiled_verify") or {}
    row["compiled_verify_calls"] = cv.get("compiled_calls") if isinstance(cv, dict) else None
    row["compiled_verify_fallbacks"] = cv.get("fallback_calls") if isinstance(cv, dict) else None
    row.update({"client_wall_s": req["wall_s"], "client_ttft_s": req["ttft_s"],
                "text_sha256": req["text_sha256"], "text_chars": len(req["text"]),
                "client_error": req["error"]})
    return row


def ensure_worktree(mtplx_repo: Path, commit: str, work_dir: Path) -> Path:
    wt = work_dir.resolve() / "worktrees" / commit
    if not wt.exists():
        subprocess.run(["git", "-C", str(mtplx_repo), "worktree", "add", "-q", "--detach", str(wt), commit],
                       check=True)
    head = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if not head.startswith(commit):
        raise RuntimeError(f"worktree {wt} is at {head}, expected {commit}")
    return wt


def run_cell(args, plan: dict, cell: dict, tokenizer) -> dict:
    arm = plan["arms"][cell["arm"]]
    ctx = plan["contexts"][cell["context"]]
    name = f"{cell['context']}-{cell['kv']}-{cell['arm']}"
    cell_dir = Path(args.work_dir) / "cells" / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    wt = ensure_worktree(Path(args.mtplx_repo), arm["commit"], Path(args.work_dir))

    if ctx["prompt"] == "telemetry":
        prompt, raw_tokens = build_telemetry_prompt(tokenizer, ctx["target_tokens"])
    else:
        prompt = (ROOT / ctx["prompt"]).read_text().strip()
        raw_tokens = len(tokenizer.encode(prompt).ids)
    max_tokens = ctx.get("max_tokens", plan["max_tokens"])
    n_requests = ctx.get("requests", plan["requests"])

    env = dict(os.environ)
    env.update({"PYTHONPATH": str(wt), "PYTHONUNBUFFERED": "1",
                "MTPLX_REQUEST_LOG_JSONL": str(cell_dir / "request-log.jsonl")})
    env.update(arm.get("env") or {})
    cmd = [args.python, "-m", "mtplx.cli", "serve", "--model", args.model, "--profile", "turbo",
           "--depth", "3", "--host", "127.0.0.1", "--port", str(args.port), "--no-auth", "--yes",
           "--warmup-tokens", "0", "--ssd-session-cache", "off", "--context-window", "262144",
           "--max-tokens", str(max_tokens), "--paged-kv-quantization", cell["kv"]]

    def log(event: dict) -> None:
        print(json.dumps({"cell": name, "t": round(time.time(), 1), **event}), flush=True)

    free_port(args.port)
    log({"event": "start", "raw_prompt_tokens": raw_tokens, "commit": arm["commit"]})
    thermal = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    log_fh = (cell_dir / "server.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    trace = MemoryTrace(cell_dir / "memory-trace.jsonl", args.port)
    trace.start()
    base = f"http://127.0.0.1:{args.port}"
    row: dict = {"cell": name, "arm": cell["arm"], "commit": arm["commit"], "context": cell["context"],
                 "kv": cell["kv"], "target_tokens": ctx.get("target_tokens"), "raw_prompt_tokens": raw_tokens,
                 "max_tokens": max_tokens, "env": arm.get("env") or {}, "started_at": time.time(),
                 "thermal_before": thermal.strip(), "power_before": power.strip().splitlines()[0] if power.strip() else "",
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "requests": []}
    try:
        load_started = time.time()
        ready = False
        while time.time() - load_started < 900 and proc.poll() is None:
            try:
                h = get_json(f"{base}/health", timeout=5)
                if h.get("ok") is True or h.get("status") in ("ok", "ready"):
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(3)
        if not ready:
            row["error"] = f"server not ready (exit={proc.poll()})"
            return row
        health = get_json(f"{base}/health", timeout=30)
        (cell_dir / "health.json").write_text(json.dumps(health, indent=1))
        row["load_s"] = time.time() - load_started
        row["mtplx_version"] = health.get("version") or health.get("mtplx_version")
        row["prefill_chunk_tokens"] = ((health.get("scheduler") or {}).get("config") or {}).get("prefill_chunk_tokens")
        row["mlx_version"] = (health.get("mlx_runtime") or {}).get("version")
        log({"event": "ready", "load_s": round(row["load_s"], 1)})
        payload = {
            "model": health.get("model_id") or health.get("model") or "mtplx",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 123,
            "enable_thinking": False, "stream": True, "stream_options": {"include_usage": True},
        }
        for i in range(n_requests):
            req = stream_request(f"{base}/v1/chat/completions", payload, ctx["timeout_s"])
            latest = None
            try:
                metrics = get_json(f"{base}/metrics", timeout=60)
                (cell_dir / f"metrics-r{i + 1}.json").write_text(json.dumps(metrics, indent=1))
                latest = metrics.get("latest")
            except Exception as exc:
                row["metrics_error"] = repr(exc)
            if i == 0:
                (cell_dir / "output-r1.txt").write_text(req["text"])
            r = summarize_request(req, latest)
            row["requests"].append(r)
            log({"event": "request_done", "rep": i + 1, "ttft_s": r["ttft_s"], "decode_tok_s": r["decode_tok_s"],
                 "prompt_tokens": r["prompt_tokens"], "cached": r["cached_tokens"],
                 "completion": r["completion_tokens"], "sha": r["text_sha256"][:10], "err": r["client_error"]})
            if req["error"]:
                break
    finally:
        row["server_exit_before_kill"] = proc.poll()
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            except Exception:
                pass
        free_port(args.port)
        trace.stop_event.set()
        trace.join(timeout=10)
        log_fh.close()
        wired = swap = 0
        for line in (cell_dir / "memory-trace.jsonl").read_text().splitlines():
            s = json.loads(line)
            wired = max(wired, s.get("wired_bytes") or 0)
            try:
                swap = max(swap, float(s.get("swapusage", "").split("used = ")[1].split("M")[0]))
            except (IndexError, ValueError):
                pass
        row["max_wired_bytes"] = wired
        row["max_swap_used_mb"] = swap
        row["finished_at"] = time.time()
        (cell_dir / "row.json").write_text(json.dumps(row, indent=1))
        log({"event": "cell_done", "server_exit": proc.poll()})
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default=str(ROOT / "configs" / "plan.json"))
    p.add_argument("--mtplx-repo", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--python", required=True, help="Python with MLX and MTPLX dependencies")
    p.add_argument("--work-dir", default=str(ROOT / "work"))
    p.add_argument("--out", default=str(ROOT / "results" / "raw-cells.jsonl"))
    p.add_argument("--port", type=int, default=18091)
    p.add_argument("--only", nargs="*", help="run only these cell names (context-kv-arm)")
    p.add_argument("--skip-done", action="store_true", help="skip cells already present in --out")
    args = p.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.skip_done and out.exists():
        done = {json.loads(l)["cell"] for l in out.read_text().splitlines() if l.strip()}
    for cell in plan["order"]:
        name = f"{cell['context']}-{cell['kv']}-{cell['arm']}"
        if (args.only and name not in args.only) or name in done:
            continue
        row = run_cell(args, plan, cell, tokenizer)
        row["host"] = {"machine": platform.machine(), "macos": platform.mac_ver()[0]}
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        time.sleep(plan.get("cooldown_s", 60))


if __name__ == "__main__":
    main()
