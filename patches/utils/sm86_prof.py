# SPDX-License-Identifier: Apache-2.0
"""SM86 decode-path micro-profiler.

Env-gated (VLLM_SM86_PROFILE=1) CUDA-event timing for the pure-PyTorch
reference ops on the per-token decode path. Records start/end events per call
into per-label buckets; periodically synchronizes, folds elapsed time into
running totals, and prints an aggregate table. CUDA events are async (no host
sync at record time), so steady-state decode is not serialized by measurement.

Periodic dump (vs atexit) because the pyref runs inside TP worker processes
that exit on SIGTERM without firing atexit. Dump cadence is driven by the
trigger label (one call per sparse layer per token).
"""
import os

import torch

_ENABLED = os.environ.get("VLLM_SM86_PROFILE", "").strip() in ("1", "true", "True")
# Dump every N trigger-label calls. 41 sparse layers/token -> ~5 tokens.
_TRIGGER = "mla_decode.pv"
_EVERY = int(os.environ.get("VLLM_SM86_PROFILE_EVERY", "200"))

# Pending (start, end) event pairs per label, drained at each dump.
_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
# Running totals folded in at each dump.
_total_ms: dict[str, float] = {}
_total_n: dict[str, int] = {}
_trigger_count = 0


def enabled() -> bool:
    return _ENABLED


class _Span:
    __slots__ = ("label", "start")

    def __init__(self, label: str):
        self.label = label
        self.start = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        _events.setdefault(self.label, []).append((self.start, end))
        if self.label == _TRIGGER:
            global _trigger_count
            _trigger_count += 1
            if _trigger_count % _EVERY == 0:
                _dump()
        return False


class _Nop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NOP = _Nop()


def span(label: str):
    """Time a region under `label`. No-op unless VLLM_SM86_PROFILE=1."""
    if not _ENABLED:
        return _NOP
    return _Span(label)


def _dump():
    torch.cuda.synchronize()
    for label, pairs in _events.items():
        ms = sum(s.elapsed_time(e) for s, e in pairs)
        _total_ms[label] = _total_ms.get(label, 0.0) + ms
        _total_n[label] = _total_n.get(label, 0) + len(pairs)
    _events.clear()
    rows = sorted(_total_ms.items(), key=lambda kv: -kv[1])
    grand = sum(_total_ms.values()) or 1.0
    pid = os.getpid()
    lines = [f"\n=== SM86PROF pid{pid} trigger#{_trigger_count} ==="]
    lines.append(f"{'label':32s} {'total_ms':>11s} {'calls':>8s} {'us/call':>9s} {'%':>6s}")
    for label, total_ms in rows:
        n = _total_n[label]
        per = (total_ms / n * 1000) if n else 0.0
        lines.append(
            f"{label:32s} {total_ms:11.1f} {n:8d} {per:9.1f} {100*total_ms/grand:6.1f}"
        )
    lines.append(f"{'TOTAL':32s} {grand:11.1f} [SM86PROF pid{pid}]")
    print("\n".join(lines), flush=True)
