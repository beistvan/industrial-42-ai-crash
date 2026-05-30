#!/usr/bin/env python3
"""Tiny Leonardo GPU sanity check for Slurm jobs."""
from __future__ import annotations

import json
import os
import platform
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - environment dependent
    print("ERROR: could not import torch", repr(exc), file=sys.stderr)
    raise SystemExit(2)

payload = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": getattr(torch.version, "cuda", None),
    "device_count": torch.cuda.device_count(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    "slurm_gpus": os.environ.get("SLURM_GPUS"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
if torch.cuda.is_available():
    payload["devices"] = [
        {
            "idx": i,
            "name": torch.cuda.get_device_name(i),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
            "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
        }
        for i in range(torch.cuda.device_count())
    ]
    x = torch.randn(512, 512, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    payload["matmul_checksum"] = round(float(y[0, 0].detach().cpu()), 6)
else:
    payload["devices"] = []

print(json.dumps(payload, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible. Submit through Slurm on boost_usr_prod with --gpus-per-task=1.")
