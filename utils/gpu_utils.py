# cell 1
# utils/gpu_utils.py

import os
import torch

from config.config import (
    CUDA_LAUNCH_BLOCKING,
    TORCH_USE_CUDA_DSA,
    MIN_COMPUTE_CAPABILITY,
)


# ------------------------------------------------------------------
# CUDA ENVIRONMENT FLAGS
# ------------------------------------------------------------------

def set_cuda_environment_flags() -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = CUDA_LAUNCH_BLOCKING
    os.environ["TORCH_USE_CUDA_DSA"]   = TORCH_USE_CUDA_DSA
    print(f"CUDA env flags set | "
          f"CUDA_LAUNCH_BLOCKING={CUDA_LAUNCH_BLOCKING} | "
          f"TORCH_USE_CUDA_DSA={TORCH_USE_CUDA_DSA}")


# ------------------------------------------------------------------
# GPU DIAGNOSTICS
# ------------------------------------------------------------------

def print_gpu_diagnostics() -> dict:
    print("=" * 55)
    print("  GPU DIAGNOSTICS")
    print("=" * 55)
    print(f"PyTorch version     : {torch.__version__}")
    print(f"CUDA available      : {torch.cuda.is_available()}")

    diag = {
        "pytorch_version"       : torch.__version__,
        "cuda_available"        : torch.cuda.is_available(),
        "cuda_version"          : None,
        "device_name"           : None,
        "compute_capability"    : None,
        "compute_capability_int": None,
    }

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        cc  = cap[0] * 10 + cap[1]

        diag["cuda_version"]           = torch.version.cuda
        diag["device_name"]            = torch.cuda.get_device_name(0)
        diag["compute_capability"]     = f"{cap[0]}.{cap[1]}"
        diag["compute_capability_int"] = cc

        print(f"CUDA version        : {torch.version.cuda}")
        print(f"Device name         : {torch.cuda.get_device_name(0)}")
        print(f"Compute capability  : {cap[0]}.{cap[1]}")

        # Multi-GPU info
        n_gpus = torch.cuda.device_count()
        print(f"GPU count           : {n_gpus}")
        for i in range(n_gpus):
            mem = torch.cuda.get_device_properties(i).total_memory
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"| Memory: {mem / 1e9:.1f} GB")

        # Compatibility warning
        if cap[0] < 7:
            print(f"\n  WARNING: Compute Capability {cap[0]}.{cap[1]} detected!")
            print("   P100 = 6.0 → Some model kernels may not be compatible")
            print("   Applying compatibility fixes...\n")
    else:
        print("No CUDA GPU detected → CPU-only environment")

    print("=" * 55)
    return diag


# ------------------------------------------------------------------
# SAFE DEVICE DETECTION
# ------------------------------------------------------------------

def get_safe_device(min_compute_capability: int = MIN_COMPUTE_CAPABILITY) -> str:
    if not torch.cuda.is_available():
        print("  No GPU found → using CPU")
        return "cpu"

    cap = torch.cuda.get_device_capability(0)
    cc  = cap[0] * 10 + cap[1]

    print(f"  GPU Compute Capability: {cap[0]}.{cap[1]} (cc={cc})")

    if cc >= min_compute_capability:
        n_gpus = torch.cuda.device_count()
        print(f"  CUDA device is compatible → using GPU "
              f"({n_gpus} device(s) available)")
        return "cuda"
    else:
        print(f"  GPU cc={cc} < required cc={min_compute_capability} "
              f"→ falling back to CPU")
        return "cpu"


# ------------------------------------------------------------------
# MULTI-GPU UTILITY
# ------------------------------------------------------------------

def get_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def is_multi_gpu() -> bool:
    return get_device_count() > 1


# ------------------------------------------------------------------
# FULL SETUP ENTRY POINT
# ------------------------------------------------------------------

def setup_gpu_environment() -> str:
    set_cuda_environment_flags()
    print_gpu_diagnostics()
    device = get_safe_device()
    print(f"\n  Selected device: {device}")
    return device