# Lesson: phases/00-setup-and-tooling/03-gpu-setup-and-cloud/docs/en.md
# Sources: PyTorch CUDA API docs and NVIDIA System Management Interface.
# This script is intentionally safe by default for CPU-only machines.
# Run with --benchmark to execute the heavier matrix multiplication demo.

import argparse
import time


def load_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def estimate_fp16_params(total_memory_bytes):
    return round((total_memory_bytes / 2) / 1e9, 1)


def bytes_to_gb(total_memory_bytes):
    return round(total_memory_bytes / 1e9, 1)


def collect_gpu_info(torch_module=None):
    if torch_module is None:
        return {
            "torch_installed": False,
            "cuda_available": False,
            "message": "PyTorch not installed. Run: pip install torch",
        }

    info = {
        "torch_installed": True,
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "cuda_available": bool(torch_module.cuda.is_available()),
    }

    if not info["cuda_available"]:
        info["message"] = (
            "No CUDA GPU detected. That is fine for most lessons; "
            "use Google Colab for GPU-heavy lessons."
        )
        return info

    props = torch_module.cuda.get_device_properties(0)
    total_memory = props.total_memory
    info.update(
        {
            "cuda_version": torch_module.version.cuda,
            "gpu_name": torch_module.cuda.get_device_name(0),
            "memory_gb": bytes_to_gb(total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
            "estimated_fp16_params_b": estimate_fp16_params(total_memory),
            "message": "CUDA GPU detected.",
        }
    )
    return info


def run_benchmark(torch_module, size):
    if torch_module is None or not torch_module.cuda.is_available():
        return {"ran": False, "message": "Benchmark skipped because CUDA is unavailable."}

    a_cpu = torch_module.randn(size, size)
    b_cpu = torch_module.randn(size, size)

    start = time.time()
    _ = a_cpu @ b_cpu
    cpu_time = time.time() - start

    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")
    torch_module.cuda.synchronize()

    start = time.time()
    _ = a_gpu @ b_gpu
    torch_module.cuda.synchronize()
    gpu_time = time.time() - start

    return {
        "ran": True,
        "size": size,
        "cpu_time": cpu_time,
        "gpu_time": gpu_time,
        "speedup": cpu_time / gpu_time if gpu_time else float("inf"),
    }


def print_gpu_info(info):
    print("=== GPU Check ===")

    if not info["torch_installed"]:
        print(info["message"])
        return

    print(f"PyTorch version: {info['torch_version']}")
    print(f"CUDA available: {info['cuda_available']}")

    if not info["cuda_available"]:
        print(info["message"])
        return

    print(f"CUDA version: {info['cuda_version']}")
    print(f"GPU: {info['gpu_name']}")
    print(f"Memory: {info['memory_gb']:.1f} GB")
    print(f"Compute capability: {info['compute_capability']}")
    print(f"Estimated max model size (fp16): ~{info['estimated_fp16_params_b']:.1f}B parameters")


def print_benchmark(result):
    print("\n=== CPU vs GPU Benchmark ===")
    if not result["ran"]:
        print(result["message"])
        return
    print(f"CPU matrix multiply ({result['size']}x{result['size']}): {result['cpu_time']:.3f}s")
    print(f"GPU matrix multiply ({result['size']}x{result['size']}): {result['gpu_time']:.3f}s")
    print(f"Speedup: {result['speedup']:.0f}x")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check CUDA GPU availability for the GPU setup lesson.")
    parser.add_argument("--benchmark", action="store_true", help="run the CPU vs GPU matrix benchmark")
    parser.add_argument("--size", type=int, default=1024, help="matrix size for --benchmark")
    args = parser.parse_args(argv)

    torch_module = load_torch()
    info = collect_gpu_info(torch_module)
    print_gpu_info(info)

    if args.benchmark:
        print_benchmark(run_benchmark(torch_module, args.size))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
