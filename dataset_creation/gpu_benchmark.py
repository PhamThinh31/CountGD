import torch
import GPUtil
import subprocess
import platform

def print_gpu_info():
    print("="*80)
    print("GPU HARDWARE INFORMATION")
    print("="*80)

    if torch.cuda.is_available():
        print(f"\nCUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n--- GPU {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")

            props = torch.cuda.get_device_properties(i)
            print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"Multi-Processor Count: {props.multi_processor_count}")

            torch.cuda.set_device(i)
            print(f"Current Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")
            print(f"Current Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

        gpus = GPUtil.getGPUs()
        if gpus:
            print("\n--- GPU Utilization ---")
            for gpu in gpus:
                print(f"GPU {gpu.id}: {gpu.name}")
                print(f"  Load: {gpu.load*100:.1f}%")
                print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")
                print(f"  Temperature: {gpu.temperature}°C")

    else:
        print("\nCUDA Available: No")
        print("Running on CPU")

    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Processor: {platform.processor()}")

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n" + "="*80)
            print("NVIDIA-SMI OUTPUT")
            print("="*80)
            print(result.stdout)
    except FileNotFoundError:
        print("\nnvidia-smi not found")

    print("="*80)


if __name__ == "__main__":
    print_gpu_info()
