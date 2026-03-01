import sys
import psutil
import torch

def verify_environment():
    print("=== System Environment Verification ===")
    
    # 1. Python & PyTorch Versions
    print(f"\n[1] Framework Status")
    print(f"    Python Version : {sys.version.split(' ')[0]}")
    print(f"    PyTorch Version: {torch.__version__}")
    
    # 2. CUDA & GPU Verification
    print("\n[2] GPU Computing Status")
    cuda_available = torch.cuda.is_available()
    print(f"    CUDA Available : {cuda_available}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"    Device Name    : {gpu_name}")
        print(f"    Total VRAM     : {vram_gb:.2f} GB")
    else:
        print("    WARNING: CUDA is not available. PyTorch will fallback to CPU.")
        print("    CRITICAL: Training deep learning models on Ryzen CPU will be extremely slow.")
        print("    Action: Reinstall PyTorch with the correct CUDA index-url.")

    # 3. System RAM Verification (Crucial for CICIoT2023)
    print("\n[3] System Memory (RAM) Status")
    vm = psutil.virtual_memory()
    total_ram = vm.total / (1024**3)
    available_ram = vm.available / (1024**3)
    
    print(f"    Total RAM      : {total_ram:.2f} GB")
    print(f"    Available RAM  : {available_ram:.2f} GB")
    print(f"    Usage          : {vm.percent}%")
    
    # Logic untuk antisipasi Out of Memory (OOM)
    if available_ram < 8.0:
        print("\n[!] BOTTLENECK WARNING: You have less than 8GB of free RAM.")
        print("    Loading CICIoT2023 chunks right now carries a high risk of Out of Memory (OOM) crashes.")
        print("    Action: Close heavy background apps (browsers, Discord, games) before running data ingestion.")
    else:
        print("\n[+] RAM availability is acceptable.")
        print("    Note: Strict chunking (chunksize parameter) is still mandatory for CICIoT2023 ingestion.")

if __name__ == "__main__":
    verify_environment()