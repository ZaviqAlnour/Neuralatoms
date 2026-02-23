import os
import time
import torch
import numpy as np
import psutil
from pathlib import Path

# Constants for benchmarking
BENCHMARK_FILE = Path("vault/disk_test.tmp")
DATA_SIZE_MB = 100  # 100MB test
FRAME_RATE_TARGET = 120  # Hz
BYTES_PER_FRAME_EST = 1024 * 50  # 50KB per frame (conservative for high-freq joint data)

def check_cuda():
    print("--- GPU Check (T4) ---")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA status: ACTIVE")
        print(f"Device: {gpu_name}")
        print(f"Total VRAM: {gpu_mem:.2f} GB")
    else:
        print("CUDA status: NOT FOUND. Ensure you are connected to the T4 GPU runtime.")

def check_disk_speed():
    print("\n--- NVMe Write Speed Check ---")
    data = np.random.bytes(DATA_SIZE_MB * 1024 * 1024)
    
    start_time = time.time()
    with open(BENCHMARK_FILE, "wb") as f:
        f.write(data)
        os.fsync(f.fileno())
    end_time = time.time()
    
    duration = end_time - start_time
    write_speed = DATA_SIZE_MB / duration
    
    # Calculate required speed for 120Hz streaming
    required_mbps = (FRAME_RATE_TARGET * BYTES_PER_FRAME_EST) / (1024 * 1024)
    
    print(f"Write Speed: {write_speed:.2f} MB/s")
    print(f"Target streaming requirement (120Hz @ 50KB/frame): {required_mbps:.2f} MB/s")
    
    if write_speed > required_mbps:
        print("Latency Status: PASS (SSD can handle high-frequency stream)")
    else:
        print("Latency Status: WARNING (Potential bottleneck for high-frequency streaming)")
        
    if BENCHMARK_FILE.exists():
        BENCHMARK_FILE.unlink()

def configure_mujoco():
    print("\n--- MuJoCo Environment Configuration ---")
    models_dir = Path("models").absolute()
    os.environ["MUJOCO_DIR"] = str(models_dir)
    # Add models dir to search path logic
    print(f"MuJoCo Path set to: {models_dir}")
    print("Optimization: Pointing models/ as the local synapse repository.")

def main():
    print("NEURALATOMS SYSTEM INITIALIZATION CHECK")
    print("=======================================")
    check_cuda()
    check_disk_speed()
    configure_mujoco()
    print("=======================================")
    print("System Check Complete.")

if __name__ == "__main__":
    main()
