import sys
import os
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))
from atoms_io import AtomsSchema, save_atoms

def stress_test(duration_sec: int = 5, target_fps: int = 120):
    print(f"STRESS TEST: Simulating {duration_sec}s capture at {target_fps}Hz")
    print("---------------------------------------------------------")
    
    header = AtomsSchema.create_header("stress_test_subject", capture_fps=target_fps)
    temp_file = Path("vault/stress_test.atoms")
    
    total_frames = duration_sec * target_fps
    frame_interval = 1.0 / target_fps
    
    frames = []
    
    write_times = []
    
    print(f"Generating {total_frames} frames of synthetic joint data...")
    # Simulate data generation (72 joint values + 72 rotation values)
    for i in range(total_frames):
        timestamp = i * frame_interval
        # Random noise to simulate real-time joint data
        joints = np.random.uniform(-1, 1, 72).tolist()
        rotations = np.random.uniform(-1, 1, 72).tolist()
        frames.append(AtomsSchema.create_frame(timestamp, joints, rotations))
        
    print(f"Starting Pipe Stream to NVMe SSD: {temp_file}...")
    
    start_time = time.time()
    save_atoms(str(temp_file), header, frames, compress=True)
    end_time = time.time()
    
    total_time = end_time - start_time
    file_size_mb = temp_file.stat().st_size / (1024 * 1024)
    
    print("\nPROFILER RESULTS:")
    print(f"Total time to write {total_frames} frames: {total_time:.4f} seconds")
    print(f"Average time per frame write: {(total_time/total_frames)*1000:.4f} ms")
    print(f"File Size (Compressed): {file_size_mb:.2f} MB")
    print(f"Effective throughput: {file_size_mb / total_time:.2f} MB/s")
    
    # Requirement Check: 120Hz means 1 frame every 8.33ms. 
    # Our batch write should be efficient enough that it doesn't block the next frame capture.
    # In a real streaming scenario, we'd write in chunks, but for this test, we verify bulk capability.
    
    if total_time < duration_sec:
        print("\nSTATUS: PASS - Disk IO is faster than real-time capture.")
    else:
        print("\nSTATUS: WARNING - Disk IO is slower than real-time capture. Optimization required.")

    if temp_file.exists():
        temp_file.unlink()

if __name__ == "__main__":
    stress_test()
