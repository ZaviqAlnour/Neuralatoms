import json
import os
import time
import gzip
from typing import List, Dict, Any, Optional
from pathlib import Path

class AtomsSchema:
    VERSION = "1.0"
    
    @staticmethod
    def create_header(subject_id: str, capture_fps: int = 120) -> Dict[str, Any]:
        return {
            "version": AtomsSchema.VERSION,
            "subject_id": subject_id,
            "timestamp_start": time.time(),
            "capture_fps": capture_fps,
            "units": "meters/radians/seconds",
            "metadata": {}
        }

    @staticmethod
    def create_frame(timestamp: float, joints_3d: List[float], pose_rot: List[float]) -> Dict[str, Any]:
        """
        joints_3d: Flattened list of XYZ coordinates (e.g., 24 joints * 3 = 72 values)
        pose_rot: Flattened list of rotation vectors/matrices
        """
        return {
            "t": timestamp,
            "pos": joints_3d,
            "rot": pose_rot,
            # Placeholder for physics vectors (Phase 3)
            "physics": {
                "torque": [], 
                "friction": []
            }
        }

def save_atoms(filepath: str, header: Dict[str, Any], frames: List[Dict[str, Any]], compress: bool = True):
    """
    Saves the data as a .atoms file (compressed JSON).
    """
    data = {
        "header": header,
        "frames": frames
    }
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        with gzip.open(filepath, 'wt', encoding='UTF-8') as f:
            json.dump(data, f)
    else:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

def load_atoms(filepath: str) -> Dict[str, Any]:
    """
    Loads a .atoms file.
    """
    if filepath.endswith('.atoms') or filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='UTF-8') as f:
            return json.load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)

if __name__ == "__main__":
    # Internal test
    header = AtomsSchema.create_header("test_subject_001")
    frames = [AtomsSchema.create_frame(0.0, [0]*72, [0]*72)]
    
    test_path = "vault/test_capture.atoms"
    save_atoms(test_path, header, frames)
    print(f"Saved test .atoms file to {test_path}")
    
    loaded = load_atoms(test_path)
    print(f"Loaded schema version: {loaded['header']['version']}")
