"""
End-to-End Integration Test for NeuralAtoms Master Pipeline.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest import mock

# ROOT
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.atoms_io import load_atoms_npz


@pytest.fixture
def workspace():
    """Create a temporary workspace for the pipeline test."""
    with tempfile.TemporaryDirectory(prefix="neuralatoms_e2e_") as td:
        ws = Path(td)
        # Replicate project structure in temp
        for d in ["src", "vault/kinematics", "vault/dynamics", "vault/context", "vault/processed", "models"]:
            (ws / d).mkdir(parents=True)
        yield ws


def test_full_pipeline_synthetic(workspace, monkeypatch):
    """
    Test the pipeline by mocking the engine calls and verifying data fusion.
    """
    # 1. Create dummy input video
    dummy_video = workspace / "test_run.mp4"
    dummy_video.write_bytes(b"dummy_video_content")
    
    # 2. Mock individual module extraction paths
    # We want to verify that main_pipeline.py correctly identifies and merges files.
    
    kin_data = {"pose_body": np.zeros((10, 23, 3)), "fps": 30, "n_frames": 10, "trans": np.zeros((10, 3)), "pose_root": np.zeros((10, 3))}
    dyn_data = {"joint_torques": np.zeros((10, 24, 3))}
    ctx_data = {"surface_label": "concrete", "mu_static": 0.8}
    
    # Pre-populate vault/ with expected module outputs (as if subprocesses ran)
    np.savez_compressed(workspace / "vault/kinematics/test_run.atoms.npz", **kin_data)
    np.savez_compressed(workspace / "vault/dynamics/test_run.dynamics.npz", **dyn_data)
    np.savez_compressed(workspace / "vault/context/test_run.context.npz", **ctx_data)
    
    # 3. Patch subprocess.run so we don't actually search for WHAM/MuJoCo
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        
        # Override _ROOT and paths in main_pipeline to use workspace
        import main_pipeline
        monkeypatch.setattr(main_pipeline, "_ROOT", workspace)
        
        # Execute main_pipeline's logic
        with mock.patch("sys.argv", ["main_pipeline.py", "--video", str(dummy_video)]):
            main_pipeline.main()
            
        # Verify that subprocess was called with PI foundation upgrades (default is true)
        # We check the first call (kinematics) as an example
        args_passed = mock_run.call_args_list[0][0][0]
        assert "--video" in args_passed
            
    # 4. Verify Final Output
    final_path = workspace / "vault/processed/test_run.atoms.npz"
    assert final_path.exists(), "Final .atoms file was not created!"
    
    fused = load_atoms_npz(final_path)
    assert "pose_body" in fused
    assert "joint_torques" in fused
    assert "surface_label" in fused
    assert fused["surface_label"] == "concrete"
    assert "total_pipeline_time_s" in fused
    assert "2.0-PI" in str(fused["pipeline_version"])
    print("\nE2E Fusion Verification (PI Foundation 2.0): PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
