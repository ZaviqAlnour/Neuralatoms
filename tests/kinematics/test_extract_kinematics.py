"""
Tests for NeuralAtoms Kinematic Engine Module 1.

All tests use synthetic data — no GPU, WHAM, or video files required.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.kinematics.extract_kinematics import (
    verify_environment,
    compute_derivatives,
    check_floor_consistency,
    save_atoms_kinematics,
    SMPL_MODEL_FILENAME,
)
from src.utils.atoms_io import load_atoms_npz as load_atoms_kinematics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir():
    """Create and cleanup a temporary directory."""
    d = tempfile.mkdtemp(prefix="neuralatoms_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_atoms_data():
    """Generate synthetic kinematic data matching the .atoms.npz schema."""
    n_frames = 60
    fps = 30.0
    dt = 1.0 / fps

    # Linear motion along X at 1 m/s
    t = np.arange(n_frames) * dt
    trans = np.column_stack([t * 1.0, np.ones(n_frames) * 0.9, np.zeros(n_frames)])

    velocity, acceleration, smoothed = compute_derivatives(trans, fps)

    return {
        "pose_body": np.random.randn(n_frames, 23, 3).astype(np.float32),
        "pose_root": np.random.randn(n_frames, 3).astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": np.random.randn(10).astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "acceleration": acceleration.astype(np.float32),
        "fps": np.float32(fps),
        "n_frames": np.int32(n_frames),
        "frame_ids": np.arange(n_frames, dtype=np.int32),
        "smoothed": np.bool_(smoothed),
        "extraction_time_s": np.float32(1.5),
        "source_video": np.array("test_video"),
    }


# ---------------------------------------------------------------------------
# 1. Environment Verification
# ---------------------------------------------------------------------------
class TestVerifyEnvironment:
    def test_missing_model(self, tmp_dir):
        """Should return False when SMPL model file is absent."""
        result = verify_environment(models_dir=tmp_dir)
        assert result is False

    def test_present_model(self, tmp_dir):
        """Should pass SMPL check when model file exists (WHAM still missing)."""
        # Create a dummy model file
        model_path = tmp_dir / SMPL_MODEL_FILENAME
        model_path.write_bytes(b"dummy_model_data")

        # Still fails overall because WHAM won't be importable in test env,
        # but the SMPL-specific check should pass
        with mock.patch("src.kinematics.extract_kinematics._WHAM_AVAILABLE", True):
            result = verify_environment(models_dir=tmp_dir)
            assert result is True


# ---------------------------------------------------------------------------
# 2. Derivative Computation
# ---------------------------------------------------------------------------
class TestComputeDerivatives:
    def test_linear_motion(self):
        """
        Linear motion at 1 m/s along X:
          - velocity X ≈ 1.0 m/s (constant)
          - acceleration ≈ 0.0 m/s²
        """
        n = 100
        fps = 30.0
        dt = 1.0 / fps
        t = np.arange(n) * dt
        trans = np.column_stack([t * 1.0, np.zeros(n), np.zeros(n)])

        vel, acc, smoothed = compute_derivatives(trans, fps)

        assert vel.shape == (n, 3)
        assert acc.shape == (n, 3)
        assert smoothed is False

        # Interior velocity along X should be ~1.0 m/s
        np.testing.assert_allclose(vel[5:-5, 0], 1.0, atol=1e-6)
        # Acceleration should be near zero
        np.testing.assert_allclose(acc[5:-5, :], 0.0, atol=1e-4)

    def test_stationary(self):
        """Stationary subject → zero velocity and acceleration."""
        n = 50
        fps = 60.0
        trans = np.tile([0.0, 0.9, 0.0], (n, 1))

        vel, acc, smoothed = compute_derivatives(trans, fps)

        np.testing.assert_allclose(vel, 0.0, atol=1e-10)
        np.testing.assert_allclose(acc, 0.0, atol=1e-10)
        assert smoothed is False

    def test_spike_triggers_smoothing(self):
        """
        Inject an artificial spike in translation.
        Acceleration should exceed threshold → smoothing triggered.
        """
        n = 100
        fps = 30.0
        dt = 1.0 / fps
        t = np.arange(n) * dt
        trans = np.column_stack([t * 0.5, np.ones(n) * 0.9, np.zeros(n)])

        # Inject spike at frame 50
        trans[50, 0] += 5.0  # Massive displacement spike

        vel, acc, smoothed = compute_derivatives(trans, fps, accel_threshold=50.0)

        assert smoothed is True
        # Post-smoothing acceleration should be lower than the raw spike
        assert np.max(np.abs(acc)) < 5000.0  # Raw spike would be >> 5000

    def test_short_sequence(self):
        """Should handle very short sequences without crashing."""
        trans = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], dtype=np.float64)
        vel, acc, smoothed = compute_derivatives(trans, fps=30.0)

        assert vel.shape == (3, 3)
        assert acc.shape == (3, 3)


# ---------------------------------------------------------------------------
# 3. Floor Consistency
# ---------------------------------------------------------------------------
class TestFloorConsistency:
    def test_above_floor(self):
        """Trajectory above floor → should pass."""
        n = 50
        trans = np.column_stack([
            np.zeros(n),
            np.ones(n) * 0.9,  # Y = 0.9m (above floor)
            np.zeros(n)
        ])
        assert check_floor_consistency(trans, floor_threshold=-0.05) is True

    def test_below_floor(self):
        """Trajectory dipping below floor → should fail."""
        n = 50
        trans = np.column_stack([
            np.zeros(n),
            np.ones(n) * 0.9,
            np.zeros(n)
        ])
        trans[25, 1] = -0.1  # Below floor at frame 25

        assert check_floor_consistency(trans, floor_threshold=-0.05) is False

    def test_at_exact_threshold(self):
        """At exact threshold → should pass (not strictly below)."""
        n = 50
        trans = np.column_stack([
            np.zeros(n),
            np.ones(n) * (-0.05),  # Exactly at threshold
            np.zeros(n)
        ])
        assert check_floor_consistency(trans, floor_threshold=-0.05) is True


# ---------------------------------------------------------------------------
# 4. Output Schema (.atoms.npz)
# ---------------------------------------------------------------------------
class TestAtomsOutput:
    def test_save_and_load_roundtrip(self, tmp_dir, synthetic_atoms_data):
        """Save and load should produce identical data."""
        output_path = save_atoms_kinematics(synthetic_atoms_data, tmp_dir)

        assert output_path.exists()
        assert output_path.suffix == ".npz"
        assert ".atoms" in output_path.name

        loaded = load_atoms_kinematics(str(output_path))

        # Check all required keys present
        required_keys = [
            "pose_body", "pose_root", "trans", "betas",
            "velocity", "acceleration", "fps", "n_frames",
        ]
        for key in required_keys:
            assert key in loaded, f"Missing key: {key}"

        # Check shapes
        n = int(loaded["n_frames"])
        assert loaded["pose_body"].shape == (n, 23, 3)
        assert loaded["pose_root"].shape == (n, 3)
        assert loaded["trans"].shape == (n, 3)
        assert loaded["betas"].shape == (10,)
        assert loaded["velocity"].shape == (n, 3)
        assert loaded["acceleration"].shape == (n, 3)

    def test_data_integrity(self, tmp_dir, synthetic_atoms_data):
        """Verify saved values match original data."""
        output_path = save_atoms_kinematics(synthetic_atoms_data, tmp_dir)
        loaded = load_atoms_kinematics(str(output_path))

        np.testing.assert_array_almost_equal(
            loaded["trans"], synthetic_atoms_data["trans"]
        )
        np.testing.assert_array_almost_equal(
            loaded["velocity"], synthetic_atoms_data["velocity"]
        )

    def test_load_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_atoms_kinematics("/nonexistent/path/data.atoms.npz")

    def test_output_directory_creation(self, tmp_dir, synthetic_atoms_data):
        """Should create nested output directories if they don't exist."""
        nested_dir = tmp_dir / "nested" / "deep" / "kinematics"
        output_path = save_atoms_kinematics(synthetic_atoms_data, nested_dir)

        assert output_path.exists()
        assert nested_dir.exists()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
