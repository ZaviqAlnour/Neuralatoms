"""
Tests for NeuralAtoms Dynamics Engine Module 2.

All tests use synthetic data and mocked MuJoCo — no GPU or real video needed.
MuJoCo itself IS imported where available (XML parsing test), otherwise mocked.
"""

from __future__ import annotations

import os
import sys
import csv
import tempfile
import shutil
from pathlib import Path
from unittest import mock
from typing import Dict

import numpy as np
import pytest

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dynamics.body_model import (
    BodyModel,
    DEMPSTER_MASS_FRACTIONS,
    SMPL_TO_SEGMENT,
    TRACKED_JOINT_INDICES,
)
from src.dynamics.mujoco_xml import build_mjcf, KINEMATIC_TREE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def default_body_model():
    return BodyModel.build(total_mass_kg=75.0, height_m=1.75)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="neuralatoms_dyn_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_atoms(tmp_dir) -> str:
    """Write a minimal synthetic .atoms.npz and return its path."""
    n = 60
    fps = 30.0
    t = np.arange(n) / fps

    data = {
        "pose_body":     np.random.randn(n, 23, 3).astype(np.float32) * 0.1,
        "pose_root":     np.zeros((n, 3), dtype=np.float32),
        "trans":         np.column_stack([
                             t * 0.5,
                             np.ones(n) * 0.9,
                             np.zeros(n),
                         ]).astype(np.float32),
        "betas":         np.zeros(10, dtype=np.float32),
        "velocity":      np.zeros((n, 3), dtype=np.float32),
        "acceleration":  np.zeros((n, 3), dtype=np.float32),
        "fps":           np.float32(fps),
        "n_frames":      np.int32(n),
        "frame_ids":     np.arange(n, dtype=np.int32),
        "smoothed":      np.bool_(False),
        "extraction_time_s": np.float32(1.0),
        "source_video":  np.array("test_walk"),
    }
    path = tmp_dir / "test_walk.atoms.npz"
    np.savez_compressed(str(path), **data)
    return str(path)


# ---------------------------------------------------------------------------
# 1. BodyModel — mass distribution
# ---------------------------------------------------------------------------
class TestBodyModel:
    def test_mass_fractions_sum_to_one(self):
        """Dempster fractions must sum to ~1.0 (enforced by auto-normalization)."""
        total = sum(DEMPSTER_MASS_FRACTIONS.values())
        # After auto-normalization in body_model.py, this must be exactly 1.0
        assert abs(total - 1.0) < 1e-6, f"Fractions sum to {total:.6f}"

    def test_75kg_total_mass(self):
        """Segment masses for 75 kg body must sum to 75 kg."""
        bm = BodyModel.build(75.0, 1.75)
        computed = bm.total_computed_mass()
        assert abs(computed - 75.0) < 0.5, f"Total mass = {computed:.3f} kg"

    def test_custom_mass_scales(self):
        """90 kg body should have proportionally heavier segments."""
        bm75 = BodyModel.build(75.0)
        bm90 = BodyModel.build(90.0)
        ratio = bm90.get("left_thigh").mass_kg / bm75.get("left_thigh").mass_kg
        assert abs(ratio - 90.0 / 75.0) < 0.01

    def test_all_segments_present(self, default_body_model):
        """All Dempster segments must exist in the built model."""
        for seg_name in DEMPSTER_MASS_FRACTIONS:
            assert seg_name in default_body_model.segments

    def test_inertia_positive(self, default_body_model):
        """All segment inertia diagonals must be positive."""
        for seg in default_body_model.segments.values():
            Ixx, Iyy, Izz = seg.inertia_cylinder()
            assert Ixx > 0
            assert Iyy > 0
            assert Izz > 0

    def test_smpl_joint_mapping_coverage(self):
        """All 24 SMPL joints (0-23) must have a segment mapping."""
        for idx in range(24):
            assert idx in SMPL_TO_SEGMENT, f"SMPL joint {idx} not mapped"

    def test_segment_for_smpl_joint(self, default_body_model):
        """segment_for_smpl_joint should return valid Segment objects."""
        seg = default_body_model.segment_for_smpl_joint(1)  # l_hip → left_thigh
        assert seg.name == "left_thigh"
        assert seg.mass_kg > 0


# ---------------------------------------------------------------------------
# 2. MuJoCo XML Builder
# ---------------------------------------------------------------------------
class TestMuJoCoXML:
    def test_xml_is_string(self, default_body_model):
        xml = build_mjcf(default_body_model)
        assert isinstance(xml, str)
        assert len(xml) > 500

    def test_xml_has_required_tags(self, default_body_model):
        xml = build_mjcf(default_body_model)
        assert "<mujoco" in xml
        assert "<worldbody>" in xml
        assert "freejoint" in xml
        assert "floor" in xml
        assert "pelvis" in xml

    def test_xml_has_kinematic_segments(self, default_body_model):
        xml = build_mjcf(default_body_model)
        for (seg_name, _, _, _, _) in KINEMATIC_TREE:
            if seg_name in default_body_model.segments:
                assert seg_name in xml, f"Segment '{seg_name}' missing from XML"

    def test_xml_parses_with_mujoco(self, default_body_model):
        """If MuJoCo is installed, verify XML compiles without error."""
        try:
            import mujoco as mj
            xml = build_mjcf(default_body_model)
            model = mj.MjModel.from_xml_string(xml)
            assert model.nq > 0
            assert model.nv > 0
            assert model.nbody > 1
        except ImportError:
            pytest.skip("MuJoCo not installed — skipping XML compile test")

    def test_xml_gravity_default(self, default_body_model):
        xml = build_mjcf(default_body_model, gravity=-9.81)
        assert "-9.8100" in xml or "-9.81" in xml

    def test_xml_hinge_limits(self, default_body_model):
        """Knee hinges should have joint limits encoded."""
        xml = build_mjcf(default_body_model)
        assert 'limited="true"' in xml


# ---------------------------------------------------------------------------
# 3. Kinematics loading
# ---------------------------------------------------------------------------
class TestLoadKinematics:
    def test_load_valid_atoms(self, synthetic_atoms):
        from src.dynamics.extract_dynamics import load_kinematics
        data = load_kinematics(synthetic_atoms)
        assert "pose_body" in data
        assert "trans" in data
        assert int(data["n_frames"]) == 60

    def test_load_missing_file(self):
        from src.dynamics.extract_dynamics import load_kinematics
        with pytest.raises(FileNotFoundError):
            load_kinematics("/nonexistent/path.atoms.npz")

    def test_load_incomplete_schema(self, tmp_dir):
        from src.dynamics.extract_dynamics import load_kinematics
        bad_path = tmp_dir / "bad.atoms.npz"
        np.savez(str(bad_path), only_this_key=np.zeros(3))
        with pytest.raises(ValueError, match="Invalid atoms file schema"):
            load_kinematics(str(bad_path))


# ---------------------------------------------------------------------------
# 4. axis_angle_to_quat
# ---------------------------------------------------------------------------
class TestAxisAngleToQuat:
    def test_zero_rotation(self):
        from src.dynamics.extract_dynamics import axis_angle_to_quat
        q = axis_angle_to_quat(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-8)

    def test_90_deg_z_rotation(self):
        from src.dynamics.extract_dynamics import axis_angle_to_quat
        aa = np.array([0.0, 0.0, np.pi / 2])  # 90° around Z
        q = axis_angle_to_quat(aa)
        assert q.shape == (4,)
        # w = cos(45°) ≈ 0.7071
        assert abs(q[0] - np.cos(np.pi / 4)) < 1e-6
        # Unit quaternion
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6

    def test_output_is_unit_quaternion(self):
        from src.dynamics.extract_dynamics import axis_angle_to_quat
        rng = np.random.default_rng(42)
        for _ in range(100):
            aa = rng.uniform(-np.pi, np.pi, size=3)
            q = axis_angle_to_quat(aa)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 5. compute_qvel_qacc
# ---------------------------------------------------------------------------
class TestComputeQvelQacc:
    def test_shapes(self):
        from src.dynamics.extract_dynamics import compute_qvel_qacc
        n, nq = 50, 20
        qpos = np.zeros((n, nq))
        qvel, qacc = compute_qvel_qacc(qpos, fps=30.0, smooth=False)
        assert qvel.shape == (n, nq)
        assert qacc.shape == (n, nq)

    def test_constant_qpos_gives_zero_derivatives(self):
        from src.dynamics.extract_dynamics import compute_qvel_qacc
        n, nq = 100, 15
        qpos = np.ones((n, nq)) * 0.5
        qvel, qacc = compute_qvel_qacc(qpos, fps=60.0, smooth=False)
        np.testing.assert_allclose(qvel, 0.0, atol=1e-10)
        np.testing.assert_allclose(qacc, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Inverse dynamics (mocked)
# ---------------------------------------------------------------------------
class TestInverseDynamicsMocked:
    """Test the inverse dynamics loop structure using MuJoCo mock objects."""

    def _make_mock_model(self, nq=45, nv=42, njnt=14):
        model = mock.MagicMock()
        model.nq = nq
        model.nv = nv
        model.njnt = njnt
        return model

    def _make_mock_data(self, nq=45, nv=42):
        data = mock.MagicMock()
        data.qpos = np.zeros(nq)
        data.qvel = np.zeros(nv)
        data.qacc = np.zeros(nv)
        data.qfrc_inverse = np.zeros(nv)
        return data

    def test_inverse_dynamics_output_shape(self):
        """Torque array shape must be (N, nv)."""
        from src.dynamics.extract_dynamics import run_inverse_dynamics

        n, nq, nv = 30, 45, 42
        model = self._make_mock_model(nq, nv)
        data  = self._make_mock_data(nq, nv)

        qpos = np.zeros((n, nq))
        qvel = np.zeros((n, nv))
        qacc = np.zeros((n, nv))

        _captured_qacc = []
        def fake_mj_inverse(m, d):
            d.qfrc_inverse = np.random.randn(nv) * 10
        
        with mock.patch("src.dynamics.extract_dynamics.mj") as mock_mj:
            mock_mj.mj_inverse.side_effect = lambda m, d: None
            mock_mj.mjtJoint = mock.MagicMock()
            # Make data.qfrc_inverse return a realistic array per call
            data.qfrc_inverse = np.array([5.0] * nv)
            torques = run_inverse_dynamics(model, data, qpos, qvel, qacc)

        assert torques.shape == (n, nv)

    def test_qpos_qvel_qacc_are_set_per_frame(self):
        """Verify the loop sets qpos/qvel/qacc on data for each frame."""
        from src.dynamics.extract_dynamics import run_inverse_dynamics

        n, nq, nv = 5, 10, 9
        model = self._make_mock_model(nq, nv)
        data  = self._make_mock_data(nq, nv)

        qpos = np.arange(n * nq, dtype=float).reshape(n, nq)
        qvel = np.zeros((n, nv))
        qacc = np.zeros((n, nv))

        assigned_qpos = []
        def capture_inverse(m, d):
            assigned_qpos.append(d.qpos.copy() if hasattr(d.qpos, 'copy') else list(d.qpos))

        with mock.patch("src.dynamics.extract_dynamics.mj") as mock_mj:
            mock_mj.mj_inverse.side_effect = capture_inverse
            data.qfrc_inverse = np.zeros(nv)
            run_inverse_dynamics(model, data, qpos, qvel, qacc)

        assert len(assigned_qpos) == n


# ---------------------------------------------------------------------------
# 7. Force profile CSV
# ---------------------------------------------------------------------------
class TestForceProfileCSV:
    def test_csv_column_schema(self, tmp_dir, synthetic_atoms, default_body_model):
        """CSV must have correct column headers for all tracked joints."""
        try:
            import mujoco as mj
        except ImportError:
            pytest.skip("MuJoCo not installed — skipping CSV test")

        from src.dynamics.mujoco_xml import build_mjcf
        from src.dynamics.extract_dynamics import (
            load_kinematics, kinematics_to_qpos,
            compute_qvel_qacc, run_inverse_dynamics,
            extract_joint_torques, save_dynamics_output,
        )

        atoms_data = load_kinematics(synthetic_atoms)
        xml = build_mjcf(default_body_model)
        model = mj.MjModel.from_xml_string(xml)
        data  = mj.MjData(model)

        n = int(atoms_data["n_frames"])
        torques = np.zeros((n, model.nv))
        joint_torques = np.zeros((n, model.njnt, 3))
        qpos = np.zeros((n, model.nq))
        qvel = np.zeros((n, model.nv))
        qacc = np.zeros((n, model.nv))

        npz_path, csv_path = save_dynamics_output(
            torques, joint_torques, qpos, qvel, qacc,
            atoms_data, model, tmp_dir, "test_walk", 75.0
        )

        assert csv_path.exists()

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)

        assert "frame" in header
        assert "time_s" in header
        # Check all 8 tracked joints have x/y/z columns
        for jlabel in ["l_ankle", "r_ankle", "l_knee", "r_knee",
                        "l_hip", "r_hip", "spine1", "spine2"]:
            for ax in ["x", "y", "z"]:
                col = f"{jlabel}_{ax}_Nm"
                assert col in header, f"Missing column: {col}"

        assert len(rows) == n


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
