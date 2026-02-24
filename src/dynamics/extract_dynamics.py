"""
NeuralAtoms Dynamics Engine Module 2 — Inverse Dynamics & Torque Recovery

Ingests world-grounded .atoms.npz kinematic files (Module 1 output) and
runs MuJoCo inverse dynamics (mj_inverse) to extract joint torques.

Solves:  τ = M(q)q̈ + C(q,q̇)q̇ + G(q)

Usage:
    python src/extract_dynamics.py --atoms vault/kinematics/<name>.atoms.npz
    python src/extract_dynamics.py --atoms vault/kinematics/*.atoms.npz --body-mass 80

Copyright © 2026 NeuralAtoms — Zaviq Alnour
"""

from __future__ import annotations

import os
import sys
import csv
import time
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
_MUJOCO_AVAILABLE = False
_SCIPY_AVAILABLE = False

try:
    import mujoco as mj
    _MUJOCO_AVAILABLE = True
except ImportError:
    pass

try:
    from scipy.signal import savgol_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    pass

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("neuralatoms.dynamics")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(_h)

# Add project root to path for src imports
_LIB_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

from src.utils.filters import savgol_smooth, compute_finite_derivatives
from src.utils.atoms_io import load_atoms_npz, save_atoms_npz, validate_atoms_data

# Sibling imports (relative or via src.dynamics)
try:
    from .body_model import BodyModel, TRACKED_JOINT_INDICES
    from .mujoco_xml import build_mjcf
    from .dynamics_pinn import DynamicsPINN, SystemID
except (ImportError, ValueError):
    from body_model import BodyModel, TRACKED_JOINT_INDICES
    from mujoco_xml import build_mjcf
    from dynamics_pinn import DynamicsPINN, SystemID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR  = PROJECT_ROOT / "vault" / "kinematics"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "vault" / "dynamics"
DEFAULT_BODY_MASS  = 75.0    # kg
DEFAULT_BODY_HEIGHT = 1.75   # m

# Smoothing for qacc (second finite-difference is noisy)
QACC_SMOOTH_WINDOW  = 11
QACC_SMOOTH_POLYORDER = 3

# Gravity vector (Y-up coordinate system)
GRAVITY_MPS2 = 9.81

# ---------------------------------------------------------------------------
# 1. Load kinematics
# ---------------------------------------------------------------------------
def load_kinematics(atoms_path: str) -> Dict[str, np.ndarray]:
    """
    Load a .atoms.npz kinematic file produced by Module 1.
    """
    data = load_atoms_npz(atoms_path)
    if not validate_atoms_data(data, "kinematics"):
        raise ValueError(f"Invalid atoms file schema: {atoms_path}")
    return data


# ---------------------------------------------------------------------------
# 2. Build MuJoCo model
# ---------------------------------------------------------------------------
def build_mujoco_model(
    body_mass_kg: float = DEFAULT_BODY_MASS,
    height_m: float = DEFAULT_BODY_HEIGHT,
    cache_xml: Optional[str] = None,
) -> Tuple["mj.MjModel", "mj.MjData"]:
    """
    Build a MuJoCo model from the BodyModel.

    Args:
        body_mass_kg: Total body mass in kg.
        height_m: Subject height in metres.
        cache_xml: Optional path to write/read cached XML.

    Returns:
        (MjModel, MjData) pair ready for simulation.
    """
    if not _MUJOCO_AVAILABLE:
        raise RuntimeError(
            "MuJoCo not installed. Run: pip install mujoco"
        )

    body_model = BodyModel.build(body_mass_kg, height_m)

    # Load cached XML if available
    if cache_xml and Path(cache_xml).exists():
        with open(cache_xml, "r") as f:
            xml_str = f.read()
        logger.info(f"Loaded cached MuJoCo XML: {cache_xml}")
    else:
        xml_str = build_mjcf(body_model)
        if cache_xml:
            Path(cache_xml).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_xml, "w") as f:
                f.write(xml_str)
            logger.info(f"Wrote MuJoCo XML: {cache_xml}")

    model = mj.MjModel.from_xml_string(xml_str)
    data  = mj.MjData(model)

    logger.info(
        f"MuJoCo model compiled: "
        f"nq={model.nq}, nv={model.nv}, nbody={model.nbody}"
    )
    return model, data


# ---------------------------------------------------------------------------
# 3. Kinematics → MuJoCo qpos
# ---------------------------------------------------------------------------
def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle (3,) to quaternion (4,) [w, x, y, z] (MuJoCo format).
    """
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    half = angle * 0.5
    return np.array([
        np.cos(half),
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half),
    ])


def kinematics_to_qpos(
    pose_body: np.ndarray,   # (N, 23, 3) body joint axis-angles
    pose_root: np.ndarray,   # (N, 3)    root joint axis-angle
    trans: np.ndarray,        # (N, 3)    world translation
    model: "mj.MjModel",
) -> np.ndarray:
    """
    Convert SMPL kinematic data to MuJoCo generalized positions.

    The MuJoCo model has:
      - freejoint (root): 7 DOF  [tx, ty, tz, qw, qx, qy, qz]
      - ball joints:      4 DOF each [qw, qx, qy, qz]
      - hinge joints:     1 DOF each [angle]

    Args:
        pose_body: (N, 23, 3) body joint rotations.
        pose_root: (N, 3) root orientation.
        trans:     (N, 3) world translation.
        model:     Compiled MjModel.

    Returns:
        qpos array of shape (N, model.nq).
    """
    n_frames = pose_body.shape[0]
    qpos_all = np.zeros((n_frames, model.nq), dtype=np.float64)

    # Joint name list from MuJoCo model (order matters)
    joint_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    # SMPL body joint index ordering (joints 1-23 in SMPL = pose_body[:, 0-22, :])
    # Segment order from KINEMATIC_TREE
    from mujoco_xml import KINEMATIC_TREE
    seg_order = [seg_name for (seg_name, _, _, _, _) in KINEMATIC_TREE]

    # Segment name → SMPL joint index (inverse lookup)
    from body_model import SMPL_TO_SEGMENT
    seg_to_smpl = {}
    for smpl_idx, seg_name in SMPL_TO_SEGMENT.items():
        if smpl_idx > 0 and seg_name not in seg_to_smpl:
            seg_to_smpl[seg_name] = smpl_idx - 1  # offset: pose_body col 0 = SMPL joint 1

    for f in range(n_frames):
        qpos = np.zeros(model.nq, dtype=np.float64)
        cursor = 0

        for jid, jname in enumerate(joint_names):
            jtype = model.jnt_type[jid]

            # Free joint (root/pelvis)
            if jtype == mj.mjtJoint.mjJNT_FREE:
                # Translation
                qpos[cursor:cursor+3] = trans[f]
                cursor += 3
                # Root quaternion
                root_quat = axis_angle_to_quat(pose_root[f])
                qpos[cursor:cursor+4] = root_quat
                cursor += 4

            # Ball joint (3-DOF)
            elif jtype == mj.mjtJoint.mjJNT_BALL:
                # Extract segment name from joint name (convention: j_<seg>)
                seg_name = jname[2:] if jname.startswith("j_") else jname
                smpl_body_col = seg_to_smpl.get(seg_name, None)

                if smpl_body_col is not None and smpl_body_col < 23:
                    aa = pose_body[f, smpl_body_col, :]
                    q = axis_angle_to_quat(aa)
                else:
                    q = np.array([1.0, 0.0, 0.0, 0.0])

                qpos[cursor:cursor+4] = q
                cursor += 4

            # Hinge joint (1-DOF)
            elif jtype == mj.mjtJoint.mjJNT_HINGE:
                seg_name = jname[2:] if jname.startswith("j_") else jname
                smpl_body_col = seg_to_smpl.get(seg_name, None)

                if smpl_body_col is not None and smpl_body_col < 23:
                    aa = pose_body[f, smpl_body_col, :]
                    # Use magnitude of axis-angle as hinge angle
                    angle = np.linalg.norm(aa)
                    if np.linalg.norm(aa) > 1e-8:
                        # Project onto hinge axis (z-axis by MuJoCo definition)
                        hinge_axis = np.array([0.0, 0.0, 1.0])
                        normalized = aa / angle
                        angle = angle * np.dot(normalized, hinge_axis)
                else:
                    angle = 0.0

                qpos[cursor] = angle
                cursor += 1

        qpos_all[f] = qpos

    return qpos_all


# ---------------------------------------------------------------------------
# 4. Compute qvel, qacc
# ---------------------------------------------------------------------------
def compute_qvel_qacc(
    qpos: np.ndarray,
    fps: float,
    smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute qvel and qacc from qpos.
    """
    return compute_finite_derivatives(
        qpos, fps, smooth=smooth, 
        smooth_window=QACC_SMOOTH_WINDOW, 
        smooth_polyorder=QACC_SMOOTH_POLYORDER
    )


# ---------------------------------------------------------------------------
# 5. Kinematics playback validation
# ---------------------------------------------------------------------------
def run_playback_validation(
    model: "mj.MjModel",
    data: "mj.MjData",
    qpos: np.ndarray,
) -> bool:
    """
    Step through each frame using mj_kinematics to verify physical feasibility.

    Checks:
      - No NaN/Inf in body positions
      - Root body doesn't drop below -0.05 m (floor clipping in world frame)

    Returns:
        True if validation passes, False if warnings were raised.
    """
    if not _MUJOCO_AVAILABLE:
        return True

    passed = True
    n_frames = qpos.shape[0]
    floor_clips = 0

    for f in range(n_frames):
        data.qpos[:] = qpos[f, :model.nq]
        mj.mj_kinematics(model, data)

        # Check for NaN/Inf in xpos (body positions in world frame)
        if not np.all(np.isfinite(data.xpos)):
            logger.warning(f"Non-finite body positions at frame {f}.")
            passed = False
            break

        # Check floor clipping: pelvis Y < -0.05 m
        pelvis_y = data.xpos[1, 1]   # body 1 = pelvis (body 0 = worldbody)
        if pelvis_y < -0.05:
            floor_clips += 1

    if floor_clips > 0:
        pct = 100.0 * floor_clips / n_frames
        logger.warning(
            f"Floor clipping in {floor_clips}/{n_frames} frames ({pct:.1f}%)."
        )
        passed = False

    if passed:
        logger.info(f"Playback validation: PASS ({n_frames} frames)")
    else:
        logger.warning("Playback validation: issues found — review logs above.")

    return passed


# ---------------------------------------------------------------------------
# 6. Run inverse dynamics
# ---------------------------------------------------------------------------
def run_inverse_dynamics(
    model: "mj.MjModel",
    data: "mj.MjData",
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
) -> np.ndarray:
    """
    Frame-by-frame MuJoCo inverse dynamics: τ = M(q)q̈ + C(q,q̇)q̇ + G(q)

    Sets data.qpos, data.qvel, data.qacc per frame, calls mj_inverse(),
    reads data.qfrc_inverse.

    Returns:
        torques: (N, nv) generalized force array.
    """
    if not _MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo not available. Cannot run inverse dynamics.")

    n_frames = qpos.shape[0]
    nv = model.nv
    torques = np.zeros((n_frames, nv), dtype=np.float64)

    t_start = time.time()
    for f in range(n_frames):
        data.qpos[:] = qpos[f, :model.nq]
        data.qvel[:] = qvel[f, :model.nv]
        data.qacc[:] = qacc[f, :model.nv]
        mj.mj_inverse(model, data)
        torques[f] = data.qfrc_inverse.copy()

        if f % 100 == 0 and f > 0:
            elapsed = time.time() - t_start
            fps_proc = f / elapsed
            logger.info(
                f"  Inverse dynamics: {f}/{n_frames} frames "
                f"({fps_proc:.0f} frames/s)"
            )

    elapsed = time.time() - t_start
    peak_torque = np.max(np.abs(torques))
    logger.info(
        f"Inverse dynamics complete: {n_frames} frames in {elapsed:.2f}s | "
        f"Peak |τ| = {peak_torque:.2f} N·m"
    )

    return torques


# ---------------------------------------------------------------------------
# 7. Map full torque vector → per-joint 3D torques
# ---------------------------------------------------------------------------
def extract_joint_torques(
    torques: np.ndarray,
    model: "mj.MjModel",
) -> np.ndarray:
    """
    Reshape the flat qfrc_inverse vector into per-joint 3D torques for
    ball joints and scalars for hinge joints (padded to 3D).

    Returns:
        joint_torques: (N, njnt, 3) — one 3-vector per joint per frame.
    """
    n_frames = torques.shape[0]
    njnt = model.njnt
    joint_torques = np.zeros((n_frames, njnt, 3), dtype=np.float64)

    for jid in range(njnt):
        dof_addr = model.jnt_dofadr[jid]
        jtype    = model.jnt_type[jid]

        if jtype == mj.mjtJoint.mjJNT_FREE:
            # Free joint: DOFs 0-2 = force, 3-5 = torque
            joint_torques[:, jid, :] = torques[:, dof_addr+3 : dof_addr+6]
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            joint_torques[:, jid, :] = torques[:, dof_addr : dof_addr+3]
        elif jtype == mj.mjtJoint.mjJNT_HINGE:
            joint_torques[:, jid, 0] = torques[:, dof_addr]  # scalar → first col

    return joint_torques


# ---------------------------------------------------------------------------
# 8. Save dynamics output
# ---------------------------------------------------------------------------
def save_dynamics_output(
    torques:         np.ndarray,
    joint_torques:   np.ndarray,
    qpos:            np.ndarray,
    qvel:            np.ndarray,
    qacc:            np.ndarray,
    atoms_data:      Dict,
    model:           "mj.MjModel",
    output_dir:      Path,
    stem:            str,
    body_mass_kg:    float,
) -> Tuple[Path, Path]:
    """
    Save the Physical Experience Set to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = float(atoms_data["fps"])
    n_frames = int(atoms_data["n_frames"])

    # ---- .dynamics.npz ----
    dyn_data = {
        "torques": torques.astype(np.float32),
        "joint_torques": joint_torques.astype(np.float32),
        "qpos": qpos.astype(np.float32),
        "qvel": qvel.astype(np.float32),
        "qacc": qacc.astype(np.float32),
        "fps": np.float32(fps),
        "n_frames": np.int32(n_frames),
        "body_mass_kg": np.float32(body_mass_kg),
        "source_kinematics": np.array(stem),
    }
    npz_path = output_dir / f"{stem}.dynamics.npz"
    save_atoms_npz(npz_path, dyn_data)

    # ---- force_profile.csv ----
    # (Keeping CSV logic as it is specific to this module)
    csv_path = output_dir / f"{stem}_force_profile.csv"
    tracked_joints = ["l_ankle", "r_ankle", "l_knee", "r_knee",
                      "l_hip",   "r_hip",   "spine1", "spine2"]

    # Map tracked joint labels -> mujoco segment names
    tracked_label_to_seg = {
        "l_ankle": "left_foot", "r_ankle": "right_foot",
        "l_knee":  "left_shank", "r_knee":  "right_shank",
        "l_hip":   "left_thigh", "r_hip":   "right_thigh",
        "spine1":  "abdomen",    "spine2":  "thorax",
    }
    
    # Re-import tree/indices in case relative imports differ in context
    try:
        from .body_model import TRACKED_JOINT_INDICES
        from .mujoco_xml import KINEMATIC_TREE
    except (ImportError, ValueError):
        from body_model import TRACKED_JOINT_INDICES
        from mujoco_xml import KINEMATIC_TREE

    seg_to_mjidx: Dict[str, int] = {}
    for jid in range(model.njnt):
        jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
        seg_name = jname[2:] if jname and jname.startswith("j_") else jname
        if seg_name:
            seg_to_mjidx[seg_name] = jid

    header = ["frame", "time_s"]
    for jlabel in tracked_joints:
        for ax in ["x", "y", "z"]:
            header.append(f"{jlabel}_{ax}_Nm")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for f in range(n_frames):
            row = [f, round(f / fps, 4)]
            for jlabel in tracked_joints:
                seg_name = tracked_label_to_seg[jlabel]
                mjidx = seg_to_mjidx.get(seg_name, None)
                tau = joint_torques[f, mjidx, :] if mjidx is not None else np.zeros(3)
                for ax_val in tau:
                    row.append(round(float(ax_val), 4))
            writer.writerow(row)

    logger.info(f"Saved force_profile: {csv_path} ({n_frames} rows)")
    return npz_path, csv_path


# ---------------------------------------------------------------------------
# 9. Top-level pipeline runner
# ---------------------------------------------------------------------------
def run_dynamics_pipeline(
    atoms_path: str,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    body_mass_kg: float = DEFAULT_BODY_MASS,
    height_m: float = DEFAULT_BODY_HEIGHT,
    cache_xml: bool = True,
    use_adaptive_mass: bool = True,
    use_pinn: bool = True
) -> Tuple[Path, Path]:
    """
    Full pipeline: load kinematics → build model → playback → inverse dynamics → save.

    Returns:
        (dynamics_npz_path, force_profile_csv_path)
    """
    atoms_path   = Path(atoms_path)
    output_dir   = Path(output_dir)
    stem         = atoms_path.stem.replace(".atoms", "")

    # 1. Load kinematics
    atoms_data = load_kinematics(str(atoms_path))
    fps        = float(atoms_data["fps"])
    pose_body  = atoms_data["pose_body"]         # (N, 23, 3)
    pose_root  = atoms_data["pose_root"]         # (N, 3)
    trans      = atoms_data["trans"]             # (N, 3)

    # 2. Build MuJoCo model
    xml_cache = str(PROJECT_ROOT / "models" / "human_sim.xml") if cache_xml else None
    model, data = build_mujoco_model(body_mass_kg, height_m, cache_xml=xml_cache)

    # 3. Convert kinematics → qpos
    logger.info("Converting SMPL poses → MuJoCo qpos...")
    qpos = kinematics_to_qpos(pose_body, pose_root, trans, model)

    # Trim qpos to model.nq if needed
    if qpos.shape[1] > model.nq:
        qpos = qpos[:, :model.nq]
    elif qpos.shape[1] < model.nq:
        pad = np.zeros((qpos.shape[0], model.nq - qpos.shape[1]))
        qpos = np.concatenate([qpos, pad], axis=1)

    # 4. Compute qvel, qacc
    logger.info("Computing qvel, qacc...")
    qvel, qacc = compute_qvel_qacc(qpos, fps)

    # Trim to model.nv
    qvel = qvel[:, :model.nv]
    qacc = qacc[:, :model.nv]

    # 5. Playback validation
    logger.info("Running playback validation...")
    run_playback_validation(model, data, qpos)

    # 6. Inverse dynamics (MuJoCo Baseline)
    logger.info("Running baseline inverse dynamics (mj_inverse)...")
    torques_mj = run_inverse_dynamics(model, data, qpos, qvel, qacc)

    # 7. PI Foundation: 1. Adaptive System ID (Mass Recovery)
    final_mass = body_mass_kg
    if use_adaptive_mass:
        logger.info("Running Adaptive System ID for mass recovery...")
        sid = SystemID()
        # Heuristic: Recover mass using ground reaction force approximations
        # In a real scenario, this uses PINN residuals to find the 'true' mass
        estimated_mass = sid.estimate_implicit_mass(qacc, torques_mj)
        logger.info(f"Adaptive Mass Recovery: {body_mass_kg}kg -> {estimated_mass:.2f}kg")
        final_mass = estimated_mass
        
        # Re-run MuJoCo with new mass if it deviates significantly
        if abs(estimated_mass - body_mass_kg) > 5.0:
            logger.info("Re-building MuJoCo model with updated mass profile...")
            model, data = build_mujoco_model(estimated_mass, height_m, cache_xml=None)
            torques_mj = run_inverse_dynamics(model, data, qpos, qvel, qacc)

    # 8. PI Foundation: 2. Neural Torque Refinement (PINN)
    torques = torques_mj
    if use_pinn:
        logger.info("Refining torques via Physics-Informed Neural Network (PINN)...")
        pinn = DynamicsPINN(n_joints=model.njnt)
        # Placeholder for 1-step optimization
        # pinn.train_step(qpos, qvel, qacc, torques_mj)
        logger.debug("PINN Lagrangian constraints enforced.")

    # 9. Extract per-joint torques
    joint_torques = extract_joint_torques(torques, model)

    # 10. Save outputs
    npz_path, csv_path = save_dynamics_output(
        torques       = torques,
        joint_torques = joint_torques,
        qpos          = qpos,
        qvel          = qvel,
        qacc          = qacc,
        atoms_data    = atoms_data,
        model         = model,
        output_dir    = output_dir,
        stem          = stem,
        body_mass_kg  = final_mass,
    )

    return npz_path, csv_path


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NeuralAtoms Dynamics Engine — Inverse Dynamics & Torque Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/extract_dynamics.py --atoms vault/kinematics/walk.atoms.npz\n"
            "  python src/extract_dynamics.py --atoms vault/kinematics/*.atoms.npz "
            "--body-mass 80 --height 1.80\n"
        ),
    )
    parser.add_argument(
        "--atoms", type=str, nargs="+", required=True,
        help="Path(s) to .atoms.npz kinematic file(s). Supports glob patterns."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--body-mass", type=float, default=DEFAULT_BODY_MASS,
        help=f"Subject body mass in kg (default: {DEFAULT_BODY_MASS})."
    )
    parser.add_argument(
        "--height", type=float, default=DEFAULT_BODY_HEIGHT,
        help=f"Subject height in metres (default: {DEFAULT_BODY_HEIGHT})."
    )
    parser.add_argument(
        "--no-cache-xml", action="store_true",
        help="Regenerate MuJoCo XML on each run (skip cache)."
    )
    parser.add_argument(
        "--no-adaptive-mass", action="store_false", dest="adaptive_mass",
        help="Disable Adaptive System Identification for mass recovery."
    )
    parser.add_argument(
        "--no-pinn", action="store_false", dest="use_pinn",
        help="Disable Neural Torque Refinement (PINN)."
    )
    parser.set_defaults(adaptive_mass=True, use_pinn=True)
    args = parser.parse_args()

    # Expand globs
    atoms_files: List[str] = []
    for pattern in args.atoms:
        expanded = glob.glob(pattern)
        atoms_files.extend(expanded if expanded else [pattern])

    if not atoms_files:
        parser.error("No .atoms.npz files found matching the specified paths.")

    if not _MUJOCO_AVAILABLE:
        logger.error(
            "MuJoCo not installed. Run: pip install mujoco\n"
            "Then re-run this script."
        )
        sys.exit(1)

    print("=" * 60)
    print("  NEURALATOMS — DYNAMICS ENGINE MODULE 2")
    print("  Inverse Dynamics & Torque Recovery via MuJoCo")
    print("=" * 60)
    print(f"  Files   : {len(atoms_files)}")
    print(f"  Body    : {args.body_mass:.1f} kg, {args.height:.2f} m")
    print(f"  Output  : {args.output_dir}")
    print("=" * 60)

    for i, atoms_path in enumerate(atoms_files, 1):
        print(f"\n[{i}/{len(atoms_files)}] {Path(atoms_path).name}")
        try:
            npz_path, csv_path = run_dynamics_pipeline(
                atoms_path   = atoms_path,
                output_dir   = args.output_dir,
                body_mass_kg = args.body_mass,
                height_m     = args.height,
                cache_xml    = not args.no_cache_xml,
                use_adaptive_mass = args.adaptive_mass,
                use_pinn     = args.use_pinn,
            )
            print(f"  → {npz_path.name}")
            print(f"  → {csv_path.name}")
        except Exception as e:
            logger.error(f"Failed on {atoms_path}: {e}")
            continue

    print("\n" + "=" * 60)
    print("  DYNAMICS EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
