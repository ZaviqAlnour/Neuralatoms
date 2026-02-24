# The NeuralAtoms 8-Layer Architecture
We are moving from pixels to **Programmable Atoms**.

---

## Layer 1: The Ingestion & Decoupling Layer (Raw Bit-Stream)

**Function:**
Decouple local NVMe storage from Cloud T4 compute.

**Mechanism:**
An asynchronous Watcher service (Python `watchdog`) detects incoming `.mp4` files in `/vault/raw`.

**Innovation — Pre-Filtering VLM:**
A lightweight model (Moondream2) performs a *Labor-Check* to discard low-value footage (static scenes, irrelevant clips) before GPU processing.
This preserves compute credits and ensures only semantically useful motion enters the pipeline.

---

## Layer 2: The Kinematic Spine (Metric Geometry)

**Function:**
Extract **metric-accurate 3D human trajectories**.

**Mechanism:**
GVHMR (Gravity-View Human Motion Recovery).

**Key Detail:**
Unlike standard motion capture, this includes a **Global Motion Predictor** that incorporates camera intrinsics and world grounding.
Outputs 3D joint quaternions in a **fixed world frame (meters)**, never pixel coordinates.

---

## Layer 3: The Semantic Voxelization Layer (Scene Reconstruction)

**Function:**
Reconstruct the environment being interacted with.

**Mechanism:**
Depth-Anything-V2 + Vosh (Voxel-Aligned Features).

**Innovation:**
Instead of detecting objects, build a **128³ Semantic Occupancy Grid**.
Each voxel is labeled (Wall, Floor, Movable Object, etc.), preventing “ghost collisions” during robotics simulation.

---

## Layer 4: The Material Intelligence Layer (Coefficient Inference)

**Function:**
Infer hidden physical properties:

* Friction (μ)
* Stiffness (k)
* Mass (m)

**Mechanism:**
Vision-to-Physics (V2P) Transformer.

**Key Detail:**
Uses SAM-2 deformation tracking.
Example: If an object compresses 2 cm under force, automatically estimate its spring constant.

---

## Layer 5: The Dynamics Engine — The Heart of NeuralAtoms

**Function:**
Recover real torques (τ) and Ground Reaction Forces (GRF).

**Mechanism:**
Differentiable MuJoCo using Newton-Euler inversion.

**Innovation — Adaptive System Identification:**
Run five parallel simulations with varying mass assumptions.
Select the model whose acceleration profile best matches the video to estimate real-world mass without measurement hardware.

---

## Layer 6: The Success & Failure Evaluator (Self-Labeling Intelligence)

**Function:**
Determine whether an action succeeded.

**Mechanism:**
SAFE — Spatio-Temporal Anomaly Factor Evaluator.

**Innovation:**
Detect drops, slips, instability, or spills automatically.
This generates **failure data**, the most valuable signal for reinforcement learning.

---

## Layer 7: The `.atoms` Encoder (Binary Physical Ledger)

**Function:**
Compress Layers 2–6 into a hardware-agnostic experience artifact.

**Mechanism:**
Protobuf-based binary encoding.

**Output:**
A `.atoms` file — a time-series ledger of forces, torques, kinematics, and voxel states.

---

## Layer 8: The Scaling & Provenance Layer (Data Marketplace)

**Function:**
Versioning, traceability, and quality scoring.

**Mechanism:**
AtomsID hashing with fidelity metrics.

High-resolution data yields high-fidelity atoms.
Low-quality sources remain usable but are down-weighted.

---

# TITAN INITIALIZATION PROMPT (EXECUTION DIRECTIVE)

## Role

You are the Lead Robotics Systems Architect.

## Objective

Replace all mockups with a **production-grade, high-throughput refinery pipeline** defined above.
