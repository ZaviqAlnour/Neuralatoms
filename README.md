# NeuralAtoms: The Language of Atoms

> **Mission:** "Decoding the hidden physical variables of human labor to provide the 'Physical Intelligence' required by the next generation of generalist robotics."

---

## 🛑 The "Atoms Bottleneck"
Current AI models are masters of pixels and silhouettes, but they lack the 'feeling' of physics. They observe motion without understanding the underlying Newtonian truths—the torque in a joint, the friction against a surface, or the mass distribution of a load. This is the **Atoms Bottleneck**: the gap between visual representation and physical reality.

## ⚡ The Solution: Our Engine
NeuralAtoms is building the world’s first Physics-Informed Data Engine. We don't just track points; we reconstruct the physical "DNA" of movement.

| Layer | Component | Description |
| :--- | :--- | :--- |
| **K** | **Kinematics** | World-grounded 3D geometry and skeletal reconstruction. |
| **D** | **Dynamics** | Inverse Dynamics for Torque and Force recovery from 2D observation. |
| **C** | **Context** | Semantic understanding of Friction, Mass, and Environment interactions. |

---

## 🛠 Technical Stack
NeuralAtoms utilizes a hybrid high-performance architecture:
- **Local Vault (Disk IO):** HP ZBook (Core i5, 512GB NVMe SSD) for high-frequency data logging at 120Hz+.
- **Remote Synapse (Compute):** Google Colab T4 GPU (Tesla T4) for deep kinematics extraction and physical compute.

---

## 📂 Project Structure

| Directory | Name | Function |
| :--- | :--- | :--- |
| `src/` | **The DNA** | Core Python logic and physics-informed algorithms. |
| `models/` | **The Synapses** | Pre-trained SMPL/WHAM weights and MuJoCo assets. |
| `vault/` | **The Ledger** | High-speed local storage for raw video and `.atoms` files. |

---

## 🚀 Usage

To initialize the internal "Pipe" and verify hardware readiness:

```bash
# Perform a system-level check on CUDA and NVMe throughput
python system_check.py
```

To run the data ingestion stress test:

```bash
python src/stress_test_pipe.py
```

---

## ⚖️ License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

**Copyright 2026 NeuralAtoms - Zaviq Alnour**
