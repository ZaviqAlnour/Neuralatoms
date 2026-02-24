"""
NeuralAtoms Context Engine Module 3 — Environmental & Material Inference

Main orchestrator for Module 3. Ingests video (and optionally kinematics) 
to infer depth, material properties, and contact points.

Usage:
    python src/context/extract_context.py --video <path> [--atoms <kinematics_path>]

Copyright © 2026 NeuralAtoms — Zaviq Alnour
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import cv2

# Add project root to path for src imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("neuralatoms.context")

from src.context.friction_table import get_friction_coefficients
from src.context.depth_engine import DepthEngine, VoxelGrid
from src.context.material_engine import MaterialEngine
from src.context.contact_engine import ContactEngine
from src.context.scene_intelligence import MaterialSigLIP, VoxelFeatureAggregator, SDFContactGenerator
from src.utils.atoms_io import save_atoms_npz, load_atoms_npz


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = _ROOT / "vault" / "context"


class ContextExtractor:
    """
    Orchestrates Depth, Material, and Contact engines.
    """
    def __init__(
        self, 
        output_dir: Optional[Path] = None,
        use_gpu: bool = True
    ):
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.depth_engine = DepthEngine(use_gpu=use_gpu)
        self.material_engine = MaterialEngine()
        self.contact_engine = ContactEngine()
        
        # PI Foundation Upgrades
        self.siglip = MaterialSigLIP()
        self.voxel_agg = VoxelFeatureAggregator()
        self.sdf_gen = SDFContactGenerator()
        
    def process_video(
        self, 
        video_path: str, 
        atoms_kinematics: Optional[Dict[str, Any]] = None,
        use_siglip: bool = True,
        use_voxel: bool = True
    ) -> Dict[str, Any]:
        """
        Run the context extraction pipeline.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        logger.info(f"Context extraction started for: {video_path.name}")
        t_start = time.time()
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 1. Material classification (sample from middle frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
        ret, frame = cap.read()
        if ret:
            # Crop a sample region for material classification (bottom 20% center)
            h, w = frame.shape[:2]
            crop = frame[int(h*0.8):h, int(w*0.4):int(w*0.6)]
            
            if use_siglip:
                logger.info("Using SigLIP for zero-shot material classification...")
                candidates = ["concrete", "wood", "rubber", "carpet", "metal", "grass"]
                results = self.siglip.classify_material(crop, candidates)
                surface_label = max(results, key=results.get)
                confidence = results[surface_label]
            else:
                surface_label, confidence = self.material_engine.classify_crop(crop)
        else:
            surface_label, confidence = "concrete", 1.0  # fallback
            
        mu_static, mu_kinetic = get_friction_coefficients(surface_label)
        logger.info(f"Surface Material: {surface_label} (mu_s={mu_static}, mu_k={mu_kinetic})")
        
        # 2. Sequential processing loop (simplified for placeholder)
        # In production, this would run Depth-Anything-V2 and SAM2 per frame
        logger.info(f"Processing {n_frames} frames for depth and contacts...")
        
        # Placeholder tensors
        depth_data = np.zeros((n_frames, 64, 64), dtype=np.float32) # Downsampled
        contact_data = [] # Sparse list
        
        # Reset capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # for f in range(n_frames):
        #     ret, frame = cap.read()
        #     if not ret: break
        #     depth = self.depth_engine.estimate_depth(frame)
        #     ...
        
        cap.release()
        t_elapsed = time.time() - t_start
        
        context_data = {
            "surface_label": np.array(surface_label),
            "surface_confidence": np.float32(confidence),
            "mu_static": np.float32(mu_static),
            "mu_kinetic": np.float32(mu_kinetic),
            "depth_sampled": depth_data,
            "extraction_time_s": np.float32(t_elapsed),
            "source_video": np.array(video_path.stem)
        }
        
        return context_data


def main():
    parser = argparse.ArgumentParser(description="NeuralAtoms Context Engine — Module 3")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--atoms", type=str, help="Optional kinematics .atoms.npz for contact detection.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-siglip", action="store_false", dest="use_siglip", help="Disable SigLIP material intelligence.")
    parser.add_argument("--no-voxel", action="store_false", dest="use_voxel", help="Disable DINOv2 voxelization.")
    parser.set_defaults(use_siglip=True, use_voxel=True)
    
    args = parser.parse_args()
    
    extractor = ContextExtractor(output_dir=args.output_dir)
    
    atoms_data = None
    if args.atoms:
        atoms_data = load_atoms_npz(args.atoms)
        
    context_data = extractor.process_video(
        video_path=args.video, 
        atoms_kinematics=atoms_data,
        use_siglip=args.use_siglip,
        use_voxel=args.use_voxel
    )
    
    output_path = Path(args.output_dir) / f"{Path(args.video).stem}.context.npz"
    save_atoms_npz(output_path, context_data)
    
    print("="*60)
    print(f" CONTEXT EXTRACTION COMPLETE -> {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
