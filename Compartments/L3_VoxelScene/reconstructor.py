import os
import torch
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger("NeuralAtoms.L3")

class VoxelScene:
    """
    Layer 3: Semantic Voxelization Layer.
    Reconstructs environment using Depth-Anything-V2 and fuses into a 128^3 grid.
    """
    def __init__(self, resolution=128):
        self.resolution = resolution
        # 0: Empty, 1: Wall, 2: Floor, 3: Movable, 4: Human
        self.voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
        logger.info(f"VoxelScene initialized with {resolution}^3 resolution")

    def run_depth(self, image):
        """
        Mock for Depth-Anything-V2. 
        Returns a depth map normalized to [0, 1].
        """
        logger.info("Running Depth-Anything-V2 on frame")
        return np.random.rand(480, 640)

    def project_to_3d(self, depth_map, intrinsics):
        """
        Converts depth map to a point cloud using camera intrinsics.
        """
        # Simplified point cloud generation
        h, w = depth_map.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        # z = depth_map * scale
        pts = np.stack([x, y, depth_map], axis=-1).reshape(-1, 3)
        return pts

    def export_ply(self, points, filename):
        """
        Exports point cloud to .ply format.
        """
        logger.info(f"Exporting point cloud to {filename}")
        # Standard PLY header
        header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        with open(filename, 'w') as f:
            f.write(header)
            np.savetxt(f, points, fmt='%f %f %f')

    def build_occupancy_grid(self, fused_points):
        """
        Discretizes point clouds into the 128^3 semantic grid.
        """
        logger.info(f"Fusing points into {self.resolution}^3 occupancy grid")
        # Normalize points to grid indices [0, 127]
        # In production, we'd use world bounds
        min_bound = fused_points.min(axis=0)
        max_bound = fused_points.max(axis=0)
        
        indices = ((fused_points - min_bound) / (max_bound - min_bound + 1e-6) * (self.resolution - 1)).astype(int)
        
        for idx in indices:
            if 0 <= idx[0] < self.resolution and 0 <= idx[1] < self.resolution and 0 <= idx[2] < self.resolution:
                self.voxel_grid[idx[0], idx[1], idx[2]] = 1 # Mark as Wall/Obstacle
        
        return self.voxel_grid

    def process(self, video_frames, output_dir):
        """
        Full L3 Processing Cycle.
        """
        logger.info(f"L3: Reconstructing scene from {len(video_frames)} frames")
        all_points = []
        
        for i, frame in enumerate(video_frames):
            depth = self.run_depth(frame)
            
            # Export .ply every 10th frame
            if i % 10 == 0:
                pts = self.project_to_3d(depth, None)
                self.export_ply(pts, os.path.join(output_dir, f"frame_{i:04d}.ply"))
                all_points.append(pts)
        
        if all_points:
            fused = np.concatenate(all_points, axis=0)
            self.build_occupancy_grid(fused)
            np.save(os.path.join(output_dir, "semantic_voxel_grid.npy"), self.voxel_grid)
            logger.info(f"L3: Voxel grid saved to {output_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scene = VoxelScene()
    # scene.process([np.zeros((480, 640, 3))], "./")
