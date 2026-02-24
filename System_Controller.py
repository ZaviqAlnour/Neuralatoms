import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("system_controller.log")
    ]
)

logger = logging.getLogger("NeuralAtoms")

# Path Configuration
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARTMENTS = {
    f"L{i}": os.path.join(ROOT_DIR, "Compartments", folder)
    for i, folder in enumerate([
        "L1_Ingestion", "L2_Kinematics", "L3_VoxelScene", 
        "L4_Materials", "L5_Dynamics", "L6_Evaluation", 
        "L7_Ledger", "L8_Scale"
    ], 1)
}
VAULT_RAW = os.path.join(ROOT_DIR, "vault", "raw")

# Ensure vault directory exists
os.makedirs(VAULT_RAW, exist_ok=True)

class PipelineHandler(FileSystemEventHandler):
    """
    Handles new file events and triggers the asynchronous refinery pipeline.
    """
    def __init__(self, executor):
        self.executor = executor

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".mp4"):
            logger.info(f"Incoming Bit-Stream Detected: {os.path.basename(event.src_path)}")
            self.executor.submit(self.run_pipeline, event.src_path)

    def run_pipeline(self, video_path):
        """
        Orchestrates the 8-layer flow.
        """
        video_name = os.path.basename(video_path)
        logger.info(f"Starting refinery pipeline for: {video_name}")
        
        try:
            # Layer 1: Ingestion & Decoupling (Pre-filtering)
            logger.info(f"L1: Executing Moondream2 Labor-Check on {video_name}")
            # Mocking the call to Layer 1 script
            time.sleep(1) 
            
            # Layer 2: Kinematic Spine (Metrics)
            logger.info(f"L2: Extracting 3D Trajectories (WHAM + GVHMR)")
            time.sleep(1)
            
            # Layer 3: Semantic Voxelization
            logger.info(f"L3: Building 128³ Semantic Occupancy Grid")
            time.sleep(1)
            
            # Layer 4: Material Intelligence
            logger.info(f"L4: Inferring Friction, Stiffness, and Mass (V2P)")
            time.sleep(1)
            
            # Layer 5: Dynamics Engine
            logger.info(f"L5: Recovering Real Torques & GRF (MuJoCo Inverse)")
            time.sleep(1)
            
            # Layer 6: Success & Failure Evaluator (SAFE)
            logger.info(f"L6: Analyzing Stability and Success Metrics")
            time.sleep(1)
            
            # Layer 7: .atoms Encoder
            logger.info(f"L7: Encoding Experience Artifact into .atoms")
            time.sleep(1)
            
            # Layer 8: Scaling & Provenance
            logger.info(f"L8: Versioning and Quality Scoring (AtomsID)")
            
            logger.info(f"Pipeline Complete for {video_name}. Artifact stored in L7_Ledger.")
            
        except Exception as e:
            logger.error(f"Pipeline Failure for {video_name}: {str(e)}")

class SystemController:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.observer = Observer()

    def start(self):
        logger.info("NeuralAtoms System Controller Initialized.")
        logger.info(f"Root: {ROOT_DIR}")
        logger.info(f"Monitoring /vault/raw for .mp4 files...")
        
        event_handler = PipelineHandler(self.executor)
        self.observer.schedule(event_handler, VAULT_RAW, recursive=False)
        self.observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            logger.info("Shutting down System Controller...")
        
        self.observer.join()
        self.executor.shutdown(wait=True)

if __name__ == "__main__":
    controller = SystemController()
    controller.start()
