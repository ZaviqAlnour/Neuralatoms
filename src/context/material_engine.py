"""
NeuralAtoms Context Engine — Material Classification

Utilizes CLIP (Contrastive Language-Image Pre-training) for zero-shot 
classification of surface materials from video frame crops.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("neuralatoms.context.material")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
MATERIAL_CANDIDATES = [
    "concrete", "wood floor", "steel", "carpet", "grass", "ice", 
    "linoleum", "ceramic tile", "wet concrete", "dirt", "gravel"
]


class MaterialEngine:
    """
    Zero-shot material classifier for environmental surfaces.
    """
    def __init__(
        self, 
        model_id: str = DEFAULT_CLIP_MODEL,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        
    def load_model(self):
        """Lazy load CLIP model and processor."""
        if not _CLIP_AVAILABLE:
            logger.error("transformers/torch not available. pip install transformers torch pillow")
            return False
            
        logger.info(f"Loading CLIP model: {self.model_id} on {self.device}...")
        self._model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        logger.info("CLIP model loaded successfully.")
        return True

    def classify_crop(
        self, 
        image_bgr: np.ndarray, 
        candidates: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """
        Classify a crop of an image into one of the candidate materials.
        
        Returns:
            (top_label, confidence_score)
        """
        if self._model is None:
            if not self.load_model():
                return "unknown", 0.0
                
        candidates = candidates or MATERIAL_CANDIDATES
        
        # Convert BGR to RGB PIL image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Prepare inputs
        inputs = self._processor(
            text=candidates, 
            images=image_pil, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            
        # Get probabilities
        logits_per_image = outputs.logits_per_image  # (1, N_candidates)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        top_idx = np.argmax(probs)
        top_label = candidates[top_idx]
        top_prob = probs[top_idx]
        
        logger.debug(f"Material prediction: {top_label} ({top_prob:.2f})")
        return top_label, top_prob


if __name__ == "__main__":
    # Internal test with dummy image
    engine = MaterialEngine()
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    label, score = engine.classify_crop(dummy_img)
    print(f"Top Material: {label} (confidence: {score:.2f})")
