"""
src/image.py
Amazon ML Challenge 2026 — Image loader subsystem stub.

This is a stub module. It provides the architectural boundaries for image
loading and preprocessing, without implementing the actual logic (which
requires knowing the competition's URL structure and image formats).
"""

from typing import List, Optional
import numpy as np
from pathlib import Path
from src.logging_utils import get_logger

logger = get_logger(__name__)


def load_image_from_url(url: str, target_size: tuple = (224, 224)) -> Optional[np.ndarray]:
    """Stub: Load an image from a URL, resize, and return as a numpy array."""
    logger.warning("load_image_from_url: STUB CALLED (returning zeros)")
    return np.zeros((*target_size, 3), dtype=np.uint8)


def load_image_from_path(path: Path, target_size: tuple = (224, 224)) -> Optional[np.ndarray]:
    """Stub: Load an image from a local path, resize, and return as a numpy array."""
    logger.warning("load_image_from_path: STUB CALLED (returning zeros)")
    return np.zeros((*target_size, 3), dtype=np.uint8)


class ImageDataset:
    """Stub: Dataset for loading images in batches."""
    def __init__(self, urls_or_paths: List[str], target_size: tuple = (224, 224)):
        self.items = urls_or_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return np.zeros((*self.target_size, 3), dtype=np.uint8)
