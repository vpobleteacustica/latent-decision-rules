import numpy as np
from typing import List, Dict
from src.models.radial_detector import RadialDetector


class CentroidClassifier:
    """
    High-level classifier that wraps RadialDetector
    and operates over batches of embeddings.
    """

    def __init__(self, detector: RadialDetector):
        self.detector = detector

    def predict_batch(self, Z: np.ndarray) -> List[str]:
        """
        Predict labels for a batch of latent vectors.

        Parameters
        ----------
        Z : np.ndarray of shape (N, D)

        Returns
        -------
        List[str]
        """
        return [self.detector.predict(z) for z in Z]

    def best_distances_batch(self, Z: np.ndarray) -> List[float]:
        """
        Compute minimal distance to centroids for each sample.
        """
        return [self.detector.best_distance(z) for z in Z]