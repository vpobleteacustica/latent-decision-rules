import numpy as np

class RadialDetector:
    """
    Radial centroid-based detector in latent space.

    For each class:
        - Compute centroid
        - Compute radial threshold based on percentile q
    """

    def __init__(self):
        self.centroids = {}
        self.thresholds = {}

    def fit(self, embeddings_by_class: dict, q: float = 0.99):
        """
        Parameters
        ----------
        embeddings_by_class : dict
            {class_name: np.ndarray of shape (N, D)}
        q : float
            Percentile used to define radial threshold.
        """

        self.centroids = {}
        self.thresholds = {}

        for class_name, Z in embeddings_by_class.items():
            if len(Z) == 0:
                continue

            centroid = Z.mean(axis=0)
            distances = np.linalg.norm(Z - centroid, axis=1)
            threshold = np.quantile(distances, q)

            self.centroids[class_name] = centroid
            self.thresholds[class_name] = threshold

    def predict(self, z: np.ndarray):
        """
        Predict class label for latent vector z.

        Returns
        -------
        str : predicted class or "NO_DETECT"
        """

        best_class = None
        best_distance = np.inf

        for class_name, centroid in self.centroids.items():
            dist = np.linalg.norm(z - centroid)

            if dist < best_distance:
                best_distance = dist
                best_class = class_name

        if best_class is None:
            return "NO_DETECT"

        if best_distance <= self.thresholds[best_class]:
            return best_class
        else:
            return "NO_DETECT"

    def best_distance(self, z: np.ndarray):
        """
        Return minimal distance to any centroid.
        """

        best_distance = np.inf

        for centroid in self.centroids.values():
            dist = np.linalg.norm(z - centroid)
            if dist < best_distance:
                best_distance = dist

        return best_distance