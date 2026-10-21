import numpy as np

class ManualClassifier:

    def __init__(self):
        self.thresholds = {}

    def fit(self, Xtrain: np.ndarray, Ytrain: np.ndarray):
        self.thresholds[0] = np.mean(Xtrain[:, 0])  # Longueur du bec
        self.thresholds[1] = np.mean(Xtrain[:, 1])  # Profondeur du bec
        self.thresholds[2] = np.mean(Xtrain[:, 2])  # Longueur des nageoires
        self.thresholds[3] = np.mean(Xtrain[:, 3])  # Masse corporelle

    def predict_one(self, x: np.ndarray):

        # 0 : bec plus court ; nageoires plus courtes en moyenne que les deux autres classes.
        if x[0] < self.thresholds[0] and x[2] < self.thresholds[2]:
            return 0

        # 1 : plus lourds, nageoires plus longues et bec moins profond que les deux autres classes.
        elif x[3] > self.thresholds[3] and x[2] > self.thresholds[2] and x[1] < self.thresholds[1]:
            return 1

        # 2 : bec plus long que la classe 0, avec une profondeur similaire ; nageoires légèrement plus longues et masse similaire.
        else:
            return 2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédire les espèces de plusieurs manchots."""
        return np.array([self.predict_one(x) for x in X])

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculer la proportion de prédictions correctes."""
        predictions = self.predict(X)
        return float(np.mean(predictions == Y))