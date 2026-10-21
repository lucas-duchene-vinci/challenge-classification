import numpy as np

def confusion_matrix(y_true, y_pred):
    n = int(max(y_true.max(), y_pred.max())) +1 # On suppose que les étiquettes sont bien (0, 1, 3) ce qui permet de faire matrice, ne fonctionnerait pas si on classifiait en (0, 4, 7) par exemple.
    cm = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            cm[i, j] = ((y_true == i) & (y_pred == j)).sum()
    return cm

def accuracy(cm: np.ndarray) -> float | int:
    return cm.diagonal().sum() / cm.sum()

def precision(cm: np.ndarray) -> float | int:
    return