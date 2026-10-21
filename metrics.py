import numpy as np

def confusion_matrix(y_true, y_pred):
    n = int(max(y_true.max(), y_pred.max())) +1 # On suppose que les étiquettes sont bien (0, 1, 2) ce qui permet de faire matrice, ne fonctionnerait pas si on classifiait en (0, 4, 7) par exemple.
    cm = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            cm[i, j] = ((y_true == i) & (y_pred == j)).sum()
    return cm

def accuracy(cm: np.ndarray) -> float:
    return float(cm.diagonal().sum()) / float(cm.sum())

def precision(cm: np.ndarray, c) -> float:
    vp = cm[c, c]
    total = cm[:, c].sum()

    if total == 0:
        return 0.0
    return float(vp) / float(total)

def recall(cm: np.ndarray, c) -> float:
    vp = cm[c, c]
    total = cm[c, :].sum()
    if total == 0:
        return 0.0
    return float(vp) / float(total)

def specificity(cm: np.ndarray, c) -> float:
    total = cm.sum()
    ligne = cm[c, :].sum()
    colone = cm[:, c].sum()
    vp = cm[c, c]

    fp = colone - vp # -> on retire les vp de la colonne pour obtenir les faux positif
    vn = total - float(ligne) - colone + vp # -> Ici on soustrait le total, la ligne et la colonne et on ajoute les vp pour obtenir le vrai negatif
    if vn + fp == 0:
        return 0.0
    return vn / (vn + fp)

def f1_score(cm: np.ndarray, c) -> float:
    vp = cm[c, c]
    fn = cm[c, :].sum() - vp # Ligne et on soustrait la diago
    fp = cm[:, c].sum() - vp  # Colonne moins la diag

    if 2 * vp + fp + fn == 0:
        return 0.0
    return float(2 * vp) / float((2 * vp + fp + fn))

def report(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    return {
        "cm": cm,
        "accuracy": accuracy(cm),
        "precision": [precision(cm, c) for c in range(n)],
        "recall": [recall(cm, c) for c in range(n)],
        "specificity": [specificity(cm, c) for c in range(n)],
        "f1_score": [f1_score(cm, c) for c in range(n)]
    }
