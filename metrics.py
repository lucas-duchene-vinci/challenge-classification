import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    y_true = Les valeurs justes, ce qui a été analyser cette valeur correspond à Ytest
    y_pred = C'est ce que le modèle pense juste après analyse des données sans la caractéristique "species"

    Renvoie une matrice, cm[i, j], la matrice contient le nombre de manchots de l'espèce i que le modèle a prédit en tant que j.
    """

    n = int(max(y_true.max(), y_pred.max())) +1 # On suppose que les étiquettes sont bien (0, 1, 2) ce qui permet de faire matrice, ne fonctionnerait pas si on classifiait en (0, 4, 7) par exemple.
    cm = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            cm[i, j] = ((y_true == i) & (y_pred == j)).sum()
    return cm


def accuracy(cm: np.ndarray) -> float:
    """
    Calcule l'exactitude (accuracy) à partir de la matrice de confusion.

    Soit le nombre de bonne réponse / par le total de la matrice.
    """
    return float(cm.diagonal().sum()) / float(cm.sum())

def precision(cm: np.ndarray, c) -> float:
    """

    c représente l'indice de l'espèce que l'on cherche à analyser, par exemple: 0 = adélie.

    La precision permet de vérifier si quand le modèle dit "c", est ce qu'il a raison ou non.

    On lit la colonne à l'indice c et on cherche quel valeur fait partie de la diagonal.
    """
    vp = cm[c, c]
    total = cm[:, c].sum()

    if total == 0:
        return 0.0
    return float(vp) / float(total)

def recall(cm: np.ndarray, c) -> float:
    """
    c sera la même chose à chaque fonction ou il apparaît

    le recall traite la ligne et chercher combien d'une espèce le modèle a-t-il trouvé parmi les vrais de son espèce
    """
    vp = cm[c, c]
    total = cm[c, :].sum()
    if total == 0:
        return 0.0
    return float(vp) / float(total)

def specificity(cm: np.ndarray, c) -> float:
    """
    Specifity est la fonction qui permet de voir parmis tous les machots ne faisant pas partie de l'espèce 'c'
    Combien le modèle a bien écarté.

    Cette métrique n'utilise pas la diagonal comme numérateur uniquement pour trouver les vrai positifs (vp)
    """
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
    """

    Le f1_score est une métrique qui combine la précision et le rappel car ces mesures à elles seules ne sont pas suffisantes.

    Car les deux métriques qu'elle combine se contredisent, f1 permet d'éviter cette contradiction et de trouver un compromis

    La moyenne sera toujours tirée vers le bas par la valeur la plus faible
    """
    vp = cm[c, c]
    fn = cm[c, :].sum() - vp # Ligne et on soustrait la diago
    fp = cm[:, c].sum() - vp  # Colonne moins la diag

    if 2 * vp + fp + fn == 0:
        return 0.0
    return float(2 * vp) / float((2 * vp + fp + fn))

def report(y_true, y_pred) -> dict:
    """
    Report est une fonciton qui va créer la matrice de confusion, et renvoyer un dictionnaire contenant tous les résultats des métrics utiliser.
    """
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
