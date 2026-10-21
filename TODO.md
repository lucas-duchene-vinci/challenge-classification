# Challenge classification — manchots

Prédire l'espèce d'un manchot (3 classes) à partir de 5 variables mesurées.

## Le dataset

`datasets/manchots.csv` — 760 lignes, 6 colonnes, en-tête présent.

| Col | Nom | Rôle |
|---|---|---|
| 0 | `species` | **cible** — 0 = Adélie, 1 = Papou, 2 = Jugulaire |
| 1 | `culmen_length_mm` | longueur du bec |
| 2 | `culmen_depth_mm` | épaisseur du bec |
| 3 | `flipper_length_mm` | longueur de la nageoire |
| 4 | `body_mass_g` | poids |
| 5 | `sex` | 0 = mâle, 1 = femelle |

**Constats après inspection :**

- 4 valeurs manquantes seulement, sur 4 lignes distinctes → 756 lignes après nettoyage (0,5 % de perte, négligeable)
- Classes **déséquilibrées** : 349 Adélie (46 %), 215 Papou (28 %), 192 Jugulaire (25 %)
- → **baseline** (toujours prédire Adélie) = **46,2 %**. Tout modèle en dessous est inutile.
- → le déséquilibre impose le **macro-F1** plutôt que l'exactitude seule, et `stratify=y` au découpage

## Livrables attendus (énoncé)

- [ ] Code : analyse + visualisation des données
- [ ] Code : création, entraînement, évaluation et **comparaison** de plusieurs modèles
- [ ] Les graphiques produits
- [ ] Les résultats, **en indiquant explicitement quelle(s) métrique(s)** ont servi
- [ ] Le modèle entraîné : `CLF_GROUPE_XXX.pkl` (pickle, protocol=5)

Remise du modèle + code + analyses à **11h30**, résultats finaux à **12h00**.

---

## À faire

### 1. `dataset.py` — fait ✅

`load_data`, `clean_data`, `split_data`, `save_data`, `load_data_split` sont en place et adaptés
(cible en colonne 0, `X = data[:, 1:]`, `y = data[:, 0]`, pas de binarisation).

Détails restants :

- [ ] valeurs par défaut sur `split_data` (`test_size=0.25, seed=42`)
- [ ] supprimer le bloc `ManchotDataset` commenté (l'historique est dans git)

### 2. `main.py` — cassé ⚠️

Il importe `ManchotDataset`, qui est commentée dans `dataset.py` → `ImportError` au lancement.

- [ ] réécrire sur le modèle de la semaine 5 : charger → nettoyer → découper → sauvegarder → explorer → régler → évaluer

### 3. `metrics.py` — à généraliser au multi-classe 🔴

C'est le gros morceau. La version semaine 5 est binaire (matrice 2×2, VN/FP/FN/VP).

- [ ] `confusion_matrix(y_true, y_pred)` → matrice **n×n**, `cm[i, j]` = éléments de classe `i` prédits en classe `j`
- [ ] `accuracy(cm)` → somme de la diagonale / somme totale (se généralise directement)
- [ ] `precision(cm, classe)` / `recall(cm, classe)` → **par classe**, plus de sens global
- [ ] `f1(cm, classe)` par classe, puis `macro_f1(cm)` = moyenne des F1
- [ ] `report(y_true, y_pred)` → matrice + métriques globales + détail par classe
- [ ] garder les gardes contre la division par zéro

Test de validation :

```python
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0])
# attendu : [[1, 1, 0],
#            [0, 2, 0],
#            [1, 0, 1]]
```

### 4. `explore.py` — à adapter

- [ ] nouvelles colonnes (les 5 mesures)
- [ ] comparer **trois** groupes au lieu de deux → moyenne ± écart-type par espèce
- [ ] `sex` est la seule catégorielle → % de chaque espèce par sexe
- [ ] **nuages de points colorés par espèce** : c'est la visualisation principale.
      Croiser `culmen_length` × `culmen_depth`, et `flipper_length` × `body_mass`.

### 5. `tuning.py` — copier-coller depuis la semaine 5 ✅

Fonctionne tel quel. Seule adaptation : `scoring="f1_macro"` au lieu de `"f1"`
(`"f1"` n'existe pas en multi-classe).

### 6. `classifiers.py` — à écrire

- [ ] `get_classifiers(best)` : Ridge, DecisionTree, KNN **normalisé** (`StandardScaler`),
      éventuellement SVC et RandomForest
- [ ] classificateur manuel ? Optionnel ici — l'énoncé ne le demande pas.
      Un `DummyClassifier(strategy="most_frequent")` suffit comme baseline.

### 7. Sauvegarde du modèle

```python
from pickle import dump
with open("CLF_GROUPE_XXX.pkl", "wb") as f:
    dump(clf, f, protocol=5)
```

- [ ] **tester l'aller-retour avant la séance** (sauvegarde → rechargement → `predict`)
- [ ] attention : si le modèle est une `Pipeline` avec `StandardScaler`, le scaler est
      sauvegardé avec — c'est voulu, le correcteur doit pouvoir appeler `predict` directement
      sur des données brutes

---

## Méthodologie à respecter

1. Découper train/test **une fois**, sauvegarder le split
2. Explorer **sur le train uniquement**
3. Régler les hyper-paramètres par **cross-validation sur le train**
4. Évaluer sur le test **une seule fois**, à la fin — pas de retour en arrière

## Repère

Ce dataset est réputé facile : bec et nageoire séparent très bien les trois espèces.
Attendre **95-99 %** d'exactitude. Plafonner à 85 % = chercher un bug, pas un meilleur modèle.
