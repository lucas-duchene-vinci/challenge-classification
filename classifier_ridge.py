
import numpy as np
from matplotlib import pyplot as plt

from dataset import load_data, clean_data, split_data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report

from metrics import report

data = load_data("datasets/manchots.csv")

data = clean_data(data)

X_train, X_test, y_train, y_test = split_data(
    data,
    test_size=0.2,
    seed=42
)


model = make_pipeline(
    StandardScaler(),
    RidgeClassifier(alpha=1.0)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)

resultats = report(y_test, y_pred)

print("Matrice de confusion :")
print(resultats["cm"])

print("\nAccuracy :", resultats["accuracy"])

noms = ["Adélie", "Papou", "Jugulaire"]

for i, nom in enumerate(noms):
    print(f"\n--- {nom} ---")
    print("Précision :", resultats["precision"][i])
    print("Rappel :", resultats["recall"][i])
    print("Spécificité :", resultats["specificity"][i])
    print("F1-score :", resultats["f1_score"][i])
