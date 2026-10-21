import pathlib

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import load_data, clean_data, split_data
from classifier import ManualClassifier
from metrics import report
from sklearn.tree import DecisionTreeClassifier
from pickle import dump

def ridge_classifier():
    print("Ridge Classifier :")
    rid_clf = make_pipeline(
    StandardScaler(),
        RidgeClassifier(alpha=1.0)
    )

    rid_clf.fit(Xtrain, Ytrain)
    predictions = rid_clf.predict(Xtest)
    results = report(Ytest, predictions)
    print_data(results)

    with open("models/rid_clf.pkl", "wb") as f:
        dump(rid_clf, f, protocol=5)

def decision_tree():
    print("\nDecision Tree Classifier :")
    dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_clf.fit(Xtrain, Ytrain)

    predictions = dt_clf.predict(Xtest)
    results = report(Ytest, predictions)

    print_data(results)

    with open("models/dt_clf.pkl", "wb") as f:
        dump(dt_clf, f, protocol=5)

def manual_classifier():
    print("Manual Classifier :")
    clf = ManualClassifier()

    clf.fit(Xtrain, Ytrain)
    predictions = clf.predict(Xtest)
    results = report(Ytest, predictions)

    print_data(results)

    with open("models/man_clf.pkl", "wb") as f:
        dump(clf, f, protocol=5)

def print_data(res):
    print(res["cm"])
    print("\nAccuracy :", res["accuracy"])
    for i, nom in enumerate(noms):
        print(f"\n--- {nom} ---")
        print("Précision :", res["precision"][i])
        print("Rappel :", res["recall"][i])
        print("Spécificité :", res["specificity"][i])
        print("F1-score :", res["f1_score"][i])

if __name__ == "__main__":
    CWD = pathlib.Path(__file__).parent
    DATA_PATH = CWD / "datasets" / "manchots.csv"

    # Chargement et préparation
    data = load_data(DATA_PATH)
    data = clean_data(data)

    Xtrain, Xtest, Ytrain, Ytest = split_data(
        data,
        test_size=0.2,
        seed=42
    )

    noms = ["Adélie", "Papou", "Jugulaire"]

    manual_classifier()
    decision_tree()
    ridge_classifier()