import pathlib

from dataset import load_data, clean_data, split_data
from classifier import ManualClassifier

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

# Classification
classifier = ManualClassifier()

classifier.fit(Xtrain, Ytrain)

predictions = classifier.predict(Xtest)

# Évaluation
accuracy = classifier.score(Xtest, Ytest)
print(f"Accuracy : {accuracy:.2%}")