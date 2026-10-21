import pathlib
from pickle import load

from dataset import load_data, clean_data, split_data
from main import print_data
from metrics import report

with open("models/rid_clf.pkl", "rb") as f:
    clf = load(f)

CWD = pathlib.Path(__file__).parent
DATA_PATH = CWD / "datasets" / "manchots_test.csv"

# Chargement et préparation
data = load_data(DATA_PATH)
data = clean_data(data)

Xtrain, Xtest, Ytrain, Ytest = split_data(
    data,
    test_size=0.25,
    seed=42
)

predictions = clf.predict(Xtest)
results = report(Ytest, predictions)
print_data(results)