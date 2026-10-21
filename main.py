from dataset import ManchotDataset
import pathlib

CWD = pathlib.Path(__file__).parent
DATA_PATH = CWD / "datasets"
dataset = ManchotDataset(path=DATA_PATH)

print(dataset.X)
print(dataset.Y)