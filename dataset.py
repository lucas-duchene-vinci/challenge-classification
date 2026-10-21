import csv
import pathlib

import numpy as np
from sklearn.model_selection import train_test_split

_COLUMNS = [
    "species",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex"
]

def load_data(path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", skip_header=1)

def clean_data(data) -> np.ndarray:
    data = data[~np.isnan(data).any(axis=1)]
    return data

def split_data(data, test_size: float, seed: int) -> tuple:
    X = data[:, 1:]
    Y = data[:, 0]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=seed)
    return Xtrain, Xtest, Ytrain, Ytest

def save_data(path, Xtrain, Xtest, Ytrain, Ytest):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest)

def load_data_split(path) -> tuple:
    d = np.load(path)
    return d["Xtrain"], d["Xtest"], d["Ytrain"], d["Ytest"]

# ------------------------------------------------------------------
# Partie désactivée volontairement — conservée pour historique
# ------------------------------------------------------------------
# class ManchotDataset:
#     def __init__(self, path: pathlib.Path):
#         if not path.exists():
#             raise ValueError("Cannot instanciate dataset: invalid path.")
#
#         self._path = path
#         self._data: list[list[float | None]] = []
#
#         with (self._path / "manchots.csv").open("r", encoding="utf-8-sig") as fp:
#             reader = csv.reader(fp)
#             next(reader, None)
#
#             for row in reader:
#                 self._add_row(row)
#
#     def _add_row(self, row: list[str]) -> None:
#         _row = []
#         for v in row:
#             if v == '?':
#                 continue
#             else:
#                 _row.append(float(v))
#         self._data.append(_row)
#
#     def select_column(self, column: int) -> list[float | None]:
#         return [row[column] for row in self._data]
#
#     @property
#     def n_rows(self) -> int:
#         return len(self._data)
#
#     @property
#     def X(self) -> list[list[float | None]]:
#         return [row[:-1] for row in self._data]
#
#     @property
#     def Y(self) -> list[float | None]:
#         return [row[-1] for row in self._data]
#
#     @staticmethod
#     def column_number(name: str) -> int:
#         return _COLUMNS.index(name)
#
#     @property
#     def asarray(self) -> np.ndarray:
#         return np.array([row
#                          for row in self._data
#                          if not any(value is None for value in row)])
#
#     def __str__(self):
#         return f"Manchot Dataset: {self.n_rows} elements"