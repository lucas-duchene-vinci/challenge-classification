import csv
import pathlib

import numpy as np

_COLUMNS = [
    "species",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex"
]

class ManchotDataset:
    def __init__(self, path: pathlib.Path):
        if not path.exists():
            raise ValueError("Cannot instanciate dataset: invalid path.")

        self._path = path
        self._data: list[list[float | None]] = []

        with (self._path / "manchots.csv").open("r", encoding="utf-8-sig") as fp:
            reader = csv.reader(fp)
            next(reader, None)

            for row in reader:
                self._add_row(row)

    def _add_row(self, row: list[str]) -> None:
        _row = []
        for v in row:
            if v == '?':
                continue
            else:
                _row.append(float(v))
        self._data.append(_row)

    def select_column(self, column: int) -> list[float | None]:
        return [row[column] for row in self._data]

    @property
    def n_rows(self) -> int:
        return len(self._data)

    @property
    def X(self) -> list[list[float | None]]:
        return [row[:-1] for row in self._data]

    @property
    def Y(self) -> list[float | None]:
        return [row[-1] for row in self._data]

    @staticmethod
    def column_number(name: str) -> int:
        return _COLUMNS.index(name)

    @property
    def asarray(self) -> np.ndarray:
        return np.array([row
                         for row in self._data
                         if not any(value is None for value in row)])

    def __str__(self):
        return f"Manchot Dataset: {self.n_rows} elements"
