import pandas as pd


def open_file(path, columns=[], dtype=None):

    file = CSVFile(path, columns=columns, dtype=dtype)

    return file


class CSVFile:

    def __init__(self, path, columns=[], dtype=None):

        self._path = path
        self._columns = columns
        self._dtype = dtype

        self._list = []

    def append(self, **kwargs):

        self._list.append(kwargs)

        return

    def close(self):

        df = pd.DataFrame(self._list, columns=self._columns, dtype=self._dtype)
        df.to_csv(self._path)

        return
