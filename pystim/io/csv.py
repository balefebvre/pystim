import pandas as pd


def open_file(path, columns=[]):

    file = CSVFile(path, columns=columns)

    return file


class CSVFile:

    def __init__(self, path, columns=[]):

        self._path = path
        self._columns = columns

        self._list = []

    def append(self, **kwargs):

        self._list.append(kwargs)

        return

    def close(self):

        df = pd.DataFrame(self._list, columns=self._columns)
        df.to_csv(self._path)

        return
