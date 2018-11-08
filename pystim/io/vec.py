import collections
import numpy as np
import os


def open_file(path, nb_displays):

    file = VecFile(path, nb_displays)

    return file


class VecFile:

    def __init__(self, path, nb_displays):

        self._path = path
        self._nb_displays = nb_displays

        self._file = open(self._path, mode='w')

        self._write_header()

    def _write_header(self):

        string = "0 {} 0 0 0\n".format(self._nb_displays)
        self._file.write(string)
        self._file.flush()

        return

    def append(self, object_):

        if isinstance(object_, str):
            line = object_
            string = "{}\n".format(line)
            self._file.write(string)
        elif isinstance(object_, (int, np.integer)):
            frame_id = object_
            string = "0 {} 0 0 0\n".format(frame_id)
            self._file.write(string)
        elif isinstance(object_, collections.Iterable):
            iterable = object_
            for object_ in iterable:
                self.append(object_)
        else:
            raise TypeError("unexpected object type: {}".format(type(object_)))

        return

    def close(self):

        os.fsync(self._file.fileno())  # force write
        self._file.close()

        return
