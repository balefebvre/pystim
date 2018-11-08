import numpy as np
import os


def open_file(path, nb_images):

    file = BinFile(path, nb_images)

    return file


class BinFile:

    def __init__(self, path, nb_images):

        self._path = path
        self._nb_images = nb_images

        self._file = open(self._path, mode='w+b',)
        self._dmd_width = 1080
        self._dmd_height = 1920
        self._nb_bits = 8

        self._write_header()

    def _write_header(self):

        header_list = [
            self._dmd_width,
            self._dmd_height,
            self._nb_images,
            self._nb_bits,
        ]
        header_array = np.array(header_list, dtype=np.uint16)
        header_bytes = header_array.tobytes()
        self._file.write(header_bytes)

        return

    def append(self, frame):

        assert frame.dtype == np.uint8, "frame.dtype: {}".format(frame.dtype)

        frame_bytes = frame.tobytes()
        self._file.write(frame_bytes)

        return

    def close(self):

        os.fsync(self._file.fileno())  # force write
        self._file.close()

        return
