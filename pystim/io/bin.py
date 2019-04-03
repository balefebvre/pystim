import numpy as np
import os


def open_file(path, nb_images, frame_width=768, frame_height=768):

    file = BinFile(path, nb_images, frame_width=frame_width, frame_height=frame_height)

    return file


class BinFile:

    def __init__(self, path, nb_images, frame_width=768, frame_height=768):

        self._path = path
        self._nb_images = nb_images
        self._frame_width = frame_width
        self._frame_height = frame_height
        # self._frame_width = 600
        # self._frame_height = 600

        self._file = open(self._path, mode='w+b',)
        self._nb_bits = 8
        self._frame_nb = -1

        self._write_header()

    def get_frame_nb(self):

        return self._frame_nb

    def _write_header(self):

        header_list = [
            self._frame_width,
            self._frame_height,
            self._nb_images,
            self._nb_bits,
        ]
        print("header_list: {}".format(header_list))
        header_array = np.array(header_list, dtype=np.int16)
        header_bytes = header_array.tobytes()
        self._file.write(header_bytes)

        return

    def append(self, frame):

        assert frame.dtype == np.uint8, "frame.dtype: {}".format(frame.dtype)

        frame_bytes = frame.tobytes()
        self._file.write(frame_bytes)
        self._frame_nb += 1

        return

    def flush(self):

        os.fsync(self._file.fileno())  # force write

        return

    def close(self):

        self.flush()
        self._file.close()

        return
