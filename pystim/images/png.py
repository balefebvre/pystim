import numpy as np
import PIL.Image

from .base import Image


class PNGImage(Image):

    @classmethod
    def load(cls, path):

        raise NotImplementedError

    def __init__(self, data):

        super().__init__()

        self._data = data

    @property
    def data(self):

        return self._data

    @property
    def min(self):

        return np.min(self._data)

    @property
    def max(self):

        return np.max(self._data)

    def save(self, path):

        # mode = 'L'  # i.e. 8-bit pixels, black and white
        # image = PIL.Image.fromarray(self._data, mode=mode)
        image = PIL.Image.fromarray(self._data)
        image.save(path)

        return


create = PNGImage
load = PNGImage.load
