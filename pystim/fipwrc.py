"""Flashed images perturbed with random checkerboards"""

import os
import tempfile


name = 'fipwrc'

default_configuration = {
    'path': os.path.join(tempfile.gettempdir(), 'pystim', name)
}


def generate(args):

    raise NotImplementedError  # TODO complete.
