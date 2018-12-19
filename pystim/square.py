import os
import tempfile

from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.utils import handle_arguments_and_configurations

name = 'square'

default_configuration = {
    'frame': {
        'width': 2000.0,  # µm
        'height': 2000.0,  # µm
        'rate': 60.0, # Hz
    },
    'size': float(16) * 30.0,  # µm
    'path': os.path.join(tempfile.gettempdir(), 'pystim', name)
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    size = config['size']
    path = config['path']

    pixel_size = 3.75  # µm

    # Create output directory (if necessary).
    if not os.path.isdir(path):
        os.makedirs(path)
    print(path)

    # TODO create white frame.
    # TODO create black frame.
    nb_images = 2

    # TODO create .bin file.
    bin_filename = "{}.bin".format(name)
    bin_path = os.path.join(path, bin_filename)
    bin_file = open_bin_file(bin_path, nb_images, frame_width=image_width, frame_height=image_height)
    # TODO complete.
    bin_file.close()

    nb_displays = 0  # TODO correct.

    # TODO create .vec file.
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(path, vec_filename)
    vec_file = open_vec_file(vec_path, nb_displays)
    # TODO complete.
    vec_file.close()

    return
