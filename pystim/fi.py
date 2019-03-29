import os
import tempfile

from pystim.datasets.van_hateren import fetch as fetch_van_hateren
from pystim.datasets.van_hateren import load_image as load_van_hateren_image
from pystim.utils import handle_arguments_and_configurations


name = 'fi'

default_configuration = {
    'path': os.path.join(tempfile.gettempdir(), "pystim", name),
    'image_nbs': [1, 2],
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    path = config['path']
    if not os.path.isdir(path):
        os.makedirs(path)
    print(path)

    images_path = os.path.join(path, "images")
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    # TODO get configuration parameters.
    image_nbs = config['image_nbs']

    # TODO collect images (unprepared).
    fetch_van_hateren(image_nbs=image_nbs)

    for image_nb in image_nbs:
        image = load_van_hateren_image(image_nb)
        image = image.prepare()  # TODO convert from iml to png and rescale image.
        image_filename = "image_{i:04d}.png".format(i=image_nb)
        image_path = os.path.join(images_path, image_filename)
        image.save(image_path)

    raise NotImplementedError  # TODO complete.
