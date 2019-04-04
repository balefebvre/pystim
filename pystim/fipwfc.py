"""Flashed images perturbed with frozen checkerboards"""

import math
import numpy as np
import os
import scipy as sp
import scipy.interpolate
import tempfile

from pystim.datasets import fetch as fetch_image
from pystim.datasets import get as get_dataset
from pystim.images.png import create as create_png_image
from pystim.utils import compute_horizontal_angles
from pystim.utils import compute_vertical_angles
from pystim.utils import handle_arguments_and_configurations


name = 'fipwfc'

default_configuration = {
    'path': os.path.join(tempfile.gettempdir(), 'pystim', name),
    'images': {
        0: ('grey', 127),
        1: ('van Hateren', 5),
        2: ('van Hateren', 31),
        3: ('van Hateren', 46),
        4: ('van Hateren', 39),
    },
    'eye_diameter': 1.2e-2,  # m
    # 'eye_diameter': 1.2e-2,  # m  # human
    # 'eye_diameter': 2.7e-3,  # m  # axolotl
    'frame': {
        'width': 864,  # px
        'height': 864,  # px
        'duration': 0.3,  # s
        'resolution': 3.5e-6,  # m / pixel  # fixed by the setup
    },
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    base_path = config['path']
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    print("Generation in {}.".format(base_path))

    # TODO create directories (if necessary).
    images_path = os.path.join(base_path, 'images')
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    # TODO get configuration parameters.
    image_keys = config['images']
    eye_diameter = config['eye_diameter']
    frame_resolution = config['frame']['resolution']
    frame_width = config['frame']['width']
    frame_height = config['frame']['height']

    # TODO fetch images.
    image_nbs = np.array(list(image_keys.keys()), dtype=int)
    for image_nb in image_nbs:
        image_key = image_keys[str(image_nb)]
        fetch_image(*image_key)

    # TODO extract images.
    for image_nb in image_nbs:
        # Check if image already exists.
        image_filename = "image_{i:01d}.png".format(i=image_nb)
        image_path = os.path.join(images_path, image_filename)
        if os.path.isfile(image_path):
            continue
        # Load image data.
        image_key = image_keys[str(image_nb)]
        dataset_name = image_key[0]
        dataset = get_dataset(dataset_name)
        data = dataset.load_data(*image_key[1:])
        # TODO cut out central sub-regions.
        a_x = dataset.get_horizontal_angles()
        a_y = dataset.get_vertical_angles()
        print(a_x.shape)
        print(a_y.shape)
        print(data.shape)
        rbs = sp.interpolate.RectBivariateSpline(a_x, a_y, data)
        angular_resolution = math.atan(frame_resolution / eye_diameter) * (180.0 / math.pi)
        a_x = compute_horizontal_angles(width=frame_width, angular_resolution=angular_resolution)
        a_y = compute_vertical_angles(height=frame_height, angular_resolution=angular_resolution)
        a_x, a_y = np.meshgrid(a_x, a_y)
        data = rbs.ev(a_x, a_y)
        data = data.transpose()
        # Prepare image data.
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        # mad = np.std(data)
        centered_data = data - median
        if mad > 1.e-13:
            centered_n_reduced_data = centered_data / mad
        else:
            centered_n_reduced_data = centered_data
        normalized_data = centered_n_reduced_data
        scaled_data = 0.1 * normalized_data
        shifted_n_scaled_data = scaled_data + 0.5
        data = shifted_n_scaled_data
        data[data < 0.0] = 0.0
        data[data > 1.0] = 1.0
        data = np.array(254.0 * data, dtype=np.uint8)  # 0.0 -> 0 and 1.0 -> 254 such that 0.5 -> 127
        data = np.transpose(data)
        data = np.flipud(data)
        image = create_png_image(data)
        image.save(image_path)

    raise NotImplementedError  # TODO complete

# TODO complete.
