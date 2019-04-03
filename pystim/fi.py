import math
import numpy as np
import os
import scipy as sp
import scipy.interpolate
import tempfile
import tqdm
import warnings

import pystim.datasets.van_hateren as vh

from pystim.images.png import create as create_png_image
from pystim.images.png import load as load_png_image
from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
# from pystim.io.csv import open_file as open_csv_file
from pystim.utils import compute_horizontal_angles
from pystim.utils import compute_vertical_angles
from pystim.utils import float_frame_to_uint8_frame
from pystim.utils import get_grey_frame
from pystim.utils import handle_arguments_and_configurations


name = 'fi'

default_configuration = {
    'path': os.path.join(tempfile.gettempdir(), "pystim", name),
    # 'image_nbs': [1, 2],  # TODO remove?
    'eye_diameter': 1.2e-2,  # m
    # 'eye_diameter': 1.2e-2,  # m  # human
    # 'eye_diameter': 2.7e-3,  # m  # axolotl
    'normalized_value_median': 0.5,
    'normalized_value_mad': 0.01,
    'display_rate': 40.0,  # Hz
    'frame': {
        'width': 864,  # px
        'height': 864,  # px
        'duration': 0.3,  # s
        'resolution': 3.5e-6,  # m / pixel  # fixed by the setup
    },
    'seed': 42,
    'verbose': True,
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    base_path = config['path']
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    print("Generation in: {}".format(base_path))

    images_path = os.path.join(base_path, "images")
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    # Get configuration parameters.
    normalized_value_median = config['normalized_value_median']
    normalized_value_mad = config['normalized_value_mad']
    frame_resolution = config['frame']['resolution']
    frame_width = config['frame']['width']
    frame_height = config['frame']['height']
    eye_diameter = config['eye_diameter']
    seed = config['seed']
    verbose = config['verbose']
    # TODO complete?

    # Fetch van Hateren images.
    vh.fetch(download_if_missing=False, verbose=verbose)

    # Select unsaturated van Hateren image numbers.
    vh_image_nbs = vh.get_image_nbs()
    are_saturated = vh.get_are_saturated(verbose=verbose)
    are_unsaturated = np.logical_not(are_saturated)
    assert vh_image_nbs.size == are_unsaturated.size, "{} != {}".format(vh_image_nbs.size, are_unsaturated.size)
    unsaturated_vh_image_nbs = vh_image_nbs[are_unsaturated]

    # Extract images from the van Hateren images.
    for vh_image_nb in tqdm.tqdm(unsaturated_vh_image_nbs):
        # Check if image already exists.
        image_filename = "image_{i:04d}.png".format(i=vh_image_nb)
        image_path = os.path.join(images_path, image_filename)
        if os.path.isfile(image_path):
            continue
        # Cut out central sub-region.
        a_x = vh.get_horizontal_angles()
        a_y = vh.get_vertical_angles()
        luminance_data = vh.load_luminance_data(vh_image_nb)
        rbs = sp.interpolate.RectBivariateSpline(a_x, a_y, luminance_data)
        angular_resolution = math.atan(frame_resolution / eye_diameter) * (180.0 / math.pi)
        a_x = compute_horizontal_angles(width=frame_width, angular_resolution=angular_resolution)
        a_y = compute_vertical_angles(height=frame_height, angular_resolution=angular_resolution)
        a_x, a_y = np.meshgrid(a_x, a_y)
        luminance_data = rbs.ev(a_x, a_y)
        luminance_data = luminance_data.transpose()
        # Prepare image data.
        luminance_median = np.median(luminance_data)
        luminance_mad = np.median(np.abs(luminance_data - luminance_median))
        centered_luminance_data = luminance_data - luminance_median
        if luminance_mad > 0.0:
            centered_n_reduced_luminance_data = centered_luminance_data / luminance_mad
        else:
            centered_n_reduced_luminance_data = centered_luminance_data
        normalized_luminance_data = centered_n_reduced_luminance_data
        scaled_data = normalized_value_mad * normalized_luminance_data
        shifted_n_scaled_data = scaled_data + normalized_value_median
        data = shifted_n_scaled_data
        if np.count_nonzero(data < 0.0) > 0:
            string = "some pixels are negative in image {} (consider changing the configuration, 'normalized_value_mad': {})"
            message = string.format(vh_image_nb, normalized_value_mad / ((normalized_value_median - np.min(data)) / (normalized_value_median - 0.0)))
            warnings.warn(message)
        data[data < 0.0] = 0.0
        if np.count_nonzero(data > 1.0) > 0:
            string = "some pixels saturate in image {} (consider changing the configuration, 'normalized_value_mad': {})"
            message = string.format(vh_image_nb, normalized_value_mad / ((np.max(data) - normalized_value_median) / (1.0 - normalized_value_median)))
            warnings.warn(message)
        data[data > 1.0] = 1.0
        data = np.array(254.0 * data, dtype=np.uint8)  # 0.0 -> 0 and 1.0 -> 254 such that 0.5 -> 127
        data = np.transpose(data)
        data = np.flipud(data)
        image = create_png_image(data)
        # Save image.
        image.save(image_path)

    # Get image numbers and paths.
    image_nbs = []
    image_paths = {}
    for k, vh_image_nb in enumerate(unsaturated_vh_image_nbs):
        image_nb = k
        image_filename = "image_{i:04d}.png".format(i=vh_image_nb)
        image_path = os.path.join(images_path, image_filename)
        assert image_nb not in image_nbs
        assert os.path.isfile(image_path)
        image_nbs.append(image_nb)
        image_paths[image_nb] = image_path
    image_nbs = np.array(image_nbs)
    nb_images = len(image_nbs)

    # Set image ordering of each repetition.
    nb_repetitions = 5  # i.e. 5 x ~3000 images -> 5 x ~15 min = 1 h 15 min
    repetition_orderings = {}
    np.random.seed(seed)
    for repetition_nb in range(0, nb_repetitions):
        ordering = np.copy(image_nbs)
        np.random.shuffle(ordering)
        repetition_orderings[repetition_nb] = ordering

    # Create .bin file.
    nb_bin_images = nb_images  # TODO correct (add grey image).
    bin_filename = "fi.bin"
    bin_path = os.path.join(base_path, bin_filename)
    if not os.path.isfile(bin_path):
        print("Start creating .bin file...")
        # # Open .bin file.
        bin_file = open_bin_file(bin_path, nb_bin_images, frame_width=frame_width, frame_height=frame_height)
        # TODO add grey image.
        # # Add grey frame.
        grey_frame =  get_grey_frame(width=frame_width, height=frame_height, luminance=0.5)
        grey_frame = float_frame_to_uint8_frame(grey_frame)
        bin_file.append(grey_frame)
        # # Add images.
        for image_nb in tqdm.tqdm(image_nbs):
            image_path = image_paths[image_nb]
            image = load_png_image(image_path)
            frame = image.data
            bin_file.append(frame)
        # # Close .bin file.
        bin_file.close()
        #
        print("End of .bin file creation.")
    else:
        print("'{}' already exists.".format(bin_path))

    # Create .vec file.
    print("Start creating .vec file...")
    nb_displays = None  # TODO correct.
    # # Open .vec file.
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(base_path, vec_filename)
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    # TODO adaptation.
    for repetition_nb in range(0, nb_repetitions):
        pass  # TODO complete.
    # # Close .vec file.
    vec_file.close()
    #
    print("End of .vec file creation.")

    # TODO generate the repetition file.

    return
