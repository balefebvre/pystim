"""Flashed images perturbed with frozen checkerboards"""

import collections
import math
import numpy as np
import os
import scipy as sp
import scipy.interpolate
import tempfile
import tqdm

import pystim.datasets.checkerboard as cb

from pystim.datasets import fetch as fetch_image
from pystim.datasets import get as get_dataset
from pystim.datasets.checkerboard import fetch as fetch_patterns
from pystim.datasets.checkerboard import load_data as load_checkerboard_data
from pystim.images.png import create as create_png_image
from pystim.images.png import load as load_png_image
from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.io.csv import open_file as open_csv_file
from pystim.utils import compute_horizontal_angles
from pystim.utils import compute_vertical_angles
from pystim.utils import float_frame_to_uint8_frame
from pystim.utils import get_grey_frame
from pystim.utils import handle_arguments_and_configurations


name = 'fipwfc'

default_configuration = {
    'path': os.path.join(tempfile.gettempdir(), 'pystim', name),
    'images': {
        0: ('grey', 127),
        1: ('van Hateren', 5),
        2: ('van Hateren', 31),
        # 3: ('van Hateren', 46),  # saturated image
        3: ('van Hateren', 2219),
        # TODO comment the following line.
        # 4: ('van Hateren', 39),
    },
    'perturbations': {
        # 'pattern_nbs': list(range(0, 2)),
        # 'pattern_nbs': list(range(0, 18)),
        'pattern_nbs': list(range(0, 5)),
        'nb_horizontal_checks': 56,
        'nb_vertical_checks': 56,
        # 'amplitudes': [float(a) / float(256) for a in [10, 28]],  # TODO remove?
        # 'amplitudes': [float(a) / float(256) for a in [-28, +28]],  # TODO remove?
        # 'amplitudes': [float(a) / float(256) for a in [2, 4, 7, 10, 14, 18, 23, 28]],  # TODO remove?
        # 'amplitudes': [float(a) / float(256) for a in [-30, -20, -15, -10, +10, +15, +20, +30]],
        # 'amplitudes': [-15.0, +15.0],
        # 'amplitudes': [-30.0, -20.0, -15.0, -10.0, +10.0, +15.0, +20.0, +30.0],
        'amplitudes': [-22.5, -15.0, -7.5, +7.5, +15.0, +22.5],
        # 'resolution': 50.0,  # µm / pixel
        'resolution': float(15) * 3.5,  # µm / pixel
    },
    'eye_diameter': 1.2e-2,  # m
    # 'eye_diameter': 1.2e-2,  # m  # human
    # 'eye_diameter': 2.7e-3,  # m  # axolotl
    'display_rate': 40.0,  # Hz
    'adaptation_duration': 60.0,  # s
    'flash_duration': 0.3,  # s
    'inter_flash_duration': 0.3,  # s
    'frame': {
        'width': 864,  # px
        'height': 864,  # px
        'resolution': 3.5e-6,  # m / pixel  # fixed by the setup
    },
    'nb_unperturbed_flashes_per_image': 5,
    # 'nb_repetitions': 2,
    # 'nb_repetitions': 10,
    'nb_repetitions': 20,
    'seed': 42,
    'force': True,
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    base_path = config['path']
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    print("Generation in {}.".format(base_path))

    # Create directories (if necessary).
    images_dirname = 'images'
    images_path = os.path.join(base_path, images_dirname)
    if not os.path.isdir(images_path):
        os.makedirs(images_path)
    patterns_dirname = 'patterns'
    patterns_path = os.path.join(base_path, patterns_dirname)
    if not os.path.isdir(patterns_path):
        os.makedirs(patterns_path)
    frames_dirname = 'frames'
    frames_path = os.path.join(base_path, frames_dirname)
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)

    # Get configuration parameters.
    image_keys = config['images']
    pattern_nbs = config['perturbations']['pattern_nbs']
    amplitude_values = config['perturbations']['amplitudes']
    eye_diameter = config['eye_diameter']
    display_rate = config['display_rate']
    adaptation_duration = config['adaptation_duration']
    flash_duration = config['flash_duration']
    inter_flash_duration = config['inter_flash_duration']
    frame_width = config['frame']['width']
    frame_height = config['frame']['height']
    frame_resolution = config['frame']['resolution']
    nb_unperturbed_flashes_per_image = config['nb_unperturbed_flashes_per_image']
    nb_repetitions = config['nb_repetitions']
    seed = config['seed']
    force = config['force']

    # Fetch images.
    image_nbs = np.array(list(image_keys.keys()), dtype=int)
    for image_nb in image_nbs:
        image_key = image_keys[str(image_nb)]
        fetch_image(*image_key)
    # nb_images = len(image_nbs)

    # Fetch patterns.
    pattern_nbs = np.array(pattern_nbs)
    fetch_patterns(image_nbs=pattern_nbs)
    nb_patterns = len(pattern_nbs)

    # Fetch amplitudes.
    amplitude_nbs = np.arange(0, len(amplitude_values))
    nb_amplitudes = len(amplitude_nbs)

    # Prepare image parameters.
    images_params = collections.OrderedDict()
    for image_nb in image_nbs:
        filename = 'image_{nb:01d}_data.npy'.format(nb=image_nb)
        path = os.path.join(images_dirname, filename)
        images_params[image_nb] = collections.OrderedDict([
            ('path', path)
        ])

    # Prepare pattern parameters.
    patterns_params = collections.OrderedDict()
    for pattern_nb in pattern_nbs:
        filename = 'pattern_{nb:01d}_data.npy'.format(nb=pattern_nb)
        path = os.path.join(patterns_dirname, filename)
        patterns_params[pattern_nb] = collections.OrderedDict([
            ('path', path)
        ])

    def get_image_data(image_nb):

        # Load image data.
        image_key = image_keys[str(image_nb)]
        dataset_name = image_key[0]
        dataset = get_dataset(dataset_name)
        data = dataset.load_data(*image_key[1:])
        # Cut out central sub-regions.
        a_x = dataset.get_horizontal_angles()
        a_y = dataset.get_vertical_angles()
        rbs = sp.interpolate.RectBivariateSpline(a_x, a_y, data, kx=1, ky=1)
        angular_resolution = math.atan(frame_resolution / eye_diameter) * (180.0 / math.pi)
        a_x = compute_horizontal_angles(width=frame_width, angular_resolution=angular_resolution)
        a_y = compute_vertical_angles(height=frame_height, angular_resolution=angular_resolution)
        a_x, a_y = np.meshgrid(a_x, a_y)
        data = rbs.ev(a_x, a_y)
        data = data.transpose()
        # TODO remove the following lines (deprecated)?
        # # Prepare image data.
        # median = np.median(data)
        # mad = np.median(np.abs(data - median))
        # # mad = np.std(data)
        # centered_data = data - median
        # if mad > 1.e-13:
        #     centered_n_reduced_data = centered_data / mad
        # else:
        #     centered_n_reduced_data = centered_data
        # normalized_data = centered_n_reduced_data
        # scaled_data = 0.1 * normalized_data
        # shifted_n_scaled_data = scaled_data + 0.5
        # data = shifted_n_scaled_data
        # TODO keep the following normalization?
        # # Prepare image data.
        # mean = np.mean(data)
        # scaled_data = data / mean if mean > 0.0 else data
        # shifted_n_scaled_data = 0.2 * scaled_data  # TODO correct?
        # data = shifted_n_scaled_data
        # TODO keep the following normalization?
        mean_luminance = 0.25
        std_luminance = 0.05
        luminance_data = data
        log_luminance_data = np.log(1.0 + luminance_data)
        log_mean_luminance = np.mean(log_luminance_data)
        log_std_luminance = np.std(log_luminance_data)
        normalized_log_luminance_data = log_luminance_data - log_mean_luminance
        if log_std_luminance > 1e-13:
            normalized_log_luminance_data = normalized_log_luminance_data / log_std_luminance
        normalized_log_luminance_data = 0.2 * normalized_log_luminance_data
        normalized_luminance_data = np.exp(normalized_log_luminance_data) - 1.0
        normalized_luminance_data = normalized_luminance_data - np.mean(normalized_luminance_data)
        if np.std(normalized_luminance_data) > 1e-13:
            normalized_luminance_data = normalized_luminance_data / np.std(normalized_luminance_data)
        luminance_data = std_luminance * normalized_luminance_data + mean_luminance
        # Save image data.
        image_data_path = os.path.join(base_path, images_params[image_nb]['path'])
        np.save(image_data_path, luminance_data)
        # Prepare image.
        data = np.copy(luminance_data)
        data[data < 0.0] = 0.0
        data[data > 1.0] = 1.0
        data = np.array(254.0 * data, dtype=np.uint8)  # 0.0 -> 0 and 1.0 -> 254 such that 0.5 -> 127
        data = np.transpose(data)
        data = np.flipud(data)
        image = create_png_image(data)
        # Save image.
        image_image_filename = "image_{nb:01d}_image.png".format(nb=image_nb)
        image_image_path = os.path.join(images_path, image_image_filename)
        image.save(image_image_path)

        return luminance_data

    def get_pattern_data(pattern_nb):

        # Load pattern data.
        data = load_checkerboard_data(pattern_nb, with_borders=0.5)
        data = 2.0 * (data - 0.5) / 254.0
        # Save pattern data.
        pattern_data_path = os.path.join(base_path, patterns_params[pattern_nb]['path'])
        np.save(pattern_data_path, data[1:-1, 1:-1])  # without borders
        # Project.
        # a_x = cb.get_horizontal_angles()
        # a_y = cb.get_vertical_angles()
        # rbs = sp.interpolate.RectBivariateSpline(a_x, a_y, data, kx=1, ky=1)
        a_x = cb.get_horizontal_angles(with_borders=True)
        a_y = cb.get_vertical_angles(with_borders=True)
        a_x, a_y = np.meshgrid(a_x, a_y)
        a = np.stack((np.ravel(a_x), np.ravel(a_y)), axis=1)  # i.e. stack along 2nd axis
        ni = sp.interpolate.NearestNDInterpolator(a, np.ravel(data))
        angular_resolution = math.atan(frame_resolution / eye_diameter) * (180.0 / math.pi)
        a_x = compute_horizontal_angles(width=frame_width, angular_resolution=angular_resolution)
        a_y = compute_vertical_angles(height=frame_height, angular_resolution=angular_resolution)
        a_x, a_y = np.meshgrid(a_x, a_y)
        # data = rbs.ev(a_x, a_y)
        # data = data.transpose()
        a = np.stack((np.ravel(a_x), np.ravel(a_y)), axis=1)  # i.e. stack along 2nd axis
        data = ni(a)
        shape = (frame_width, frame_height)
        data = np.reshape(data, shape)
        data = data.transpose()
        # Create pattern image.
        image_data = data
        image_data = 254.0 * image_data
        image_data = np.array(image_data, dtype=np.uint8)  # 0.0 -> 0 and 1.0 -> 254 such that 0.5 -> 127
        image_data = np.transpose(image_data)
        image_data = np.flipud(image_data)
        pattern = create_png_image(image_data)
        # Save pattern.
        pattern_image_filename = "pattern_{nb:01d}_image.png".format(nb=pattern_nb)
        pattern_image_path = os.path.join(patterns_path, pattern_image_filename)
        pattern.save(pattern_image_path)

        return data

    def get_frame_path(image_nb, pattern_nb, amplitude_nb):

        if nb_unperturbed_flashes_per_image == 0:
            frame_nb = ((image_nb * nb_patterns) + pattern_nb) * nb_amplitudes + amplitude_nb
        else:
            if pattern_nb is None and amplitude_nb is None:
                frame_nb = image_nb + (image_nb * nb_patterns) * nb_amplitudes
            else:
                frame_nb = (image_nb + 1) + ((image_nb * nb_patterns) + pattern_nb) * nb_amplitudes + amplitude_nb
        filename = "frame_{nb:01d}.png".format(nb=frame_nb)
        path = os.path.join(frames_path, filename)

        return path

    # Set condition parameters.
    condition_nb = 0
    conditions_params = collections.OrderedDict()
    frame_paths = collections.OrderedDict()
    unperturbed_condition_nbs = []
    for image_nb in image_nbs:
        if nb_unperturbed_flashes_per_image > 0:
            assert condition_nb not in conditions_params
            conditions_params[condition_nb] = collections.OrderedDict([
                ('image_nb', image_nb),
                ('pattern_nb', None),
                ('amplitude_nb', None),
            ])
            frame_paths[condition_nb] = get_frame_path(image_nb, None, None)
            unperturbed_condition_nbs.append(condition_nb)
            condition_nb += 1
        for pattern_nb in pattern_nbs:
            for amplitude_nb in amplitude_nbs:
                assert condition_nb not in conditions_params
                conditions_params[condition_nb] = collections.OrderedDict([
                    ('image_nb', image_nb),
                    ('pattern_nb', pattern_nb),
                    ('amplitude_nb', amplitude_nb),

                ])
                frame_paths[condition_nb] = get_frame_path(image_nb, pattern_nb, amplitude_nb)
                condition_nb += 1
    condition_nbs = np.array(list(conditions_params.keys()))
    unperturbed_condition_nbs = np.array(unperturbed_condition_nbs)
    nb_conditions = len(condition_nbs)
    nb_unperturbed_conditions = len(unperturbed_condition_nbs)

    # Create frames.
    image_data = None
    for image_nb in tqdm.tqdm(image_nbs):
        if nb_unperturbed_flashes_per_image > 0:
            frame_path = get_frame_path(image_nb, None, None)
            if not os.path.isfile(frame_path) or force:
                # Get image data.
                image_data = get_image_data(image_nb)
                # Create frame data.
                data = image_data
                # Create frame image.
                data[data < 0.0] = 0.0
                data[data > 1.0] = 1.0
                data = np.array(254.0 * data, dtype=np.uint8)  # 0.0 -> 0 and 1.0 -> 254 such that 0.5 -> 127
                data = np.transpose(data)
                data = np.flipud(data)
                image = create_png_image(data)
                # Save frame image.
                image.save(frame_path)
        pattern_data = None
        for pattern_nb in tqdm.tqdm(pattern_nbs):
            for amplitude_nb in tqdm.tqdm(amplitude_nbs):
                frame_path = get_frame_path(image_nb, pattern_nb, amplitude_nb)
                if os.path.isfile(frame_path) and not force:
                    continue
                # Get image data.
                if image_data is None:
                    image_data = get_image_data(image_nb)
                # Get pattern data.
                if pattern_data is None:
                    pattern_data = get_pattern_data(pattern_nb)
                # Get amplitude value.
                amplitude_value = amplitude_values[amplitude_nb]
                # Create frame data.
                data = image_data + amplitude_value * pattern_data
                # Create frame image.
                data[data < 0.0] = 0.0
                data[data > 1.0] = 1.0
                data = np.array(254.0 * data, dtype=np.uint8)  # 0.0 -> 0 and 1.0 -> 254 such that 0.5 -> 127
                data = np.transpose(data)
                data = np.flipud(data)
                image = create_png_image(data)
                # Save frame image.
                image.save(frame_path)
            pattern_data = None
        image_data = None

    # Set image ordering for each repetition.
    repetition_orderings = collections.OrderedDict()
    np.random.seed(seed)
    # # Remove, keep or duplicate unperturbed condition numbers.
    assert isinstance(nb_unperturbed_flashes_per_image, int)
    assert nb_unperturbed_flashes_per_image >= 0
    if nb_unperturbed_flashes_per_image == 0:
        selected_condition_nbs = np.setdiff1d(condition_nbs, unperturbed_condition_nbs)
    elif nb_unperturbed_flashes_per_image == 1:
        selected_condition_nbs = condition_nbs
    else:
        selected_condition_nbs = np.concatenate(
            [condition_nbs] + [unperturbed_condition_nbs for _ in range(0, nb_unperturbed_flashes_per_image - 1)]
        )
    selected_condition_nbs = np.sort(selected_condition_nbs)
    for repetition_nb in range(0, nb_repetitions):
        # ordering = np.copy(condition_nbs)
        ordering = np.copy(selected_condition_nbs)
        np.random.shuffle(ordering)
        repetition_orderings[repetition_nb] = ordering

    # Create conditions .csv file.
    conditions_csv_filename = '{}_conditions.csv'.format(name)
    conditions_csv_path = os.path.join(base_path, conditions_csv_filename)
    print("Start creating conditions .csv file...")
    conditions_csv_file = open_csv_file(conditions_csv_path, columns=['image_nb', 'pattern_nb', 'amplitude_nb'])
    for condition_nb in condition_nbs:
        condition_params = conditions_params[condition_nb]
        conditions_csv_file.append(**condition_params)
    conditions_csv_file.close()
    print("End of conditions .csv file creation.")

    # Create images .csv file.
    images_csv_filename = '{}_images.csv'.format(name)
    images_csv_path = os.path.join(base_path, images_csv_filename)
    print("Start creating images .csv file...")
    images_csv_file = open_csv_file(images_csv_path, columns=['path'])
    for image_nb in image_nbs:
        image_params = images_params[image_nb]
        images_csv_file.append(**image_params)
    images_csv_file.close()
    print("End of images .csv file creation.")

    # Create patterns .csv file.
    patterns_csv_filename = '{}_patterns.csv'.format(name)
    patterns_csv_path = os.path.join(base_path, patterns_csv_filename)
    print("Start creating patterns .csv file...")
    patterns_csv_file = open_csv_file(patterns_csv_path, columns=['path'])
    for pattern_nb in pattern_nbs:
        pattern_params = patterns_params[pattern_nb]
        patterns_csv_file.append(**pattern_params)
    patterns_csv_file.close()
    print("End of patterns .csv file creation.")

    # Create amplitudes .csv file.
    amplitudes_csv_filename = '{}_amplitudes.csv'.format(name)
    amplitudes_csv_path = os.path.join(base_path, amplitudes_csv_filename)
    print("Start creating amplitudes .csv file...")
    amplitudes_csv_file = open_csv_file(amplitudes_csv_path, columns=['value'])
    for amplitude_nb in amplitude_nbs:
        value = amplitude_values[amplitude_nb]
        amplitudes_csv_file.append(value=value)
    amplitudes_csv_file.close()
    print("End of patterns .csv file creation.")

    # Create .bin file.
    print("Start creating .bin file...")
    bin_filename = '{}.bin'.format(name)
    bin_path = os.path.join(base_path, bin_filename)
    nb_bin_images = 1 + nb_conditions  # i.e. grey image and other conditions
    bin_frame_nbs = {}
    # Open .bin file.
    bin_file = open_bin_file(bin_path, nb_bin_images, frame_width=frame_width, frame_height=frame_height, reverse=False, mode='w')
    # Add grey frame.
    grey_frame = get_grey_frame(frame_width, frame_height, luminance=0.5)
    grey_frame = float_frame_to_uint8_frame(grey_frame)
    bin_file.append(grey_frame)
    bin_frame_nbs[None] = bin_file.get_frame_nb()
    # Add frames.
    for condition_nb in tqdm.tqdm(condition_nbs):
        frame_path = frame_paths[condition_nb]
        frame = load_png_image(frame_path)
        bin_file.append(frame.data)
        bin_frame_nbs[condition_nb] = bin_file.get_frame_nb()
    # Close .bin file.
    bin_file.close()
    # ...
    print("End of .bin file creation.")

    # Create .vec file.
    print("Start creating .vec file...")
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(base_path, vec_filename)
    csv_filename = "{}_trials.csv".format(name)
    csv_path = os.path.join(base_path, csv_filename)
    # ...
    nb_displays_during_adaptation = int(np.ceil(adaptation_duration * display_rate))
    nb_displays_per_flash = int(np.ceil(flash_duration * display_rate))
    nb_displays_per_inter_flash = int(np.ceil(inter_flash_duration * display_rate))
    nb_displays_per_repetition = ((nb_conditions - nb_unperturbed_conditions) + nb_unperturbed_flashes_per_image * nb_unperturbed_conditions) * (nb_displays_per_flash + nb_displays_per_inter_flash)
    nb_displays = nb_displays_during_adaptation + nb_repetitions * nb_displays_per_repetition
    # Open .vec file.
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    # Open .csv file.
    csv_file = open_csv_file(csv_path, columns=['condition_nb', 'start_display_nb', 'end_display_nb'])
    # Add adaptation.
    bin_frame_nb = bin_frame_nbs[None]  # i.e. default frame (grey)
    for _ in range(0, nb_displays_during_adaptation):
        vec_file.append(bin_frame_nb)
    # Add repetitions.
    for repetition_nb in tqdm.tqdm(range(0, nb_repetitions)):
        condition_nbs = repetition_orderings[repetition_nb]
        for condition_nb in condition_nbs:
            # Add flash.
            start_display_nb = vec_file.get_display_nb() + 1
            bin_frame_nb = bin_frame_nbs[condition_nb]
            for _ in range(0, nb_displays_per_flash):
                vec_file.append(bin_frame_nb)
            end_display_nb = vec_file.get_display_nb()
            csv_file.append(condition_nb=condition_nb, start_display_nb=start_display_nb, end_frame_nb=end_display_nb)
            # Add inter flash.
            bin_frame_nb = bin_frame_nbs[None]  # i.e. default frame (grey)
            for _ in range(0, nb_displays_per_inter_flash):
                vec_file.append(bin_frame_nb)
    # Close .csv file.
    csv_file.close()
    # Close .vec file.
    vec_file.close()
    # ...
    print("End of .vec file creation.")

    return
