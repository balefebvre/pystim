import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

from PIL.Image import fromarray

from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.utils import handle_arguments_and_configurations


name = 'dg'

default_configuration = {
    'frame': {
        'rate': 10.0,  # 60.0,  # Hz  # TODO undo.
        'width': 2000.0,  # µm  # TODO correct.
        'height': 2000.0,  # µm  # TODO correct.
        # 'horizontal_offset': 0.0,  # µm
        # 'vertical_offset': 0.0,  # µm
    },
    'spatial_frequencies': [600.0],  # µm
    'speeds': [450.0],  # µm / s
    'contrasts': [1.0],
    'directions': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],  # rad
    'trial_duration': 5.0,  # s
    'intertrial_duration': 1.67,  # s
    'nb_repetitions': 1,  # TODO replace with 5.
    'path': os.path.join(tempfile.gettempdir(), "pystim", name),
}


def linspace(pixel_size, width=None, height=None):

    dmd_width_in_px = 1920  # px
    dmd_height_in_px = 1080  # px
    dmd_width_in_um = dmd_width_in_px * pixel_size
    dmd_height_in_um = dmd_height_in_px * pixel_size

    x = np.linspace(0.0, dmd_width_in_um, num=dmd_width_in_px, endpoint=False) - 0.5 * dmd_width_in_um
    y = np.linspace(0.0, dmd_height_in_um, num=dmd_height_in_px, endpoint=False) - 0.5 * dmd_height_in_um

    frame_width = width if width is not None else dmd_width_in_um
    frame_height = height if height is not None else dmd_height_in_um
    frame_horizontal_offset = 0.0  # µm
    frame_vertical_offset = 0.0  # µm

    x_min = frame_horizontal_offset - frame_width / 2.0
    x_max = frame_horizontal_offset + frame_width / 2.0
    y_min = frame_vertical_offset - frame_height / 2.0
    y_max = frame_vertical_offset + frame_height / 2.0

    xm = np.logical_and(x_min <= x, x <= x_max)
    ym = np.logical_and(y_min <= y, y <= y_max)
    x = x[xm]
    y = y[ym]

    return x, y


def shape(pixel_size, width=None, height=None):

    x, y = linspace(pixel_size, width=width, height=height)

    return y.size, x.size


def meshgrid(pixel_size, width=None, height=None):

    x, y = linspace(pixel_size, width=width, height=height)
    xv, yv = np.meshgrid(x, y, indexing='xy')

    return xv, yv


def get_combinations(spatial_frequencies, speeds, contrasts, directions):

    sf = np.sort(spatial_frequencies)
    s = np.sort(speeds)
    c = np.sort(contrasts)
    d = np.sort(directions)

    sf_indices = np.arange(0, len(sf))
    s_indices = np.arange(0, len(s))
    c_indices = np.arange(0, len(c))
    d_indices = np.arange(0, len(d))

    combinations = {
        'condition': {
            0: {
                'name': 'spatial_frequency',
                'values': sf,
            },
            1: {
                'name': 'speed',
                'values': s,
            },
            2: {
                'name': 'contrast',
                'values': c,
            },
            3: {
                'name': 'direction',
                'values': d,
            },
        },
        'combination': {
            k: combination
            for k, combination in enumerate(itertools.product(sf_indices, s_indices, c_indices, d_indices))
        }
    }

    return combinations


def get_grey_frame(width, height, luminance=0.5):

    shape = (height, width)
    dtype = np.float
    frame = luminance * np.ones(shape, dtype=dtype)

    return frame


def float_frame_to_uint8_frame(float_frame):

    dtype = np.uint8
    dinfo = np.iinfo(dtype)
    float_frame = float_frame * dinfo.max
    float_frame[float_frame < dinfo.min] = dinfo.min
    float_frame[dinfo.max + 1 <= float_frame] = dinfo.max
    uint8_frame = float_frame.astype(dtype)

    return uint8_frame


def save_frame(path, frame):

    image = fromarray(frame)
    image.save(path)

    return


def get_frame(frame_id, pixel_size, direction=0.0, spatial_frequency=600.0, contrast=1.0, speed=450.0, width=None, height=None, rate=60.0):

    xv, yv = meshgrid(pixel_size, width=width, height=height)
    pv = xv + yv * 1j

    angle = direction * np.pi
    t = float(frame_id) / rate
    d = speed * t
    u = np.exp(1j * angle)  # TODO handle special cases (e.g. horizontal, vertical).
    dv = pv * u
    theta = (dv.real - d) / spatial_frequency
    frame = 0.5 + (contrast / 2.0) * np.ones_like(theta)
    mask = (theta % 1.0) >= 0.5
    frame[mask] = 0.5 - (contrast / 2.0)

    return frame


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    # Experimental rig parameters.
    # TODO 1. Get pixel size (i.e. ? µm).
    pixel_size = 3.75  # µm

    # Display parameters.
    # TODO 1. Get frame rate (i.e. 60 Hz).
    frame_rate = config['frame']['rate']
    # TODO 2. Get frame width (i.e. ? µm).
    frame_width = config['frame']['width']
    # TODO 3. Get frame height (i.e. ? µm).
    frame_height = config['frame']['height']
    # # TODO 4. Get frame horizontal offset (i.e. 0.0 µm).
    # frame_horizontal_offset = config['frame']['horizontal_offset']
    # # TODO 5. Get frame vertical offset (i.e. 0.0 µm).
    # frame_vertical_offset = config['frame']['vertical_offset']

    # Stimulus parameters.
    # TODO 1. Get spatial frequency (i.e. 600.0 µm).
    spatial_frequencies = config['spatial_frequencies']
    # TODO 2. Get speed (i.e. 450.0 µm / s or 0.75 Hz).
    speeds = config['speeds']
    # TODO 3. Get contrast (i.e. 100.0 %).
    contrasts = config['contrasts']
    # TODO 3. Get directions (i.e. 0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75).
    directions = config['directions']
    # TODO 4. Get trial duration (i.e. 5.0 s).
    trial_duration = config['trial_duration']
    # TODO 5. Get intertrial duration (i.e. 1.67 s).
    intertrial_duration = config['intertrial_duration']
    # TODO 6. Get number of repetitions (i.e. 5).
    nb_repetitions = config['nb_repetitions']

    path = config['path']
    if not os.path.isdir(path):
        os.makedirs(path)
    frames_path = os.path.join(path, "frames")
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)

    frame_height_in_px, frame_width_in_px = shape(pixel_size, width=frame_width, height=frame_height)
    print(frame_height_in_px)
    print(frame_width_in_px)

    # TODO Get combinations.
    combinations = get_combinations(spatial_frequencies, speeds, contrasts, directions)
    nb_combinations = len(combinations['combination'])

    nb_trials = nb_combinations * nb_repetitions
    stimulus_duration = nb_trials * trial_duration + (nb_trials - 1) * intertrial_duration
    print("stimulus durations: {} s ({} min)".format(stimulus_duration, stimulus_duration / 60.0))
    # TODO improve feedback.

    nb_images_per_trial = int(trial_duration * frame_rate)
    nb_images = 1 + nb_images_per_trial * nb_combinations

    # TODO Create .bin file.
    # Create .bin file.
    bin_filename = "{}.bin".format(name)
    bin_path = os.path.join(path, bin_filename)
    bin_file = open_bin_file(bin_path, nb_images)
    # Get grey frame.
    grey_frame = get_grey_frame(frame_width_in_px, frame_height_in_px)
    grey_frame = float_frame_to_uint8_frame(grey_frame)
    # Save frame in .bin file.
    bin_file.append(grey_frame)
    # Save frame as .png file.
    grey_frame_filename = "grey.png"
    grey_frame_path = os.path.join(frames_path, grey_frame_filename)
    save_frame(grey_frame_path, grey_frame)
    for combination_index in combinations['combination']:
        combination = combinations['combination'][combination_index]
        sf_index = combination[0]
        sf = combinations['condition'][0]['values'][sf_index]
        s_index = combination[1]
        s = combinations['condition'][1]['values'][s_index]
        c_index = combination[2]
        c = combinations['condition'][2]['values'][c_index]
        d_index = combination[3]
        d = combinations['condition'][3]['values'][d_index]
        for frame_id in range(0, nb_images_per_trial):
            frame = get_frame(frame_id, pixel_size, spatial_frequency=sf, speed=s, contrast=c, direction=d, width=frame_width, height=frame_height, rate=frame_rate)
            frame = float_frame_to_uint8_frame(frame)
            # Save frame in .bin file.
            bin_file.append(frame)
            # Save frame as .png file.
            frame_filename = "frame_c{}_f{}.png".format(combination_index, frame_id)
            frame_path = os.path.join(frames_path, frame_filename)
            save_frame(frame_path, frame)
    bin_file.close()

    # TODO Create .vec file.

    raise NotImplementedError()
