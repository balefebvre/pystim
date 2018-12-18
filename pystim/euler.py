import numpy as np
import os
import tempfile

from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.utils import handle_arguments_and_configurations
from pystim.utils import shape
from pystim.utils import get_grey_frame
from pystim.utils import float_frame_to_uint8_frame


name = 'euler'

default_configuration = {
    'frame': {
        'width': 3000.0,  # µm
        'height': 3000.0,  # µm
        'rate': 60.0,  # Hz
    },
    'pattern': {
        'step': {
            'd_ante': 2.0,  # s
            'd': 3.0,  # s
            'd_post': 3.0,  # s
        },
        'frequency_chirp': {
            'd_ante': 2.0,  # s
            'd': 8.0,  # s
            'd_post': 1.0,  # s
            'nb_periods': 32,
        },
        'amplitude_chirp': {
            'd_ante': 1.0,  # s
            'd': 8.0,  # s
            'd_post': 2.0,  # s
            'nb_periods': 16,
        },
    },
    'initial_adaptation_duration': 1.0,  # s
    'intertrial_duration': 2.0,  # s
    'nb_repetitions': 2,  # TODO replace with 25.
    'path': os.path.join(tempfile.gettempdir(), "pystim", name)
}


def get_pattern(config, frame_rate):

    step_config = config['step']
    step_d_ante = step_config['d_ante']
    step_d = step_config['d']
    step_d_post = step_config['d_post']
    assert (step_d_ante * frame_rate).is_integer()
    assert (step_d * frame_rate).is_integer()
    assert (step_d_post * frame_rate).is_integer()
    step_duration = step_d_ante + step_d + step_d_post

    frequency_chirp_config = config['frequency_chirp']
    frequency_chirp_d_ante = frequency_chirp_config['d_ante']
    frequency_chirp_d = frequency_chirp_config['d']
    frequency_chirp_d_post = frequency_chirp_config['d_post']
    assert (frequency_chirp_d_ante * frame_rate).is_integer()
    assert (frequency_chirp_d * frame_rate).is_integer()
    assert (frequency_chirp_d_post * frame_rate).is_integer()
    frequency_chirp_duration = frequency_chirp_d_ante + frequency_chirp_d + frequency_chirp_d_post

    amplitude_chirp_config = config['amplitude_chirp']
    amplitude_chirp_d_ante = amplitude_chirp_config['d_ante']
    amplitude_chirp_d = amplitude_chirp_config['d']
    amplitude_chirp_d_post = amplitude_chirp_config['d_post']
    assert (amplitude_chirp_d_ante * frame_rate).is_integer()
    assert (amplitude_chirp_d * frame_rate).is_integer()
    assert (amplitude_chirp_d_post * frame_rate).is_integer()
    amplitude_chirp_duration = amplitude_chirp_d_ante + amplitude_chirp_d + amplitude_chirp_d_post

    duration = step_duration + frequency_chirp_duration + amplitude_chirp_duration

    nb_displays = int(np.round(duration * frame_rate))
    pattern = 0.5 * np.ones(nb_displays, dtype=np.float)

    # Create step.
    i_min = int(np.round(step_d_ante))
    i_max = int(np.round(step_d_ante + step_d))
    pattern[i_min:i_max+1] = 1.0

    return pattern


def digitize(pattern):

    dtype = np.uint8
    nb_bins = np.iinfo(dtype).max - np.iinfo(dtype).min
    bins = np.linspace(0.0, 1.0, num=nb_bins, endpoint=False)
    bins = bins[1:]  # remove first element

    indices = np.digitize(pattern, bins)

    return indices


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    pixel_size = 3.75  # µm
    # dmd_width = 1920  # px
    # dmd_height = 1080  # px

    frame_width_in_um = config['frame']['width']
    frame_height_in_um = config['frame']['height']
    frame_rate = config['frame']['rate']
    nb_repetitions = config['nb_repetitions']

    path = config['path']
    if not os.path.isdir(path):
        os.makedirs(path)

    dtype = np.uint8
    nb_grey_levels = np.iinfo(dtype).max - np.iinfo(dtype).min + 1
    nb_images = nb_grey_levels

    frame_height_in_px, frame_width_in_px = shape(pixel_size, width=frame_width_in_um, height=frame_height_in_um)

    # TODO create .bin file.
    # Create .bin file.
    bin_filename = "{}.bin".format(name)
    bin_path = os.path.join(path, bin_filename)
    bin_file = open_bin_file(bin_path, nb_images, frame_width=frame_width_in_px, frame_height=frame_height_in_px)
    for k in range(0, nb_grey_levels):
        grey_level = float(k) / nb_grey_levels
        frame = get_grey_frame(frame_width_in_px, frame_height_in_px, luminance=grey_level)
        frame = float_frame_to_uint8_frame(frame)
        bin_file.append(frame)
        bin_file.flush()
    bin_file.close()

    pattern_config = config['pattern']
    pattern = get_pattern(pattern_config, frame_rate)
    intertrial_duration = config['intertrial_duration']
    assert (intertrial_duration * frame_rate).is_integer()

    # nb_displays_per_trial = int(np.round(trial_duration * frame_rate))
    nb_displays_per_trial = pattern.size
    nb_displays_per_intertrial = int(np.round(intertrial_duration * frame_rate))

    nb_trials = nb_repetitions
    nb_intertrials = nb_trials - 1
    nb_displays = nb_trials * nb_displays_per_trial + nb_intertrials * nb_displays_per_intertrial

    frame_indices = digitize(pattern)

    # TODO create .vec file.
    # Create .vec file.
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(path, vec_filename)
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    # TODO add adaptation.
    for _ in range(0, nb_repetitions):
        for k in range(0, nb_displays_per_trial):
            frame_id = frame_indices[k]
            vec_file.append(frame_id)
            vec_file.flush()
        for _ in range(0, nb_displays_per_intertrial):
            frame_id = 0
            vec_file.append(frame_id)
            vec_file.flush()
    vec_file.close()

    return
