import os
import tempfile

from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.io.csv import open_file as open_csv_file
from pystim.utils import float_frame_to_uint8_frame
from pystim.utils import handle_arguments_and_configurations
from pystim.utils import shape
from pystim.utils import get_grey_frame


name = 'flashes'

default_configuration = {
    'frame': {
        'width': 3024.00,  # µm
        'height': 3024.00,  # µm
        'rate': 50.0,  # Hz
        'resolution': 3.5e-6,  # m / pixel  # fixed by the setup
    },
    'adaptation_duration': 30.0,  # s
    'flash': {
        'luminance': 1.0,  # arb. unit (from 0 to 1)
        'duration': 2.0,  # s
        'inter-duration': 8.0,  # s
    },
    'nb_repetitions': 10,  # test value
    # 'nb_repetitions': 60,  # i.e. ~10 min (for drug application/recovery)
    'path': os.path.join(tempfile.gettempdir(), 'pystim', name)
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    frame_width_in_um = config['frame']['width']
    frame_height_in_um = config['frame']['height']
    frame_resolution = config['frame']['resolution']
    frame_rate = config['frame']['rate']
    adaptation_duration = config['adaptation_duration']
    flash_luminance = config['flash']['luminance']
    flash_duration = config['flash']['duration']
    flash_inter_duration = config['flash']['inter-duration']
    nb_repetitions = config['nb_repetitions']
    path = config['path']

    pixel_size = frame_resolution * 1e+6  # µm

    # Create output directory (if necessary).
    if not os.path.isdir(path):
        os.makedirs(path)
    print(path)

    # Collect frame parameters.
    frame_height_in_px, frame_width_in_px = shape(pixel_size, width=frame_width_in_um, height=frame_height_in_um)
    # Create black frame.
    black_frame = get_grey_frame(frame_width_in_px, frame_height_in_px, luminance=0.0)
    black_frame = float_frame_to_uint8_frame(black_frame)
    # Create white frame.
    white_frame = get_grey_frame(frame_width_in_px, frame_height_in_px, luminance=flash_luminance)
    white_frame = float_frame_to_uint8_frame(white_frame)

    nb_images = 2

    # Create .bin file.
    bin_filename = "{}.bin".format(name)
    bin_path = os.path.join(path, bin_filename)
    bin_file = open_bin_file(
        bin_path, nb_images, frame_width=frame_width_in_px, frame_height=frame_height_in_px, mode='w'
    )
    bin_file.append(black_frame)
    bin_file.append(white_frame)
    bin_file.close()

    nb_displays_adaptation = int(adaptation_duration * frame_rate)
    nb_displays_per_flash = int(flash_duration * frame_rate)
    nb_displays_per_inter_flash = int(flash_inter_duration * frame_rate)
    nb_displays_per_repetition = nb_displays_per_flash + nb_displays_per_inter_flash
    nb_displays = nb_displays_adaptation + nb_displays_per_repetition * nb_repetitions

    # Create .vec and .csv files.
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(path, vec_filename)
    vec_file = open_vec_file(vec_path, nb_displays)
    csv_filename = "{}.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    csv_columns = ['condition_nb', 'start_display_nb', 'end_display_nb']
    csv_file = open_csv_file(csv_path, columns=csv_columns)
    black_frame_index = 0
    white_frame_index = 1
    for _ in range(0, nb_displays_adaptation):
        vec_file.append(black_frame_index)  # i.e. black
    for _ in range(0, nb_repetitions):
        start_display_nb = vec_file.get_display_nb() + 1
        for _ in range(0, nb_displays_per_flash):
            vec_file.append(white_frame_index)  # i.e. white
        end_display_nb = vec_file.get_display_nb()
        for _ in range(0, nb_displays_per_inter_flash):
            vec_file.append(black_frame_index)  # i.e. black
        csv_file.append(condition_nb=0, start_display_nb=start_display_nb, end_display_nb=end_display_nb)
    vec_file.close()
    csv_file.close()

    return
