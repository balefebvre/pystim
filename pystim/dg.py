import numpy as np

from pystim.utils import handle_arguments_and_configurations


name = 'dg'

default_configuration = {
    'frame': {
        'rate': 60.0,  # Hz
        'width': 1000.0,  # µm  # TODO correct.
        'height': 1000.0,  # µm  # TODO correct.
        # 'horizontal_offset': 0.0,  # µm
        # 'vertical_offset': 0.0,  # µm
    },
    'spatial_frequency': 600.0,  # µm
    'speed': 450.0,  # µm / s
    'contrast': 1.0,
    'directions': [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],  # °
    'trial_duration': 5.0,  # s
    'intertrial_duration': 1.67,  # s
    'nb_repetitions': 1,  # TODO replace with 5.
}


def meshgrid(pixel_size, width=None, height=None):

    dmd_width_in_px = 1920  # px
    dmd_height_in_px = 1080  # px
    dmd_width_in_um = dmd_width_in_px * pixel_size
    dmd_height_in_um = dmd_height_in_px * pixel_size

    x = np.linspace(0.0, dmd_width_in_um, num=dmd_width_in_px, endpoint=False) - 0.5 * dmd_width_in_um
    y = np.linspace(dmd_height_in_um, 0.0, num=dmd_height_in_px, endpoint=False) - 0.5 * dmd_height_in_um

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

    xv, yv = np.meshgrid(x, y, indexing='xy')

    return xv, yv


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
    spatial_frequency = config['spatial_frequency']
    # TODO 2. Get speed (i.e. 450.0 µm / s or 0.75 Hz).
    speed = config['speed']
    # TODO 3. Get contrast (i.e. 100.0 %).
    contrast = config['contrast']
    # TODO 3. Get directions (i.e. 0.0°, 45.0°, 90.0°, 135.0°, 180.0°, 225.0°, 270.0°, 315.0°).
    directions = config['directions']
    # TODO 4. Get trial duration (i.e. 5.0 s).
    trial_duration = config['trial_duration']
    # TODO 5. Get intertrial duration (i.e. 1.67 s).
    intertrial_duration = config['intertrial_duration']
    # TODO 6. Get number of repetitions (i.e. 5).
    nb_repetitions = config['nb_repetitions']

    xv, yv = meshgrid(pixel_size, width=frame_width, height=frame_height)

    frame = (xv - np.min(xv)) / (np.max(xv) - np.min(xv))

    # TODO complete.

    print(frame)

    # TODO Generate combinations.

    # TODO Create .bin file.

    # TODO Create .vec file.

    raise NotImplementedError()
