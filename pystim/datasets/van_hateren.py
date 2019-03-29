import numpy as np
import os
import re
import tqdm
import urllib.request
import urllib.error

from .base import Bunch
from .base import get_path as get_base_path
from ..images.iml import load as load_iml_image
from ..images.imc import load as load_imc_image
from ..images.png import create as create_png_image


min_image_nb = 1
max_image_nb = 4212
all_image_nbs = list(range(min_image_nb, max_image_nb + 1))
missing_image_nbs = [
    2829, 2831, 2835, 2837, 2839, 2846, 2859, 2860, 2863, 2864, 2865, 2872,
    2873, 2875, 2876, 2877, 2879, 2880, 2882, 2885, 2886, 2888, 2890, 2891,
    2892, 2893, 2922, 2924, 2925, 2927, 2936, 2938, 2939, 2941, 2963, 2964,
    2965, 2969, 2971, 2974, 2975, 2977, 2984, 3000, 3001,
]

dtype = np.uint16
height = 1024
width = 1536

# sup_value = 6282
# sup_value = 12564  # i.e. 2*6282=12564, image 980
sup_value = 25119  # i.e. 2*2*6282=25128, image 1910
# uint12: 4095, uint13: 8191, uint14: 16383, uint15: 32767, uint16: 65535

# max_luminance = 8669.16  # cd/m²
max_luminance = 70302.4  # cd/m²


def get_reference_path():

    path = os.path.join(get_base_path(), 'van_hateren')

    return path


def get_settings_filename():

    filename = 'camerasettings.txt'

    return filename


def get_settings_path():

    reference_path = get_reference_path()
    filename = get_settings_filename()
    path = os.path.join(reference_path, filename)

    return path


def get_directory(format_='iml'):

    directory = format_

    return directory


def get_filename(image_nb, format_='iml'):

    filename_string = 'imk{:05d}.{}'
    filename = filename_string.format(image_nb, format_)

    return filename


def get_path(image_nb, format_='iml'):

    reference_path = get_reference_path()
    directory = get_directory(format_=format_)
    filename = get_filename(image_nb, format_=format_)
    path = os.path.join(reference_path, directory, filename)

    return path


def get_settings_url():

    url = "http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/camerasettings.txt"

    return url


def get_url(image_nb, format_='iml'):

    url_string = "http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/{f}/imk{i:05d}.{f}"
    url = url_string.format(i=image_nb, f=format_)

    return url


def fetch(image_nbs=None, format_='iml', download_if_missing=True, verbose=False):
    """Fetch the Van Hateren dataset from the Max Planck Institute.

    Arguments:
        image_nbs: iterable
            Specify the numbers of the images to be fetched.
        format_: string
            ...
        download_if_missing: boolean (optional)
            If False, raise a IOError if the data is not locally available
            instead of trying to download the data from the source site.
            The default value is True.
        verbose: boolean (optional)
            ...
            The default value is False.
    """

    reference_path = get_reference_path()
    directory = get_directory(format_)
    path = os.path.join(reference_path, directory)
    if not os.path.isdir(path):
        os.makedirs(path)

    if image_nbs is None:
        image_nbs = get_image_nbs(fetched_only=False)
    else:
        for image_nb in image_nbs:
            assert image_nb in all_image_nbs and image_nb not in missing_image_nbs, image_nb

    # Fetch settings.
    path = get_settings_path()
    if not os.path.isfile(path):
        if not download_if_missing:
            raise IOError("data not found and `download_if_missing` is False.")
        else:
            url = get_settings_url()
            if verbose:
                print("Load {} to file file://{}".format(url, path))
            urllib.request.urlretrieve(url, path)

    # Fetch images.
    paths = []
    for image_nb in image_nbs:
        path = get_path(image_nb, format_=format_)
        if not os.path.isfile(path):
            if not download_if_missing:
                raise IOError("data not found and `download_if_missing` is False.")
            else:
                url = get_url(image_nb, format_=format_)
                if verbose:
                    print("Load {} to file://{}.".format(url, path))
                try:
                    urllib.request.urlretrieve(url, path)
                except urllib.error.HTTPError as error:
                    if error.code == 404:  # i.e. not found
                        # msg = "{} not found.".format(url)
                        # raise urllib.error.HTTPError(url=error.url, code=error.code, msg=msg, hdrs=error.hdrs, fp=error.fp) from None
                        print(image_nb)
                except urllib.error.URLError as error:
                    raise urllib.error.URLError(reason=error.reason, filename=error.filename) from None
        paths.append(path)

    return Bunch(paths=paths)


def get_image_nbs(format_='iml', fetched_only=True):

    image_nbs = all_image_nbs.copy()
    for image_nb in missing_image_nbs:
        image_nbs.remove(image_nb)

    if fetched_only:
        image_nbs = [
            image_nb
            for image_nb in image_nbs
            if os.path.isfile(get_path(image_nb, format_=format_))
        ]
    image_nbs = np.array(image_nbs)

    return image_nbs


def load_settings():

    path = get_settings_path()

    with open(path, mode='r') as file:
        text = file.read()

    lines = text.split('\n')
    lines = lines[3:]  # remove header lines

    settings = {}
    for line in lines:
        line = line.strip()  # remove strip spaces
        line = re.sub(' +', ' ', line)  # remove duplicated spaces
        elements = line.split(' ')
        assert len(elements) == 5, elements
        image_nb = int(elements[0])
        image_settings = {
            'iso': int(elements[1]),  # ISO setting (i.e. electronic equivalent)
            'aperture': float(elements[2]),  # aperture
            'shutter': int(elements[3]),  # reciprocal shutter time (1/s)
            'factor': float(elements[4]),  # factor for converting pixel values to luminance
                                           # (cd/m2, luminance=factor*pixel value)
        }
        settings[image_nb] = image_settings

    return settings


def load_image_settings(image_nb):

    assert min_image_nb <= image_nb <= max_image_nb, image_nb

    settings = load_settings()
    image_settings = settings[image_nb]

    return image_settings


def load_raw_image(image_nb, format_='iml'):

    path = get_path(image_nb, format_=format_)
    if format_ == 'iml':
        image = load_iml_image(path, dtype, width, height)
    elif format_ == 'imc':
        image = load_imc_image(path, dtype, width, height)
    else:
        raise ValueError("unknown format value: {}".format(format_))

    return image


def load_luminance_data(image_nb, format_='iml'):

    image = load_raw_image(image_nb, format_=format_)
    image_settings = load_image_settings(image_nb)
    factor = image_settings['factor']
    data = factor * image.data.astype('float')

    return data


def load_image(image_nb, format_='iml'):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    normalized_luminance_data = luminance_data / max_luminance
    assert 0.0 <= np.min(normalized_luminance_data), np.min(luminance_data)
    assert np.max(normalized_luminance_data) <= 1.0, np.max(luminance_data)
    dtype = 'uint8'
    dinfo = np.iinfo(dtype)
    assert dinfo.min == 0, dinfo.min
    max_value = float(dinfo.max + 1)
    factor = max_value / max_luminance
    data = factor * luminance_data
    data[float(dinfo.max) <= data] = float(dinfo.max)  # correct max. values
    assert float(dinfo.min) <= np.min(data), np.min(data)
    assert np.max(data) <= float(dinfo.max), np.max(data)
    data = data.astype(dtype)
    image = create_png_image(data)

    return image


def get_min_luminances(verbose=False):

    image_nbs = get_image_nbs()
    if verbose:
        image_nbs = tqdm.tqdm(image_nbs)
    min_luminances = np.array([
        np.min(load_luminance_data(image_nb))
        for image_nb in image_nbs
    ])

    return min_luminances


def get_max_luminances(verbose=False):

    image_nbs = get_image_nbs()
    if verbose:
        image_nbs = tqdm.tqdm(image_nbs)
    max_luminances = np.array([
        np.max(load_luminance_data(image_nb))
        for image_nb in image_nbs
    ])

    return max_luminances


def get_mean_values():

    image_nbs = get_image_nbs()
    mean_values = np.array([
        load_image(image_nb).mean
        for image_nb in image_nbs
    ])

    return mean_values


def get_std_values():

    image_nbs = get_image_nbs()
    std_values = np.array([
        load_image(image_nb).std
        for image_nb in image_nbs
    ])

    return std_values

