"""Flashed images perturbed with checkerboard"""

import array
import io
import math
# import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import scipy.interpolate
import scipy.ndimage
import tempfile

# from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL.Image import fromarray
from PIL.Image import open as open_image
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve

from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.utils import handle_arguments_and_configurations


name = 'fipwc'

default_configuration = {
    'path': os.path.join(tempfile.gettempdir(), "pystim", name),
    'reference_images': {
        # 'indices': [5, 31, 39, 46],  # a selection of promising images
        'indices': [5, 31, 46],
    },
    'perturbations': {
        'pattern_indices': list(range(1, 2 + 1)),
        # 'pattern_indices': list(range(1, 18 + 1)),
        'amplitude_indices': [10, 28],
        # 'amplitude_indices': [2, 4, 7, 10, 14, 18, 23, 28],
    },
    # 'display_rate': 50.0,  # Hz
    'display_rate': 40.0,  # Hz
    'frame': {
        # 'width': 1920',  # DMD's limit
        'width': 768,
        # 'height': 1080,  # DMD's limit
        'height': 768,
        'duration': 0.3,  # s
        # 'resolution': 0.42, # µm / pixel  # fixed by the setup
        'resolution': 0.7,  # µm / pixel  # fixed by the setup
    },
    # 'frame_width': 768,  # TODO remove.
    # 'frame_height': 768,  # TODO remove.
    # image_resolution = 3.3  # µm / pixel  # fixed by the eye (monkey)
    'image_resolution': 0.8,  # µm / pixel  # fixed by the eye (salamander)
    'background_luminance': 0.5,  # arb. unit
    'mean_luminance': 0.5,  # arb. unit
    # 'std_luminance': 0.06,  # arb. unit
    'std_luminance': 0.2,  # arb. unit
    # 'frame_duration': 0.3,  # s  # TODO remove.
    # 'frame_duration': 0.3,  # s  # TODO remove.
    # 'frame_resolution': 0.7,  # µm / pixel  # TODO remove.
    'nb_repetitions': 1,
    # 'nb_repetitions': 20,
}


def load_resource(url):

    with urlopen(url) as handle:
        resource = handle.read()

    return resource


def get_palmer_resource_locator(path, index):

    assert 1 <= index <= 6

    frame_index = 1200 * (index - 1) + 600
    filename = "frame{i:04d}.png".format(i=frame_index)
    path = os.path.join(path, filename)
    url = "file://localhost" + path

    return url


def get_palmer_resource_locators(path=None, indices=None):

    if indices is None:
        indices = range(1, 7)

    urls = [
        get_palmer_resource_locator(path, index)
        for index in indices
    ]

    return urls


def convert_palmer_resource_to_image(resource):

    data = io.BytesIO(resource)
    image = open_image(data)
    data = image.getdata()
    data = np.array(data, dtype=np.uint8)
    image = data.reshape(image.size)

    image = image.astype(np.float)
    image = image / (2.0 ** 8)

    return image


def get_van_hateren_resource_locator(index, format_='iml', scheme='file', path=None, mirror='Lies'):

    filename = "imk{:05d}.{}".format(index, format_)

    if scheme == 'file':
        assert path is not None
        netloc = "localhost"
        path = os.path.join(path, filename)
        url = "{}://{}{}".format(scheme, netloc, path)
    elif scheme == 'http':
        if mirror == 'Lies':
            assert 1 <= index <= 4212
            assert format_ in ['iml', 'imc']
            url_string = "http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/{f}/imk{i:05d}.{f}"
            url = url_string.format(i=index, f=format_)
        elif mirror == 'Ivanov':
            assert 1 <= index <= 99
            if format_ == 'iml':
                url = "http://pirsquared.org/research/vhatdb/imk{i:05d}.{f}".format(i=index, f=format_)
            elif format_ == 'imc':
                url = "http://pirsquared.org/research/vhatdb/{f}/imk{i:05d}.{f}".format(i=index, f=format_)
            else:
                raise ValueError("unexpected format value: {}".format(format_))
        else:
            raise ValueError("unexpected mirror value: {}".format(mirror))
    else:
        raise ValueError("unexpected scheme value: {}".format(scheme))

    return url


def get_van_hateren_resource_locators(path=None, indices=None, format_='iml', mirror='Lies'):

    assert format_ in ['iml', 'imc']

    if indices is None:
        if mirror == 'Lies':
            indices = range(0, 4213)
        elif mirror == 'Ivanov':
            indices = range(1, 100)
        else:
            raise ValueError("unexpected mirror value: {}".format(mirror))

    urls = [
        get_van_hateren_resource_locator(index, format_=format_, path=path, mirror=mirror)
        for index in indices
    ]

    return urls


def convert_van_hateren_resource_to_image(resource):

    data = array.array('H', resource)
    data.byteswap()
    data = np.array(data, dtype=np.uint16)
    image = data.reshape(1024, 1536)
    image = image.astype(np.float)
    image = image / (2.0 ** (12 + 1))

    return image


# def generate(args):
#
#     _ = args  # TODO remove.
#
#     # dataset = 'Palmer'
#     dataset = 'van Hateren'  # TODO cache the van Hateren's natural images.
#
#     # TODO collect the reference images (i.e. natural images).
#     if dataset == 'Palmer':
#         path = os.path.join("~", "spot_sensitivity_context_dependent")
#         path = os.path.expanduser(path)
#         urls = get_palmer_resource_locators(path=path, indices=[1])
#     elif dataset == 'van Hateren':
#         path = tempfile.TemporaryDirectory()
#         urls = get_van_hateren_resource_locators(path=path, indices=[5])
#     else:
#         raise ValueError("unexpected dataset value: {}".format(dataset))
#     # Print resource locators.
#     for url in urls:
#         print(url)
#     # Load resources.
#     resources = [
#         load_resource(url)
#         for url in urls
#     ]
#     # Convert resources to images.
#     if dataset == 'Palmer':
#         images = [
#             convert_palmer_resource_to_image(resource)
#             for resource in resources
#         ]
#     elif dataset == 'van Hateren':
#         images = [
#             convert_van_hateren_resource_to_image(resource)
#             for resource in resources
#         ]
#     else:
#         raise ValueError("unexpected dataset value: {}".format(dataset))
#
#     # TODO clean the following experimental lines.
#
#     image = images[0]
#
#     # Plot reference image.
#     fig, ax = plt.subplots()
#     im = ax.imshow(image, cmap='gray', vmin=-1.0, vmax=1.0)
#     div = make_axes_locatable(ax)
#     cax = div.append_axes('right', size='5%', pad=0.05)
#     plt.colorbar(im, cax=cax)
#
#     image_height, image_width = image.shape
#     # image_resolution = 3.3  # µm / pixel  # fixed by the eye (monkey)
#     image_resolution = 0.8  # µm / pixel  # fixed by the eye (salamander)
#
#     # frame_shape = frame_height, frame_width = 1080, 1920  # fixed by the DMD
#     frame_shape = frame_height, frame_width = 768, 768  # fixed by the DMD
#     # frame_resolution = 0.42  # µm / pixel  # fixed by the setup  # TODO check this value.
#     frame_resolution = 0.7  # µm / pixel
#
#     background_luminance = 0.5
#
#     if frame_resolution <= image_resolution:
#         # Up-sample the image (interpolation).
#         image_x = image_resolution * np.arange(0, image_height)
#         image_x = image_x - np.mean(image_x)
#         image_y = image_resolution * np.arange(0, image_width)
#         image_y = image_y - np.mean(image_y)
#         image_z = image
#         spline = scipy.interpolate.RectBivariateSpline(image_x, image_y, image_z, kx=1, ky=1)
#         frame_x = frame_resolution * np.arange(0, frame_height)
#         frame_x = frame_x - np.mean(frame_x)
#         mask_x = np.logical_and(
#             np.min(image_x) - 0.5 * image_resolution <= frame_x,
#             frame_x <= np.max(image_x) + 0.5 * image_resolution
#         )
#         frame_y = frame_resolution * np.arange(0, frame_width)
#         frame_y = frame_y - np.mean(frame_y)
#         mask_y = np.logical_and(
#             np.min(image_y) - 0.5 * image_resolution <= frame_y,
#             frame_y <= np.max(image_y) + 0.5 * image_resolution
#         )
#         frame_z = spline(frame_x[mask_x], frame_y[mask_y])
#         frame_i_min = np.min(np.nonzero(mask_x))
#         frame_i_max = np.max(np.nonzero(mask_x)) + 1
#         frame_j_min = np.min(np.nonzero(mask_y))
#         frame_j_max = np.max(np.nonzero(mask_y)) + 1
#         frame = background_luminance * np.ones(frame_shape, dtype=np.float)
#         frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_z
#     else:
#         # Down-sample the image (decimation).
#         image_frequency = 1.0 / image_resolution
#         frame_frequency = 1.0 / frame_resolution
#         cutoff = frame_frequency / image_frequency
#         sigma = math.sqrt(2.0 * math.log(2.0)) / (2.0 * math.pi * cutoff)
#         filtered_image = scipy.ndimage.gaussian_filter(image, sigma=sigma)
#         # see https://en.wikipedia.org/wiki/Gaussian_filter for a justification of this formula
#         image_x = image_resolution * np.arange(0, image_height)
#         image_x = image_x - np.mean(image_x)
#         image_y = image_resolution * np.arange(0, image_width)
#         image_y = image_y - np.mean(image_y)
#         image_z = filtered_image
#         spline = scipy.interpolate.RectBivariateSpline(image_x, image_y, image_z, kx=1, ky=1)
#         frame_x = frame_resolution * np.arange(0, frame_height)
#         frame_x = frame_x - np.mean(frame_x)
#         mask_x = np.logical_and(
#             np.min(image_x) - 0.5 * image_resolution <= frame_x,
#             frame_x <= np.max(image_x) + 0.5 * image_resolution
#         )
#         frame_y = frame_resolution * np.arange(0, frame_width)
#         frame_y = frame_y - np.mean(frame_y)
#         mask_y = np.logical_and(
#             np.min(image_y) - 0.5 * image_resolution <= frame_y,
#             frame_y <= np.max(image_y) + 0.5 * image_resolution
#         )
#         frame_z = spline(frame_x[mask_x], frame_y[mask_y])
#         frame_i_min = np.min(np.nonzero(mask_x))
#         frame_i_max = np.max(np.nonzero(mask_x)) + 1
#         frame_j_min = np.min(np.nonzero(mask_y))
#         frame_j_max = np.max(np.nonzero(mask_y)) + 1
#         frame = background_luminance * np.ones(frame_shape, dtype=np.float)
#         frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_z
#
#     mean_luminance = 0.5  # arb. unit
#     std_luminance = 0.06  # arb. unit
#
#     frame_roi = frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max]
#     print("mean (luminance): {}".format(np.mean(frame_roi)))
#     print("std (luminance): {}".format(np.std(frame_roi)))
#     print("min (luminance): {}".format(np.min(frame_roi)))
#     print("max (luminance): {}".format(np.max(frame_roi)))
#     frame_roi = frame_roi - np.mean(frame_roi)
#     frame_roi = frame_roi / np.std(frame_roi)
#     frame_roi = frame_roi * std_luminance
#     frame_roi = frame_roi + mean_luminance
#     print("mean (luminance): {}".format(np.mean(frame_roi)))
#     print("std (luminance): {}".format(np.std(frame_roi)))
#     print("min (luminance): {}".format(np.min(frame_roi)))
#     print("max (luminance): {}".format(np.max(frame_roi)))
#     frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_roi
#
#     # TODO clean the previous experimental lines..
#
#     # Plot reference frame.
#     fig, ax = plt.subplots()
#     im = ax.imshow(frame, cmap='gray', vmin=0.0, vmax=1.0)
#     div = make_axes_locatable(ax)
#     cax = div.append_axes('right', size='2%', pad=0.05)
#     plt.colorbar(im, cax=cax)
#
#     # TODO clean the following experimental lines.
#
#     # TODO create the image perturbations (i.e. the checkerboards).
#     # perturbation_shape = perturbation_height, perturbation_width = (14, 26)
#     perturbation_shape = perturbation_height, perturbation_width = (10, 10)
#     perturbation = np.random.choice(a=[-1.0, +1.0], size=perturbation_shape)
#     perturbation_resolution = 50.0  # µm / pixel
#     perturbation_x = perturbation_resolution * np.arange(0, perturbation_height)
#     perturbation_x = perturbation_x - np.mean(perturbation_x)
#     perturbation_y = perturbation_resolution * np.arange(0, perturbation_width)
#     perturbation_y = perturbation_y - np.mean(perturbation_y)
#     perturbation_z = perturbation
#     perturbation_x_, perturbation_y_ = np.meshgrid(perturbation_x, perturbation_y)
#     perturbation_points = np.stack((perturbation_x_.flatten(), perturbation_y_.flatten()), axis=-1)
#     perturbation_data = perturbation_z.flatten()
#     interpolate = scipy.interpolate.NearestNDInterpolator(perturbation_points, perturbation_data)
#     mask_x = np.logical_and(
#         np.min(perturbation_x) - 0.5 * perturbation_resolution <= frame_x,
#         frame_x <= np.max(perturbation_x) + 0.5 * perturbation_resolution
#     )
#     mask_y = np.logical_and(
#         np.min(perturbation_y) - 0.5 * perturbation_resolution <= frame_y,
#         frame_y <= np.max(perturbation_y) + 0.5 * perturbation_resolution
#     )
#     frame_x_, frame_y_ = np.meshgrid(frame_x[mask_x], frame_y[mask_y])
#     frame_x_ = frame_x_.transpose().flatten()
#     frame_y_ = frame_y_.transpose().flatten()
#     frame_points_ = np.stack((frame_x_, frame_y_), axis=-1)
#     frame_data_ = interpolate(frame_points_)
#     frame_z_ = np.reshape(frame_data_, (frame_x[mask_x].size, frame_y[mask_y].size))
#     frame_i_min = np.min(np.nonzero(mask_x))
#     frame_i_max = np.max(np.nonzero(mask_x)) + 1
#     frame_j_min = np.min(np.nonzero(mask_y))
#     frame_j_max = np.max(np.nonzero(mask_y)) + 1
#     perturbation_frame = np.zeros(frame_shape, dtype=np.float)
#     perturbation_frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_z_
#
#     # Plot perturbation image.
#     fig, ax = plt.subplots()
#     im = ax.imshow(perturbation_frame, cmap='gray', vmin=-1.0, vmax=+1.0)
#     div = make_axes_locatable(ax)
#     cax = div.append_axes('right', size='2%', pad=0.05)
#     plt.colorbar(im, cax=cax)
#
#     # TODO create the perturbed images.
#
#     # Create perturbed reference image.
#     perturbation_amplitude = - 2.0 / (2.0 ** 8)
#     frame_ = frame + perturbation_amplitude * perturbation_frame
#
#     # Plot perturbed frame.
#     fig, ax = plt.subplots()
#     im = ax.imshow(frame_, cmap='gray', vmin=0.0, vmax=1.0)
#     div = make_axes_locatable(ax)
#     cax = div.append_axes('right', size='2%', pad=0.05)
#     plt.colorbar(im, cax=cax)
#
#     # TODO generate all the reference frames and all the perturbed frames.
#     # TODO 1. get the list of reference images
#     # TODO 2. get the list of perturbation patterns (or number of perturbation patterns to generate)
#     # TODO 3. get the list of perturbation amplitudes
#     # TODO 4. generate all the reference frames
#     # TODO 5. generate all the perturbed frames
#
#     # TODO create the .bin file.
#     # TODO 1. open the .bin file.
#     # TODO 2. for each reference frame
#     # TODO   - load the frame
#     # TODO   - append the frame to the .bin file
#     # TODO 3. for each perturbation
#     # TODO   - for each reference frame
#     # TODO     - load the corresponding perturbed frame
#     # TODO     - append the frame to the .bin file
#     # TODO 4. close the .bin file
#
#     # TODO create the interleaving of the perturbed images.
#     # TODO 1. list all the possible combinations (i.e. reference image, perturbation pattern, perturbation amplitude)
#     # TODO 2. for each repetition
#     # TODO   - sample a random order of the combinations
#     # TODO   - append this sequence to ...
#
#     # TODO create the .vec file.
#
#     # TODO clean the previous experimental lines.
#
#     plt.show()  # TODO save images to disk instead.
#
#     raise NotImplementedError()


def get_checkerboard_locator(index, scheme='file', path=None):

    filename = "checkerboard{:05d}.png".format(index)

    if scheme == 'file':
        netloc = "localhost"
        path = os.path.join(path, filename)
        url = "{}://{}{}".format(scheme, netloc, path)
    else:
        raise ValueError("unexpected scheme value: {}".format(scheme))

    return url


def get_resource_locator(index, dataset='van Hateren', **kwargs):

    if dataset == 'van Hateren':
        url = get_van_hateren_resource_locator(index, **kwargs)
    elif dataset == 'Palmer':
        url = get_palmer_resource_locator(index, **kwargs)
    elif dataset == 'checkerboard':
        url = get_checkerboard_locator(index, **kwargs)
    else:
        raise ValueError("unexpected dataset value: {}".format(dataset))

    return url


def is_resource(url):

    url = urlparse(url)
    if url.scheme == 'file':
        ans = os.path.isfile(url.path)
        print("{} is resource: {}".format(url.path, ans))
    elif url.scheme == 'http':
        ans = True
    else:
        raise ValueError("unexpected url scheme: {}".format(url.scheme))

    return ans


def collect_reference_image(index, dataset='van Hateren', **kwargs):

    local_url = get_resource_locator(index, dataset=dataset, scheme='file', **kwargs)
    if not is_resource(local_url):
        remote_url = get_resource_locator(index, dataset=dataset, scheme='http', **kwargs)
        local_url = urlparse(local_url)
        local_path = local_url.path
        urlretrieve(remote_url, local_path)

    return


def generate_perturbation_pattern(index, path=None):

    assert path is not None

    np.random.seed(seed=index)
    dtype = np.uint8
    info = np.iinfo(dtype)
    a = np.array([info.min, info.max], dtype=dtype)
    # height = 14
    # width = 26
    height = 10
    width = 10
    shape = (height, width)
    pattern = np.random.choice(a=a, size=shape)
    image = fromarray(pattern)
    image.save(path)

    return


def collect_perturbation_pattern(index, path=None):

    assert path is not None

    url = get_resource_locator(index, dataset='checkerboard', scheme='file', path=path)
    if not is_resource(url):
        url = urlparse(url)
        path = url.path
        generate_perturbation_pattern(index, path=path)

    return


def load_perturbation_pattern(index, path):

    url = get_resource_locator(index, dataset='checkerboard', scheme='file', path=path)
    url = urlparse(url)
    path = url.path

    image = open_image(path)
    data = image.getdata()
    data = np.array(data, dtype=np.uint8)
    width, height = image.size
    data = data.reshape(height, width)

    data = data.astype(np.float)
    data = data / np.iinfo(np.uint8).max

    return data


def load_reference_image(index, path):

    dtype = np.uint16
    height = 1024
    width = 1536

    url = get_resource_locator(index, dataset='van Hateren', scheme='file', path=path)
    url = urlparse(url)
    path = url.path

    with open(path, mode='rb') as handle:
        data_bytes = handle.read()
    data = array.array('H', data_bytes)
    data.byteswap()
    data = np.array(data, dtype=dtype)
    data = data.reshape(height, width)

    data = data.astype(np.float)
    data = data / np.iinfo(dtype).max

    return data


def get_frame(image, config):

    image_height, image_width = image.shape
    image_resolution = config['image_resolution']
    frame_width = config['frame']['width']
    frame_height = config['frame']['height']
    frame_shape = frame_height, frame_width
    frame_resolution = config['frame']['resolution']

    background_luminance = config['background_luminance']

    if frame_resolution <= image_resolution:
        # Up-sample the image (interpolation).
        image_x = image_resolution * np.arange(0, image_height)
        image_x = image_x - np.mean(image_x)
        image_y = image_resolution * np.arange(0, image_width)
        image_y = image_y - np.mean(image_y)
        image_z = image
        spline = scipy.interpolate.RectBivariateSpline(image_x, image_y, image_z, kx=1, ky=1)
        frame_x = frame_resolution * np.arange(0, frame_height)
        frame_x = frame_x - np.mean(frame_x)
        mask_x = np.logical_and(
            np.min(image_x) - 0.5 * image_resolution <= frame_x,
            frame_x <= np.max(image_x) + 0.5 * image_resolution
        )
        frame_y = frame_resolution * np.arange(0, frame_width)
        frame_y = frame_y - np.mean(frame_y)
        mask_y = np.logical_and(
            np.min(image_y) - 0.5 * image_resolution <= frame_y,
            frame_y <= np.max(image_y) + 0.5 * image_resolution
        )
        frame_z = spline(frame_x[mask_x], frame_y[mask_y])
        frame_i_min = np.min(np.nonzero(mask_x))
        frame_i_max = np.max(np.nonzero(mask_x)) + 1
        frame_j_min = np.min(np.nonzero(mask_y))
        frame_j_max = np.max(np.nonzero(mask_y)) + 1
        frame = background_luminance * np.ones(frame_shape, dtype=np.float)
        frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_z
    else:
        # Down-sample the image (decimation).
        image_frequency = 1.0 / image_resolution
        frame_frequency = 1.0 / frame_resolution
        cutoff = frame_frequency / image_frequency
        sigma = math.sqrt(2.0 * math.log(2.0)) / (2.0 * math.pi * cutoff)
        filtered_image = scipy.ndimage.gaussian_filter(image, sigma=sigma)
        # see https://en.wikipedia.org/wiki/Gaussian_filter for a justification of this formula
        image_x = image_resolution * np.arange(0, image_height)
        image_x = image_x - np.mean(image_x)
        image_y = image_resolution * np.arange(0, image_width)
        image_y = image_y - np.mean(image_y)
        image_z = filtered_image
        spline = scipy.interpolate.RectBivariateSpline(image_x, image_y, image_z, kx=1, ky=1)
        frame_x = frame_resolution * np.arange(0, frame_height)
        frame_x = frame_x - np.mean(frame_x)
        mask_x = np.logical_and(
            np.min(image_x) - 0.5 * image_resolution <= frame_x,
            frame_x <= np.max(image_x) + 0.5 * image_resolution
        )
        frame_y = frame_resolution * np.arange(0, frame_width)
        frame_y = frame_y - np.mean(frame_y)
        mask_y = np.logical_and(
            np.min(image_y) - 0.5 * image_resolution <= frame_y,
            frame_y <= np.max(image_y) + 0.5 * image_resolution
        )
        frame_z = spline(frame_x[mask_x], frame_y[mask_y])
        frame_i_min = np.min(np.nonzero(mask_x))
        frame_i_max = np.max(np.nonzero(mask_x)) + 1
        frame_j_min = np.min(np.nonzero(mask_y))
        frame_j_max = np.max(np.nonzero(mask_y)) + 1
        frame = background_luminance * np.ones(frame_shape, dtype=np.float)
        frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_z

    limits = frame_i_min, frame_i_max, frame_j_min, frame_j_max

    return frame, limits


def get_reference_frame(reference_image, config):

    frame, limits = get_frame(reference_image, config)
    i_min, i_max, j_min, j_max = limits

    mean_luminance = config['mean_luminance']
    std_luminance = config['std_luminance']

    frame_roi = frame[i_min:i_max, j_min:j_max]
    frame_roi = frame_roi - np.mean(frame_roi)
    frame_roi = frame_roi / np.std(frame_roi)
    frame_roi = frame_roi * std_luminance
    frame_roi = frame_roi + mean_luminance
    print(np.std(frame_roi))
    frame[i_min:i_max, j_min:j_max] = frame_roi

    return frame


def get_perturbation_frame(perturbation_image):

    # frame_shape = frame_height, frame_width = 1080, 1920  # fixed by the DMD
    frame_shape = frame_height, frame_width = 768, 768  # fixed by the DMD
    # frame_resolution = 0.42  # µm / pixel  # fixed by the setup  # TODO check this value.
    frame_resolution = 0.7  # µm / pixel

    frame_x = frame_resolution * np.arange(0, frame_height)
    frame_x = frame_x - np.mean(frame_x)
    frame_y = frame_resolution * np.arange(0, frame_width)
    frame_y = frame_y - np.mean(frame_y)

    perturbation = perturbation_image
    perturbation_height, perturbation_width = perturbation.shape
    perturbation_resolution = 50.0  # µm / pixel
    perturbation_x = perturbation_resolution * np.arange(0, perturbation_height)
    perturbation_x = perturbation_x - np.mean(perturbation_x)
    perturbation_y = perturbation_resolution * np.arange(0, perturbation_width)
    perturbation_y = perturbation_y - np.mean(perturbation_y)
    perturbation_z = perturbation
    perturbation_x_, perturbation_y_ = np.meshgrid(perturbation_x, perturbation_y)
    perturbation_points = np.stack((perturbation_x_.flatten(), perturbation_y_.flatten()), axis=-1)
    perturbation_data = perturbation_z.flatten()
    interpolate = scipy.interpolate.NearestNDInterpolator(perturbation_points, perturbation_data)
    mask_x = np.logical_and(
        np.min(perturbation_x) - 0.5 * perturbation_resolution <= frame_x,
        frame_x <= np.max(perturbation_x) + 0.5 * perturbation_resolution
    )
    mask_y = np.logical_and(
        np.min(perturbation_y) - 0.5 * perturbation_resolution <= frame_y,
        frame_y <= np.max(perturbation_y) + 0.5 * perturbation_resolution
    )
    frame_x_, frame_y_ = np.meshgrid(frame_x[mask_x], frame_y[mask_y])
    frame_x_ = frame_x_.transpose().flatten()
    frame_y_ = frame_y_.transpose().flatten()
    frame_points_ = np.stack((frame_x_, frame_y_), axis=-1)
    frame_data_ = interpolate(frame_points_)
    frame_z_ = np.reshape(frame_data_, (frame_x[mask_x].size, frame_y[mask_y].size))
    i_min = np.min(np.nonzero(mask_x))
    i_max = np.max(np.nonzero(mask_x)) + 1
    j_min = np.min(np.nonzero(mask_y))
    j_max = np.max(np.nonzero(mask_y)) + 1
    frame = np.zeros(frame_shape, dtype=np.float)
    frame[i_min:i_max, j_min:j_max] = frame_z_

    frame_roi = frame[i_min:i_max, j_min:j_max]
    print("mean (luminance): {}".format(np.mean(frame_roi)))
    print("std (luminance): {}".format(np.std(frame_roi)))
    print("min (luminance): {}".format(np.min(frame_roi)))
    print("max (luminance): {}".format(np.max(frame_roi)))
    frame_roi = -1.0 + 2.0 * frame_roi
    print("mean (luminance): {}".format(np.mean(frame_roi)))
    print("std (luminance): {}".format(np.std(frame_roi)))
    print("min (luminance): {}".format(np.min(frame_roi)))
    print("max (luminance): {}".format(np.max(frame_roi)))
    frame[i_min:i_max, j_min:j_max] = frame_roi

    return frame


def float_frame_to_uint8_frame(float_frame):

    dtype = np.uint8
    dinfo = np.iinfo(dtype)
    float_frame = float_frame * dinfo.max
    float_frame[float_frame < dinfo.min] = dinfo.min
    float_frame[dinfo.max + 1 <= float_frame] = dinfo.max
    uint8_frame = float_frame.astype(dtype)

    return uint8_frame


def get_perturbed_frame(reference_image, perturbation_pattern, perturbation_amplitude, config):

    reference_frame = get_reference_frame(reference_image, config)
    perturbation_frame = get_perturbation_frame(perturbation_pattern)
    frame = reference_frame + perturbation_amplitude * perturbation_frame

    return frame


def get_combinations(reference_images_indices, perturbation_patterns_indices, perturbation_amplitudes_indices):

    index = 1
    combinations = {}

    for i in reference_images_indices:
        combinations[index] = (i, 0, 0)
        index += 1
        for j in perturbation_patterns_indices:
            for k in perturbation_amplitudes_indices:
                combinations[index] = (i, j, k)
                index += 1

    return combinations


def save_frame(path, frame):

    image = fromarray(frame)
    image.save(path)

    return


def get_grey_frame(luminance=0.5):

    # height = 1080  # DMD height
    # width = 1920  # DMD width
    height = 768  # DMD height
    width = 768  # DMD width
    shape = (height, width)
    dtype = np.float
    frame = luminance * np.ones(shape, dtype=dtype)

    return frame


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    path = config['path']
    if not os.path.isdir(path):
        os.makedirs(path)

    reference_images_path = os.path.join(path, "reference_images")
    if not os.path.isdir(reference_images_path):
        os.makedirs(reference_images_path)
    perturbation_patterns_path = os.path.join(path, "perturbation_patterns")
    if not os.path.isdir(perturbation_patterns_path):
        os.makedirs(perturbation_patterns_path)
    frames_path = os.path.join(path, "frames")
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)

    reference_images_indices = config['reference_images']['indices']
    for index in reference_images_indices:
        collect_reference_image(index, path=reference_images_path)

    perturbation_patterns_indices = config['perturbations']['pattern_indices']
    for index in perturbation_patterns_indices:
        collect_perturbation_pattern(index, path=perturbation_patterns_path)

    perturbation_amplitudes_indices = config['perturbations']['amplitude_indices']
    perturbation_amplitudes = {
        k: float(k) / np.iinfo(np.uint8).max
        for k in perturbation_amplitudes_indices
    }

    for index in reference_images_indices:
        _ = load_reference_image(index, reference_images_path)

    for index in perturbation_patterns_indices:
        _ = load_perturbation_pattern(index, perturbation_patterns_path)

    nb_reference_images = len(reference_images_indices)
    nb_perturbation_patterns = len(perturbation_patterns_indices)
    nb_perturbation_amplitudes = len(perturbation_amplitudes_indices)
    nb_images = 1 + nb_reference_images * (1 + nb_perturbation_patterns * nb_perturbation_amplitudes)

    combinations = get_combinations(reference_images_indices, perturbation_patterns_indices,
                                    perturbation_amplitudes_indices)

    display_rate = config['display_rate']
    frame_duration = config['frame']['duration']

    nb_repetitions = config['nb_repetitions']
    nb_combinations = len(combinations)
    nb_frame_displays = int(display_rate * frame_duration)
    assert display_rate * frame_duration == float(nb_frame_displays)
    nb_displays = nb_frame_displays + nb_repetitions * nb_combinations * 2 * nb_frame_displays

    display_time = float(nb_displays) / display_rate
    print("display time: {} s ({} min)".format(display_time, display_time / 60.0))
    # TODO improve feedback.

    # Create .bin file.
    bin_filename = "fipwc.bin"
    bin_path = os.path.join(path, bin_filename)
    bin_file = open_bin_file(bin_path, nb_images)
    # Get grey frame.
    grey_frame = get_grey_frame()
    grey_frame = float_frame_to_uint8_frame(grey_frame)
    # Save frame in .bin file.
    print(grey_frame.shape)
    bin_file.append(grey_frame)
    # Save frame as .png file.
    grey_frame_filename = "grey.png"
    grey_frame_path = os.path.join(frames_path, grey_frame_filename)
    save_frame(grey_frame_path, grey_frame)
    for reference_image_index in reference_images_indices:
        # Get reference frame.
        reference_image = load_reference_image(reference_image_index, reference_images_path)
        reference_frame = get_reference_frame(reference_image, config)
        reference_frame = float_frame_to_uint8_frame(reference_frame)
        # Save frame in .bin file.
        print(reference_frame.shape)
        bin_file.append(reference_frame)
        # Save frame as .png file.
        reference_frame_filename = "reference{:05d}.png".format(reference_image_index)
        reference_frame_path = os.path.join(frames_path, reference_frame_filename)
        save_frame(reference_frame_path, reference_frame)
    for reference_image_index in reference_images_indices:
        reference_image = load_reference_image(reference_image_index, reference_images_path)
        for perturbation_pattern_index in perturbation_patterns_indices:
            perturbation_pattern = load_perturbation_pattern(perturbation_pattern_index, perturbation_patterns_path)
            for perturbation_amplitude_index in perturbation_amplitudes_indices:
                perturbation_amplitude = perturbation_amplitudes[perturbation_amplitude_index]
                perturbed_frame = get_perturbed_frame(reference_image, perturbation_pattern, perturbation_amplitude,
                                                      config)
                perturbed_frame = float_frame_to_uint8_frame(perturbed_frame)
                # Save frame in .bin file.
                print(perturbed_frame.shape)
                bin_file.append(perturbed_frame)
                # Save frame as .png file.
                perturbed_frame_filename = "perturbed_r{:05d}_p{:05d}_a{:05d}.png".format(reference_image_index,
                                                                                          perturbation_pattern_index,
                                                                                          perturbation_amplitude_index)
                perturbed_frame_path = os.path.join(frames_path, perturbed_frame_filename)
                save_frame(perturbed_frame_path, perturbed_frame)
    bin_file.close()

    # Create .vec file.
    vec_filename = "fipwc.vec"
    vec_path = os.path.join(path, vec_filename)
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    grey_frame_id = 0
    for _ in range(0, nb_frame_displays):
        vec_file.append(grey_frame_id)
    for k in range(0, nb_repetitions):
        combination_indices = np.array(list(combinations.keys()))
        np.random.shuffle(combination_indices)
        for combination_index in combination_indices:
            combination_frame_id = combination_index
            for _ in range(0, nb_frame_displays):
                vec_file.append(combination_frame_id)
            for _ in range(0, nb_frame_displays):
                vec_file.append(grey_frame_id)
    vec_file.close()

    return
