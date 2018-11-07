import array
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import scipy.interpolate
import scipy.ndimage
import tempfile

from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL.Image import open as open_image
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve


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
            url = "http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/{f}/imk{i:05d}.{f}".format(i=index, f=format_)
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
        get_van_hateren_resource_locator(path, index, format_=format_, mirror=mirror)
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


def generate(args):

    _ = args  # TODO remove.

    # dataset = 'Palmer'
    dataset = 'van Hateren'  # TODO cache the van Hateren's natural images.

    # TODO collect the reference images (i.e. natural images).
    if dataset == 'Palmer':
        path = os.path.join("~", "spot_sensitivity_context_dependent")
        path = os.path.expanduser(path)
        urls = get_palmer_resource_locators(path=path, indices=[1])
    elif dataset == 'van Hateren':
        path = tempfile.TemporaryDirectory()
        urls = get_van_hateren_resource_locators(path=path, indices=[5])
    else:
        raise ValueError("unexpected dataset value: {}".format(dataset))
    # Print resource locators.
    for url in urls:
        print(url)
    # Load resources.
    resources = [
        load_resource(url)
        for url in urls
    ]
    # Convert resources to images.
    if dataset == 'Palmer':
        images = [
            convert_palmer_resource_to_image(resource)
            for resource in resources
        ]
    elif dataset == 'van Hateren':
        images = [
            convert_van_hateren_resource_to_image(resource)
            for resource in resources
        ]
    else:
        raise ValueError("unexpected dataset value: {}".format(dataset))

    # TODO clean the following experimental lines.

    image = images[0]

    # Plot reference image.
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='gray', vmin=-1.0, vmax=1.0)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)

    image_height, image_width = image.shape
    # image_resolution = 3.3  # µm / pixel  # fixed by the eye (monkey)
    image_resolution = 0.8  # µm / pixel  # fixed by the eye (salamander)

    frame_shape = frame_height, frame_width = 1080, 1920  # fixed by the DMD
    # frame_resolution = 0.42  # µm / pixel  # fixed by the setup  # TODO check this value.
    frame_resolution = 0.7  # µm / pixel

    background_luminance = 0.5

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
        mask_x = np.logical_and(np.min(image_x) - 0.5 * image_resolution <= frame_x, frame_x <= np.max(image_x) + 0.5 * image_resolution)
        frame_y = frame_resolution * np.arange(0, frame_width)
        frame_y = frame_y - np.mean(frame_y)
        mask_y = np.logical_and(np.min(image_y) - 0.5 * image_resolution <= frame_y, frame_y <= np.max(image_y) + 0.5 * image_resolution)
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
        mask_x = np.logical_and(np.min(image_x) - 0.5 * image_resolution <= frame_x, frame_x <= np.max(image_x) + 0.5 * image_resolution)
        frame_y = frame_resolution * np.arange(0, frame_width)
        frame_y = frame_y - np.mean(frame_y)
        mask_y = np.logical_and(np.min(image_y) - 0.5 * image_resolution <= frame_y, frame_y <= np.max(image_y) + 0.5 * image_resolution)
        frame_z = spline(frame_x[mask_x], frame_y[mask_y])
        frame_i_min = np.min(np.nonzero(mask_x))
        frame_i_max = np.max(np.nonzero(mask_x)) + 1
        frame_j_min = np.min(np.nonzero(mask_y))
        frame_j_max = np.max(np.nonzero(mask_y)) + 1
        frame = background_luminance * np.ones(frame_shape, dtype=np.float)
        frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_z

    mean_luminance = 0.5  # arb. unit
    std_luminance = 0.06  # arb. unit

    frame_roi = frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max]
    print("mean (luminance): {}".format(np.mean(frame_roi)))
    print("std (luminance): {}".format(np.std(frame_roi)))
    print("min (luminance): {}".format(np.min(frame_roi)))
    print("max (luminance): {}".format(np.max(frame_roi)))
    frame_roi = frame_roi - np.mean(frame_roi)
    frame_roi = frame_roi / np.std(frame_roi)
    frame_roi = frame_roi * std_luminance
    frame_roi = frame_roi + mean_luminance
    print("mean (luminance): {}".format(np.mean(frame_roi)))
    print("std (luminance): {}".format(np.std(frame_roi)))
    print("min (luminance): {}".format(np.min(frame_roi)))
    print("max (luminance): {}".format(np.max(frame_roi)))
    frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_roi

    # TODO clean the previous experimental lines..

    # Plot reference frame.
    fig, ax = plt.subplots()
    im = ax.imshow(frame, cmap='gray', vmin=0.0, vmax=1.0)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='2%', pad=0.05)
    plt.colorbar(im, cax=cax)

    # TODO clean the following experimental lines.

    # TODO create the image perturbations (i.e. the checkerboards).
    perturbation_shape = perturbation_height, perturbation_width = (14, 26)
    perturbation = np.random.choice(a=[-1.0, +1.0], size=perturbation_shape)
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
    mask_x = np.logical_and(np.min(perturbation_x) - 0.5 * perturbation_resolution <= frame_x, frame_x <= np.max(perturbation_x) + 0.5 * perturbation_resolution)
    mask_y = np.logical_and(np.min(perturbation_y) - 0.5 * perturbation_resolution <= frame_y, frame_y <= np.max(perturbation_y) + 0.5 * perturbation_resolution)
    frame_x_, frame_y_ = np.meshgrid(frame_x[mask_x], frame_y[mask_y])
    frame_x_ = frame_x_.transpose().flatten()
    frame_y_ = frame_y_.transpose().flatten()
    frame_points_ = np.stack((frame_x_, frame_y_), axis=-1)
    frame_data_ = interpolate(frame_points_)
    frame_z_ = np.reshape(frame_data_, (frame_x[mask_x].size, frame_y[mask_y].size))
    frame_i_min = np.min(np.nonzero(mask_x))
    frame_i_max = np.max(np.nonzero(mask_x)) + 1
    frame_j_min = np.min(np.nonzero(mask_y))
    frame_j_max = np.max(np.nonzero(mask_y)) + 1
    perturbation_frame = np.zeros(frame_shape, dtype=np.float)
    perturbation_frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_z_

    # Plot perturbation image.
    fig, ax = plt.subplots()
    im = ax.imshow(perturbation_frame, cmap='gray', vmin=-1.0, vmax=+1.0)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='2%', pad=0.05)
    plt.colorbar(im, cax=cax)

    # TODO create the perturbed images.

    # Create perturbed reference image.
    perturbation_amplitude = - 2.0 / (2.0 ** 8)
    frame_ = frame + perturbation_amplitude * perturbation_frame

    # Plot perturbed frame.
    fig, ax = plt.subplots()
    im = ax.imshow(frame_, cmap='gray', vmin=0.0, vmax=1.0)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='2%', pad=0.05)
    plt.colorbar(im, cax=cax)

    # TODO generate all the reference frames and all the perturbed frames.
    # TODO 1. get the list of reference images
    # TODO 2. get the list of perturbation patterns (or number of perturbation patterns to generate)
    # TODO 3. get the list of perturbation amplitudes
    # TODO 4. generate all the reference frames
    # TODO 5. generate all the perturbed frames

    # TODO create the .bin file.
    # TODO 1. open the .bin file.
    # TODO 2. for each reference frame
    # TODO   - load the frame
    # TODO   - append the frame to the .bin file
    # TODO 3. for each perturbation
    # TODO   - for each reference frame
    # TODO     - load the corresponding perturbed frame
    # TODO     - append the frame to the .bin file
    # TODO 4. close the .bin file

    # TODO create the interleaving of the perturbed images.
    # TODO 1. list all the possible combinations (i.e. reference image, perturbation pattern, perturbation amplitude)
    # TODO 2. for each repetition
    # TODO   - sample a random order of the combinations
    # TODO   - append this sequence to ...

    # TODO create the .vec file.

    # TODO clean the previous experimental lines.

    plt.show()  # TODO save images to disk instead.

    raise NotImplementedError()


def get_resource_locator(index, dataset='van Hateren', **kwargs):

    if dataset == 'van Hateren':
        url = get_van_hateren_resource_locator(index, **kwargs)
    elif dataset == 'Palmer':
        url = get_palmer_resource_locator(index, **kwargs)
    else:
        raise ValueError("unexpected dataset value: {}".format(dataset))

    return url


def is_reference_image(url):

    url = urlparse(url)
    if url.scheme == 'file':
        ans = os.path.isfile(url.path)
    elif url.scheme == 'http':
        ans = True
    else:
        raise ValueError("unexpected url scheme: {}".format(url.scheme))

    return ans


def collect_reference_images(indices=None, dataset='van Hateren', **kwargs):

    assert indices is not None

    for index in indices:
        local_url = get_resource_locator(index, dataset=dataset, scheme='file', **kwargs)
        if not is_reference_image(local_url):
            remote_url = get_resource_locator(index, dataset=dataset, scheme='http', **kwargs)
            local_url = urlparse(local_url)
            local_path = local_url.path
            urlretrieve(remote_url, local_path)

    # TODO for each reference image index
    # TODO get the local url
    # TODO   if the reference image doesn't exist locally
    # TODO     get the remote url
    # TODO     load the reference image
    # TODO     save the reference image

    # TODO complete

    return


def generate(args):

    _ = args
    path = '/tmp/tmps5xrjpxh'

    if path is None:
        path = tempfile.mkdtemp()
    else:
        if not os.path.isdir(path):
            os.makedirs(path)

    print(path)

    refrence_images_path = os.path.join(path, "reference_images")
    if not os.path.isdir(refrence_images_path):
        os.makedirs(refrence_images_path)
    collect_reference_images([1, 2, 3], path=refrence_images_path)
    collect_perturbation_patterns(path=os.path.join(path, "perturbation_patterns"))

    # TODO generate all the reference frames and all the perturbed frames.
    # TODO 1. get the list of reference images
    # TODO 2. get the list of perturbation patterns (or number of perturbation patterns to generate)
    # TODO 3. get the list of perturbation amplitudes
    # TODO 4. generate all the reference frames
    # TODO 5. generate all the perturbed frames

    return
