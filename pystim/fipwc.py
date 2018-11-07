import array
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import scipy.interpolate

from PIL.Image import open as open_image
from urllib.request import urlopen


def load_resource(url):

    with urlopen(url) as handle:
        resource = handle.read()

    return resource


def get_palmer_resource_locator(index):

    assert 1 <= index <= 6

    frame_index = 1200 * (index - 1) + 600
    filename = "frame{i:04d}.png".format(i=frame_index)
    path = os.path.join("~", "spot_sensitivity_context_dependent", filename)
    path = os.path.expanduser(path)
    url = "file://localhost" + path

    return url


def get_palmer_resource_locators(indices=None):

    if indices is None:
        indices = range(1, 7)

    urls = [
        get_palmer_resource_locator(index)
        for index in indices
    ]

    return urls


def convert_palmer_resource_to_image(resource):

    data = io.BytesIO(resource)
    image = open_image(data)
    data = image.getdata()
    data = np.array(data, dtype=np.uint8)
    image = data.reshape(image.size)

    return image


def get_van_hateren_resource_locator(index, format_='iml', mirror='Lies'):

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

    return url


def get_van_hateren_resource_locators(indices=None, format_='iml', mirror='Lies'):

    assert format_ in ['iml', 'imc']

    if indices is None:
        if mirror == 'Lies':
            indices = range(0, 4213)
        elif mirror == 'Ivanov':
            indices = range(1, 100)
        else:
            raise ValueError("unexpected mirror value: {}".format(mirror))

    urls = [
        get_van_hateren_resource_locator(index, format_=format_, mirror=mirror)
        for index in indices
    ]

    return urls


def convert_van_hateren_resource_to_image(resource):

    data = array.array('H', resource)
    data.byteswap()
    data = np.array(data, dtype='uint16')
    image = data.reshape(1024, 1536)

    return image


def generate(args):

    _ = args  # TODO remove.

    # dataset = 'Palmer'
    dataset = 'van Hateren'  # TODO cache the van Hateren's natural images.

    # TODO collect the reference images (i.e. natural images).
    if dataset == 'Palmer':
        urls = get_palmer_resource_locators(indices=[1])
    elif dataset == 'van Hateren':
        urls = get_van_hateren_resource_locators(indices=[1])
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
    # Plot images.
    for image in images:
        plt.figure()
        plt.imshow(image, cmap='gray')

    image = images[0]
    x = np.linspace(0.0, 1.0, num=300)
    y = np.linspace(0.0, 1.0, num=300)
    z = image
    spline = scipy.interpolate.RectBivariateSpline(x, y, z, bbox=(0.0, 1.0, 0.0, 1.0))
    x_ = np.linspace(0.0, 1.0, num=1080)
    y_ = np.linspace(0.0, 1.0, num=1920)
    z_ = spline(x_, y_)
    image = z_

    plt.figure()
    plt.imshow(image, cmap='gray')

    plt.show()

    # TODO create the image perturbations (i.e. the checkerboards).
    # TODO create the perturbed images.
    # TODO create the grey image.
    # TODO create the .bin file.
    # TODO create the interleaving of the perturbed images.
    # TODO create the .vec file.

    raise NotImplementedError()
