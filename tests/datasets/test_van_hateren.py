import pystim


def test_fetch_van_hateren():

    image_nbs = list(range(1, 10))
    pystim.datasets.fetch_van_hateren(image_nbs=image_nbs)

    return
