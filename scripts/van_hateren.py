import matplotlib.pyplot as plt
import numpy as np

import pystim.datasets.van_hateren as vh


# # Fetch dataset.
# vh.fetch(image_nbs=[1])


# Test load settings.
image_nb = 1
settings = vh.load_settings()
print("image {} settings: {}".format(image_nb, settings[image_nb]))


# Plot image.
# image_nb = 3119
image_nb = 5
luminance_data = vh.load_luminance_data(image_nb)
print("min. lum.: {}".format(np.min(luminance_data)))
print("max. lum.: {}".format(np.max(luminance_data)))
image = vh.load_image(image_nb)
image_settings = vh.load_image_settings(image_nb)
print("min. pixel value: {}".format(image.min))
print("max. pixel value: {}".format(image.max))
print("image setting: {}".format(image_settings))
ax = plt.subplot()
# ax.imshow(luminance_data, cmap='gray')
ax.imshow(image.data, cmap='gray')
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")


# image_nbs = vh.get_image_nbs()

# # Plot histogram of minimum luminances.
# min_values = vh.get_min_values()
# ax = plt.subplot()
# ax.hist(min_values, bins=200)
# ax.set_xlabel("min. pixel value")
# ax.set_ylabel("nb. images")

# # Plot histogram of maximum luminances.
# max_values = vh.get_max_luminances(verbose=True)
# _, ax = plt.subplots()
# ax.hist(max_values, bins=200)
# ax.set_xlabel("max. lum.")
# ax.set_ylabel("nb. images")
# max_value = np.max(max_values)
# print("max. lum: {}".format(max_value))
# image_nbs = vh.get_image_nbs()
# selection = np.nonzero(max_values == max_value)
# image_nbs = image_nbs[selection]
# print("image with max. lum.: {}".format(image_nbs))

# # Plot histogram of mean pixel values.
# mean_values = vh.get_mean_values()
# ax = plt.subplot()
# ax.hist(mean_values, bins=200)
# ax.set_xlabel("mean pixel value")
# ax.set_ylabel("nb. images")

# # Plot histogram of std pixel values.
# std_values = vh.get_std_values()
# ax = plt.subplot()
# ax.hist(std_values, bins=200)
# ax.set_xlabel("std pixel value")
# ax.set_ylabel("nb. images")

plt.show()
