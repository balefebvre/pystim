import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

import pystim.datasets.van_hateren as vh


verbose = True


plt.close()


# # Fetch dataset.
# vh.fetch(image_nbs=[1])


# Test load settings.
image_nb = 1
settings = vh.load_settings()
print("image {} settings: {}".format(image_nb, settings[image_nb]))


# # Plot image.
# # image_nb = 3119
# # image_nb = 5
# # image_nb = 31
# image_nb = 46
# luminance_data = vh.load_luminance_data(image_nb)
# print("min. lum.: {}".format(np.min(luminance_data)))
# print("max. lum.: {}".format(np.max(luminance_data)))
# image = vh.load_image(image_nb)
# image_settings = vh.load_image_settings(image_nb)
# print("min. pixel value: {}".format(image.min))
# print("max. pixel value: {}".format(image.max))
# print("image setting: {}".format(image_settings))
# ax = plt.subplot()
# # ax.imshow(luminance_data, cmap='gray')
# ax.imshow(image.data, cmap='gray', vmin=image.inf, vmax=image.sup)
# ax.set_xlabel("x (px)")
# ax.set_ylabel("y (px)")


# Create PNG for each image.
image_nbs = vh.get_image_nbs()
for image_nb in tqdm.tqdm(image_nbs):
    path = vh.get_path(image_nb)
    path = os.path.splitext(path)[0] + ".png"
    if not os.path.isfile(path):
        image = vh.load_image(image_nb)
        image.save(path)


# Load metadata.
image_nb = 1
image_metadata = vh.load_image_metadata(image_nb)
print(image_metadata)

image_nbs = vh.get_image_nbs()


# # Plot histogram of minimum luminances.
# min_luminances = vh.get_min_luminances(verbose=verbose)
# _, ax = plt.subplots()
# ax.hist(min_luminances, bins=200)
# ax.set_xlabel("min. lum. (cd/m²)")
# ax.set_ylabel("nb. images")


# # Plot histogram of maximum luminances.
# max_luminances = vh.get_max_luminances(verbose=verbose)
# _, ax = plt.subplots()
# ax.hist(max_luminances, bins=200)
# ax.set_xlabel("max. lum. (cd/m²)")
# ax.set_ylabel("nb. images")


# max_luminance = np.max(max_luminances)
# print("max. lum (cd/m²): {}".format(max_luminance))
# selection = np.nonzero(max_luminances == max_luminance)
# print("image with max. lum.: {}".format(image_nbs[selection]))


# # Plot histogram of mean luminances.
# mean_luminances = vh.get_mean_luminances(verbose=verbose)
# _, ax = plt.subplots()
# ax.hist(mean_luminances, bins=200)
# ax.set_xlabel("mean lum. (cd/m²)")
# ax.set_ylabel("nb. images")


# # Plot histogram of median luminances.
# median_luminances = vh.get_median_luminances(verbose=verbose)
# _, ax = plt.subplots()
# ax.hist(median_luminances, bins=200)
# ax.set_xlabel("median lum. (cd/m²)")
# ax.set_ylabel("nb. images")


# # Plot histogram of std luminances.
# std_luminances = vh.get_std_luminances(verbose=verbose)
# _, ax = plt.subplots()
# ax.hist(std_luminances, bins=200)
# ax.set_xlabel("std lum. (cd/m²)")
# ax.set_ylabel("nb. images")


# # Plot histogram of mad luminances.
# mad_luminances = vh.get_mad_luminances(verbose=verbose)
# _, ax = plt.subplots()
# ax.hist(mad_luminances, bins=200)
# ax.set_xlabel("mad lum. (cd/m²)")
# ax.set_ylabel("nb. images")


# # Define saturation values (by hand).
# settings = vh.load_settings()
# unique_settings = settings.get_unique()
# for s in unique_settings:
#     image_nbs = settings.get_image_nbs(**s)
#     max_values = vh.get_max_values(image_nbs=image_nbs)
#     max_values.sort()
#     _, ax = plt.subplots()
#     ax.plot(max_values, marker='.')
#     iso_value = s['iso']
#     if iso_value == 200:
#         v_th = 6266
#     elif iso_value == 400:
#         v_th = 12551
#     elif iso_value == 800:
#         v_th = 25102
#     else:
#         raise NotImplementedError
#     ax.axhline(v_th, color='tab:red')
#     ax.set_xlabel("image nb. (sorted by max. value)")
#     ax.set_ylabel("max. values")
#     ax.set_title("iso: {iso}, aperture: {aperture}, shutter: {shutter}".format(**s))
#     plt.show()


# # Plot histogram of saturation factors.
# saturation_factors = vh.get_saturation_factors(verbose=verbose)
# _, ax = plt.subplots()
# ax.hist(100.0 * saturation_factors, bins=200)
# ax.set_xlabel("saturation factor (%)")
# ax.set_ylabel("nb. images")
# _, ax = plt.subplots()
# # ...
# saturation_factors_sorted = np.sort(saturation_factors)
# ax.plot(100.0 * saturation_factors_sorted, marker='.')
# ax.set_xlabel("image nb. (sorted by sat. factor)")
# ax.set_ylabel("sat. factor")

# # Display saturated image numbers.
# are_saturated = vh.get_are_saturated(verbose=verbose)
# saturated_image_nbs = image_nbs[are_saturated]
# print("saturated images: {}".format(saturated_image_nbs))
# print("nb. saturated images: {} ({} %)".format(len(saturated_image_nbs), 100.0 * len(saturated_image_nbs) / len(image_nbs)))


# Plot random saturates image.
are_saturated = vh.get_are_saturated()
saturated_image_nbs = image_nbs[are_saturated]
image_nb = np.random.choice(saturated_image_nbs)
image = vh.load_normalized_image(image_nb, max_luminance='image')
saturation_mask = vh.load_saturation_mask(image_nb)
print("nb. saturated pixels: {}".format(np.count_nonzero(saturation_mask)))
print("nb. pixels: {}".format(np.size(saturation_mask)))
print("saturation factor: {}".format(100.0 * np.count_nonzero(saturation_mask) / np.size(saturation_mask)))
data = np.transpose(image.data)
data = np.flipud(data)
red_data = np.copy(data)
red_data[saturation_mask] = 255
green_data = np.copy(data)
green_data[saturation_mask] = 0
blue_data = np.copy(data)
blue_data[saturation_mask] = 0
image_data = np.dstack((red_data, green_data, blue_data))
_, ax = plt.subplots()
ax.imshow(image_data)


# # Plot histogram of maximum luminances (for non saturated images).
are_saturated = vh.get_are_saturated()
are_not_saturated = np.logical_not(are_saturated)
not_saturated_image_nbs = image_nbs[are_not_saturated]
mean_luminances = vh.get_mean_luminances(image_nbs=not_saturated_image_nbs, verbose=True)
std_luminances = vh.get_std_luminances(image_nbs=not_saturated_image_nbs, verbose=True)
max_luminances = vh.get_max_luminances(image_nbs=not_saturated_image_nbs, verbose=True)
max_centered_luminances = (max_luminances / mean_luminances - 1.0) / std_luminances + 1.0
max_normalized_luminances = (max_luminances - mean_luminances) / std_luminances
# selection = max_luminances < 18018.0
# not_saturated_image_nbs = not_saturated_image_nbs[selection]
# max_luminances = max_luminances[selection]
_, ax = plt.subplots()
ax.hist(100.0 * max_luminances, bins=200)
ax.set_xlabel("max. lum. (cd/m²)")
ax.set_ylabel("nb. images")
_, ax = plt.subplots(nrows=3)
# ...
ml_indices = np.argsort(max_luminances)
mcl_indices = np.argsort(max_centered_luminances)
mnl_indices = np.argsort(max_normalized_luminances)
# ...
max_luminances_sorted = max_luminances[ml_indices]
max_centered_luminances_sorted = max_centered_luminances[mcl_indices]
max_normalized_luminances_sorted = max_normalized_luminances[mnl_indices]
ax[0].plot(max_luminances_sorted, marker='.', color='tab:blue')
ax[0].set_xlabel("image nb. (sorted by max. lum.)")
ax[0].set_ylabel("max. lum. (cd/m²)")
ax[1].plot(max_centered_luminances_sorted, marker='.', color='tab:red')
ax[1].set_xlabel("image nb. (sorted by max. cen. lum.)")
ax[1].set_ylabel("max. cen. lum. (cd/m²)")
ax[2].plot(max_normalized_luminances_sorted, marker='.', color='tab:green')
ax[2].set_xlabel("image nb. (sorted by max. nor. lum.)")
ax[2].set_ylabel("max. nor. lum.")


# Plot image with min., median and max. luminance.
target_lum = np.median(max_luminances)
index = np.where(max_luminances == target_lum)[0][0]
image_nb = not_saturated_image_nbs[index]
# image = vh.load_image(image_nb)
# data = image.data
data = vh.load_luminance_data(image_nb)
data = np.transpose(data)
data = np.flipud(data)
fig, ax = plt.subplots()
im = ax.imshow(data, cmap='gray', vmin=0)
fig.colorbar(im, label='lum. (cd/m²)', fraction=0.146, pad=0.04)
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
ax.set_title("van Hateren {}".format(image_nb))
# fig.tight_layout()


# Plot histogram of generated image.
# image_name = 'image_0001'
# image_name = 'image_0694'
image_name = 'image_1_image'
image_path = '/tmp/pystim/fipwfc/images/{}.png'.format(image_name)
from pystim.images.png import load as load_png
image = load_png(image_path)
data = image.data
fig, ax = plt.subplots()
im = ax.imshow(data, cmap='gray', vmin=0, vmax=255)
fig.colorbar(im, label='grey value', fraction=0.146, pad=0.04)
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
ax.set_title(image_name)
fig, ax = plt.subplots()
ax.hist(np.ravel(image.data), bins=255, range=(0, 255), color='black')
ax.set_xlabel('grey value')
ax.set_ylabel('nb. pixels')
ax.set_title(image_name)


# TODO remove a two pixel wide border all around the images.
# TODO optimize the normalization procedure (find the optimal std reduction).

# # Plot histogram of saturation luminances.
# are_saturated = vh.get_are_saturated(verbose=verbose)
# saturated_image_nbs = image_nbs[are_saturated]
# saturation_luminances = vh.get_max_luminances(image_nbs=saturated_image_nbs)
# _, ax = plt.subplots()
# ax.hist(saturation_luminances, bins=200)
# ax.set_xlabel("saturation lum.")
# ax.set_ylabel("nb. images")


# saturation_factors = vh.get_saturation_factors(image_nbs=saturated_image_nbs)
# unique_saturation_luminances, counts = np.unique(saturation_luminances, return_counts=True)
# median_saturation_factors = np.array([
#     np.median(saturation_factors[saturation_luminances == usl])
#     for usl in unique_saturation_luminances
# ])
# # indices = np.argsort(counts)
# indices = np.argsort(median_saturation_factors)
# unique_saturation_luminances = unique_saturation_luminances[indices]
# counts = counts[indices]
# median_saturation_factors = median_saturation_factors[indices]
# for usl, c, msf in zip(unique_saturation_luminances, counts, median_saturation_factors):
#     print("{}, {}, {:.2f} %".format(usl, c, 100.0 * msf))


# # Display the settings of saturated images for a given saturation luminance.
# are_saturated = vh.get_are_saturated(verbose=verbose)
# saturated_image_nbs = image_nbs[are_saturated]
# saturation_luminances = vh.get_max_luminances(image_nbs=saturated_image_nbs)
# min_sat_lum = 4403.0
# max_sat_lum = 4481.0
# selection = np.logical_and(
#     min_sat_lum <= saturation_luminances,
#     saturation_luminances <= max_sat_lum
# )
# selected_image_nbs = saturated_image_nbs[selection]
# for image_nb in selected_image_nbs:
#     image_settings = vh.load_image_settings(image_nb)
#     print("{}: {}".format(image_nb, image_settings))


# # Plot histogram of min. values.
# min_values = vh.get_min_values(verbose=verbose)
# _, ax = plt.subplots()
# ax.hist(min_values, bins=200)
# ax.set_xlabel("min. values (grey level)")
# ax.set_ylabel("nb. images")


# # Plot histogram of max. values.
# max_values = vh.get_max_values(verbose=verbose)
# _, ax = plt.subplots()
# ax.hist(max_values, bins=200)
# ax.set_xlabel("max. values (grey level)")
# ax.set_ylabel("nb. images")


# saturation_factors = vh.get_saturation_factors(image_nbs=saturated_image_nbs)
# max_values = vh.get_max_values(image_nbs=saturated_image_nbs)
# unique_max_values, counts = np.unique(max_values, return_counts=True)
# median_saturation_factors = np.array([
#     np.median(saturation_factors[max_values == uml])
#     for uml in unique_max_values
# ])
# # indices = np.argsort(counts)
# indices = np.argsort(median_saturation_factors)
# unique_max_values = unique_max_values[indices]
# counts = counts[indices]
# median_saturation_factors = median_saturation_factors[indices]
# for umv, c, msf in zip(unique_max_values, counts, median_saturation_factors):
#     print("{}, {}, {:.2f} %".format(umv, c, 100.0 * msf))


# # max_value = 6306
# # max_value = 6286
# # max_value = 6308
# max_value = 1462
# indices = np.where(max_values == max_value)[0]
# index = indices[0]
# image_nb = saturated_image_nbs[index]
# print("image_nb: {}".format(image_nb))
# # image = vh.load_image(image_nb)
# image = vh.load_raw_image(image_nb)
# _, ax = plt.subplots()
# ax.imshow(image.data, cmap='gray', vmin=0.0)
# ax.set_xlabel("x (px)")
# ax.set_ylabel("y (px)")


# _, ax = plt.subplots()
# # range_ = (6000, 6256)
# range_ = None
# ax.hist(np.ravel(image.data), bins=200, range=range_)
# ax.set_xlabel("value (grey level)")
# ax.set_ylabel("nb. pixels")


plt.show()
