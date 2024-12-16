import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from icecream import ic
from tqdm import tqdm

from ImageProcessing import load_bin_file
from TraceIO import hdf_data, timeit


def radial_average(image, step=0.25, show=False):
    mean_im = np.zeros_like(image, dtype=float)
    for r in np.arange(0, np.min(image.shape) // 2, step):
        mask = bandpass_filter(
            image,
            high=r - step / 2,
            low=r + step / 2,
            width=step,
            centered=True,
            cut_offset=False,
        )
        mask = mask * (np.sum(mask * image) / np.sum(mask))
        mean_im += mask
    image /= np.sum(image)
    mean_im /= np.sum(mean_im)
    if show:
        imshow_multiple(
            [image, mean_im, np.abs(image - mean_im)],
            titles=["Image", "Mean image", "Difference"],
            vrange=[0, 5 / np.prod(image.shape)],
        )
    return mean_im


def show_frames(frames, circles=None):
    window_name = "Frames"
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 300, 300)

    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        try:
            circle = circles[i]
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])

            cv2.circle(frame, center, radius, (0, 0, 255), 1)  # Red color, thickness 2
        except (TypeError, IndexError):
            pass
        cv2.imshow(window_name, frame)
        cv2.waitKey(100)
    cv2.destroyAllWindows()


def distance_from_center(image, offset=[0, 0]):
    (rows, cols) = image.shape
    center_row, center_col = offset[0] + rows // 2, offset[1] + cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)
    return distance


def bandpass_filter(
    image, low=None, high=None, width=1, centered=False, cut_offset=True
):
    np.seterr(over="ignore")
    r = distance_from_center(image)

    with np.errstate(over="ignore"):
        mask = np.ones_like(r)
        if low is not None:
            mask *= 1 / (1 + np.exp(-(low - r) / (width / 4)))
        if high is not None:
            mask *= 1 / (1 + np.exp(-(r - high) / (width / 4)))
    if cut_offset:
        mask[len(mask) // 2] = 0
    if not centered:
        mask = np.fft.ifftshift(mask)
    return mask


def quadratic_fit(data):
    def model(p, x, y):
        a, b, c, d, e, f = p
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

    def error(p, x, y, z):
        return np.sum((model(p, x, y) - z) ** 2)

    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x = x.flatten()
    y = y.flatten()
    z = data.flatten()

    p0 = np.zeros(6)
    res = minimize(error, p0, args=(x, y, z))

    a, b, c, d, e, f = res.x
    x_max = (2 * b * d - c * e) / (c**2 - 4 * a * b)
    y_max = (2 * a * e - c * d) / (c**2 - 4 * a * b)

    return x_max, y_max


def imshow_multiple(images, titles=None, circles=None, radius=10, vrange=None):
    if titles is None:
        titles = [f"Image {i}" for i in range(len(images))]
    size = 5
    n = len(images)
    _, axes = plt.subplots(1, n, figsize=(n * size, 5))
    for i, (im, title) in enumerate(zip(images, titles)):

        rows, cols = im.shape
        extent = [-cols // 2, cols // 2, -rows // 2, rows // 2]
        im_plot = axes[i].imshow(im, cmap="gray", origin="lower", extent=extent)
        axes[i].set_title(title)

        if vrange is not None:
            im_plot.set_clim(vrange)

        # Create an axes divider and append a colorbar to the right of the image
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_plot, cax=cax)

        try:
            axes[i].scatter(*circles[i][:2], c="r", marker="+")
            circle = plt.Circle(
                circles[i][:2], radius=circles[i][-1], color="r", fill=False
            )
            axes[i].add_artist(circle)
        except (IndexError, TypeError):
            pass
    plt.show()


def find_peak(image, centered=True):
    max_index = np.argmax(image)
    max_indices = np.asarray(np.unravel_index(max_index, image.shape))

    roi = image[
        max_indices[0] - 1 : max_indices[0] + 2,
        max_indices[1] - 1 : max_indices[1] + 2,
    ]
    sub_pixel_max = quadratic_fit(roi)
    sub_pixel_max = np.asarray(
        (
            sub_pixel_max[0] + max_indices[1] - 0.5,
            sub_pixel_max[1] + max_indices[0] - 0.5,
        )
    )

    if centered:
        sub_pixel_max -= np.asarray(image.shape) // 2
    return sub_pixel_max


def find_bead_center(image, centered=True, show=False):
    mask = bandpass_filter(
        image, high=15, low=25, width=2, centered=False, cut_offset=True
    )
    fft = mask * np.fft.fft2(image)

    cc = np.fft.fftshift(np.abs(np.fft.ifft2(fft**2))) / np.prod(image.shape)
    peak = find_peak(cc)
    coords = peak / 2

    if show:
        imshow_multiple([image, cc], circles=[[*coords, 15], [*peak, 5]])
    if not centered:
        coords += np.asarray(image.shape) // 2
    return coords


def get_z(image, z_ref, lut, mask, show=True):
    fft = np.abs(np.fft.fftshift(np.fft.fft2(image))) * mask
    fft /= np.sum(fft)
    diff = np.asarray([np.sum(np.abs(l - fft) ** 0.5) for l in lut])

    # diff = np.exp(-diff) ** 0.5
    # diff /= np.sum(diff)

    # fit parabola to peak and yield maximum position
    width = 0.5
    index = np.argmin(diff)
    selection = np.abs(z_ref - z_ref[index]) < width * 2
    weight = np.exp(-((z_ref[selection] - z_ref[index]) ** 2) / width**2)
    weight = np.ones_like(z_ref[selection])

    poly = np.polyfit(z_ref[selection], diff[selection], 2, w=np.asarray(weight))
    new_z = -poly[1] / (2 * poly[0])

    if show:
        plt.plot(z_ref, diff, marker="o")
        plt.vlines(z_ref[index], 0, 1, color="r")
        plt.vlines(new_z, 0, 1, color="g")

        x_fit = np.linspace(z_ref[selection][0], z_ref[selection][-1], 100)
        fit = np.polyval(poly, x_fit)
        plt.plot(x_fit, fit)
        plt.ylim([0, 1.3 * np.max(diff)])
        plt.show()

    return new_z


def create_lut(images, average=None):
    mask = bandpass_filter(images[0], high=5, low=35, width=5, centered=True)
    size = np.prod(images[0].shape)
    lut = [np.abs(mask * np.fft.fftshift(np.fft.fft2(im))) / size for im in images]
    lut = [l / np.sum(l) for l in lut]
    if average is not None:
        lut = [radial_average(l, step=average) for l in tqdm(lut, desc="Averaging LUT")]
    return lut, mask


def process_lut(z, images, show=False):
    z_ref, lut, mask = create_lut(z, images, average=0.25)

    if show:
        vmax = 10 / np.sum(np.prod(images[0].shape))
        imshow_multiple([lut[10], lut[40], lut[-10]], vrange=[0, vmax])

    image_nr = np.random.randint(0, len(images))

    new_z = get_z(images[image_nr].T, z_ref, lut, mask, show=True)

    return

    z = [get_z(im, z_ref, lut, mask) for im in images]
    plt.plot(z_ref, z)

    plt.show

    # imange_nr = np.random.randint(0, len(images))

    # diff = np.asarray([np.sum((l - lut[imange_nr].T) ** 2) for l in lut])
    # diff = np.exp(-(diff**3))
    # diff /= np.sum(diff)
    # plt.plot(z, diff, marker="o")
    # plt.vlines(z[imange_nr], 0, 0.25, color="r")
    # plt.show()

    return lut


def test(z, frames):
    process_lut(z, frames)


if __name__ == "__main__":
    # print("This is a module, not a standalone script.")
    filename = r"d:\users\noort\data\20241211\data_153.hdf"
    filename = r"d:\users\noort\data\20241212\data_006.hdf"
    frames = load_bin_file(filename)
    data = hdf_data(filename)

    z = data.traces["Focus (mm)"].values * 1000
    lut, mask = create_lut(frames, average=0.25)

    z_new = [get_z(im, z, lut, mask, show=False) for im in frames]
    z = np.asarray(z_new)
    z_new = [get_z(im, z, lut, mask, show=False) for im in frames]

    weight = np.ones_like(z)
    weight[:10] = 0
    weight[-10:] = 0
    poly = np.polyfit(z, z_new, 1, w=weight)
    fit = np.polyval(poly, z)

    plt.plot(z, z_new - fit, marker="o")
    plt.ylim([-0.25, 0.25])
    plt.show()

    # test(z, frames)

    # find_bead_center(frames[-19], show=True)
    # process_lut(z, frames)

    # show_frames(frames)

    # ac = find_center(frames[-25])

    # ac = [find_center(frame) for frame in frames]

    # center = [np.argmax(a) for a in ac]

    # plt.imshow(find_center(frames[10]))
    # plt.colorbar()
    # plt.show()
