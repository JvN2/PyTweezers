import cv2
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from icecream import ic
from tqdm import tqdm
import tkinter as tk
import warnings
from pathlib import Path


from ImageProcessing import load_bin_file
from TraceIO import hdf_data, timeit


import tkinter as tk
from tkinter import ttk
import threading
import time


def gausian_filter(y, sigma, x=None):

    def mask(x, sigma):
        mask = np.exp(-(x**2) / (2 * sigma**2))
        return mask / np.sum(mask)

    if x is None:
        x = np.arange(len(y))

    y = [np.sum(mask(x - i, sigma) * y) for i in x]
    return np.asarray(y)


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_max = (2 * b * d - c * e) / (c**2 - 4 * a * b)
        y_max = (2 * a * e - c * d) / (c**2 - 4 * a * b)

    return x_max, y_max


def distance_from_center(image, offset=np.zeros(2)):
    (rows, cols) = image.shape
    center_row, center_col = rows // 2 - offset[0], cols // 2 - offset[1]
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)
    return distance


def bandpass_filter(
    image,
    low=None,
    high=None,
    width=1,
    centered=False,
    cut_dc=True,
    offset=np.zeros(2),
    normalised=False,
):
    np.seterr(over="ignore")
    r = distance_from_center(image, -offset)

    with np.errstate(over="ignore"):
        mask = np.ones_like(r)
        if low is not None:
            mask *= 1 / (1 + np.exp(-(low - r) / (width / 4)))
        if high is not None:
            mask *= 1 / (1 + np.exp(-(r - high) / (width / 4)))
    if cut_dc:
        mask[len(mask) // 2] = 0
    if not centered:
        mask = np.fft.ifftshift(mask)
    if normalised:
        mask /= np.sum(mask)
    return mask


def filter_image(image, low=None, high=None, width=1):
    filter = bandpass_filter(image, low, high, width, centered=False, cut_dc=True)
    im = np.fft.ifft2(np.fft.fft2(image) * filter)
    return np.real(im)


def imshow_multiple(
    images,
    titles=None,
    circles=None,
    vrange=None,
    ncols=3,
    x_extent=None,
    y_extent=None,
):
    if titles is None:
        titles = [f"Image {i}" for i in range(len(images))]
    size = 5

    if type(images) is list:
        try:
            images = np.asarray(images)
            if images.ndim == 2:
                images = [images]
        except ValueError:
            pass

    ncols = min(ncols, len(images))
    nrows = np.ceil(len(images) / ncols).astype(int)

    _, axes = plt.subplots(nrows, ncols=ncols, figsize=(ncols * size, nrows * size))

    # Ensure axes is always a 2D array for consistent indexing
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)
    axes = axes.flatten()

    for i, (im, title) in enumerate(zip(images, titles)):
        rows, cols = im.shape
        extent = [-cols // 2, cols // 2 - 1, -rows // 2, rows // 2 - 1]
        if x_extent is not None:
            extent[:1] = x_extent
        if y_extent is not None:
            extent[2:] = y_extent
        im_plot = axes[i].imshow(im, cmap="gray", origin="lower", extent=extent)
        axes[i].set_title(title)

        if vrange is not None:
            im_plot.set_clim(vrange)

        # Create an axes divider and append a colorbar to the right of the image
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im_plot, cax=cax)

        # Format the colorbar ticks to use scientific notation
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        cbar.ax.yaxis.set_major_formatter(formatter)

        try:
            axes[i].scatter(*circles[i][:2], c="r", marker="+", s=5000)
            circle = plt.Circle(
                circles[i][:2], radius=circles[i][-1], color="r", fill=False
            )
            axes[i].add_artist(circle)
        except (IndexError, TypeError):
            pass

    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show(block=False)

    # Move the window to the top left corner
    def move_window():
        window = plt.get_current_fig_manager().window
        window.wm_geometry("+0+0")

    def on_close(event=None):
        plt.close("all")
        root.quit()

    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.after(1, move_window)  # Move the window after a short delay
    window = plt.get_current_fig_manager().window
    window.bind("<Destroy>", on_close)

    root.mainloop()


def find_peak1d(y, x, width=0.5, show=False):
    # fit parabola to peak and yield maximum position
    max_index = np.argmax(y)
    weight = np.exp(-((x[max_index] - x) ** 2) / width**2)
    selection = weight > 0.01
    poly = np.polyfit(x[selection], y[selection], 2, w=weight[selection])
    x_max = -poly[1] / (2 * poly[0])

    if show:
        plt.figure(figsize=(12, 3))
        plt.plot(x, y, marker="o", color="blue", alpha=0.3, linestyle="None")
        x_fit = np.linspace(x[selection][0], x[selection][-1], 100)
        fit = np.polyval(poly, x_fit)
        plt.plot(x_fit, fit, color="blue")
        plt.ylim([1.2 * np.min(y), 1.2 * np.max(y)])
        plt.xlim([x[0], x[-1]])

        plt.plot(width)
        plt.tight_layout(pad=2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    return x_max


def find_peak2d(image, centered=True):
    max_index = np.argmax(image)
    max_indices = np.asarray(np.unravel_index(max_index, image.shape))

    image /= np.sum(image)
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


def find_bead_center(image, fft_mask, show=False, shift_image_to_center=False):
    fft = fft_mask * np.fft.fftshift(np.fft.fft2(image))
    autocorrelation = np.fft.ifftshift(
        np.abs(np.fft.ifft2(fft**2)) / np.prod(image.shape)
    )

    coords = find_peak2d(autocorrelation) / 2

    if shift_image_to_center:
        image = shift(image, coords)
        image[image == 0] = np.median(image)
        image = np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fftshift(fft_mask)))
        coords *= 0
        return image

    if show:
        imshow_multiple(
            [image, autocorrelation],
            circles=[[*coords, 15], [*coords * 2, 5]],
            titles=[f"Image ({coords[0]:.2f})", "CC"],
        )

    return coords


def find_focus(lut, z, show=False):
    mask1 = bandpass_filter(
        lut[0],
        high=10,
        low=20,
        width=0.5,
        centered=True,
        cut_dc=True,
        normalised=True,
    )
    mask2 = bandpass_filter(
        lut[0],
        high=10,
        low=30,
        width=0.5,
        centered=True,
        cut_dc=True,
        normalised=True,
    )

    fft = [np.abs(np.fft.fftshift(np.abs(np.fft.fft2(im)))) for im in lut]
    width = [
        (np.sum(mask2 * im) - np.sum(mask1 * im))
        / (np.sum(mask2 * im) + np.sum(mask1 * im))
        for im in fft
    ]

    z_range = 0.5
    z_focus = find_peak1d(-np.asarray(width), z, width=z_range, show=show)

    mask3 = bandpass_filter(
        lut[0],
        high=8,
        low=35,
        width=1,
        centered=False,
        cut_dc=True,
        normalised=True,
    )

    if show:
        z_ranges = [-z_range, 0, z_range]

        indices_to_display = [
            np.argmin((z - z_focus - z_step) ** 2) for z_step in z_ranges
        ]

        titles = [
            f"focus -{z_ranges[-1]} um",
            f"focus = {z_focus:.2f} um",
            f"focus +{z_ranges[-1]} um",
        ]

        imshow_multiple(lut[indices_to_display], titles=titles)

    return z_focus


def find_modulation(y, x, show=False):
    # remove outliers
    np.clip(y, np.percentile(y, 5), np.percentile(y, 95))

    # Compute the FFT of the data to make initial guesses for the fit
    N = len(y)
    dx = (x[-1] - x[0]) / (len(x) - 1)
    yf = np.fft.fft(y)[: N // 2]
    xf = np.fft.fftfreq(N, dx)[: N // 2]
    peak = np.argmax(np.abs(yf))
    amplitude = np.percentile(np.abs(y), 80)
    period = 1 / xf[peak]
    phase = np.angle(yf[peak])

    def func(x, p, phi, a=amplitude):
        return a * np.sin(2 * np.pi * x / p + phi)

    popt, _ = curve_fit(
        lambda x, p, phi: func(x, p, phi, amplitude),
        x,
        y,
        p0=[period, phase],
    )

    fit = func(x, *popt, amplitude)

    if show:
        plt.figure(figsize=(12, 3))
        plt.plot(x, y, "o", label="Data", color="blue", alpha=0.3)
        # plt.plot(x, y - fit, "o", label="Data", color="red", alpha=0.3)
        x = np.linspace(x[0], x[-1], 1000)
        plt.plot(x, func(x, *popt, amplitude), label="Fit", color="blue")
        plt.hlines(0, x[0], x[-1], color="k")
        plt.xlim([x[0], x[-1]])
        plt.ylim([-0.5, 0.5])
        plt.tight_layout(pad=2)
        plt.xlabel("z_lut (um)")
        plt.ylabel("correction (um)")
        plt.show()

    return fit


class Tracker:
    def __init__(self, filename=None):

        if filename is None:
            # Open file dialog to select a file
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            filename = tk.filedialog.askopenfilename(
                title="Select a file",
                filetypes=(("bin files", "*.bin"), ("All files", "*.*")),
            )
            if not filename:
                raise ValueError("No file selected")

        filename = Path(filename).with_suffix(".hdf")
        self.filename = filename

        # correct focus for Focus recording errors
        focus = hdf_data(filename).traces["Focus (mm)"].values
        t = hdf_data(filename).traces["Time (s)"].values
        fit = np.polyval(np.polyfit(t, focus, 1), t)

        # correct for refractive index
        self.z_lut = fit * 1000 / 1.3333

        images = load_bin_file(filename)
        self.mask = bandpass_filter(images[0], high=8, low=25, width=2.5, centered=True)

        self.lut = self._create_lut(images, average=None)
        self.z_lut -= find_focus(images, self.z_lut, show=False)

        # mask = bandpass_filter(
        #     images[0], high=15, low=25, width=2, centered=False, cut_dc=True
        # )
        # centers = [find_bead_center(image, self.mask, show=False) for image in images]

        i_focus = np.argmin(np.abs(self.z_lut))
        find_bead_center(images[i_focus], self.mask, show=False)

        # z_correction = find_modulation(z_new - z_old, z_new, show=True)

        resample = True
        if resample:
            self._resample_lut(dz=0.5, show=False)

    def _create_lut(self, images, average=None):
        lut = [self.mask * np.fft.fftshift(np.abs(np.fft.fft2(im))) for im in images]
        lut = [l / np.sum(l) for l in lut]

        if average is not None:
            masks = None
            for i, l in enumerate(tqdm(lut, desc="Averaging LUT")):
                lut[i], masks = self._radial_average(l, masks, step=average)
        return np.asarray(lut)

    def _resample_lut(self, dz=0.3, show=False):
        # Resample lut to z_new
        z_new = np.arange(np.min(self.z_lut), np.max(self.z_lut), dz)

        new_lut = []
        for z in z_new:
            weight = np.exp(-(((self.z_lut - z) / dz) ** 2))
            weight /= np.sum(weight)
            lut = np.sum(self.lut * weight[:, np.newaxis, np.newaxis], axis=0)
            new_lut.append(lut / np.sum(lut))

        new_lut = np.asarray(new_lut)

        if show:
            width = new_lut.shape[-1] // 2
            ims = [self.lut[:, :, width], new_lut[:, :, width]]

            imshow_multiple(
                ims,
                titles=["original", "resampled"],
                ncols=3,
                vrange=[0, 1 * np.percentile(ims[-1], 95)],
                y_extent=[self.z_lut[0], self.z_lut[-1]],
            )

            plt.figure(figsize=(12, 3))

            mask1 = bandpass_filter(
                new_lut[0],
                high=30,
                low=40,
                width=1,
                centered=True,
                cut_dc=False,
                normalised=True,
            )
            mask2 = bandpass_filter(
                new_lut[0],
                high=15,
                low=40,
                width=1,
                centered=True,
                cut_dc=False,
                normalised=True,
            )

            diff_lut = [np.sum(mask1 * im) / np.sum(mask2 * im) for im in self.lut]

            x = self.z_lut
            plt.plot(
                x,
                diff_lut,
                marker="o",
                color="blue",
                alpha=0.3,
                linestyle="None",
            )
            plt.plot(x, gausian_filter(diff_lut, dz, x=x), color="black")
            plt.show()

        self.lut = new_lut
        self.z_lut = z_new

    def _radial_average(self, image, masks=None, step=0.25, show=False):
        mean_image = np.zeros_like(image)

        if masks is None:
            masks = [
                bandpass_filter(
                    image,
                    high=r - step / 2,
                    low=r + step / 2,
                    width=step,
                    centered=True,
                    cut_dc=False,
                )
                for r in np.arange(0, np.min(image.shape[-1]) // 2, step)
            ]

        for mask in masks:
            mean_image += mask * (np.sum(mask * image) / np.sum(mask))

        if show:
            imshow_multiple(
                [image, mean_image],
                titles=["Image", "Mean image"],
                vrange=[0, 5 / np.prod(image.shape)],
            )
        return mean_image, masks

    def _get_z(self, image, show=None):
        fft = np.fft.fftshift(np.abs(np.fft.fft2(image))) * self.mask
        fft /= np.sum(fft)
        diff = np.asarray(
            [np.sum(np.abs(self.mask * (l - fft))) ** 0.5 for l in self.lut]
        )

        if show:
            imshow_multiple(
                [fft, self.lut[show], self.mask * np.abs(fft - self.lut[show])],
                vrange=[0, np.max(fft)],
            )

        # fit parabola to peak and yield maximum position
        width = 0.5
        index = np.argmin(diff)

        selection = np.abs(self.z_lut - self.z_lut[index]) < width * 2
        # weight = np.exp(-((z_ref[selection] - z_ref[index]) ** 2) / width**2)
        weight = np.ones_like(self.z_lut[selection])

        # Suppress warnings for np.polyfit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            poly = np.polyfit(
                self.z_lut[selection], diff[selection], 2, w=np.asarray(weight)
            )
        new_z = -poly[1] / (2 * poly[0])
        min_value = np.polyval(poly, new_z)

        if show:
            plt.plot(self.z_lut, diff, marker="o")
            plt.vlines(self.z_lut[index], 0, 1, color="r")
            plt.vlines(new_z, 0, 1, color="g")

            x_fit = np.linspace(
                self.z_lut[selection][0], self.z_lut[selection][-1], 100
            )
            fit = np.polyval(poly, x_fit)
            plt.plot(x_fit, fit)
            plt.ylim([0, 1.3 * np.max(diff)])
            plt.show()

        return new_z, 1 / min_value

    def get_coords(self, images, show=False):
        coords = np.asarray(
            [
                np.append(find_bead_center(image, self.mask), self._get_z(image))
                for image in images
            ]
        )

        if show:
            colors = ["k", "r", "b", "g"]
            labels = ["X (pix)", "Y (pix)", "Z (um)", "A (a.u)"]
            for i, coord in enumerate(coords.T):
                plt.plot(
                    coord,
                    marker="o",
                    alpha=0.3,
                    color=colors[i],
                    linestyle="None",
                    label=labels[i],
                )
                plt.xlabel("frame")
                plt.legend()
            plt.show()
        return coords


if __name__ == "__main__":
    filename1 = r"data\data_006.hdf"
    filename2 = r"data\data_153.hdf"
    # filename = r"d:\users\noort\data\20241219\data_002.hdf"
    # filename = r"d:\users\noort\data\20241220\data_003.hdf"
    tracker = Tracker(filename2)
    images = load_bin_file(filename2)
    tracker.get_coords(images, show=True)

    if False:
        # test()
        # print("This is a module, not a standalone script.")
        filename = r"d:\users\noort\data\20241211\data_153.hdf"
        # filename = r"d:\users\noort\data\20241212\data_006.hdf"
        frames = load_bin_file(filename)
        data = hdf_data(filename)

        z = data.traces["Focus (mm)"].values * 1000
        lut, mask = create_lut(frames, average=0.25)

        z_new = [get_z(im, z, lut, mask, show=False) for im in frames]
        z = np.asarray(z_new)
        z_new = [get_z(im, z, lut, mask, show=False) for im in frames]

        poly = np.polyfit(z, z_new, 1)
        fit = np.polyval(poly, z)

        plt.plot(z, z_new - fit, marker="o")
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
