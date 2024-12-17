import cv2
import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from icecream import ic
from tqdm import tqdm
import tkinter as tk

from ImageProcessing import load_bin_file
from TraceIO import hdf_data, timeit


import tkinter as tk
from tkinter import ttk
import threading
import time


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


def imshow_multiple(images, titles=None, circles=None, vrange=None, ncols=3):
    if titles is None:
        titles = [f"Image {i}" for i in range(len(images))]
    size = 5

    if type(images) != list:
        images = [images]

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
        im_plot = axes[i].imshow(im, cmap="gray", origin="lower", extent=extent)
        axes[i].set_title(title)

        if vrange is not None:
            im_plot.set_clim(vrange)

        # Create an axes divider and append a colorbar to the right of the image
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.15)
        plt.colorbar(im_plot, cax=cax)

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


def find_peak1d(data, x, width=0.5, show=False):
    # fit parabola to peak and yield maximum position
    index = np.argmax(data)

    selection = np.abs(x - x[index]) < width * 2
    # weight = np.exp(-((z_ref[selection] - z_ref[index]) ** 2) / width**2)
    weight = np.ones_like(x[selection])

    poly = np.polyfit(x[selection], data[selection], 2, w=np.asarray(weight))
    x_max = -poly[1] / (2 * poly[0])

    if show:
        plt.figure(figsize=(12, 3))
        plt.plot(x, data, marker="o")
        plt.vlines(x[index], 0, 1, color="r")
        plt.vlines(x_max, 0, 1, color="g")

        x_fit = np.linspace(x[selection][0], x[selection][-1], 100)
        fit = np.polyval(poly, x_fit)
        plt.plot(x_fit, fit)
        plt.ylim([np.min(data), 1.3 * np.max(data)])

        plt.plot(width)
        plt.tight_layout(pad=2)
        plt.xlabel("x")
        plt.legend()
        plt.show()

    return x_max


def find_peak2d(image, centered=True):
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
    mask = bandpass_filter(image, high=15, low=25, width=2, centered=False, cut_dc=True)
    fft = mask * np.fft.fft2(image)

    cc = np.fft.fftshift(np.abs(np.fft.ifft2(fft**2))) / np.prod(image.shape)
    peak = find_peak2d(cc)
    coords = peak / 2

    if show:
        imshow_multiple([image, cc], circles=[[*coords, 25], [*peak, 5]])
    if not centered:
        coords += np.asarray(image.shape) // 2
    return coords


def find_focus(lut, centers, show=False):
    mask1 = bandpass_filter(
        lut[0], high=10, low=20, width=0.5, centered=True, cut_dc=True, normalised=True
    )
    mask2 = bandpass_filter(
        lut[0], high=10, low=30, width=0.5, centered=True, cut_dc=True, normalised=True
    )

    fft = [np.abs(np.fft.fftshift(np.abs(np.fft.fft2(im)))) for im in lut]

    width = [np.sum(mask2 * im) / np.sum(mask1 * im) for im in fft]

    fft = [im * mask2 for im in fft]
    if show:
        # imshow_multiple(
        #     fft[::15],
        #     ncols=4,
        #     vrange=[0, 0.5],
        #     circles=[[*c * 0, 25] for c in centers[::15]],
        #     titles=[f"Width: {w:.2f}" for w in width],
        # )
        plt.figure(figsize=(12, 3))
        plt.plot(width)
        plt.tight_layout(pad=2)
        plt.xlabel("z_lut")
        plt.ylabel("z_new")
        plt.legend()
        plt.show()

        find_peak1d(np.asarray(width), np.arange(len(width)), width=2, show=True)


class Tracker:
    def __init__(self, filename=None):

        def func(x, a, p, phi):
            return a * np.sin(2 * np.pi * x / p + phi)

        if filename is None:
            # Open file dialog to select a file
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            filename = tk.filedialog.askopenfilename(
                title="Select a file",
                filetypes=(("HDF5 files", "*.hdf5 *.h5"), ("All files", "*.*")),
            )
            if not filename:
                raise ValueError("No file selected")

        # convert to um and correct for air-water interface
        self.z_lut = hdf_data(filename).traces["Focus (mm)"].values * 1000 / 1.3333

        images = load_bin_file(filename)
        self.mask = bandpass_filter(images[0], high=2, low=35, width=2.5, centered=True)
        self.lut = self._create_lut(images, average=0.4)

        self.z_new = [self._get_z(im, show=False) for im in images]

        sigma = np.diff(self.z_lut, prepend=0)
        sigma = np.abs(sigma - np.max(sigma))

        x = self.z_lut[60:90]
        y = x - self.z_new[60:90]
        # Fit the function to the data
        popt, pcov = curve_fit(func, x, y, p0=[0.2, 2, 0])
        a, p, phi = popt

        if False:
            # Plot the data and the fit
            plt.figure(figsize=(12, 3))
            plt.plot(
                self.z_lut,
                self.z_lut - self.z_new - func(self.z_lut, *popt),
                "o-",
                label="Data",
            )
            plt.hlines(0, self.z_lut[0], self.z_lut[-1], color="k")
            plt.xlim([self.z_lut[0], self.z_lut[-1]])
            plt.ylim([-0.5, 0.5])
            plt.tight_layout(pad=2)
            plt.xlabel("z_lut")
            plt.ylabel("z_new")
            plt.legend()
            plt.show()

        centers = [
            find_bead_center(image, centered=True, show=False) for image in images
        ]

        find_focus(images, centers, show=True)

    def _create_lut(self, images, average=None):
        lut = [self.mask * np.fft.fftshift(np.abs(np.fft.fft2(im))) for im in images]
        lut = [l / np.sum(l) for l in lut]

        if average is not None:
            masks = None
            for i, l in enumerate(tqdm(lut, desc="Averaging LUT")):
                lut[i], masks = self._radial_average(l, masks, step=average)
        return lut

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

        poly = np.polyfit(
            self.z_lut[selection], diff[selection], 2, w=np.asarray(weight)
        )
        new_z = -poly[1] / (2 * poly[0])

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

        return new_z


if __name__ == "__main__":
    filename = r"data\data_006.hdf"
    # filename = r"data\data_153.hdf"
    tracker = Tracker(filename)
    # test2(filename)

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
