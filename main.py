# see https://pylablib.readthedocs.io/en/latest/devices/cameras_basics.html#cameras-basics

import sys

import cv2
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from pylablib.devices import IMAQ
from scipy.optimize import curve_fit
from tqdm import tqdm

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def get_roi(im, width, center=None, size=None):
    if center is None:
        center = np.asarray(np.shape(im)) // 2
    bl = np.asarray(center) - width // 2
    bl = np.clip(bl, 0, np.asarray(np.shape(im)) - width)
    tr = bl + width
    roi = im[bl[0]:tr[0], bl[1]:tr[1]]
    if size is not None:
        roi = cv2.resize(roi, size)
    return roi, bl + width // 2


def set_roi(im, center, roi):
    width = len(roi[0])
    start = (np.asarray(center) - width // 2).astype(int)
    im[start[0]: start[0] + width, start[1]: start[1] + width] = roi
    return im


def aquire_images(n_frames=100, show=True):
    cam = IMAQ.IMAQCamera()
    # print(IMAQ.list_cameras())
    # image = cam.snap()
    # print(cam.get_detector_size())
    # attrs = cam.get_all_grabber_attribute_values()

    cam.setup_acquisition(mode="sequence", nframes=n_frames)
    cam.start_acquisition()
    cv2.namedWindow('live cam', cv2.WINDOW_NORMAL)

    for i in tqdm(range(n_frames)):
        cam.wait_for_frame()
        frame, info = cam.read_oldest_image(return_info=True)
        if show:
            im = get_roi(frame, 500)
            cv2.imshow('live cam', im)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cam.stop_acquisition()
    return frame


def create_circular_mask(width, size=None, center=None, steepness=3):
    if size is None:
        size = [width, width]
    if center is None:
        center = -0.5 + np.asarray(size) / 2
    x = np.outer(np.linspace(0, size[0] - 1, size[0]), np.ones(size[1]))
    y = np.outer(np.ones(size[0]), np.linspace(0, size[1] - 1, size[1]))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return 1 - 1 / (1 + np.exp(-(r - width / 2) * steepness))


def create_ref_image(period=10, width=100, size=100):
    x = np.outer(np.linspace(-size / 2, size / 2, size), np.ones(size)) - 0.5
    r = (x ** 2 + x.T ** 2) ** 0.5 + 1
    im = 0.5 * (np.cos(2 * np.pi * r / width) + 1)
    im[r >= width / 2] = 0
    return im * np.cos(2 * np.pi * r / period)


def fit_peak(X, Y, Z, show=False):
    # Our function to fit is a two-dimensional Gaussian
    def gaussian(x, y, x0, y0, sigma, A):
        return A * np.exp(-((x - x0) / sigma) ** 2 - ((y - y0) / sigma) ** 2)

    def _gaussian(M, *args):
        x, y = M
        arr = gaussian(x, y, *args)
        return arr

    center = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    pars = (X[center], Y[center], 2.0, np.max(Z))

    xdata = np.vstack((X.ravel(), Y.ravel()))
    pars, pcov = curve_fit(_gaussian, xdata, Z.ravel(), pars, ftol=0.5, xtol=0.5)
    fit = gaussian(X, Y, *pars)

    residuals = Z - fit
    ss_res = np.sum(residuals ** 2)  # residual sum of squares
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)  # total sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # r-squared value
    pars = np.append(pars, r_squared)
    return fit, np.asarray(pars)


def find_beads(im, roi_size=100, n_max=100, treshold=None, show=False):
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def bin2d(a, K):
        assert a.shape[0] % K == 0
        assert a.shape[1] % K == 0
        m_bins = a.shape[0] // K
        n_bins = a.shape[1] // K
        res = np.zeros((m_bins, n_bins), dtype=a.dtype)
        for i in nb.prange(res.shape[0]):
            for ii in range(i * K, (i + 1) * K):
                for j in range(res.shape[1]):
                    TMP = res[i, j]
                    for jj in range(j * K, (j + 1) * K):
                        TMP += a[ii, jj]
                    res[i, j] = TMP
        return res

    reduction = 2  # scale down image by factor 2
    reduced_im = bin2d(im, reduction)
    reduced_im -= np.median(reduced_im)

    cc = np.ones_like(reduced_im)
    for period in tqdm([6.5, 8.7, 10.3, 12.7], postfix='Cross correlating with ref images'):
        ref_im = create_ref_image(period, size=len(reduced_im))
        cc *= np.abs(np.fft.ifft2(np.fft.fft2(reduced_im) * np.conjugate(np.fft.fft2(ref_im))))
    cc = np.abs(np.fft.fftshift(cc))
    cc /= np.percentile(cc, 99.999)

    X, Y = np.meshgrid(np.arange(len(cc)), np.arange(len(cc)))

    coords_list = []
    for _ in tqdm(range(n_max), postfix='find_beads: Finding peaks in cross-corelation'):
        max_index = np.unravel_index(np.argmax(cc, axis=None), cc.shape)
        arrays = []
        for roi in [X, Y, cc]:
            roi, center = get_roi(roi, roi_size, max_index)
            arrays.append(roi)
        try:
            fit, pars = fit_peak(*arrays)
            if pars[3] < 0.01:
                break
            coords_list.append(
                [int(pars[0] * reduction), int(pars[1] * reduction), pars[2] * reduction, pars[3], pars[-1]])
        except (RuntimeError, ValueError) as e:
            pass
        cc = set_roi(cc, center, 0 * fit)

    coords = pd.DataFrame(coords_list, columns=['X (pix)', 'Y (pix)', 'width (pix)', 'amplitude (a.u.)', 'R2'])

    if show:
        print(coords.head(30))
        fig = plt.figure(figsize=(12, 12))
        # im = cv2.resize(cc_start, (size * reduction, size * reduction))
        plt.imshow(im, origin='lower', vmax=np.percentile(im, 99.9), cmap='gray')
        for i, row in coords.iterrows():
            color = 'green' if row['R2'] > treshold else 'red'
            # if color == 'green':
            box = plt.Rectangle((row['X (pix)'] - roi_size / 2, row['Y (pix)'] - roi_size / 2), roi_size, roi_size,
                                edgecolor=color, facecolor='none')
            fig.gca().add_artist(box)
            plt.text(row['X (pix)'] - roi_size / 2, row['Y (pix)'] + roi_size / 1.9, f'{i}: {row["R2"]:.2f}',
                     color=color)
        plt.show()

    coords = coords[coords['R2'] > treshold]
    coords.sort_values('Y (pix)', inplace=True, ascending=False)
    coords.reset_index(drop=True, inplace=True)
    return coords


if __name__ == '__main__':
    filename = r'data\data_024.jpg'
    # filename = r'data\test.jpg'
    # frame = aquire_images(500)
    # cv2.imwrite(r'c:\tmp\test.jpg', frame)
    im = cv2.imread(filename)[:, :, 0].astype(float)
    im, _ = get_roi(im, 3500)

    coords = find_beads(im, 100, 200, 0.5, show=True)

    print(coords)

    # plt.imshow(cc.T, origin='lower', vmax=np.percentile(cc, 99.9), cmap='gray')
    # plt.scatter(coords['X (pix)'] / 2, coords['Y (pix)'] / 2, edgecolors='red', facecolors='none')
    # for i, row in coords.iterrows():
    #     plt.text(row['X (pix)'] / 2, row['Y (pix)'] / 2, f' {i}: {row["R2"]:.2f}', color='red')
    #
    # plt.show()
    # plt.imshow(im, origin='lower')

    # plt.plot(im[50,])
    # plt.show()
    # cv2.imshow('image', im.astype(np.uint8))
    # cv2.waitKey(1500)
