# see https://pylablib.readthedocs.io/en/latest/devices/cameras_basics.html#cameras-basics

import cv2
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from pylablib.devices import IMAQ
from scipy.optimize import curve_fit
from tqdm import tqdm


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

# @nb.njit(parallel=True, fastmath=True, cache=True)
def fit_peak(Z, show=False, center=[0, 0]):
    # Our function to fit is a two-dimensional Gaussian
    def gaussian(x, y, x0, y0, sigma, A):
        return A * np.exp(-((x - x0) / sigma) ** 2 - ((y - y0) / sigma) ** 2)

    # This is the callable that is passed to curve_fit. M is a (2,N) array
    # where N is the total number of data points in Z, which will be ravelled
    # to one dimension.
    def _gaussian(M, *args):
        x, y = M
        arr = gaussian(x, y, *args)
        return arr

    Z = np.asarray(Z)
    N = len(Z)
    X, Y = np.meshgrid(np.linspace(0, N - 1, N) - N / 2 + center[0],
                       np.linspace(0, N - 1, N) - N / 2 + center[1])
    pars = (center[0], center[1], 2.0, np.max(Z))

    xdata = np.vstack((X.ravel(), Y.ravel()))
    pars, pcov = curve_fit(_gaussian, xdata, Z.ravel(), pars)
    fit = gaussian(X, Y, *pars)

    residuals = Z - fit
    ss_res = np.sum(residuals ** 2)  # residual sum of squares
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)  # total sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # r-squared value
    pars = np.append(pars, r_squared)
    return fit, np.asarray(pars)


def find_beads(im, roi_size=100, n_max=100, treshold = 0.85):
    im = bin2d(im, 2) # scale down image by factor 2
    im -= np.median(im)
    size = len(im)

    cc = np.ones_like(im)
    for period in tqdm([6.5, 8.7, 10.3, 12.7], postfix='Cross correlating with ref images'):
        ref_im = create_ref_image(period, size=size)
        cc *= np.abs(np.fft.ifft2(np.fft.fft2(im) * np.conjugate(np.fft.fft2(ref_im))))
    cc = np.fft.fftshift(cc).T

    coords = pd.DataFrame(columns=['X (pix)', 'Y (pix)', 'amplitude (a.u.)',  'R2'])
    for i in tqdm(range(n_max), postfix='Finding beads'):
        max_index = np.asarray(np.unravel_index(np.argmax(cc, axis=None), cc.shape))
        roi, center = get_roi(cc, roi_size, max_index)
        roi /= roi_size**8
        try:
            fit, pars = fit_peak(roi, center=center)
            if pars[-1] > treshold:
                coords.loc[i] = [pars[0] + center[0], pars[1] + center[1], pars[3], pars[-1]]
        except RuntimeError:
            pass
        cc = set_roi(cc, center, roi * 0)
    coords = coords[coords['R2'] > 0.8].sort_values('Y (pix)')
    return coords.reset_index(drop=True)


if __name__ == '__main__':
    filename = r'data\test.jpg'
    # frame = aquire_images(500)
    # cv2.imwrite(r'c:\tmp\test.jpg', frame)
    size = 1500
    im = cv2.imread(filename)[:, :, 0].astype(float)
    # im, _ = get_roi(im, size, (4000, 4200))

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.imshow(im, origin='lower', vmax=np.percentile(im, 99.9), cmap='gray')
    coords = find_beads(im, 100, 100, 0.8)
    # plt.imshow(im, origin='lower')

    print(coords)
    plt.scatter(coords['X (pix)'], coords['Y (pix)'], edgecolors='red', facecolors='none')
    # plt.imshow(im)
    plt.show()
    # plt.plot(im[50,])
    # plt.show()
    # cv2.imshow('image', im.astype(np.uint8))
    # cv2.waitKey(1500)
