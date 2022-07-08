# see https://pylablib.readthedocs.io/en/latest/devices/cameras_basics.html#cameras-basics

import sys

import cv2
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from nptdms import TdmsFile
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


def get_fft(im, low_pass=None, high_pass=None):
    fft_im = np.fft.fftshift(np.fft.fft2(im)) / np.prod(np.shape(im))
    if low_pass is not None:
        fft_im, _ = get_roi(fft_im, low_pass)
    if high_pass is not None:
        fft_im *= 1 - create_circular_mask(high_pass, np.shape(fft_im), steepness=2)
    return fft_im


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
        center = np.asarray(size) / 2
    x = np.outer(np.linspace(0, size[0] - 1, size[0]), np.ones(size[1]))
    y = np.outer(np.ones(size[0]), np.linspace(0, size[1] - 1, size[1]))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return 1 - 1 / (1 + np.exp(-(r - width / 2) * steepness))


def create_ref_image(period=10, width=100, size=100):
    x, y = np.asarray(np.meshgrid(np.arange(size), np.arange(size))) - size / 2
    r = (x ** 2 + y ** 2) ** 0.5 + 1
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

    M = np.vstack((X.ravel(), Y.ravel()))
    pars, pcov = curve_fit(_gaussian, M, Z.ravel(), pars, ftol=0.5, xtol=0.5)
    fit = gaussian(X, Y, *pars)

    residuals = Z - fit
    ss_res = np.sum(residuals ** 2)  # residual sum of squares
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)  # total sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # r-squared value
    pars = np.append(pars, r_squared)
    return fit, np.asarray(pars)


def calc_weight(x, x_array, width):
    weight = np.exp(-2 * (x - x_array) ** 2 / width ** 2)
    return weight / np.sum(weight)


def multiply_along_axis(A, B, axis):
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)


class BeadTracking():
    def __init__(self, filename=None, pix_um=0.225, n=1.33):
        self.refraction_index = n
        self.pix_um = pix_um
        self._from_file(filename)
        return

    def _from_file(self, filename, highpass=5, lowpass=50):
        if '.tdms' in filename:
            z_calib = np.asarray(TdmsFile(filename)['Tracking data']['Focus (mm)'])
            ims = np.fromfile(filename.replace('.tdms', 'ROI.bin'), np.uint8)
            roi_size = np.sqrt(np.size(ims) / np.size(z_calib)).astype(np.int8)
            ims = ims.reshape([-1, roi_size, roi_size])

            z_calib = 1000 * z_calib / self.refraction_index  # to um
            lut = [np.abs(get_fft(im, lowpass, highpass)) for im in ims]

            self._highpass = highpass
            self._lowpass = lowpass
            self.z_calib, self.lut = self._resample_lut(z_calib, lut)
        return

    def _resample_lut(self, z_calib, lut, step=0.3):
        z_array = np.linspace(np.min(z_calib), np.max(z_calib), int((np.max(z_calib) - np.min(z_calib)) / step))
        new_lut = [np.sum(multiply_along_axis(lut, calc_weight(z_calib, z, step), 0), axis=0) for z in z_array]
        return z_array, new_lut

    def find_beads(self, im, roi_size=100, n_max=100, treshold=None, show=False):
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

        X, Y = np.meshgrid(reduction * np.arange(len(cc)), reduction * np.arange(len(cc)))

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
                coords_list.append(pars)
            except (RuntimeError, ValueError) as e:
                pass
            cc = set_roi(cc, center, 0 * fit)

        coords = pd.DataFrame(coords_list, columns=['X0 (pix)', 'Y0 (pix)', 'width (pix)', 'amplitude (a.u.)', 'R2'])

        if show:
            print(coords.head(30))
            fig = plt.figure(figsize=(12, 12))
            # im = cv2.resize(cc_start, (size * reduction, size * reduction))
            plt.imshow(im, origin='lower', vmax=np.percentile(im, 99.9), cmap='gray')
            for i, row in coords.iterrows():
                color = 'green' if row['R2'] > treshold else 'red'
                # if color == 'green':
                box = plt.Rectangle((row['X0 (pix)'] - roi_size / 2, row['Y0 (pix)'] - roi_size / 2), roi_size,
                                    roi_size,
                                    edgecolor=color, facecolor='none')
                fig.gca().add_artist(box)
                plt.text(row['X0 (pix)'] - roi_size / 2, row['Y0 (pix)'] + roi_size / 1.9, f'{i}: {row["R2"]:.2f}',
                         color=color)
            plt.show()

        coords = coords[coords['R2'] > treshold]
        coords.sort_values('Y0 (pix)', inplace=True, ascending=False)
        coords.reset_index(drop=True, inplace=True)
        self.globals = coords
        return coords

    def get_roi_xyz(self, im, center=[0, 0], width_um=0.4, unit='pix'):
        def calc_extreme(self, x, y):
            try:
                denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
                A = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
                B = (x[2] * x[2] * (y[0] - y[1]) + x[1] * x[1] * (y[2] - y[0]) + x[0] * x[0] * (y[1] - y[2])) / denom
                return -B / (2 * A)
            except IndexError:
                return np.NaN

        fft_im = get_fft(im, self._lowpass, self._highpass)

        cc = fft_im * np.conjugate(fft_im).T
        cc = np.fft.fftshift(np.abs(np.fft.ifft2(cc)))
        peak = np.unravel_index(np.argmax(cc, axis=None), cc.shape)
        points = np.asarray([-1, 0, 1])
        x = calc_extreme(self, points + peak[0], cc[peak[0] - 1:peak[0] + 2, peak[1]])
        y = calc_extreme(self, points + peak[1], cc[peak[0], peak[1] - 1:peak[1] + 2])

        xy = np.asarray(np.shape(im)) * (0.5 + 0.5 * (-0.5 + np.asarray([y, x]) / self._lowpass))
        xy += center
        if unit == 'um':
            xy -= np.asarray(np.shape(im)) / 2
            xy /= self.pix_um

        msd = np.sum((self.lut - np.abs(fft_im)) ** 2, axis=(1, 2))
        weight = calc_weight(self.z_calib[np.argmin(msd)], self.z_calib, width_um)
        p = np.polyfit(self.z_calib, msd, 2, w=weight)
        z = -p[1] / (2 * p[0])

        return *xy, z

    def process_image(self, im, frame=0):
        try:
            self.traces
        except AttributeError:
            columns = [[f'{i}: X (pix)', f'{i}: Y (pix)', f'{i}: Z (um)'] for i, _ in
                       enumerate(self.globals['X0 (pix)'])]
            self.traces = pd.DataFrame(columns=np.append(['frame'], np.reshape(columns, [-1])))

        xyz = [self.get_roi_xyz(*get_roi(im, 100, (int(x), int(y)))) for x, y in
               zip(self.globals['X0 (pix)'], self.globals['Y0 (pix)'])]
        self.traces.loc[len(self.traces.index)] = np.append([frame], np.reshape(xyz, [-1]))
        return self.traces


if __name__ == '__main__':
    filename = r'data\data_024.jpg'
    # filename = r'data\test.jpg'
    # frame = aquire_images(500)
    # cv2.imwrite(r'c:\tmp\test.jpg', frame)
    im = cv2.imread(filename)[:, :, 0].astype(float)
    im, _ = get_roi(im, 3500)

    tracker = BeadTracking(filename.replace('.jpg', '.tdms'))
    tracker.find_beads(im, 100, 200, 0.5, show=False)
    for frame in tqdm(range(100)):
        tracker.process_image(im, frame)
    print(tracker.traces)

    # ims = np.fromfile(filename.replace('.jpg', 'ROI.bin'), np.uint8).reshape([-1, 100, 100])
    # pos = np.asarray([tracker.get_xyz(im) for im in ims]).T
    # i = 153
    # plt.imshow(ims[i], cmap='gray')
    # radii = np.linspace(5, 33, 5)
    # for r in radii:
    #     circle = plt.Circle(pos[:2, i], r, color='r', fill=False)
    #     plt.gca().add_patch(circle)
    #
    # plt.show()

    # plt.plot(z)
    # plt.show()
    # tracker.correct_xy(ims[200])
    # xyz = np.asarray([tracker.get_xyz(im, unit='um') for im in ims]).T
    # for trace in xyz:
    #     plt.plot(trace)
    # plt.show()

    #
    # for im in lut:
    #     cv2.imshow('image', im.astype(np.uint8))
    #     cv2.waitKey(10)

    # im = lut[74]
    # diff = [np.sum((get_fft(im, 50) - l) ** 2) for l in fft_lut]
    # plt.plot(diff, 'o')
    # plt.show()

    # for im in fft_lut:
    #     cv2.imshow('image', im)
    #     cv2.waitKey(1)

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
