from nptdms import TdmsFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm
from natsort import natsorted
from pathlib import Path
from scipy.optimize import curve_fit
from modules.TweezersSupport import time_it


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


def get_roi(im, width, center=None):
    if center is None:
        center = np.asarray(np.shape(im)) // 2
    bl = np.asarray(center) - width // 2
    bl = np.clip(bl, 0, np.asarray(np.shape(im)) - width)
    tr = bl + width
    roi = im[bl[0]:tr[0], bl[1]:tr[1]]
    return roi, bl + width // 2


def set_roi(im, center, roi):
    width = len(roi[0])
    start = (np.asarray(center) - width // 2).astype(int)
    im[start[0]: start[0] + width, start[1]: start[1] + width] = roi
    return im


def calc_fft(im, low_pass=None, high_pass=None):
    fft_im = np.fft.fftshift(np.fft.fft2(im)) / np.prod(np.shape(im))
    if low_pass is not None:
        fft_im, _ = get_roi(fft_im, low_pass)
    if high_pass is not None:
        fft_im *= 1 - create_circular_mask(high_pass, np.shape(fft_im), steepness=2)
    return fft_im


def calc_weight(x, x_array, width):
    weight = np.exp(-2 * (x - x_array) ** 2 / width ** 2)
    return weight / np.sum(weight)


def multiply_along_axis(A, B, axis):
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)


class Beads():
    def __init__(self, filename=None, pix_um=0.225, n=1.33):
        self.refraction_index = n
        self.pix_um = pix_um
        self._from_file(filename)
        return

    def _from_file(self, filename, highpass=5, lowpass=50):
        if '.tdms' in filename.suffix:
            z_calib = np.asarray(TdmsFile(filename)['Tracking data']['Focus (mm)'])
            ims = np.fromfile(str(filename).replace('.tdms', 'ROI.bin'), np.uint8)
            roi_size = np.sqrt(np.size(ims) / np.size(z_calib)).astype(np.int8)
            ims = ims.reshape([-1, roi_size, roi_size])

            z_calib = 1000 * z_calib / self.refraction_index  # to um
            lut = [np.abs(calc_fft(im, lowpass, highpass)) for im in ims]

            self._highpass = highpass
            self._lowpass = lowpass
            self.z_calib, self.lut = self._resample_lut(z_calib, lut)
            self.roi_size = roi_size
        return

    def _resample_lut(self, z_calib, lut, step=0.3):
        z_array = np.linspace(np.min(z_calib), np.max(z_calib), int((np.max(z_calib) - np.min(z_calib)) / step))
        new_lut = [np.sum(multiply_along_axis(lut, calc_weight(z_calib, z, step), 0), axis=0) for z in z_array]
        return z_array, new_lut

    def find_beads(self, im, n_max=100, treshold=None, show=False):
        def bin_2d(a, K):
            assert a.shape[0] % K == 0
            assert a.shape[1] % K == 0
            m_bins = a.shape[0] // K
            n_bins = a.shape[1] // K
            res = np.zeros((m_bins, n_bins), dtype=a.dtype)
            for i in range(res.shape[0]):
                for ii in range(i * K, (i + 1) * K):
                    for j in range(res.shape[1]):
                        TMP = res[i, j]
                        for jj in range(j * K, (j + 1) * K):
                            TMP += a[ii, jj]
                        res[i, j] = TMP
            return res

        reduction = 2  # scale down image by factor 2
        reduced_im = bin_2d(im.astype(float), reduction)
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
                roi, center = get_roi(roi, self.roi_size, max_index)
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
        coords.sort_values('Y0 (pix)', inplace=True, ignore_index=True)

        if show:
            # print(coords.head(30))
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(im, origin='lower', vmax=np.percentile(im, 99.9), cmap='Greys_r')
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
        self.coords = np.asarray([coords['X0 (pix)'], coords['Y0 (pix)']]).astype(int).T
        return coords

    def set_roi_coords(self, coords):
        self.coords = coords

    def get_xyza(self, im, center=[0, 0], width_um=0.4, unit='pix'):
        def calc_extreme(self, x, y):
            try:
                denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
                A = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
                B = (x[2] * x[2] * (y[0] - y[1]) + x[1] * x[1] * (y[2] - y[0]) + x[0] * x[0] * (y[1] - y[2])) / denom
                return -B / (2 * A)
            except IndexError:
                return np.NaN

        fft_im = calc_fft(im, self._lowpass, self._highpass)

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

        return *xy, z, np.max(cc)

    def process_roi(self, im, xy):
        return self.get_xyza(*get_roi(im, 100, *xy))
        # return np.append(xy, [3])

    def process_image(self, im, coords):
        xyz = [self.get_xyza(*get_roi(im, 100, c)) for c in coords]
        return np.reshape(xyz, [-1])


class Traces():
    def __init__(self, filename=None):
        self.from_file(filename)
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def from_file(self, filename):
        self.filename = Path(filename).with_suffix('.xlsx')
        try:
            self.traces = pd.read_excel(self.filename, 'traces')
        except ValueError:
            self.traces = pd.DataFrame()
        except FileNotFoundError:
            self.traces = pd.DataFrame()
            self.pars = pd.DataFrame()
            self.globs = pd.DataFrame().squeeze()
            return
        try:
            self.pars = pd.read_excel(self.filename, 'parameters', index_col=0)
        except:
            self.pars = pd.DataFrame()
        try:
            self.globs = pd.read_excel(self.filename, 'globals', index_col=0)
            for key, value in self.globs.iteritems():
                if isinstance(value, str):
                    if value[0] == '[' and value[-1] == ']':
                        ar = value[1:-1].split(', ')
                        ar = [a[1:-1] if a[0] == "'" else a for a in ar]
                        self.globs.at[key] = ar

        except ValueError:
            self.globs = pd.DataFrame().squeeze()
        return

    def to_file(self, filename=None):
        if filename is not None:
            self.filename = Path(filename).with_suffix('.xlsx')
        with pd.ExcelWriter(self.filename) as writer:
            if not self.traces.empty:
                self.sort_traces()
                self.traces.to_excel(writer, sheet_name='traces', index=True, float_format='%.5f')
            if not self.pars.empty:
                self.pars.to_excel(writer, sheet_name='parameters')
            if not self.globs.empty:
                self.globs.to_excel(writer, sheet_name='globals')
        return

    def sort_traces(self):
        reordered_cols = np.append([c for c in self.traces.columns if ': ' not in c],
                                   natsorted([c for c in self.traces.columns if ': ' in c]))
        self.traces = self.traces[reordered_cols]
        self.traces.sort_index(inplace=True)

    def read_log(self, filename=None):
        if filename is None:
            filename = self.filename
        with open(Path(filename).with_suffix('.log')) as f:
            contents = f.read().replace('"', '')
        contents = contents.split('\n')
        section = 'None'
        log = []
        for line in contents:
            if line is None:
                break
            if '[' in line:
                section = line[1:-1]
            if ' = ' in line:
                log.append(line.split(' = ') + [section])

        self.globs = pd.DataFrame(log, columns=['parameter', 'value', 'section']).set_index('parameter')

    def get_glob(self, parameter, section=None):
        if section is not None:
            df = self.globs[self.globs['section'] == section]
        else:
            df = self.globs
        try:
            return float(df.loc[parameter]['value'])
        except ValueError:
            return df.loc[parameter]['value']

    def set_glob(self, parameter, value, section=''):
        try:
            self.globs.loc[parameter] = [value, section]
        except ValueError:
            self.globs = pd.DataFrame(columns=['parameter', 'value', 'section'])
            self.globs.loc[0] = [parameter, value, section]
            self.globs.set_index('parameter', inplace=True)

