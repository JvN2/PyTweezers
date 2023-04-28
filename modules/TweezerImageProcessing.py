from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from nptdms import TdmsFile
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
import cv2


def create_circular_mask(width, shape=None, center=None, steepness=3, invert=False):
    if shape is None:
        shape = [width, width]
    if center is None:
        center = np.asarray(shape) / 2

    x = np.outer(np.linspace(0, shape[0] - 1, shape[0]), np.ones(shape[1]))
    y = np.outer(np.ones(shape[0]), np.linspace(0, shape[1] - 1, shape[1]))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = 1 - 1 / (1 + np.exp(-(r - width / 2) * steepness))
    if invert:
        mask = 1 - mask
    return mask


def create_ref_image(period=10, width=100, shape=[100, 100]):
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    r = np.sqrt((x - shape[1] / 2) ** 2 + (y - shape[0] / 2) ** 2)
    im = np.cos(2 * np.pi * r / period)
    im *= np.cos(2 * np.pi * r / (2 * width))
    im[r >= width / 2] = 0
    return im * np.cos(2 * np.pi * r / period).astype(float)


def fit_peak(X, Y, Z):
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
    else:
        center = np.asarray(center).astype(int)
    bl = np.asarray(center) - width // 2
    bl = np.clip(bl, 0, np.asarray(np.shape(im)) - width)
    tr = bl + width
    roi = im[bl[0]:tr[0], bl[1]:tr[1]]
    return {'image': roi, 'center': bl + width // 2}


def set_roi(im, center, roi):
    width = len(roi[0])
    start = (np.asarray(center) - width // 2).astype(int)
    im[start[0]: start[0] + width, start[1]: start[1] + width] = roi
    return im


def multiply_along_axis(A, B, axis):
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)


def create_filter(lowpass, highpass=None):
    filter = create_circular_mask(lowpass, [lowpass, lowpass], steepness=2)
    if highpass is not None:
        filter *= 1 - create_circular_mask(highpass, np.shape(filter), steepness=2)
    return filter


def calc_fft(im, filter=None):
    fft_im = np.fft.fftshift(np.fft.fft2(im)) / np.prod(np.shape(im))
    if filter is not None:
        if np.shape(im) != np.shape(filter):
            fft_im = get_roi(fft_im, np.shape(filter)[0])['image']
        fft_im *= filter
    return fft_im


def calc_weight(x, x_array, width):
    weight = np.exp(-2 * (x - x_array) ** 2 / width ** 2)
    return weight / np.sum(weight)


def zoom_fft2(im, freqs=None):
    for _ in range(2):
        im = np.fft.fft(im).T
        im = np.asarray([im[line] for line in freqs])
    return im.T


def get_peak(im):
    def calc_extreme(x, y):
        denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
        A = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
        B = (x[2] * x[2] * (y[0] - y[1]) + x[1] * x[1] * (y[2] - y[0]) + x[0] * x[0] * (y[1] - y[2])) / denom
        return -B / (2 * A)

    shape = im.shape
    peak = np.unravel_index(np.argmax(im, axis=None), shape)
    offset = np.asarray([-1, 0, 1])
    intensity = [0, 0, 0]
    im = np.abs(im)

    position = peak[0] + offset
    intensity[0] = im[position[0], peak[1]] if position[0] >= 0 else im[position[0] + shape[0], peak[1]]
    intensity[1] = im[position[1], peak[1]]
    intensity[2] = im[position[2], peak[1]] if position[2] < shape[0] else im[position[2] - shape[0], peak[1]]
    x = calc_extreme(position, intensity)

    position = peak[1] + offset
    intensity[0] = im[peak[0], position[0]] if position[0] >= 0 else im[peak[0], position[0] + shape[0]]
    intensity[1] = im[peak[0], position[1]]
    intensity[2] = im[peak[0], position[2]] if position[2] < shape[1] else im[peak[0], position[2] - shape[1]]
    y = calc_extreme(position, intensity)

    center = np.asarray([y, x])
    # center = -np.asarray([shape[1] / 2 - y, shape[0] / 2 - x])
    return center


class Beads():
    def __init__(self, lut_filename=None, pix_um=0.225, n=1.33, roi_size=128):
        self.refraction_index = n
        self.pix_um = pix_um
        self.roi_size = roi_size
        if lut_filename is not None:
            self._create_lut(lut_filename)
        return

    def find_files_with_wildcard(self, filename):
        if '*' in str(filename):
            path = Path(str(filename).split('*')[0])
            extension = str(filename).split('.')[-1]
            files = natsorted(path.glob('*'), key=lambda x: x.stem)
            matching_files = [file for file in files if str(file).split('.')[-1] == extension]
            return matching_files
        else:
            return [str(filename)]

    def read_avi(self, filename, frames=np.inf):
        filenames = self.find_files_with_wildcard(filename)
        movie = []
        for filename in tqdm(filenames, desc=f'Reading {filename}'):
            vidcap = cv2.VideoCapture(str(filename))
            num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(num_frames):
                ret, frame = vidcap.read()
                if not ret:
                    break
                movie.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if len(movie) >= frames:
                    break
            if len(movie) >= frames:
                break
            vidcap.release()
        return np.asarray(movie)

    def save_avi(self, movie, filename, fps=15):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(str(filename), fourcc, fps, movie[0].T.shape, isColor=False)
        for frame in tqdm(movie, desc='Saving avi file'):
            out.write(frame)
        out.release()

    def find_focus(self, lut):
        size = lut.shape[-1]
        high_frequencies = np.asarray(
            [np.percentile(np.abs(calc_fft(l)) * create_filter(size, size * 0.45), 95) for l in lut])
        high_frequencies /= np.asarray(
            [np.percentile(np.abs(calc_fft(l)) * create_filter(size, size * 0.85), 95) for l in lut])
        return np.argmax(high_frequencies)

    def calc_fft2d(self, im):
        fft_im = np.fft.fftshift(np.fft.fft2(im))
        n_pix = self._mask.shape[0]
        start_index = (im.shape[0] - n_pix) // 2
        end_index = start_index + n_pix
        fft_im = self._mask * fft_im[start_index:end_index, start_index:end_index]
        cc = np.fft.fftshift(np.abs(np.fft.ifft2(fft_im ** 2)))
        fft_im = np.abs(fft_im) / np.sum(np.abs(fft_im))
        return fft_im, cc

    def resample_lut(self, z, lut, n_points, z_width=1):
        new_z = np.linspace(np.min(z), np.max(z), n_points)
        if z[-1] < z[0]:
            new_z = new_z[::-1]
        new_lut = [np.sum(lut * calc_weight(z, z0, z_width)[:, np.newaxis, np.newaxis], axis=0) for z0 in new_z]
        return new_z, np.asarray(new_lut)

    def get_xyz(self, roi, xy_coords, show = False):
        def fit_minimum(x, y, width):
            p = np.polyfit(x, y, 2, w=calc_weight(x, x[np.argmin(y)], width))
            return -p[1] / (2 * p[0]), 1 / np.min(y)

        fft_roi, cc = self.calc_fft2d(roi)
        diff = np.abs(np.sum((self._fft_lut - np.abs(fft_roi)) ** 2, axis=(1, 2)))
        z0, a = fit_minimum(self._z_calib, diff, self._z_width_um)

        if show:
            plt.plot(self._z_calib, diff, 'o-')
            plt.title(f'z0={z0:.2f}um, a={a:.2f}')
            plt.show()

        xy = get_peak(cc)
        xy -= fft_roi.shape[0] / 2
        xy /= 2
        xy *= roi.shape[0] / fft_roi.shape[0]

        xy += xy_coords

        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(roi, cmap='Greys_r', vmax=255, vmin=0)
            ax1.set_title(f'peak @ {xy - xy_coords + roi.shape[0] / 2}')
            ax2.imshow(cc, cmap='Greys_r')
            ax2.set_title(f'peak @ {get_peak(cc)}')
            plt.tight_layout()
            plt.show()

        return np.asarray([*xy, z0, a])

    def get_settings(self):
        settings = {}
        settings['highpass'] = self._highpass
        settings['lowpass'] = self._lowpass
        settings['z_width'] = self._z_width_um
        settings['resampled_z'] = len(self._z_calib)
        return settings

    def _create_lut(self, filename, highpass=20, lowpass=90, z_width=0.25):
        filename = Path(filename)
        if '.tdms' in filename.suffix:
            z_calib = np.asarray(TdmsFile(filename)['Tracking data']['Focus (mm)']) * 1000
            self.lut_filename = str(filename).replace(r'.tdms', r'ROI.bin')
            lut = np.fromfile(self.bin_file, np.uint8)
            roi_size = np.sqrt(len(lut) / len(z_calib)).astype(np.int32)
            lut = lut.reshape([-1, roi_size, roi_size])
        elif '.xlsx' in filename.suffix:
            df = pd.read_excel(filename)
            try:
                z_calib = df['z_calib (um)'].values
                lut = self.read_avi(filename.with_suffix('.avi'))
                self.roi_size = lut.shape[-1]
                self.lut_filename = filename
            except KeyError:
                try:
                    z_calib = df['focus (um)'].values
                except KeyError:
                    raise KeyError('No focus data found in excel file')
                    return
                movie = self.read_avi(filename.with_suffix('.avi'))
                lut_file = filename.with_name(filename.stem + '_lut.avi')
                coords = self.pick_beads(movie[0], n_beads=1, title='Pick bead for LUT', filename=lut_file)
                lut = np.asarray([get_roi(im, self.roi_size, coords[0])['image'] for im in movie]).astype(np.uint8)

                focus_index = self.find_focus(lut)  # find the frame with the highest spatial frequencies
                z_calib = z_calib[focus_index:]
                z_calib = np.max(z_calib) - z_calib
                z_calib /= self.refraction_index  # to um
                z_calib -= 1  # add 1 um margin to avoid negative values
                lut = lut[focus_index:]

                self.save_avi(lut, lut_file)

                with pd.ExcelWriter(lut_file.with_suffix('.xlsx')) as writer:
                    pd.DataFrame(z_calib, columns=['z_calib (um)']).to_excel(writer, sheet_name='traces', index=True,
                                                                             float_format='%.5f')
                    pd.DataFrame(coords, columns=['x0 (pix)', 'y0 (pix)']).to_excel(writer, sheet_name='parameters')
                    globs = pd.DataFrame([['roi_pix', self.roi_size, 'settings']],
                                         columns=['parameter', 'values', 'section'])
                    globs.set_index('parameter', inplace=True)
                    globs.to_excel(writer, sheet_name='globals', index=True)
                print(f'New LUT saved to {lut_file}')

                self.lut_filename = lut_file
        else:
            raise ValueError('File type not supported')
            return

        self._highpass = highpass
        self._lowpass = lowpass
        self._z_width_um = z_width

        self._mask = create_filter(lowpass, highpass)
        self._fft_lut = np.asarray([self.calc_fft2d(frame)[0] for frame in lut])
        self._z_calib = z_calib
        # self._z_calib, self._fft_lut = self.resample_lut(z_calib, self._fft_lut, 50, z_width=z_width)
        return

    def _resample_lut(self, z_calib, lut, width_um=0.3):
        z_array = np.linspace(np.min(z_calib), np.max(z_calib), int((np.max(z_calib) - np.min(z_calib)) / width_um))
        new_lut = [np.sum(multiply_along_axis(lut, calc_weight(z_calib, z, width_um), 0), axis=0) for z in z_array]
        return z_array, new_lut

    def pick_beads(self, image, filename=None, n_beads=None, title=None):
        if title is None:
            title = 'Press <Enter> to finish'
        plt.title(title)
        plt.imshow(image, cmap='Greys_r', vmin=0, vmax=255)
        color = 'blue'
        coords_list = []
        if n_beads is None:
            n_beads = np.inf
        while True and len(coords_list) < n_beads:
            c = plt.ginput(show_clicks=True, timeout=-1)
            try:
                c = np.asarray(c).astype(np.int32)[0]
                box = plt.Rectangle([c[0] - self.roi_size / 2, c[1] - self.roi_size / 2], self.roi_size, self.roi_size,
                                    edgecolor=color, facecolor='none')
                plt.gca().add_artist(box)
                plt.text(c[0] - self.roi_size / 2, c[1] - self.roi_size / 1.9, f'{len(coords_list)}', color=color)
                plt.draw()
                coords_list.append(c[::-1])
            except IndexError:
                break
        self.coords = np.asarray(coords_list)

        if filename is not None:
            plt.savefig(filename.with_suffix('.jpg'))
        plt.close()
        return self.coords

    def find_beads(self, im, n_max=100, treshold=None, show=False):
        mask = np.fft.fftshift(create_circular_mask(20, im.shape, invert=True))
        cc = np.ones_like(im).astype(float)
        for period in tqdm([38.4, 29.3, 47.5, 52.0], postfix='Cross correlating with ref images'):
            ref_im = create_ref_image(period, shape=im.shape, width=period * 10)
            cc *= np.abs(np.fft.ifft2(mask * np.fft.fft2(im) * np.conjugate(np.fft.fft2(ref_im))))
        cc = np.abs(np.fft.fftshift(cc))
        cc /= np.percentile(cc, 99.9)

        if False:
            plt.imshow(cc, cmap='Greys_r', vmax=2.0, vmin=0, origin='lower')
            plt.show()

        X, Y = np.meshgrid(np.arange(len(cc[0])), np.arange(len(cc)))

        coords_list = []
        for _ in tqdm(range(n_max), postfix='find_beads: Finding peaks in cross-corelation'):
            max_index = np.unravel_index(np.argmax(cc, axis=None), cc.shape)
            arrays = []
            for roi in [X, Y, cc]:
                roi = get_roi(roi, self.roi_size, max_index)
                arrays.append(roi['image'])
            try:
                fit, pars = fit_peak(*arrays)
                if pars[3] < 0.4:
                    break
                coords_list.append(pars)
            except (RuntimeError, ValueError) as e:
                pass
            cc = set_roi(cc, roi['center'], 0 * fit)

        coords = pd.DataFrame(coords_list, columns=['X0 (pix)', 'Y0 (pix)', 'width (pix)', 'amplitude (a.u.)', 'R2'])
        coords.sort_values('Y0 (pix)', inplace=True, ignore_index=True)

        if show:
            print(coords.tail(5))
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(im, origin='lower', vmax=np.percentile(im, 99.9), cmap='Greys_r')
            roi_size = 100
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

    def process_image(self, im, coords=None):
        if coords is None:
            coords = self.coords
        result = np.asarray([self.get_xyz(get_roi(im, self.roi_size, c)['image'], c) for c in coords]).astype(float)
        return result


class Traces():
    def __init__(self, filename=None):
        self.traces = pd.DataFrame()
        self.globs = pd.DataFrame()
        self.pars = pd.DataFrame()
        if filename:
            self.from_file(filename)
            self.set_glob('LUT file', filename, 'LUT')
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def from_file(self, filename):
        self.filename = Path(filename).with_suffix('.xlsx')
        try:
            self.traces = pd.read_excel(self.filename, 'traces', index_col=0)
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
            try:
                self.sort_traces()
                self.traces.to_excel(writer, sheet_name='traces', index=True, float_format='%.5f')
            except AttributeError:
                pass
            try:
                self.pars.to_excel(writer, sheet_name='parameters')
            except AttributeError:
                pass
            try:
                self.globs.to_excel(writer, sheet_name='globals')
            except AttributeError:
                pass
        print(f'Data saved to {self.filename}')
        return

    def sort_traces(self):
        reordered_cols = np.append([c for c in self.traces.columns if ': ' not in c],
                                   natsorted([c for c in self.traces.columns if ': ' in c],
                                             key=lambda x: x.split(':')[1], reverse=True))
        self.traces = self.traces[reordered_cols]
        self.traces.sort_index(inplace=True)

    def add_from_excel(self, filename, column, name=None):
        if name is None:
            name = column
        self.traces[name] = pd.read_excel(filename, index_col=None)[column]
        return

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
            if section == 'files':
                return df.loc['directory']['value'] + r'\\' + df.loc[parameter]['value']
            else:
                return df.loc[parameter]['value']

    def set_glob(self, parameter, value, section=''):
        try:
            self.globs.loc[parameter] = [value, section]
        except ValueError:
            self.globs = pd.DataFrame(columns=['parameter', 'value', 'section'])
            self.globs.loc[0] = [parameter, value, section]
            self.globs.set_index('parameter', inplace=True)

    def get_par(self, parameter, bead=None):
        if bead is None:
            return self.get_glob(parameter, section='parameters')
        else:
            return self.pars.loc[bead, parameter]

    def set_par(self, parameter, value, bead=None):
        if bead is None:
            self.set_glob(parameter, value, section='parameters')
        else:
            if bead not in self.pars.index:
                self.pars.loc[bead, parameter] = value
            if parameter not in self.pars.columns:
                self.pars[parameter] = np.nan
            self.pars.loc[bead, parameter] = value

    def add_frame_coords(self, frame, coords):
        names = []
        for i, bead in enumerate(coords):
            names.extend([f'{i}: x (pix)', f'{i}: y (pix)', f'{i}: z (um)', f'{i}: a (a.u.)'])
        try:
            self.traces.loc[frame, names] = np.asarray(coords).reshape(-1)
        except AttributeError:
            self.traces = pd.DataFrame(columns=names)
            self.traces.loc[frame] = np.asarray(coords).reshape(-1)
        return
