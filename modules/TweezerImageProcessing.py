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


def calc_freqs(im, zoom=2):
    freqs = np.arange(len(im) / (2 * zoom))
    return np.append(np.flip(-np.asarray(freqs + 1)), freqs).astype(int)


def zoom_fft2(im, freqs=None):
    for _ in range(2):
        im = np.fft.fft(im).T
        im = np.asarray([im[line] for line in freqs])
    return im


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

    center = [x, y]
    center = [c - shape[i] if c > shape[i] / 2 else c for i, c in enumerate(center)]
    return np.asarray(center)


def get_xyza(roi, freqs, lut, lut_z_um, width_um=0.4, filter=None, show=False):
    fft_im = zoom_fft2(roi['image'], freqs)
    if filter is not None:
        fft_im *= filter
    cc = np.abs(np.fft.ifft2(fft_im ** 2))
    xy = roi['center'] + get_peak(cc)

    msd = np.sum((lut - np.abs(fft_im)) ** 2, axis=(1, 2))
    msd /= np.max(msd)
    weight = calc_weight(lut_z_um[np.argmin(msd)], lut_z_um, width_um)

    p = np.polyfit(lut_z_um, msd, 2, w=weight)

    if show:
        plt.scatter(lut_z_um, msd)
        lut_z_um = np.linspace(np.min(lut_z_um), np.max(lut_z_um), 1000)
        plt.plot(lut_z_um, np.polyval(p, lut_z_um), color='k')
        plt.ylim([np.min(msd), 1 + 0.5 * (1 - np.min(msd))])
        plt.xlabel('z in (um)')
        plt.ylabel('msd (a.u)')
        plt.show()

    z = -p[1] / (2 * p[0])
    return *xy, z, np.max(cc)


def proces_image(index, image, settings):
    rois = [get_roi(image, settings['size'], coord) for coord in settings['coords']]
    result = [get_xyza(roi, settings['freqs'], settings['lut'], settings['lut_z_um'], filter=settings['filter']) for roi
              in rois]
    return np.append(index, np.reshape(result, (-1)))


class Beads():
    def __init__(self, filename=None, pix_um=0.225, n=1.33, roi_size=128):
        self.refraction_index = n
        self.pix_um = pix_um
        self.roi_size = roi_size
        if filename is not None:
            self._create_lut(filename)
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
        for filename in tqdm(filenames, desc='Reading avi files'):
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

    def _create_lut(self, filename, highpass=5, zoom=2):
        filename = Path(filename)
        if '.tdms' in filename.suffix:
            z_calib = np.asarray(TdmsFile(filename)['Tracking data']['Focus (mm)']) * 1000
            self.bin_file = str(filename).replace(r'.tdms', r'ROI.bin')
            ims = np.fromfile(self.bin_file, np.uint8)
            roi_size = np.sqrt(len(ims) / len(z_calib)).astype(np.int32)
            ims = ims.reshape([-1, roi_size, roi_size])
        elif '.xlsx' in filename.suffix:
            df = pd.read_excel(filename)
            z_calib = df['Piezo Position (um)'].values
            ims = self.read_avi(filename.with_name('lut.avi'))
            roi_size = ims.shape[-1]
        else:
            raise ValueError('File type not supported')
            return

        self.freqs = calc_freqs(ims[0], zoom)
        lowpass = ims.shape[-1] // zoom
        self.filter = create_filter(lowpass, highpass)

        z_calib /= self.refraction_index  # to um
        z_calib -= np.min(z_calib)
        lut = [np.abs(zoom_fft2(im, freqs=self.freqs)) * self.filter for im in ims]

        self._highpass = highpass
        self._lowpass = lowpass

        self.width_um = 0.4
        self.z_calib, self.lut = self._resample_lut(z_calib, lut, self.width_um)
        self.roi_size = roi_size

        return

    def _resample_lut(self, z_calib, lut, width_um=0.3):
        z_array = np.linspace(np.min(z_calib), np.max(z_calib), int((np.max(z_calib) - np.min(z_calib)) / width_um))
        new_lut = [np.sum(multiply_along_axis(lut, calc_weight(z_calib, z, width_um), 0), axis=0) for z in z_array]
        return z_array, new_lut

    def pick_beads(self, image):
        plt.title('Press <Enter> to finish')
        plt.imshow(image, cmap='Greys_r', origin='lower', vmin=0, vmax=255)
        color = 'blue'
        coords_list = []
        while True:
            c = plt.ginput(show_clicks=True, timeout=-1)
            try:
                c = np.asarray(c).astype(np.int32)[0]
                box = plt.Rectangle([c[0] - self.roi_size / 2, c[1] - self.roi_size / 2], self.roi_size, self.roi_size,
                                    edgecolor=color,
                                    facecolor='none')
                plt.gca().add_artist(box)
                plt.text(c[0] - self.roi_size / 2, c[1] + self.roi_size / 1.9, f'{len(coords_list)}', color=color)
                plt.draw()
                coords_list.append(c[::-1])
            except IndexError:
                break
        self.coords = np.asarray(coords_list).astype(int)

        return self.coords

    def find_beads(self, im, n_max=100, treshold=None, show=False):
        mask = np.fft.fftshift(create_circular_mask(20, im.shape, invert=True))
        cc = np.ones_like(im).astype(float)
        for period in tqdm([38.4, 29.3, 47.5, 52.0], postfix='Cross correlating with ref images'):
            ref_im = create_ref_image(period, shape=im.shape, width=period * 10)
            cc *= np.abs(np.fft.ifft2(mask * np.fft
                                      .fft2(im) * np.conjugate(np.fft.fft2(ref_im))))
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

    def process_roi(self, im, xy):
        return get_xyza(*get_roi(im, 100, *xy))
        # return np.append(xy, [3])

    def process_image(self, im, coords=None):
        if coords is None:
            coords = self.coords

        result = [get_xyza(get_roi(im, self.roi_size, c), self.freqs, self.lut, self.z_calib, self.width_um) for c
                  in coords]
        return result

    def get_settings(self):
        settings = {'coords': self.coords, 'size': self.roi_size, 'lowpass': self._lowpass, 'highpass': self._highpass,
                    'lut': self.lut, 'lut_z_um': self.z_calib, 'pix_um': self.pix_um, 'filter': self.filter,
                    'freqs': self.freqs}
        return settings


class Traces():
    def __init__(self, filename=None):
        if filename:
            self.from_file(filename)
        self.traces = pd.DataFrame()
        self.globs = pd.DataFrame()
        self.pars = pd.DataFrame()
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
        # print(self.traces)
        # print(self.pars)
        # print(self.globs)
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

    # def set_par(self, parameter, value, bead=None):
    #     print(parameter, value, bead)
    #     if bead is None:
    #         self.set_glob(parameter, value, section='parameters')
    #     else:
    #         self.pars.loc[bead, parameter] == value

    def set_par(self, parameter, value, bead=None):
        if bead is None:
            self.set_glob(parameter, value, section='parameters')
        else:
            if bead not in self.pars.index:
                self.pars.loc[bead, parameter]= value
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
