from modules.TweezerImageProcessing import get_peak
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks


def read_avi(filename, frames=np.inf, roi=None):
    movie = []

    if roi is not None:
        roi = np.asarray(roi)
        roi_size = 128
        roi = np.asarray(roi) - roi_size // 2

    vidcap = cv2.VideoCapture(str(filename))
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(num_frames), desc=f'Reading: {filename.name}'):
        ret, frame = vidcap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        if roi is not None:
            frame = frame[roi[0]:roi[0] + roi_size, roi[1]:roi[1] + roi_size]

        movie.append(frame)
        if len(movie) >= frames:
            break
    if len(movie) >= frames:
        np.asarray(movie)
    vidcap.release()
    return np.asarray(movie)


def save_avi(movie, filename, fps=15):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(str(filename), fourcc, fps, movie[0].T.shape, isColor=False)
    for frame in tqdm(movie, desc='Saving avi file'):
        out.write(frame)
    out.release()


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


def create_filter(lowpass, highpass=None):
    filter = create_circular_mask(lowpass, [lowpass, lowpass], steepness=2)
    if highpass is not None:
        filter *= 1 - create_circular_mask(highpass, np.shape(filter), steepness=2)
    return filter


def calc_weight(x, x0, width):
    result = np.exp(-(x - x0) ** 2 / width ** 2)
    return result / np.sum(result)


def resample_lut(z, lut, n_points, z_width=1):
    new_z = np.linspace(np.min(z), np.max(z), n_points)
    new_lut = [np.sum(lut * calc_weight(z, z0, z_width)[:, np.newaxis, np.newaxis], axis=0) for z0 in new_z]
    return new_z, np.asarray(new_lut)


def calc_fft2d(im, mask):
    fft_im = np.fft.fftshift(np.fft.fft2(im))
    n_pix = mask.shape[0]
    start_index = (im.shape[0] - n_pix) // 2
    end_index = start_index + n_pix
    fft_im = mask * fft_im[start_index:end_index, start_index:end_index]
    return fft_im / np.sum(fft_im)


def fit_minimum(x, y, width):
    p = np.polyfit(x, y, 2, w=calc_weight(x, x[np.argmin(y)], width))
    return -p[1] / (2 * p[0])


def get_z(roi, z_calib, fft_lut, mask, z_width=0.5):
    fft_roi = calc_fft2d(roi, mask)
    diff = np.sum((fft_lut - fft_roi) ** 2, axis=(1, 2))
    z0 = fit_minimum(z_calib, diff, z_width)
    return z0


if __name__ == '__main__':
    filename = Path(r'C:\tmp\TeezersOSU\20230408\shift_702_291.avi')
    movie = read_avi(filename)

    im = movie[-10]


    mask = create_filter(60, 20)
    im2 = np.abs(np.fft.fftshift(np.fft.ifft2(calc_fft2d(im, mask)**2)))

    center =get_peak(im2)/2
    center += len(mask)/2
    center *= len(im)/len(mask)

    print(center)


    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im, cmap='Greys_r', vmax = 255, vmin = 0)
    ax1.set_title('Image 1')
    ax2.imshow(im2, cmap='Greys_r')
    ax2.set_title('Image 2')
    plt.tight_layout()
    plt.show()
