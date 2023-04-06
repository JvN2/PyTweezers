import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from modules.TweezerImageProcessing import create_ref_image, create_filter


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
    intensity[0] = im[position[0], peak[1]] if position[0] > 0 else im[position[0] + shape[0], peak[1]]
    intensity[1] = im[position[1], peak[1]]
    intensity[2] = im[position[2], peak[1]] if position[2] < shape[0] else im[position[2] - shape[0], peak[1]]
    x = calc_extreme(position, intensity)

    position = peak[1] + offset
    intensity[0] = im[peak[0], position[0]] if position[0] > 0 else im[peak[0], position[0] + shape[0]]
    intensity[1] = im[peak[0], position[1]]
    intensity[2] = im[peak[0], position[2]] if position[2] < shape[1] else im[peak[0], position[2] - shape[1]]
    y = calc_extreme(position, intensity)

    center = [x, y]
    center = [c - shape[i] if c > shape[i] / 2 else c for i, c in enumerate(center)]
    return np.asarray(center)


shift = [10, -15]
size = 100
image = np.random.normal(size=(size, size))
print('Shape = ' , image.shape)
for p in (7, 9, 8.5):
    image += 2 * create_ref_image(period=p, shape=image.shape, width=size)
image = np.roll(image, shift, axis=[0, 1])

repeats = 2000

freqs = calc_freqs(image)

position = []
for i in tqdm(range(repeats), postfix='np_zoom_fft2'):
    fft_im = zoom_fft2(image, freqs)
    cc = np.abs(np.fft.ifft2(fft_im ** 2))
    position = get_peak(cc)

plt.imshow(image.T, cmap='Greys_r')
plt.scatter(*(position + np.asarray(np.shape(image)) / 2), edgecolors='red', facecolors='none')
plt.title(f'center = {position[0]:.2f}, {position[1]:.2f}')
plt.show()

# plt.imshow(np.abs(create_filter(50, 5)), cmap = 'Greys_r')
plt.imshow(np.abs(fft_im), cmap = 'Greys_r')
plt.show()