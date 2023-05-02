from modules.TweezerImageProcessing import Beads
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

beads = Beads()
filename = Path(r'C:\tmp\TeezersOSU\20230408\focus_lut.avi')
images = beads.read_avi(filename)
im = images[2]

shift =-6
for shift in [-8, -6, -4]:
    im = np.asarray([np.roll(l, shift) for l in im])

    im = np.fft.fftshift(im)

    fft_im = np.fft.fftshift(np.fft.fft2(im))

    angle_im = np.angle(fft_im)
    # plt.imshow(angle_im, cmap = 'Greys_r', vmax=np.percentile(angle_im, 99))

    plt.plot(angle_im[64,:])
plt.show()
