from modules.TweezerImageProcessing import Beads, Traces, get_roi, get_xyza
from modules.TweezersSupport import color_text
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
from nptdms import TdmsFile

if __name__ == '__main__':
    # set up lookup table
    ref_filename = Path(r'data\data_024.tdms')
    tracker = Beads(ref_filename)

    z_calib = np.asarray(TdmsFile(ref_filename)['Tracking data']['Focus (mm)'])*1000
    z_calib -= np.min(z_calib)
    ims = np.fromfile(str(ref_filename).replace('.tdms', 'ROI.bin'), np.uint8)
    roi_size = np.sqrt(np.size(ims) / np.size(z_calib)).astype(np.int8)
    ims = ims.reshape([-1, roi_size, roi_size])
    coords = [[roi_size/2, roi_size/2]]

    # plt.imshow(ims[-100], cmap = 'Greys')
    # plt.show()


    z = []
    for i, im in enumerate(ims):
        show = i == len(ims)//3
        res = get_xyza(get_roi(im, roi_size), tracker.freqs, tracker.lut, tracker.z_calib, width_um=0.4, filter=None, show = show)
        z.append(res[2])
    z = np.asarray(z)
    fit = np.polyval(np.polyfit(z_calib, z, 1), z_calib)

    fig, axs = plt.subplots(2)
    fig.suptitle('Tracking z stack')
    axs[0].scatter(z_calib, z)
    axs[0].plot(z_calib, fit, color= 'k')
    axs[0].set_ylabel('z out (um)')

    axs[1].scatter(z_calib, z-fit)
    axs[1].plot(z_calib, fit*0, color= 'k')
    axs[1].set_ylabel('residue (um)')
    plt.ylim((-0.200, 0.200))
    plt.xlabel('z in (um)')

    plt.show()

    # tracker.set_roi_coords(coords)
    #
    #
    #
    #
    #
    #
    #

    #     print(tracker.process_roi(im, coords))