from modules.TweezerImageProcessing import Beads, Traces, get_roi, get_xyza
from modules.TweezersSupport import color_text
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
import cv2
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import pandas as pd


def display_movie(movie):
    def update(frame):
        plt.clf()  # Clear the current figure
        plt.imshow(movie[frame], cmap='Greys_r')  # Display the current frame
        plt.axis('off')  # Hide the axis labels

    ani = animation.FuncAnimation(plt.gcf(), update, frames=movie.shape[0], interval=50)
    plt.show()


def read_avi_to_array(filename):
    return np.stack([cv2.cvtColor(cv2.VideoCapture(filename).read()[1], cv2.COLOR_BGR2GRAY).astype(np.uint8) for i in range(int(cv2.VideoCapture(filename).get(7)))])

def create_LUT_from_tifs(ref_file, size=128, coords=None):
    if '.xlsx' in str(ref_file):
        try:
            df = pd.read_excel(ref_file)
            frames = df['Frame Number'].values
        except FileNotFoundError:
            print('No reference file found, making a new one from the tifs')
            files = ref_file.parent / 'LUT_Frames'
            frames = [int(f.stem[:-2]) for f in files.glob('*.tif')]
            df = pd.DataFrame({'Frame Number': sorted(frames)})
            df['Piezo Position (um)'] = np.linspace(10, 40, len(frames))
            df.to_excel(ref_file, index=False)
            return
    else:
        print('No reference file found')
    directory = ref_file.parent / 'LUT_Frames'
    files = [directory.joinpath(rf'{frame}-1.tif') for frame in frames]

    if coords is None:
        img = plt.imread(files[0])
        plt.imshow(img, cmap='Greys_r')
        coords = np.asarray(plt.ginput(n=1, timeout=-1)).astype(int)[0][::-1]
        plt.close()

    lut = []
    for f in tqdm(files):
        img = plt.imread(f)
        roi = get_roi(img, size, coords)
        lut.append(roi['image'])
    lut = np.asarray(lut).astype(np.uint8)
    display_movie(lut)
    lutfile = ref_file.with_name(fr'LUT_{coords[0]}_{coords[1]}.bin')
    with open(lutfile, 'wb') as f:
        lut.tofile(f)
    return lutfile

def test_z(filename, tracker):
    df = pd.read_excel(filename)
    z_calib = df['Piezo Position (um)'].values

    z_calib -= np.min(z_calib)
    ims = np.fromfile(tracker.bin_file, np.uint8)
    roi_size = np.sqrt(np.size(ims) / np.size(z_calib)).astype(np.int32)
    ims = ims.reshape([-1, roi_size, roi_size])
    coords = [[roi_size / 2, roi_size / 2]]

    z = []
    for i, im in enumerate(ims):
        res = get_xyza(get_roi(im, roi_size), tracker.freqs, tracker.lut, tracker.z_calib, width_um=0.4, filter=None)
        z.append(res[2])
    z = np.asarray(z)
    fit = np.polyval(np.polyfit(z_calib, z, 1), z_calib)

    fig, axs = plt.subplots(2)
    fig.suptitle('Tracking z stack')
    axs[0].scatter(z_calib, z)
    axs[0].plot(z_calib, fit, color='k')
    axs[0].set_ylabel('z out (um)')

    axs[1].scatter(z_calib, z - fit)
    axs[1].plot(z_calib, fit * 0, color='k')
    axs[1].set_ylabel('residue (um)')
    plt.ylim((-0.200, 0.200))
    plt.xlabel('z in (um)')

    plt.show()


if __name__ == '__main__':
    # filename = Path(r'data\data_024.tdms')
    filename = Path(r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230314\Piezo position_Frames.xlsx')

    tracker = Beads(filename)
    if False:
       test_z(filename, tracker)


    # init bead coordinates
    if False:
        from_image = True
        filename = Path(r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230314\LUT_Frames\214-1.tif')
        data = Traces(filename)
        im = plt.imread(filename)
        data.pars = tracker.find_beads(im, 200, 0.5)

        data.set_glob('roi (pix)', tracker.roi_size, 'Image processing')
        data.to_file(filename)

    if True:
        filename =  Path(r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230314\FEC AVIs_10Frames\1.avi')
        movie = read_avi_to_array(str(filename))
        data = Traces(filename)
        data.pars = tracker.find_beads(movie[0], 200, 0.5, show=True)
        data.to_file()

    # coords = np.asarray([data.pars['X0 (pix)'], data.pars['Y0 (pix)']]).astype(int).T
    # tracker.set_roi_coords(coords)
    #
    # df = test_multi_processing(tracker.get_settings(), im, show=True)
    #
    # print(df)
    #
    # plt.plot(np.diff(df.index))
    # # plt.plot(df['thread'])
    # plt.show()
    #

