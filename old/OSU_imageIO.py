from modules.TweezerImageProcessing import Beads, Traces, get_roi, get_xyza
from modules.TweezersSupport import color_text
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
import cv2
from natsort import natsorted
from nptdms import TdmsFile
import numpy as np

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import pandas as pd


def display_movie(movie):
    def update(frame):
        plt.clf()  # Clear the current figure
        plt.imshow(
            movie[frame], cmap="Greys_r", origin="lower"
        )  # Display the current frame
        # plt.axis('off')  # Hide the axis labels
        plt.title(f"Frame {frame}")  # Add a title to the axes

    ani = animation.FuncAnimation(plt.gcf(), update, frames=movie.shape[0], interval=50)
    plt.show()


def save_avi(array, filename, fps=15):
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(str(filename), fourcc, fps, array[0].T.shape, isColor=False)
    for frame in tqdm(array, desc="Saving avi file"):
        out.write(frame)
    out.release()
    print(f"Video saved to {filename}")


def find_files_with_wildcard(filename):
    if "*" in str(filename):
        path = Path(str(filename).split("*")[0])
        extension = str(filename).split(".")[-1]
        files = natsorted(path.glob("*"), key=lambda x: x.stem)
        matching_files = [
            file for file in files if str(file).split(".")[-1] == extension
        ]
        return matching_files
    else:
        return [str(filename)]


def read_avi(filename, frames=np.inf):
    filenames = find_files_with_wildcard(filename)
    movie = []
    for filename in tqdm(filenames, desc="Reading avi files"):
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


def read_tifs(filename, frames=np.inf):
    files = find_files_with_wildcard(filename)
    movie = []
    for file in tqdm(files, desc="Reading tifs"):
        movie.append(plt.imread(file).astype(np.uint8))
        if len(movie) >= frames:
            break
    return np.asarray(movie)


def movie_to_lut(movie, coords, size=128):
    lut = []
    for frame in tqdm(movie, desc="Creating LUT"):
        roi = get_roi(frame, size, coords)
        lut.append(roi["image"])
    return np.asarray(lut).astype(np.uint8)


def create_LUT_from_tifs(ref_file, size=128, coords=None):
    if ".xlsx" in str(ref_file):
        try:
            df = pd.read_excel(ref_file)
            frames = df["Frame Number"].values
        except FileNotFoundError:
            print("No reference file found, making a new one from the tifs")
            files = ref_file.parent / "LUT_Frames"
            frames = [int(f.stem[:-2]) for f in files.glob("*.tif")]
            df = pd.DataFrame({"Frame Number": sorted(frames)})
            df["Piezo Position (um)"] = np.linspace(10, 40, len(frames))
            df.to_excel(ref_file, index=False)
            return
    else:
        print("No reference file found")
    directory = ref_file.parent / "LUT_Frames"
    files = [directory.joinpath(rf"{frame}-1.tif") for frame in frames]

    if coords is None:
        img = plt.imread(files[0])
        plt.imshow(img, cmap="Greys_r")
        coords = np.asarray(plt.ginput(n=1, timeout=-1)).astype(int)[0][::-1]
        plt.close()

    lut = []
    for f in tqdm(files):
        img = plt.imread(f)
        roi = get_roi(img, size, coords)
        lut.append(roi["image"])
    lut = np.asarray(lut).astype(np.uint8)
    display_movie(lut)
    lutfile = ref_file.with_name(rf"LUT_{coords[0]}_{coords[1]}.bin")
    with open(lutfile, "wb") as f:
        lut.tofile(f)
    return lutfile


def test_z(filename, tracker):
    df = pd.read_excel(filename.with_name("piezo.xlsx"))
    z_calib = df["Piezo Position (um)"].values

    z_calib -= np.min(z_calib)
    ims = read_avi(filename.with_name("lut.avi"))
    # ims = np.fromfile(tracker.bin_file, np.uint8)
    roi_size = np.sqrt(np.size(ims) / np.size(z_calib)).astype(np.int32)
    # ims = ims.reshape([-1, roi_size, roi_size])
    # coords = [[roi_size / 2, roi_size / 2]]

    z = []
    for i, im in enumerate(ims):
        res = get_xyza(
            get_roi(im, roi_size),
            tracker.freqs,
            tracker.lut,
            tracker.z_calib,
            width_um=0.4,
            filter=None,
        )
        z.append(res[2])
    z = np.asarray(z)
    fit = np.polyval(np.polyfit(z_calib, z, 1), z_calib)

    fig, axs = plt.subplots(2)
    fig.suptitle("Tracking z stack")
    axs[0].scatter(z_calib, z)
    axs[0].plot(z_calib, fit, color="k")
    axs[0].set_ylabel("z out (um)")

    axs[1].scatter(z_calib, z - fit)
    axs[1].plot(z_calib, fit * 0, color="k")
    axs[1].set_ylabel("residue (um)")
    plt.ylim((-0.200, 0.200))
    plt.xlabel("z in (um)")

    plt.show()


if __name__ == "__main__":
    # filename = Path(r'data\data_024.tdms')
    lut_filename = Path(
        r"\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\LUT Tiffs - 10-40 um\piezo.xlsx"
    )

    filename = Path(
        r"\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\LUT Tiffs - 10-40 um\*.tif"
    )
    tracker = Beads(lut_filename)

    if False:
        filename = Path(
            r"\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\LUT Tiffs - 10-40 um\*.tif"
        )
        # movie = read_tifs_to_array(filename)
        # save_array_to_avi(movie, filename.with_name('all_frames.avi'))
        movie = read_avi_to_array(str(filename.with_name("all_frames.avi")))

        data = Beads(roi_size=128)
        coords = data.pick_beads(movie[0])[0]
        lut = movie_to_lut(movie, coords, data.roi_size)
        save_array_to_avi(lut, filename.with_name("lut.avi"))

    if False:  # Test z calibration
        test_z(lut_filename, tracker)

    if False:  # Create LUT
        from_image = True
        filename = Path(
            r"\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230314\LUT_Frames\214-1.tif"
        )
        data = Traces(filename)
        im = plt.imread(filename)
        data.pars = tracker.find_beads(im, 200, 0.5)

        data.set_glob("roi (pix)", tracker.roi_size, "Image processing")
        data.to_file(filename)

    if False:  # pick coords
        filename = Path(
            r"\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230314\FEC AVIs_10Frames\1.avi"
        )
        movie = read_avi_to_array(str(filename), 2)
        coords = tracker.pick_beads(movie[0])

    if True:  # Test tracking
        filename = Path(
            r"\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\4-8-23_FEC Trial_Selected AVIs - 1-14.5 mm\*.avi"
        )
        df = pd.read_excel(str(filename.with_name("MagnetShift.xlsx")))
        movie = tracker.read_avi(filename)
        display_movie(movie)
        save_array_to_avi(movie, str(filename).replace(r"*", r"all_frames"))
        breakpoint()

        if not hasattr(locals(), "coords"):
            coords = [[409, 617], [755, 550], [246, 477]]

        data = Traces()
        data.traces["time (s)"] = df["Time (msec)"]
        data.traces["shift (mm)"] = df["Piezo Position (um)"]

        breakpoint()

        for i in tqdm(range(len(movie)), desc="Tracking"):
            beads = tracker.process_image(movie[i], coords)
            data.add_frame_coords(i, beads)

        data.to_file(str(filename).replace(r"*", r"data"))

        # display_movie(movie)
        # animate_3darray(movie)

        # data = Traces(filename)
        # data.pars = tracker.find_beads(movie[0], 200, 0.5, show=True)
        # data.to_file()

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
