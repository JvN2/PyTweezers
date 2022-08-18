# see https://pylablib.readthedocs.io/en/latest/devices/cameras_basics.html#cameras-basics

import sys
from functools import partial

import cv2, time, ray
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from pylablib.devices import IMAQ
from scipy.optimize import curve_fit
from tqdm.auto import tqdm, trange
import multiprocessing as mp
from natsort import natsorted
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from threading import RLock as TRLock

from modules.TweezerImageProcessing import get_roi, Traces, Beads

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def time_it(f):
    @wraps(f)
    def wrap(*args, **kw):
        print(f'Running: {f.__name__:25s}', end='')
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'-> {te - ts:9.3f} s')
        time.sleep(3)
        return result

    return wrap


def aquire_frame():
    return IMAQ.IMAQCamera().snap()


def aquire_images(n_frames=100, show=True):
    cam = IMAQ.IMAQCamera()
    # print(IMAQ.list_cameras())
    image = cam.snap()
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
            cv2.imshow('live cam', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            # plt.imshow(im, cmap='gray')
            # plt.show()
    cam.stop_acquisition()
    return frame


@time_it
def using_list_comprehension(im, coords, n_frames):
    xyz = [tracker.process_image(im, coords) for _ in range(n_frames)]
    return xyz


@time_it
def using_pool(im, coords, n_frames):
    collect_results = []
    with mp.Pool() as pool:
        for frame in range(n_frames):
            collect_results.append(pool.apply_async(tracker.process_image, [im, coords, ]))
        xyz = [np.asarray(res.get()) for res in collect_results]
    return xyz


@time_it
def using_pool_partial(im, coords, n_frames):
    with mp.Pool() as pool:
        xyz = pool.map(partial(tracker.process_image, coords=coords), [im] * n_frames)
    return xyz


@time_it
def using_thread_pool(im, coords, n_frames):
    with ThreadPoolExecutor() as p:
        result = p.map(partial(tracker.process_image, coords=coords), [im] * n_frames)
    xyz = [r for r in result]
    return xyz


if __name__ == '__main__':
    filename = Path(r'data\data_024.jpg')
    ref_filename = Path(r'data\data_024.tdms')
    # filename = r'data\test.jpg'
    # frame = aquire_images(500)
    # cv2.imwrite(r'c:\tmp\test.jpg', frame)
    # im = aquire_images()

    from_file = True
    data = Traces(filename)
    tracker = Beads(ref_filename)

    if from_file:
        im = cv2.imread(str(filename))[:, :, 0].astype(float)
        im, _ = get_roi(im, 3500)
    else:
        im = aquire_frame()
        im = np.asarray(im).astype(float)
        data.pars = tracker.find_beads(im, 100, 200, 0.5, show=False)
        data.to_file()

    coords = np.asarray([data.pars['X0 (pix)'], data.pars['Y0 (pix)']]).astype(int).T
    n_frames = 10

    using_list_comprehension(im, coords, n_frames)
    using_pool(im, coords, n_frames)
    using_pool_partial(im, coords, n_frames)
    xyz = using_thread_pool(im, coords, n_frames)
    print()

    columns = np.reshape([[f'{i}: X (pix)', f'{i}: Y (pix)', f'{i}: Z (um)'] for i in data.pars.index], [-1])
    data.traces = pd.DataFrame(xyz, columns=columns)
    print(data.traces.tail())
