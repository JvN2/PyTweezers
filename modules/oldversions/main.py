# Inspired by SuperFastPython.com -> example of one producer and multiple consumers with threads
# and https://www.askpython.com/python/producer-consumer-problem for buffering
from pathlib import Path
from queue import Queue
from threading import Thread, Semaphore
from time import sleep

import numpy as np
import pandas as pd
from cv2 import imread
from pylablib.devices import IMAQ
from tqdm import tqdm

from modules.TweezerImageProcessing import get_roi, Beads, Traces
from modules.TweezersSupport import time_it

# Shared Memory variables
CAPACITY = 100
buffer = [-1 for i in range(CAPACITY)]
in_index = 0
out_index = 0
acquisition_done = True

# Declaring Semaphores
mutex = Semaphore()
vacant_buffer_index = Semaphore(CAPACITY)
full = Semaphore(0)


@time_it
def test_aquisition(tracker, image=None):
    # producer task
    def aquire_images(tracker, n_frames=100, image=None):
        global CAPACITY, buffer, in_index, out_index, acquisition_done
        global mutex, vacant_buffer_index, full

        acquisition_done = False
        if image is None:
            cam = IMAQ.IMAQCamera()
            cam.setup_acquisition(mode="sequence", nframes=n_frames)
            cam.start_acquisition()
            for _ in tqdm(range(n_frames), postfix='Acquiring images'):
                cam.wait_for_frame()
                frame, info = cam.read_oldest_image(return_info=True)
                rois = [get_roi(frame, tracker.roi_size, coord) for coord in tracker.coords]

                empty.acquire()
                mutex.acquire()
                buffer[in_index] = [info.frame_index, rois]
                in_index = (in_index + 1) % CAPACITY
                mutex.release()
                full.release()

            cam.stop_acquisition()
        else:
            for frame_index in tqdm(range(n_frames), postfix='Acquiring images: Simulation using saved image'):
                rois = [get_roi(image, tracker.roi_size, coord) for coord in tracker.coords]

                empty.acquire()
                mutex.acquire()
                buffer[in_index] = [frame_index, rois]
                in_index = (in_index + 1) % CAPACITY
                mutex.release()
                full.release()
                sleep(0.03)
        acquisition_done = True

    # consumer task
    def process_rois(tracker, n_frames, processed_data, identifier):
        global CAPACITY, buffer, in_index, out_index, counter, acquisition_done
        global mutex, vacant_buffer_index, full

        while True:
            full.acquire(timeout=0.5)
            mutex.acquire(timeout=0.5)
            item = buffer[out_index]
            out_index = (out_index + 1) % CAPACITY
            mutex.release()
            empty.release()

            result = [tracker.get_xyza(*roi) for roi in item[1]]
            result = np.asarray(np.append([item[0], identifier], np.reshape(result, (-1))))

            processed_data.put(result)
            if processed_data.qsize() >= n_frames-1:
                break

    def get_queue(q, pars=None):
        result = pd.DataFrame([q.get() for _ in range(q.qsize())])
        if pars is not None:
            result.columns = ['frame', 'cpu' ] + list(
                np.reshape([[f'{i}: {p}' for p in pars] for i in range(len(result.columns) // len(pars))], -1))
            result.set_index('frame', inplace=True, drop=True)
        return result.sort_index()

    # Start combined acquisition and processing
    n_cores = 3
    n_frames = 100
    processed_data = Queue()

    producer = [Thread(target=aquire_images, args=(tracker, n_frames, image))]
    consumers = [Thread(target=process_rois, args=(tracker, n_frames, processed_data, i), daemon=True) for i
                 in range(n_cores)]
    for proces in consumers + producer:
        proces.start()

    for proces in consumers + producer:
        proces.join()

    results = get_queue(processed_data, pars=['X (pix)', 'Y (pix)', 'Z (um)', 'A (a.u.)'])

    return results


if __name__ == '__main__':
    print()
    # set up lookup table
    ref_filename = Path(r'data\data_024.tdms')
    tracker = Beads(ref_filename)

    # find bead coordinates
    from_file = True
    filename = Path(r'data\data_024.jpg')
    data = Traces(filename)

    if from_file:
        im = imread(str(filename))[:, :, 0].astype(float).T
    else:
        im = imread(str(filename))[:, :, 0]
        data.pars = tracker.find_beads(im, 200, 0.6, show=False)
        data.set_glob('roi (pix)', tracker.roi_size, 'Image processing')
        data.to_file()
        im = None

    print(data.pars.tail(3))

    # acquire and process images
    coords = np.asarray([data.pars['X0 (pix)'], data.pars['Y0 (pix)']]).astype(int).T
    tracker.set_roi_coords(coords[:100])
    data.traces = test_aquisition(tracker, im)
    # data.to_file()
    print(data.traces.tail(3))

    if False:
        # plot positions
        selected_cols = [col for col in data.data.columns if 'X (pix)' in col]
        x = data.data[selected_cols].iloc[0].values
        selected_cols = [col for col in data.data.columns if 'Y (pix)' in col]
        y = data.data[selected_cols].iloc[0].values
        plt.imshow(im, cmap='Greys_r', origin='lower')
        plt.scatter(y - 50, x - 50, s=80, facecolors='none', edgecolors='r')
        # xy coords are not correct: to be solved
        # probably x and y direction mixed up
        plt.show()
