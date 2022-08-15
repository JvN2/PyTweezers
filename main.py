# Inspired by SuperFastPython.com -> example of one producer and multiple consumers with threads
from queue import Queue
from threading import Thread
from multiprocess import Process
from modules.TweezersSupport import time_it
from modules.TweezerImageProcessing import get_roi, Beads, Traces
from pylablib.devices import IMAQ
from tqdm import tqdm
from pathlib import Path
from cv2 import imread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@time_it
def test_aquisition(tracker, image=None):
    # producer task
    def aquire_images(tracker, queue_raw_data1, queue_raw_data2, n_frames=100, image=None):
        if image is None:
            cam = IMAQ.IMAQCamera()
            cam.setup_acquisition(mode="sequence", nframes=n_frames)
            cam.start_acquisition()
            for _ in tqdm(range(n_frames), postfix='Acquiring images'):
                cam.wait_for_frame()
                frame, info = cam.read_oldest_image(return_info=True)
                rois = [get_roi(frame, tracker.roi_size, coord) for coord in tracker.coords]
                queue_raw_data.put([info.frame_index, rois])
            cam.stop_acquisition()
        else:
            for i in tqdm(range(n_frames), postfix='Using dummy image'):
                rois = [get_roi(image, tracker.roi_size, coord) for coord in tracker.coords]
                if i // 2 == 0:
                    queue_raw_data1.put([i, rois])
                else:
                    queue_raw_data2.put([i, rois])
        queue_raw_data1.put(None)
        queue_raw_data2.put(None)

    # consumer task
    def process_rois(tracker, queue_raw_data, queue_processed_data, identifier):
        while True:
            item = queue_raw_data.get()
            if item is None:
                queue_raw_data.put(item)
                break
            # result = [[*x[1], np.mean(x[0]), np.std(x[0])] for x in item[1]]
            result = [tracker.get_xyza(roi[0], roi[1]) for roi in item[1]]

            result = np.append(item[0], np.reshape(result, (-1)))
            queue_processed_data.put(result)

    def get_queue(q, pars=None):
        result = pd.DataFrame([q.get() for _ in range(q.qsize())])
        if pars is not None:
            result.columns = ['frame'] + list(
                np.reshape([[f'{i}: {p}' for p in pars] for i in range(len(result.columns) // len(pars))], -1))
            result.set_index('frame', inplace=True, drop=True)
        return result

    n_cores = 3
    n_frames = 100
    queue_raw_data1 = Queue()
    queue_raw_data2 = Queue()
    queue_processed_data = Queue()

    producer = Thread(target=aquire_images, args=(tracker, queue_raw_data1, queue_raw_data2, n_frames, image))
    producer.start()
    producer.join()

    # consumers = [Thread(target=process_rois, args=(tracker, queue_raw_data, queue_processed_data, i), daemon=True) for i in
    #              range(n_cores)]
    consumers = [Thread(target=process_rois, args=(tracker, queue_raw_data1, queue_processed_data, 0), daemon=True),
                 Thread(target=process_rois, args=(tracker, queue_raw_data2, queue_processed_data, 1), daemon=True)]
    for consumer in consumers:
        consumer.start()

    # wait for all threads to finish

    for consumer in consumers:
        consumer.join()

    results = get_queue(queue_processed_data, pars=['X (pix)', 'Y (pix)', 'Z (um)', 'A (a.u.)'])

    return results


if __name__ == '__main__':
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
    data.to_file()
    print(data.traces.tail(3))

    if False:
        # plot positions
        selected_cols = [col for col in data.traces.columns if 'X (pix)' in col]
        x = data.traces[selected_cols].iloc[0].values
        selected_cols = [col for col in data.traces.columns if 'Y (pix)' in col]
        y = data.traces[selected_cols].iloc[0].values
        plt.imshow(im, cmap='Greys_r', origin='lower')
        plt.scatter(y - 50, x - 50, s=80, facecolors='none', edgecolors='r')
        # xy coords are not correct: to be solved
        # probably x and y direction mixed up
        plt.show()
