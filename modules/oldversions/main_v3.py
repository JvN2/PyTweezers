# Inspired by SuperFastPython.com -> example of one producer and multiple consumers with threads
# and https://www.askpython.com/python/producer-consumer-problem for buffering

# Using multithreading


from pathlib import Path
from queue import Queue
from threading import Thread
from multiprocessing import Process
from time import sleep
from time import time

import numpy as np
import pandas as pd
from cv2 import imread
from pylablib.devices import IMAQ
from tqdm import tqdm

from modules.TweezerImageProcessing import get_roi, get_xyza, Beads, Traces
from modules.TweezersSupport import time_it, color_text

# Shared Memory variables
CAPACITY = 100
buffer = [0 for i in range(CAPACITY)]


# @time_it
def test_acquisition(acquisition_settings, processing_settings, image=None):
    # producer task
    def aquire_images(settings, buffer_index, n_frames=100, image=None):
        global buffer

        index = 0
        if image is None:
            cam = IMAQ.IMAQCamera()
            cam.setup_acquisition(mode="sequence", nframes=n_frames)
            cam.start_acquisition()
            for _ in tqdm(range(n_frames), postfix='Acquiring images'):
                cam.wait_for_frame()
                frame, info = cam.read_oldest_image(return_info=True)
                rois = [get_roi(frame, tracker.roi_size, coord) for coord in settings['coords']]

            cam.stop_acquisition()
        else:
            for frame_index in tqdm(range(n_frames), postfix='Acquiring images: Simulation using saved image'):
                rois = [get_roi(image, settings['size'], coord) for coord in settings['coords']]
                buffer[index] = {'frame': frame_index, 'rois':rois}
                buffer_index.put(index)
                index = (index + 1) % CAPACITY
                sleep(0.03) # for ~25 fps
        buffer_index.put(None)

    # consumer task
    def process_rois(settings, buffer_index, processed_data, identifier):
        global buffer

        while True:
            index = buffer_index.get()
            if index is None:
                buffer_index.put(index)
                break
            item = buffer[index]

            if True:
                if item['frame'] == 0:
                    print(color_text(0, 50, 0, 'Real processing ;-)'))
                result = [get_xyza(roi['image'], settings['lut'], settings['lut_z_um'], roi['center']) for roi in item['rois']]
            else:
                if item['frame'] == 0:
                    print(color_text(0, 50, 0, 'Dummy processing ;-<'))
                result = [[np.NaN] * 4 for roi in item['rois']]
                sleep(0.2) # for ~20 fps @ 1CPU
            result = np.asarray(np.append([item['frame'], identifier], np.reshape(result, (-1))))
            processed_data.put(result)

    def get_queue(q, pars=None):
        result = pd.DataFrame([q.get() for _ in range(q.qsize())])
        if pars is not None:
            numbered_pars = [par for par in pars if '%' in par]
            shared_pars = [par for par in pars if '%' not in par]

            result.columns = shared_pars + list(
                np.reshape([[p.replace('%', f'{i}') for p in numbered_pars] for i in
                            range(len(result.columns) // len(numbered_pars))], -1))
            result.set_index('frame', inplace=True, drop=True)
        return result.sort_index()

    # Start combined acquisition and processing
    n_threads = 1
    processed_data = Queue()
    buffer_index = Queue()
    n_images = 100

    threads = [Thread(target=process_rois, args=(processing_settings, buffer_index, processed_data, i)) for i in range(n_threads)]
    threads.append(Thread(target=aquire_images, args=(acquisition_settings, buffer_index, n_images, image)))
    t_start = time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    print(color_text(0, 100, 0,
                     f'Acquisition + processing time = {time() - t_start:.3f} s for {n_threads} processing threads'))

    try:
        results = get_queue(processed_data,
                            pars=['frame', 'thread', '%: X (pix)', '%: Y (pix)', '%: Z (um)', '%: A (a.u.)'])
        return results
    except ValueError:
        return None


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
        data.pars = tracker.find_beads(im, 200, 0.6, show=True)
        data.set_glob('roi (pix)', tracker.roi_size, 'Image processing')
        data.to_file()
        # im = None

    # print(data.pars.tail(3))

    # acquire and process images
    coords = np.asarray([data.pars['X0 (pix)'], data.pars['Y0 (pix)']]).astype(int).T
    tracker.set_roi_coords(coords[:50])

    data.traces = test_acquisition(tracker.get_acquisition_settings(), tracker.get_settings(), im)
    # data.to_file()
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
