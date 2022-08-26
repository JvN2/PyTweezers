from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import imread
from tqdm import tqdm
from multiprocessing import Process, Manager

from modules.TweezerImageProcessing import proces_image, Beads, Traces, get_roi, get_xyza
from modules.TweezersSupport import color_text



# producer task
def aquire_images(settings, raw_data, n_frames=100, image=None):
    if image is None:
        print('IMAQ not implemented')
        # cam = IMAQ.IMAQCamera()
        # cam.setup_acquisition(mode="sequence", nframes=n_frames)
        # cam.start_acquisition()
        # for _ in tqdm(range(n_frames), postfix='Acquiring images'):
        #     cam.wait_for_frame()
        #     frame, info = cam.read_oldest_image(return_info=True)
        #     rois = [get_roi(frame, settings['size'], coord) for coord in settings['coords']]
        #     raw_data.put({'frame': info.frame_index, 'rois': rois})
        # cam.stop_acquisition()
    else:
         for frame_index in tqdm(range(n_frames), postfix='Acquiring images: Simulation using saved image'):
            rois = [get_roi(image, settings['size'], coord) for coord in settings['coords']]
            raw_data.put({'frame': frame_index, 'rois': rois})
    raw_data.put(None)

# consumer task
def process_rois(settings, raw_data, processed_data, identifier):
    while True:
        item = raw_data.get()
        if item is None:
            raw_data.put(None)
            break

        result = [get_xyza(roi, settings['lut'], settings['lut_z_um'], filter=settings['filter']) for roi in item['rois']]

        result = np.asarray(np.append([item['frame']], np.reshape(result, (-1))))
        processed_data.put(result)

def test_multi_processing(settings, image, n_images = 10):
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
    n_cores = 4
    manager = Manager()
    processed_data = manager.Queue()
    raw_data = manager.Queue()
    n_images = 100

    consumers = [Process(target=process_rois, args=(settings, raw_data, processed_data, i,)) for
                 i in range(n_cores)]
    producer = Process(target=aquire_images, args=(settings, raw_data, n_images, image))

    t_start = time()
    for process in [producer] + consumers:
        process.start()

    for process in [producer] + consumers:
        process.join()

    dt = time() - t_start
    print(color_text(0, 100, 0,
                     f'Acquisition + processing time = {dt:.3f} s for {n_cores} processing threads; {n_images/dt:.1f} frames/s'))

    try:
        results = get_queue(processed_data, pars=['frame', '%: X (pix)', '%: Y (pix)', '%: Z (um)', '%: A (a.u.)'])
        return results
    except ValueError:
        return None


def convert_result(result, pars=None):
    result = pd.DataFrame(result)

    if pars is not None:
        numbered_pars = [par for par in pars if '%' in par]
        shared_pars = [par for par in pars if '%' not in par]

        result.columns = shared_pars + list(
            np.reshape([[p.replace('%', f'{i}') for p in numbered_pars] for i in
                        range(len(result.columns) // len(numbered_pars))], -1))
        result.set_index('frame', inplace=True, drop=True)
    return result.sort_index()


def test_acquisition(settings, image=None, show=False):
    n_images = 50
    results = []
    start = time()
    for i in tqdm(range(n_images), postfix='Acquiring images: Simulation using saved image'):
        results.append(proces_image(i, image, settings))
    print(color_text(0, 100, 0, f'Computed in {n_images / (time() - start):.3f} frames/s'))
    df = convert_result(results, pars=['frame', '%: X (pix)', '%: Y (pix)', '%: Z (um)', '%: A (a.u.)'])
    print(df.tail(4))

    if show:
        x = [df.iloc[0][f'{i}: X (pix)'] for i, _ in enumerate(settings['coords'])]
        y = [df.iloc[0][f'{i}: Y (pix)'] for i, _ in enumerate(settings['coords'])]
        plt.imshow(image.T, cmap='Greys_r')
        plt.scatter(x, y)
        plt.show()


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
    tracker.set_roi_coords(coords)

    # data.traces = test_acquisition(tracker.get_settings(), im, show=True)
    # test2(im, tracker.get_settings())
    test_multi_processing(tracker.get_settings(), im)