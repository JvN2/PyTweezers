from multiprocessing import Process, Manager
from pathlib import Path
from time import time, sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import imread
from tqdm import tqdm

from modules.TweezerImageProcessing import proces_image, Beads, Traces, get_roi, get_xyza
from modules.TweezersSupport import color_text


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class IMAQ_dummy():
    def __init__(self, image, framerate=25):
        self.framerate = framerate
        self.image = image
        self.frame_info = AttrDict()
        return

    def setup_acquisition(self, nframes, mode=None):
        self.nframes = nframes
        return

    def start_acquisition(self):
        self.start = time()
        self.last_frame = -1
        return

    def wait_for_frame(self):
        current_frame = int((time() - self.start) * self.framerate)
        while current_frame < self.last_frame:
            sleep(0.05)
            current_frame = int((time() - self.start) * self.framerate)
        self.last_frame += 1
        return

    def read_oldest_image(self, return_info=True):
        self.frame_info.update({'frame_index': self.last_frame})
        return im, self.frame_info

    def stop_acquisition(self):
        return


def get_queue(q, pars=None):
    df = pd.DataFrame([q.get() for _ in range(q.qsize())])
    if pars is not None:
        numbered_pars = [par for par in pars if '%' in par]
        shared_pars = [par for par in pars if '%' not in par]

        df.columns = shared_pars + list(
            np.reshape([[p.replace('%', f'{i}') for p in numbered_pars] for i in
                        range(len(df.columns) // len(numbered_pars))], -1))
        df.set_index('frame', inplace=True, drop=True)
    return df.sort_index()


def framegrabber_init(n_frames=100, image=None):
    if image is None:
        cam = IMAQ.IMAQCamera()
    else:
        cam = IMAQ_dummy(image)
    cam.setup_acquisition(mode="sequence", nframes=n_frames)
    cam.start_acquisition()
    return cam


def framegrabber_run(cam, settings, queue):
    cam.wait_for_frame()
    frame, info = cam.read_oldest_image(return_info=True)
    rois = [get_roi(frame, settings['size'], coord) for coord in settings['coords']]
    result = [get_xyza(roi, settings['freqs'], settings['lut'], settings['lut_z_um'], filter=settings['filter'])
              for roi in rois]
    result = np.asarray(np.append([info.frame_index], np.reshape(result, (-1))))
    queue.put(result)
    return


def framegrabber_close(cam, queue):
    cam.stop_acquisition()
    df = get_queue(queue, pars=['frame', '%: X (pix)', '%: Y (pix)', '%: Z (um)', '%: A (a.u.)'])
    return df


def test_multi_processing(settings, image, show= False):
    # Start combined acquisition and processing
    n_cores = 4
    manager = Manager()
    queue = manager.Queue()
    n_images = 100

    cam = framegrabber_init(n_images, image)

    t_start = time()
    for i in tqdm(range(100), postfix='Aquiring and processing'):
        framegrabber_run(cam, settings, queue)
    # processes = [Process(target=grab_and_process_images, args=(cam, settings, queue, i,)) for i in range(n_cores)]
    #

    # for process in processes:
    #     process.start()
    dt = time() - t_start
    print(color_text(0, 100, 0,
                     f'Acquisition + processing time = {dt:.3f} s for {n_cores} processing threads; {n_images / dt:.1f} frames/s'))

    df = framegrabber_close(cam, queue)

    if show:
        x = [df.iloc[0][f'{i}: X (pix)'] for i, _ in enumerate(settings['coords'])]
        y = [df.iloc[0][f'{i}: Y (pix)'] for i, _ in enumerate(settings['coords'])]
        plt.imshow(image.T, cmap='Greys_r')
        plt.scatter(x, y, edgecolors='red', facecolors='none')
        plt.show()

    return df

if __name__ == '__main__':
    print()

    # set up lookup table
    ref_filename = Path(r'data\data_024.tdms')
    tracker = Beads(ref_filename)

    # find bead coordinates
    from_file = False
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

    settings = tracker.get_settings()
    print(settings)
    # test_multi_processing(tracker.get_settings(), im, show=True)
