# Inspired by SuperFastPython.com -> example of one producer and multiple consumers with threads
from pathlib import Path
from queue import Queue
from threading import Thread
from time import time

import numpy as np
import pandas as pd
from cv2 import imread
from pylablib.devices import IMAQ
from tqdm import tqdm

from modules.TweezerImageProcessing import get_roi, Beads, Traces
from modules.TweezersSupport import color_text
from modules.TweezersSupport import time_it


@time_it
def test_aquisition(tracker, image=None):
    # producer task
    def aquire_images(tracker, queue_raw_data, n_frames=100, image=None):
        if image is None:
            cam = IMAQ.IMAQCamera()
            cam.setup_acquisition(mode="sequence", nframes=n_frames)
            cam.start_acquisition()
            for _ in tqdm(range(n_frames), postfix="Acquiring images"):
                cam.wait_for_frame()
                frame, info = cam.read_oldest_image(return_info=True)
                rois = [
                    get_roi(frame, tracker.roi_size, coord) for coord in tracker.coords
                ]
                queue_raw_data.put([info.frame_index, rois])
            cam.stop_acquisition()
        else:
            for i in tqdm(range(n_frames), postfix="Using dummy image"):
                rois = [
                    get_roi(image, tracker.roi_size, coord) for coord in tracker.coords
                ]
                queue_raw_data.put([i, rois])
        queue_raw_data.put(None)

    # consumer task
    def process_rois(tracker, queue_raw_data, queue_processed_data, identifier):
        while True:
            item = queue_raw_data.get()
            if item is None:
                queue_raw_data.put(item)
                break
            # sleep(1/5)
            result = [tracker.get_xyza(*roi) for roi in item[1]]
            # result = [[np.NaN]*4 for roi in item[1]]

            result = np.append(item[0], np.reshape(result, (-1)))
            queue_processed_data.put(result)

    def get_queue(q, pars=None):
        result = pd.DataFrame([q.get() for _ in range(q.qsize())])
        if pars is not None:
            numbered_pars = [par for par in pars if "%" in par]
            shared_pars = [par for par in pars if "%" not in par]

            result.columns = shared_pars + list(
                np.reshape(
                    [
                        [p.replace("%", f"{i}") for p in numbered_pars]
                        for i in range(len(result.columns) // len(numbered_pars))
                    ],
                    -1,
                )
            )
            result.set_index("frame", inplace=True, drop=True)
        return result.sort_index()

    # Start combined acquisition and processing
    n_cores = 4
    raw_data = Queue()
    processed_data = Queue()
    n_frames = 100

    producer = [Thread(target=aquire_images, args=(tracker, raw_data, n_frames, image))]
    consumers = [
        Thread(
            target=process_rois,
            args=(tracker, raw_data, processed_data, i),
            daemon=True,
        )
        for i in range(n_cores)
    ]

    t_start = time()
    for proces in consumers + producer:
        proces.start()

    for proces in consumers + producer:
        proces.join()

    print(
        color_text(
            0,
            100,
            0,
            f"Acquisition + processing time = {time() - t_start:.3f} s for {n_cores} processing threads",
        )
    )

    try:
        results = get_queue(
            processed_data,
            pars=[
                "frame",
                "cpu",
                "%: X (pix)",
                "%: Y (pix)",
                "%: Z (um)",
                "%: A (a.u.)",
            ],
        )
        return results
    except ValueError:
        return None


if __name__ == "__main__":
    # set up lookup table
    ref_filename = Path(r"data\data_024.tdms")
    tracker = Beads(ref_filename)

    # find bead coordinates
    from_file = True
    filename = Path(r"data\data_024.jpg")
    data = Traces(filename)

    if from_file:
        im = imread(str(filename))[:, :, 0].astype(float).T
    else:
        im = imread(str(filename))[:, :, 0]
        data.pars = tracker.find_beads(im, 200, 0.6, show=False)
        data.set_glob("roi (pix)", tracker.roi_size, "Image processing")
        data.to_file()
        im = None

    print(data.pars.tail(3))

    # acquire and process images
    coords = np.asarray([data.pars["X0 (pix)"], data.pars["Y0 (pix)"]]).astype(int).T
    tracker.set_roi_coords(coords[:100])
    data.traces = test_aquisition(tracker, im)
    # data.to_file()
    print(data.traces.tail(3))

    if False:
        # plot positions
        selected_cols = [col for col in data.data.columns if "X (pix)" in col]
        x = data.data[selected_cols].iloc[0].values
        selected_cols = [col for col in data.data.columns if "Y (pix)" in col]
        y = data.data[selected_cols].iloc[0].values
        plt.imshow(im, cmap="Greys_r", origin="lower")
        plt.scatter(y - 50, x - 50, s=80, facecolors="none", edgecolors="r")
        # xy coords are not correct: to be solved
        # probably x and y direction mixed up
        plt.show()
