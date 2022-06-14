# see https://pylablib.readthedocs.io/en/latest/devices/cameras_basics.html#cameras-basics

from pylablib.devices import IMAQ
from PIL import Image, ImageSequence

import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import cv2
from tqdm import tqdm


def get_roi(im, width, center = None, size = None):
    if center is None:
        center = np.asarray(np.shape(im))//2
    bl = np.asarray(center) -width//2
    bl = np.clip(bl, 0, len(im))
    tr = bl + width
    roi = im[bl[0]:tr[0], bl[1]:tr[1]]
    if size is not None:
        roi = cv2.resize(roi, size)
    return roi

def aquire_images(n_frames=100, show = True):
    cam = IMAQ.IMAQCamera()
    # print(IMAQ.list_cameras())
    # image = cam.snap()
    # print(cam.get_detector_size())
    # attrs = cam.get_all_grabber_attribute_values()
    # for a in attrs:
    #     print(a, attrs[a])

    cam.setup_acquisition(mode="sequence", nframes=n_frames)  # could be combined with start_acquisition, or kept separate
    cam.start_acquisition()
    cv2.namedWindow('live cam', cv2.WINDOW_NORMAL)

    for i in tqdm(range(n_frames)):  # acquisition loop
        cam.wait_for_frame()  # wait for the next available frame
        frame, info = cam.read_oldest_image(return_info=True)  # get the oldest image which hasn't been read yet
        if show:
            im = get_roi(frame, 500)
            cv2.imshow('live cam', im)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cam.stop_acquisition()
    return frame


filename = r'data\test.jpg'
# frame = aquire_images(500)
# cv2.imwrite(r'c:\tmp\test.jpg', frame)

im = cv2.imread(filename)
im = get_roi(im, 500)
cv2.imshow('image', im)
cv2.waitKey(1500)
