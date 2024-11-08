import copy
import queue
import cv2
import numpy as np
from vmbpy import *

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 480
FRAME_WIDTH = 480


def resize_if_required(frame: Frame) -> np.ndarray:
    cv_frame = frame.as_opencv_image()

    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        cv_frame = cv_frame[..., np.newaxis]

    return cv_frame

class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        # self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.quit = False

    def run(self):
        IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit'
        KEY_CODE_ENTER = 13
        frames = {}
        alive = True
        # self.log.info('\'FrameConsumer\' started.')

        while alive:
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

                if frame:
                    frames[cam_id] = frame
                else:
                    frames.pop(cam_id, None)
                frames_left -= 1

            if frames:
                cv_images = [resize_if_required(frames[cam_id]) for cam_id in sorted(frames.keys())]
                cv2.imshow(IMAGE_CAPTION, np.concatenate(cv_images, axis=1))
            else:
                # cv2.imshow(IMAGE_CAPTION, create_dummy_frame())
                pass

            if KEY_CODE_ENTER == cv2.waitKey(10) or self.quit:
                cv2.destroyAllWindows()
                alive = False

        # self.log.info('\'FrameConsumer\' terminated.')
    def stop(self):
        self.quit = True