import copy
import queue
import threading
from typing import Optional
from vmbpy import *

from ImageProcessing import FrameConsumer

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 480
FRAME_WIDTH = 480


def enqueue_frame(q: queue.Queue, frame_nr, frame: Optional[Frame]):
    try:
        q.put_nowait((frame_nr, frame))

    except queue.Full:
        pass


def adjust_cam_setting(cam: Camera, feat_name: str, feat_value: int):
    feat = cam.get_feature_by_name(feat_name)
    try:
        feat.set(feat_value)

    except VmbFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()
        if feat_value <= min_:
            val = min_
        elif feat_value >= max_:
            val = max_
        else:
            val = (((feat_value - min_) // inc) * inc) + min_
        feat.set(val)


class FrameProducer(threading.Thread):
    def __init__(self, frame_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.log = Log.get_instance()

        vmb = VmbSystem.get_instance()
        with vmb:
            for cam in vmb.get_all_cameras():
                self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                enqueue_frame(self.frame_queue, 0, frame_cpy)
        cam.queue_frame(frame)


    def setup_camera(self):
        adjust_cam_setting(self.cam, "Height", FRAME_HEIGHT)
        adjust_cam_setting(self.cam, "Width", FRAME_WIDTH)
        print(self.cam.get_all_features())
        # Try to enable automatic exposure time setting
        # try:
        #     self.cam.ExposureAuto.set('Once')

        # except (AttributeError, VmbFeatureError):
        #     self.log.info('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(
        #                   self.cam.get_id()))

        self.cam.set_pixel_format(PixelFormat.Mono8)

    def start(self):
        try:
            with self.cam:
                self.setup_camera()
                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()
                finally:
                    self.cam.stop_streaming()
        except VmbCameraError:
            pass
        finally:
            enqueue_frame(self.frame_queue, 0, None)

    def stop(self):
        self.killswitch.set()