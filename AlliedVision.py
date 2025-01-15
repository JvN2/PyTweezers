import copy
import queue
import threading
from typing import Optional
from ImageProcessing import FrameConsumer
from icecream import ic
import cv2


from vmbpy import *

FRAME_QUEUE_SIZE = 10


def print_preamble():
    print("////////////////////////////////////////")
    print("/// VmbPy Multithreading Example ///////")
    print("////////////////////////////////////////\n")
    print(flush=True)


def add_camera_id(frame: Frame, cam_id: str) -> Frame:
    cv2.putText(
        frame.as_opencv_image(),
        "Cam: {}".format(cam_id),
        org=(0, 30),
        fontScale=1,
        color=255,
        thickness=1,
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )
    return frame


def try_put_frame(
    q: queue.Queue,
    cam: Camera,
    frame: Optional[Frame],
    frame_num,
    digital_input_state=0,
):
    try:
        q.put_nowait((cam.get_id(), frame, frame_num, digital_input_state))
    except queue.Full:
        pass


def set_nearest_value(cam: Camera, feat_name: str, feat_value: int):
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

        msg = (
            "Camera {}: Failed to set value of Feature '{}' to '{}': "
            "Using nearest valid value '{}'. Note that, this causes resizing "
            "during processing, reducing the frame rate."
        )
        Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))


class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue, settings_new={}):
        threading.Thread.__init__(self)
        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()
        self.settings_new = settings_new
        self.framerate = None

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                frame_num = frame.get_id()
                line_status = cam.LineStatus.get()
                try_put_frame(self.frame_queue, cam, frame_cpy, frame_num, line_status)
        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    # def get_framerate(self):
    #     return self.cam.get_framerate()

    def setup_camera(self):
        set_nearest_value(self.cam, "Height", self.settings_new.camera__pix[0])
        set_nearest_value(self.cam, "Width", self.settings_new.camera__pix[1])
        set_nearest_value(self.cam, "OffsetX", 456)  # 456 (1936 -1024)/2
        set_nearest_value(self.cam, "OffsetY", 0)  # 96(1216 -1024)/2
        self.cam.set_pixel_format(PixelFormat.Mono8)

        try:
            self.cam.TriggerMode.set("Off")
            self.cam.LineSelector.set("Line2")
            self.cam.ExposureTime.set(self.settings_new.exposure_time__us)

            self.framerate = self.cam.get_feature_by_name("AcquisitionFrameRate").get()
            # ic("Frame producer", self.framerate)

            # print("Getting camera features:")
            # for f in self.cam.get_all_features():
            #     try:
            #         print(f"{f.get_name()}: {f.get()}")
            #     except Exception as e:
            #         # print(f"Error getting feature: {e}")
            #         pass
            # print("Camera settings configured successfully.")

        except Exception as e:
            print(f"Error camera set up: {e}")

    def run(self):
        self.log.info("Thread 'FrameProducer({})' started.".format(self.cam.get_id()))
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
            try_put_frame(self.frame_queue, self.cam, None, -1, False)
        self.log.info(
            "Thread 'FrameProducer({})' terminated.".format(self.cam.get_id())
        )


class CameraApplication:
    def __init__(self, settings={}, root=None, data_queue=None):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()
        self.settings = settings
        self.consumer = FrameConsumer(
            self.frame_queue,
            self.settings,
            root=root,
            data_queue=data_queue,
        )
        self.framerate = None

    def __call__(self, cam: Camera, event: CameraEvent):
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(
                    cam, self.frame_queue, self.settings
                )
                self.producers[cam.get_id()].start()
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()
                self.framerate = self.producers[cam.get_id()].framerate
                # ic("CameraApplication", self.framerate)

    def list_camera_features(self, cam: Camera):
        with cam:
            for feature in cam.get_all_features():
                try:
                    feature_name = feature.get_name()
                    feature_value = feature.get()
                    is_writeable = feature.is_writeable()
                    print(
                        f"Feature name: {feature_name}, Value: {feature_value}, Writable: {is_writeable}"
                    )
                except VmbFeatureError as e:
                    # print(f"Feature name: {feature.get_name()}, Error: {e}")
                    pass
                except AttributeError:
                    pass

    def run(self):
        log = Log.get_instance()
        log.setLevel(LogLevel.Warning)
        vmb = VmbSystem.get_instance()
        vmb.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)
        log.info("'Application' started.")

        with vmb:
            for cam in vmb.get_all_cameras():
                # self.list_camera_features(cam)
                self.producers[cam.get_id()] = FrameProducer(
                    cam, self.frame_queue, self.settings
                )

            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            vmb.register_camera_change_handler(self)
            self.consumer.run()
            vmb.unregister_camera_change_handler(self)

            with self.producers_lock:
                for producer in self.producers.values():
                    producer.stop()
                for producer in self.producers.values():
                    producer.join()

        log.info("'Application' terminated.")

    def stop(self):
        with self.producers_lock:
            for producer in self.producers.values():
                producer.stop()
            for producer in self.producers.values():
                producer.join()
        self.consumer.stop()


# if __name__ == "__main__":
#     print_preamble()
#     settings = {"roi_size (pix)": 50, "rois": [(50, 50), (100, 200), (198, 350)]}
#     settings["height (pix)"] = 400
#     settings["width (pix)"] = 400
#     settings["zoom"] = 2
#     app = CameraApplication(settings, tracker=settings)
#     threading.Thread(target=app.run).start()
#     sleep(2)
#     app.stop()
