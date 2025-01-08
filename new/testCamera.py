import threading
import queue
import cv2
import numpy as np
from vmbpy import *

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 480
FRAME_WIDTH = 640


def resize_if_required(frame: Frame) -> np.ndarray:
    img = frame.as_opencv_image()
    if img.shape[0] != FRAME_HEIGHT or img.shape[1] != FRAME_WIDTH:
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
    return img


class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        super().__init__()
        self.cam = cam
        self.frame_queue = frame_queue
        self.running = True
        self.n_frames = 0

    def run(self):
        with self.cam:
            print("FrameProducer started")
            self.cam.start_streaming(self)
            while self.running:
                frame = self.cam.get_frame()
                self.frame_queue.put(frame)
                print("Produced frame:", self.n_frames)
                self.n_frames += 1

    def stop(self):
        self.running = False
        self.cam.stop_streaming()


class FrameConsumer(threading.Thread):
    def __init__(self, frame_queue: queue.Queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.alive = True

    def run(self):
        print("FrameConsumer started")
        while self.alive:
            try:
                frame = self.frame_queue.get(timeout=1)
                img = resize_if_required(frame)
                cv2.imshow("Frame", img)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    self.alive = False
            except queue.Empty:
                continue
        cv2.destroyAllWindows()

    def stop(self):
        self.alive = False
        cv2.destroyAllWindows()


class CameraApplication:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()
        self.consumer = None
        self.stop_event = threading.Event()

    def __call__(self, cam: Camera, event: CameraEvent):
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()
                print(f"Camera {cam.get_id()} detected and producer started")
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()
                print(f"Camera {cam.get_id()} missing and producer stopped")

    def start(self):
        self.consumer = FrameConsumer(self.frame_queue)
        self.consumer.start()

    def stop(self):
        if self.consumer:
            self.consumer.stop()
            self.consumer.join()
        with self.producers_lock:
            for producer in self.producers.values():
                producer.stop()
                producer.join()
            self.producers.clear()
        print("Application stopped")

    def run(self):
        self.start()
        while not self.stop_event.is_set():
            pass
        self.stop()

    def wait_for_stop_signal(self):
        input("Press 's' to stop the application...\n")
        self.stop_event.set()


if __name__ == "__main__":
    with VmbSystem.get_instance() as vmb:
        app = CameraApplication()
        vmb.register_camera_change_handler(app)

        stop_thread = threading.Thread(target=app.wait_for_stop_signal)
        stop_thread.start()

        app.run()

        stop_thread.join()
        vmb.unregister_camera_change_handler(app)
