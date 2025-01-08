import copy
import queue
import threading
from typing import Optional

import cv2
import numpy

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 480
FRAME_WIDTH = 640


def try_put_frame(q: queue.Queue, cam_id: str, frame: numpy.ndarray):
    try:
        q.put_nowait((cam_id, frame))
    except queue.Full:
        pass


class Webcam:
    def __init__(self, cam_id=0):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running = False
        self.killswitch = threading.Event()

    def start(self):
        self.running = True
        threading.Thread(target=self._run).start()

    def stop(self):
        self.running = False
        self.killswitch.set()
        self.cap.release()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                # Ensure the frame is in the correct format
                if (
                    frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
                    and frame.dtype == numpy.uint8
                ):
                    try_put_frame(self.frame_queue, self.get_id(), frame)
                else:
                    print(
                        f"Captured frame has invalid shape or type: {frame.shape}, {frame.dtype}"
                    )
            threading.Event().wait(1 / 30)  # Simulate 30 FPS

    def get_id(self):
        return f"webcam_{self.cam_id}"

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None


class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.frame_queue = frame_queue
        self.running = True

    def run(self):
        alive = True
        while alive:
            if not self.frame_queue.empty():
                cam_id, frame = self.frame_queue.get()
                # Ensure the frame is a valid NumPy array
                if isinstance(frame, numpy.ndarray):
                    # Ensure the frame has the correct shape and type
                    if (
                        frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
                        and frame.dtype == numpy.uint8
                    ):
                        # Process the frame (dummy processing here)
                        cv2.imshow(cam_id, frame)
                    else:
                        print(
                            f"Invalid frame shape or type: {frame.shape}, {frame.dtype}"
                        )
                else:
                    print(f"Invalid frame type: {type(frame)}")
            if cv2.waitKey(10) == 13 or not self.running:  # 13 is the Enter key
                cv2.destroyAllWindows()
                alive = False

        print("'FrameConsumer' terminated.")


class CameraApplication:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producers = {}
        self.producers_lock = threading.Lock()
        self.running = False
        self.consumer = FrameConsumer(self.frame_queue)

    def __call__(self, cam: Webcam, event: str):
        # New camera was detected. Create FrameProducer, add it to active FrameProducers
        if event == "Detected":
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()

        # An existing camera was disconnected, stop associated FrameProducer.
        elif event == "Missing":
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()


class FrameProducer:
    def __init__(self, cam: Webcam, frame_queue: queue.Queue):
        self.cam = cam
        self.frame_queue = frame_queue
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._run).start()

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            frame = self.cam.get_frame()
            if frame is not None:
                try_put_frame(self.frame_queue, self.cam.get_id(), frame)
            threading.Event().wait(1 / 30)  # Simulate 30 FPS


if __name__ == "__main__":
    cam_app = CameraApplication()
    webcam = Webcam()
    cam_app(webcam, "Detected")
    webcam.start()
    cam_app.consumer.run()
    webcam.stop()
    cam_app(webcam, "Missing")
