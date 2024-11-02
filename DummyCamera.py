import queue
import threading
import cv2
import numpy as np
from time import sleep

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 480
FRAME_WIDTH = FRAME_HEIGHT


def plot_rois(frame: np.ndarray, settings=None) -> np.ndarray:
    for i, center in enumerate(settings["rois"]):
        cv2.rectangle(
            frame,
            np.asarray(center) - settings["roi_size"] // 2,
            np.asarray(center) - settings["roi_size"] // 2 + settings["roi_size"],
            (255, 0, 0),
            1,
        )
        cv2.putText(
            frame,
            f"{i}",
            [
                center[0] - settings["roi_size"] // 2,
                center[1] - 5 - settings["roi_size"] // 2,
            ],
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 0, 0),
            1,
        )
    return frame


def create_dummy_frame(mean=128, settings=None) -> np.ndarray:
    frame = np.random.poisson(mean, (FRAME_HEIGHT, FRAME_WIDTH, 1)).astype(np.uint8)
    return frame


class CameraApplication:
    def __init__(self):
        self.running = False
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producer_thread = None
        self.consumer_thread = None

    def get_camera_id(self):
        return "dummy_camera"

    def start(self, settings=None):
        self.settings = settings
        self.frame_nr = -1
        self.running = True
        self.producer_thread = threading.Thread(target=self._produce_frames)
        self.producer_thread.start()
        self.consumer_thread = threading.Thread(target=self._consume_frames)
        self.consumer_thread.start()

    def stop(self):
        self.running = False
        if self.producer_thread is not None:
            self.producer_thread.join()
        if self.consumer_thread is not None:
            self.consumer_thread.join()
        self.running = False

    def exit(self):
        self.stop()
        self.frame_queue.queue.clear()
        self.frame_queue = None

    def _process_frame(self, frame):
        # Dummy processing
        return frame

    def _produce_frames(self):
        while self.running:
            frame = create_dummy_frame(settings=self.settings)
            self.frame_nr += 1
            try:
                self.frame_queue.put_nowait((self.frame_nr, frame))
            except queue.Full:
                pass

            threading.Event().wait(1 / 30)  # Simulate 30 FPS

    def _consume_frames(self):
        alive = True
        while alive:
            if not self.frame_queue.empty():
                frame_nr, frame = self.frame_queue.get()
                # Process the frame (dummy processing here)

                # display frames
                if frame_nr % 1 == 0:
                    cv_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    plot_rois(cv_frame, self.settings)
                    cv2.imshow(self.get_camera_id(), cv_frame)
                    cv2.namedWindow(self.get_camera_id())
                    cv2.setMouseCallback(self.get_camera_id(), self._mouse_callback)
            if cv2.waitKey(10) == 13 or not self.running:  # 13 is the Enter key
                cv2.destroyAllWindows()
                alive = False

    def _mouse_callback(self, event, x, y, flags, param):
        def distance2(coords):
            return (coords[0] - x) ** 2 + (coords[1] - y) ** 2

        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if self.settings["rois"]:
                    closest_coord = min(self.settings["rois"], key=distance2)
                self.settings["rois"].remove(closest_coord)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.settings["rois"].append((x, y))


if __name__ == "__main__":
    cam = CameraApplication()

    settings = {"roi_size": 100, "rois": [(70, 190), (100, 200), (198, 350)]}
    cam.start(settings)
    sleep(3)
    cam.stop()
