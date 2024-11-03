import queue
import threading
import cv2
import numpy as np
from time import sleep
from concurrent.futures import ThreadPoolExecutor


FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 480
FRAME_WIDTH = FRAME_HEIGHT


def get_roi(frame, center, size):
    x, y = center
    x0, x1 = x - size // 2, x + size // 2
    y0, y1 = y - size // 2, y + size // 2
    return frame[y0:y1, x0:x1]


def process_single_roi(args):
    frame_nr, frame, center, size = args
    roi = get_roi(frame, center, size)
    return [*center, roi.min(), roi.max()]


def process_frame(frame_nr: np.int32, frame: np.ndarray, settings=None) -> np.ndarray:
    row = np.asarray([frame_nr])
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                process_single_roi,
                [
                    (frame_nr, frame, center, settings["roi_size"])
                    for center in settings["rois"]
                ],
            )
        )
    for result in results:
        row = np.append(row, result)
    print(row)
    return frame


def plot_rois(cv_frame: np.ndarray, settings=None) -> np.ndarray:
    for i, center in enumerate(settings["rois"]):
        cv2.rectangle(
            cv_frame,
            np.asarray(center) - settings["roi_size"] // 2,
            np.asarray(center) - settings["roi_size"] // 2 + settings["roi_size"],
            (255, 0, 0),
            1,
        )
        cv2.putText(
            cv_frame,
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
    return cv_frame


def apply_zoom(self, cv_frame):
    h, w, _ = cv_frame.shape
    center_x, center_y = w // 2, h // 2
    new_w, new_h = int(w / self.zoom_level), int(h / self.zoom_level)
    top_left_x = max(center_x - new_w // 2, 0)
    top_left_y = max(center_y - new_h // 2, 0)
    bottom_right_x = min(center_x + new_w // 2, w)
    bottom_right_y = min(center_y + new_h // 2, h)
    cropped_frame = cv_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    resized_frame = cv2.resize(cropped_frame, (w, h))
    return resized_frame


def create_dummy_frame(mean=200, settings=None) -> np.ndarray:
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

            threading.Event().wait(1 / 60)  # Simulate 30 FPS

    def _consume_frames(self):
        alive = True
        while alive:
            if not self.frame_queue.empty():
                frame_nr, frame = self.frame_queue.get()
                # Process the frame (dummy processing here)
                # process_frame(frame_nr, frame, self.settings)
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
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if len(self.settings["rois"]) == 0:
                    self.settings["rois"].append([x, y])
                    return
                dist = np.abs(
                    np.asarray([c - np.asarray([x, y]) for c in self.settings["rois"]])
                )
                in_roi = np.all((dist < self.settings["roi_size"] // 2), axis=1)
                index = np.where(in_roi)[0]
                if len(index) > 0:
                    self.settings["rois"].pop(index[0])
                else:
                    margin = self.settings["roi_size"] // 2
                    if (margin < x < FRAME_WIDTH - margin) & (
                        margin < y < FRAME_HEIGHT - margin
                    ):
                        self.settings["rois"].append([x, y])
            # elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            # move cursor to center


if __name__ == "__main__":
    cam = CameraApplication()

    settings = {"roi_size": 100, "rois": [(70, 190), (100, 200), (198, 350)]}
    cam.start(settings)
    sleep(3)
    cam.stop()
