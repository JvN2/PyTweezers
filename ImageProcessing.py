import queue
import cv2
import numpy as np
from time import sleep
from vmbpy import *

FRAME_QUEUE_SIZE = 10


def plot_rois(cv_frame: np.ndarray, settings=None) -> np.ndarray:
    cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_GRAY2RGB)
    for i, center in enumerate(settings.get("rois", [])):
        color = (0, 255, 255) if i == settings.get("selected", 0) else (255, 0, 0)
        cv2.rectangle(
            cv_frame,
            np.asarray(center) - settings["roi_size (pix)"] // 2,
            np.asarray(center)
            - settings["roi_size (pix)"] // 2
            + settings["roi_size (pix)"],
            color,
            1,
        )
        cv2.putText(
            cv_frame,
            f"{i}",
            [
                center[0] - settings["roi_size (pix)"] // 2,
                center[1] - 5 - settings["roi_size (pix)"] // 2,
            ],
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            color,
            1,
        )
    return cv_frame


def extract_rois(frame: np.ndarray, settings=None) -> np.ndarray:
    half_size = settings.get("roi_size (pix)", 100) // 2
    rois = [
        frame[y - half_size : y + half_size, x - half_size : x + half_size, 0]
        for x, y in settings.get("rois", [])
    ]
    return np.array(rois)


class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue, settings: dict):
        self.frame_queue = frame_queue
        self.quit = False
        self.settings = settings

    def run(self):
        IMAGE_CAPTION = "MT camera"
        frames = {}
        alive = True

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
                cv_images = np.concatenate(
                    [
                        frames[cam_id].as_opencv_image()
                        for cam_id in sorted(frames.keys())
                    ],
                    axis=1,
                )
                rois = extract_rois(cv_images, self.settings)
                cv_images = plot_rois(cv_images, self.settings)
                cv_images = self._get_zoom(cv_images)
                cv2.imshow(IMAGE_CAPTION, cv_images)
                cv2.namedWindow(IMAGE_CAPTION)
                cv2.setMouseCallback(IMAGE_CAPTION, self._mouse_callback)

            cv2.waitKey(10)
            if self.quit:
                cv2.destroyAllWindows()
                alive = False

    def stop(self):
        self.quit = True

    def _get_zoom(self, cv_frame: np.ndarray) -> np.ndarray:
        return cv_frame
        def extract_square(image: np.ndarray, center: tuple, width: int) -> np.ndarray:
            print(center, width)
            height, img_width = image.shape[:2]
            half_width = width // 2
            x, y = center

            # Calculate the coordinates of the top-left and bottom-right corners
            x0, x1 = x - half_width, x + half_width
            y0, y1 = y - half_width, y + half_width

            # Adjust the coordinates if they exceed the image boundaries
            if x0 < 0:
                x0 = 0
                x1 = width
            if x1 > img_width:
                x1 = img_width
                x0 = img_width - width
            if y0 < 0:
                y0 = 0
                y1 = width
            if y1 > height:
                y1 = height
                y0 = height - width

            # Ensure the coordinates are within the image boundaries
            x0 = max(0, x0)
            x1 = min(img_width, x1)
            y0 = max(0, y0)
            y1 = min(height, y1)

            # Extract the square region from the image
            square = image[y0:y1, :, x0:x1, :]

            return square, (x0, y0)

        new, offset = extract_square(
            cv_frame,
            self.settings.get("rois", [0])[self.settings.get("selected", 0)],
            self.settings.get("window (pix)", 1000),
        )
        self.settings["offset (pix)"] = offset
        print(new.shape)

        return cv_frame

    # if zoom != 1:
    #     height, width = cv_frame.shape[:2]
    #     new_height, new_width = int(height * zoom), int(width * zoom)
    #     if new_height <= 0 or new_width <= 0:
    #         raise ValueError(f"Invalid dimensions after resizing: {new_height}x{new_width}")
    #     cv_frame = cv2.resize(cv_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # return cv_frame

    def _mouse_callback(self, event, x, y, flags, param):
        x, y = (
            int(x / self.settings.get("zoom (pix)", 1) + self.settings.get("offset (pix)", [0, 0])[0]),
            int(y / self.settings.get("zoom (pix)", 1) + self.settings.get("offset (pix)", [0, 0])[1]),
        )
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.settings["rois"]) == 0:
                    self.settings["rois"].append([x, y])
                    return
                dist = np.abs(
                    np.asarray([c - np.asarray([x, y]) for c in self.settings["rois"]])
                )
                in_roi = np.all((dist < self.settings["roi_size (pix)"] // 2), axis=1)
                index = np.where(in_roi)[0]
                if len(index) > 0:
                    self.settings["rois"].pop(index[0])
                    self.settings["selected"] = max(0, self.settings["selected"] - 1)
                else:
                    margin = self.settings["roi_size (pix)"] // 2
                    if (margin < x < self.settings["width (pix)"] - margin) & (
                        margin < y < self.settings["height (pix)"] - margin
                    ):
                        self.settings["rois"].append([x, y])
                        self.settings["selected"] = len(self.settings["rois"]) - 1
        elif flags & cv2.EVENT_LBUTTONDOWN:

            dist = np.abs(
                np.asarray([c - np.asarray([x, y]) for c in self.settings["rois"]])
            )
            in_roi = np.all((dist < self.settings["roi_size (pix)"] // 2), axis=1)
            index = np.where(in_roi)[0]
            if len(index) > 0:
                self.settings["selected"] = index[0]

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.settings["zoom"] *= 1.5
                if self.settings["zoom"] > 3:
                    self.settings["zoom"] = 3
            else:
                self.settings["zoom"] /= 1.5
                if self.settings["zoom"] < 0.25:
                    self.settings["zoom"] = 0.25
