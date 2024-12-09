import queue
import cv2
import numpy as np
from time import sleep
from vmbpy import *
from icecream import ic
from typing import Tuple, Optional

FRAME_QUEUE_SIZE = 10


def get_subarray(array: np.ndarray, center: tuple, size: int) -> np.ndarray:
    height, width = array.shape[:2]
    half_size = size // 2
    x, y = center

    # Calculate the coordinates of the top-left and bottom-right corners
    x0, x1 = x - half_size, x + half_size
    y0, y1 = y - half_size, y + half_size

    # Adjust the coordinates if they exceed the image boundaries
    if x0 < 0:
        x0 = 0
        x1 = size
    if x1 > width:
        x1 = width
        x0 = width - size
    if y0 < 0:
        y0 = 0
        y1 = size
    if y1 > height:
        y1 = height
        y0 = height - size

    # Ensure the coordinates are within the image boundaries
    x0 = max(0, x0)
    x1 = min(width, x1)
    y0 = max(0, y0)
    y1 = min(height, y1)

    # Extract the square sub-array from the image
    if len(array.shape) == 3:
        sub_array = array[y0:y1, x0:x1, :]
    else:
        sub_array = array[y0:y1, x0:x1]

    return (
        sub_array,
        [
            (x0 + x1) // 2,
            (y0 + y1) // 2,
        ],
        x1 - x0,
    )


def update_fov(
    fov: np.ndarray,
    center: Optional[np.ndarray] = None,
    zoom: float = 1,
    camera: Tuple[int, int] = (1000, 1000),
) -> np.ndarray:

    fov = np.asarray(fov).astype(int)
    camera = np.asarray(camera).astype(int)

    if center is None:
        center = np.sum(fov, axis=0) // 2
    else:
        center = np.asarray(center).astype(int)

    # Calculate the new half-width and half-height based on the zoom factor
    half_size = ((fov[1] - fov[0]) * zoom) // 2

    # Calculate the new FOV coordinates
    new_fov = np.array([center - half_size, center + half_size])

    # Adjust FOV coordinates if they exceed the camera boundaries
    if new_fov[0, 0] < 0:
        new_fov[:, 0] -= new_fov[0, 0]
    if new_fov[0, 1] < 0:
        new_fov[:, 1] -= new_fov[0, 1]
    if new_fov[1, 0] > camera[0]:
        new_fov[:, 0] -= new_fov[1, 0] - camera[0]
    if new_fov[1, 1] > camera[1]:
        new_fov[:, 1] -= new_fov[1, 1] - camera[1]

    # Ensure FOV is within camera boundaries
    new_fov = np.clip(new_fov, 0, camera)
    return new_fov


def extract_fov(cv_image: np.ndarray, fov: np.ndarray) -> np.ndarray:
    ic(cv_image, fov)
    return cv_image[
        fov[0][1] : fov[1][1],
        fov[0][0] : fov[1][0],
    ]


def plot_rois(
    cv_frame: np.ndarray, settings=None, fov=False, frame_nr=None, proccessing=False
) -> np.ndarray:
    cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_GRAY2RGB)
    if fov:
        roi_size = int(
            settings["roi_size (pix)"]
            * settings["window (pix)"]
            / settings["fov_size (pix)"]
        )
    else:
        roi_size = settings["roi_size (pix)"]
    for i, center in enumerate(settings.get("rois", [])):
        if fov:
            center = (np.array(center) - settings["fov_center (pix)"]) / settings[
                "fov_size (pix)"
            ]
            center += 0.5
            center = center * settings["window (pix)"]
            center = center.astype(int)

        color = (0, 255, 255) if i == settings.get("selected", 0) else (255, 0, 0)
        cv2.rectangle(
            cv_frame,
            np.asarray(center) - roi_size // 2,
            np.asarray(center) - roi_size // 2 + roi_size,
            color,
            1,
        )
        cv2.putText(
            cv_frame,
            f"{i}",
            [
                center[0] - roi_size // 2,
                center[1] - 5 - roi_size // 2,
            ],
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            color,
            1,
        )
        if frame_nr:
            if proccessing:
                color = (0, 255, 255)
            else:
                color = (255, 0, 0)
            cv2.putText(
                cv_frame,
                f"{frame_nr}",
                [
                    10,
                    cv_frame.shape[1] - 10,
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
        self.frames = []

    def run(self):
        IMAGE_CAPTION = "MT camera"
        frames = {}
        n_frames = 0
        alive = True

        while alive:
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame, frame_num, process_frames = (
                        self.frame_queue.get_nowait()
                    )
                    # print((cam_id, frame, line2))
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

                # Extract the FOV from the camera image and display it
                (
                    fov,
                    self.settings["fov_center (pix)"],
                    self.settings["fov_size (pix)"],
                ) = get_subarray(
                    cv_images,
                    self.settings["rois"][self.settings["selected"]],
                    self.settings["fov_size (pix)"],
                )

                fov = self._magnify(fov)
                fov = plot_rois(
                    fov,
                    self.settings,
                    fov=True,
                    frame_nr=frame_num,
                    proccessing=process_frames,
                )

                cv2.imshow(IMAGE_CAPTION, fov)
                cv2.namedWindow(IMAGE_CAPTION)
                cv2.setMouseCallback(IMAGE_CAPTION, self._mouse_callback)

                # extract selected roi and save it
                if self.settings.get("_aquisition mode") == "calibrate":
                    (roi, _, _) = get_subarray(
                        cv_images,
                        self.settings["rois"][self.settings["selected"]],
                        self.settings["roi_size (pix)"],
                    )
                    self.frames.append(roi)

                if (
                    n_frames >= self.settings["frames"] - 1
                    and self.settings["frames"] != 0
                ):
                    self.quit = True

                n_frames += 1

            cv2.waitKey(10)
            if self.quit:
                cv2.destroyAllWindows()
                alive = False
                if len(self.frames):
                    self.save_frames_to_binary_file(self.settings["_filename"])

    def stop(self):
        self.quit = True

    def _magnify(self, cv_frame: np.ndarray) -> np.ndarray:
        cv_frame = cv2.resize(
            cv_frame,
            (
                self.settings["window (pix)"],
                self.settings["window (pix)"],
            ),
            interpolation=cv2.INTER_NEAREST,
        )
        return cv_frame

    def _mouse_callback(self, event, x, y, flags, param):
        x, y = (
            (np.array([x, y]) / self.settings["window (pix)"] - 0.5)
            * self.settings["fov_size (pix)"]
            + self.settings["fov_center (pix)"]
        ).astype(int)

        roi_size = self.settings["roi_size (pix)"]
        roi_size = int(
            self.settings["roi_size (pix)"]
            * self.settings["window (pix)"]
            / self.settings["fov_size (pix)"]
        )

        if flags & cv2.EVENT_FLAG_CTRLKEY:
            # add or remove roi
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.settings["rois"]) == 0:
                    self.settings["rois"].append([x, y])
                    return
                dist = np.abs(
                    np.asarray([c - np.asarray([x, y]) for c in self.settings["rois"]])
                )
                in_roi = np.all((dist < roi_size // 2), axis=1)
                index = np.where(in_roi)[0]
                if len(index) > 0:
                    self.settings["rois"].pop(index[0])
                    self.settings["selected"] = max(0, self.settings["selected"] - 1)
                else:
                    margin = roi_size // 2
                    if (margin < x < self.settings["camera (pix)"][0] - margin) & (
                        margin < y < self.settings["camera (pix)"][1] - margin
                    ):
                        self.settings["rois"].append([x, y])
                        self.settings["selected"] = len(self.settings["rois"]) - 1

        elif flags & cv2.EVENT_LBUTTONDOWN:

            dist = np.abs(
                np.asarray([c - np.asarray([x, y]) for c in self.settings["rois"]])
            )
            in_roi = np.all((dist < roi_size // 2), axis=1)
            index = np.where(in_roi)[0]
            if len(index) > 0:
                self.settings["selected"] = index[0]

        if event == cv2.EVENT_MOUSEWHEEL:
            zoom = 1.25
            if flags < 0:
                self.settings["fov_size (pix)"] = int(
                    self.settings["fov_size (pix)"] * zoom
                )
            else:
                self.settings["fov_size (pix)"] = max(
                    int(self.settings["fov_size (pix)"] / zoom), 16
                )

    def save_frames_to_binary_file(self, filename: str):
        with open(filename, "wb") as f:
            for frame in self.frames:
                frame.tofile(f)

    def load_frames_from_binary_file(self, filename: str, shape: tuple, count: int):
        frames = []
        with open(filename, "rb") as f:
            for _ in range(count):
                frame = np.fromfile(f, dtype=np.uint8, count=np.prod(shape)).reshape(
                    shape
                )
                frames.append(frame)
        return frames


if __name__ == "__main__":
    # print("This is a module, not a standalone script.")
    array = np.array([[x for x in range(1000)] for y in range(1000)])
    center = (500, 1500)
    width = 10
    (subarray, center, width) = get_subarray(array.T, center, width)
    ic(subarray)
    ic(center)
    ic(width)
