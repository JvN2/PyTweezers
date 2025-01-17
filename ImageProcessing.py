import queue
import cv2
import numpy as np
import pandas as pd
import traceback

from icecream import ic
from typing import Tuple, Optional
from TraceIO import hdf_data, timed_ic
from pathlib import Path
from matplotlib import pyplot as plt
from time import time

FRAME_QUEUE_SIZE = 10
SENTINEL = None


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
    return cv_image[
        fov[0][1] : fov[1][1],
        fov[0][0] : fov[1][0],
    ]


def plot_rois(fov, settings=None, frame_nr=None, proccessing=False):

    roi_size = int(
        settings.roi_size__pix * settings.window__pix / settings.fov_size__pix
    )

    for i, center in enumerate(settings.rois):

        center = (np.array(center) - settings.fov_center__pix) / settings.fov_size__pix
        center += 0.5
        center = center * settings.window__pix
        center = center.astype(int)

        color = (0, 255, 255) if i == settings.selected else (255, 0, 0)
        cv2.rectangle(
            fov,
            np.asarray(center) - roi_size // 2,
            np.asarray(center) - roi_size // 2 + roi_size,
            color,
            1,
        )
        cv2.putText(
            fov,
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
            fov,
            f"{frame_nr}",
            [
                10,
                fov.shape[1] - 10,
            ],
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            color,
            1,
        )

    return fov


def extract_rois(frame: np.ndarray, centers, size) -> np.ndarray:
    half_size = size // 2
    rois = [
        frame[y - half_size : y + half_size, x - half_size : x + half_size, 0]
        for x, y in centers
    ]
    return np.array(rois)


class FrameConsumer:
    def __init__(
        self,
        frame_queue: queue.Queue,
        settings: dict,
        root=None,
        data_queue=None,
    ):
        self.frame_queue = frame_queue
        self.data_queue = data_queue
        self.quit = False
        self.settings = settings
        self.selected_roi_frames = []
        self.traces = []
        self.main_root = root
        self.latest_processed_frame = -1

    def run(self):
        IMAGE_CAPTION = "MT camera"
        frames = {}
        n_frames = 0
        alive = True

        self.time = time()
        try:
            while alive:
                # get all frames from the queue
                frames_left = self.frame_queue.qsize()
                while frames_left:
                    try:
                        cam_id, frame, frame_num, acquisition_in_progress = (
                            self.frame_queue.get_nowait()
                        )

                    except queue.Empty:
                        acquisition_in_progress = True
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
                    if frame_num % 1 == 0:
                        # Extract the FOV from the camera image and display it
                        try:
                            center = self.settings.rois[self.settings.selected]
                        except IndexError:
                            center = self.settings.fov_center__pix

                        (
                            fov,
                            self.settings.fov_center__pix,
                            self.settings.fov_size__pix,
                        ) = get_subarray(
                            cv_images,
                            center,
                            self.settings.fov_size__pix,
                        )

                        fov = cv2.cvtColor(self._magnify(fov), cv2.COLOR_GRAY2RGB)
                        fov = plot_rois(
                            fov,
                            self.settings,
                            frame_nr=frame_num,
                            proccessing=acquisition_in_progress,
                        )

                        cv2.imshow(IMAGE_CAPTION, fov)
                        cv2.namedWindow(IMAGE_CAPTION)
                        cv2.setMouseCallback(
                            IMAGE_CAPTION, self._mouse_callback, acquisition_in_progress
                        )

                        if n_frames == 0:
                            self.main_root.focus_force()
                    n_frames += 1

                    if (
                        acquisition_in_progress
                        and frame_num != self.latest_processed_frame
                    ):
                        # get coordinates for each roi
                        rois = extract_rois(
                            cv_images,
                            self.settings.rois,
                            self.settings.roi_size__pix,
                        )
                        coords = np.concatenate(
                            (
                                [frame_num],
                                self.settings._tracker.get_coords(rois).flatten(),
                            )
                        )

                        self.data_queue.put(coords)
                        self.latest_processed_frame = frame_num

                        # Extract selected roi and save it
                        (roi, _, _) = get_subarray(
                            cv_images,
                            self.settings.rois[self.settings.selected],
                            self.settings.roi_size__pix,
                        )
                        self.selected_roi_frames.append(roi)

                    if len(self.selected_roi_frames) and not acquisition_in_progress:
                        if self.settings._aquisition_mode == "calibrate":
                            self.save_frames_to_binary_file(self.settings._filename)
                        self.selected_roi_frames.clear()
                        self.data_queue.put(SENTINEL)

                cv2.waitKey(10)
                if self.quit:
                    cv2.destroyAllWindows()
                    alive = False

        except Exception:
            traceback.print_exc()
            self.data_queue.put(SENTINEL)

    def stop(self):
        self.quit = True

    def _magnify(self, cv_frame: np.ndarray) -> np.ndarray:
        cv_frame = cv2.resize(
            cv_frame,
            (
                self.settings.window__pix,
                self.settings.window__pix,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
        return cv_frame

    def _mouse_callback(self, event, x, y, flags, param):
        acquisition_in_progress = param
        x, y = (
            (np.array([x, y]) / self.settings.window__pix - 0.5)
            * self.settings.fov_size__pix
            + self.settings.fov_center__pix
        ).astype(int)

        roi_size = self.settings.roi_size__pix
        roi_size = int(
            self.settings.roi_size__pix
            * self.settings.window__pix
            / self.settings.fov_size__pix
        )

        if flags & cv2.EVENT_FLAG_CTRLKEY:
            # add or remove roi
            if event == cv2.EVENT_LBUTTONDOWN and not acquisition_in_progress:
                if len(self.settings.rois) == 0:
                    self.settings.rois.append([x, y])
                    return
                dist = np.abs(
                    np.asarray([c - np.asarray([x, y]) for c in self.settings.rois])
                )
                in_roi = np.all((dist < roi_size // 2), axis=1)
                index = np.where(in_roi)[0]
                if len(index) > 0:
                    self.settings.rois.pop(index[0])
                    self.settings.selected = max(0, self.settings.selected - 1)
                else:
                    margin = roi_size // 2
                    if (margin < x < self.settings.camera__pix[0] - margin) & (
                        margin < y < self.settings.camera__pix[1] - margin
                    ):
                        self.settings.rois.append([x, y])
                        self.settings.selected = len(self.settings.rois) - 1

                self.settings._last_measured_file = None

        elif flags & cv2.EVENT_LBUTTONDOWN:

            dist = np.abs(
                np.asarray([c - np.asarray([x, y]) for c in self.settings.rois])
            )
            try:
                in_roi = np.all((dist < roi_size // 2), axis=1)
                index = np.where(in_roi)[0]
                if len(index) > 0:
                    self.settings.selected = index[0]
            except IndexError:
                pass  # no rois defined

        if event == cv2.EVENT_MOUSEWHEEL:
            zoom = 1.25
            if flags < 0:
                self.settings.fov_size__pix = int(self.settings.fov_size__pix * zoom)
            else:
                self.settings.fov_size__pix = max(
                    int(self.settings.fov_size__pix / zoom), 16
                )

        self.main_root.focus_force()

    def save_frames_to_binary_file(self, filename: str):
        filename = Path(filename).with_suffix(".bin")
        with open(filename, "wb") as f:
            for frame in self.selected_roi_frames:
                frame.tofile(f)

        print(f"Saved {len(self.selected_roi_frames)} frames to {filename}")


def load_bin_file(filename: str):
    data = hdf_data(Path(filename).with_suffix(".hdf"))
    shape = [data.traces.shape[0]] + [data.settings["roi_size (pix)"]] * 2

    with open(Path(filename).with_suffix(".bin"), "rb") as f:
        frames = np.fromfile(f, dtype=np.uint8, count=np.prod(shape)).reshape(shape)
    return np.asarray(frames)


if __name__ == "__main__":
    # print("This is a module, not a standalone script.")
    filename = r"d:\users\noort\data\20241211\data_153.hdf"
    frames = load_bin_file(filename)

    ac = [find_center(frame) for frame in frames]

    show_frames(ac)

    center = [np.argmax(a) for a in ac]

    plt.imshow(find_center(frames[10]))
    plt.colorbar()
    plt.show()
