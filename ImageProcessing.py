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
                # cv_images = self._get_zoom(cv_images)
                cv2.imshow(IMAGE_CAPTION, cv_images)
                cv2.namedWindow(IMAGE_CAPTION)
                cv2.setMouseCallback(IMAGE_CAPTION, self._mouse_callback)

            cv2.waitKey(10)
            if self.quit:
                cv2.destroyAllWindows()
                alive = False

    def stop(self):
        self.quit = True

    def _update_fov(self, center=None, width=None):
        if width is None:
            width = min(self.settings["fov (pix)"][1] - self.settings["fov (pix)"][0])
        half_width = min(width, *self.settings["camera (pix)"]) // 2
        if center is None:
            center = np.sum(self.settings["fov (pix)"], axis=1) // 2

        # Calculate initial FOV coordinates
        fov = np.array([center - half_width, center + half_width])

        # Adjust FOV coordinates if they exceed the camera boundaries
        if fov[0, 0] < 0:
            fov[:, 0] -= fov[0, 0]
        if fov[0, 1] < 0:
            fov[:, 1] -= fov[0, 1]
        if fov[1, 0] > self.settings["camera (pix)"][0]:
            fov[:, 0] -= fov[1, 0] - self.settings["camera (pix)"][0]
        if fov[1, 1] > self.settings["camera (pix)"][1]:
            fov[:, 1] -= fov[1, 1] - self.settings["camera (pix)"][1]

        # Ensure FOV is within camera boundaries
        self.settings["fov (pix)"] = np.clip(fov, 0, self.settings["camera (pix)"])
        return

    def _get_zoom(self, cv_frame: np.ndarray) -> np.ndarray:
        x0 = self.settings.get("fov (pix)", 0)[0][0]
        x1 = self.settings.get("fov (pix)", self.settings["camera (pix)"])[1][0]
        y0 = self.settings.get("fov (pix)", 0)[0][1]
        y1 = self.settings.get("fov (pix)", self.settings["camera (pix)"])[1][1]
        cv_frame = cv_frame[y0:y1, :, x0:x1, :]
        return cv_frame

    def _mouse_callback(self, event, x, y, flags, param):
        x, y = (
            int(
                x / self.settings.get("zoom (pix)", 1)
                + self.settings.get("offset (pix)", [0, 0])[0]
            ),
            int(
                y / self.settings.get("zoom (pix)", 1)
                + self.settings.get("offset (pix)", [0, 0])[1]
            ),
        )
        roi_size = self.settings["roi_size (pix)"]

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
                    print(x,y)
                    if (margin < x < self.settings["camera (pix)"][0] - margin) & (
                        margin < y < self.settings["camera (pix)"][1] - margin
                    ):
                        self.settings["rois"].append([x, y])
                        self.settings["selected"] = len(self.settings["rois"]) - 1


        elif flags & cv2.EVENT_LBUTTONDOWN:
            # select roi
            dist = np.abs(
                np.asarray([c - np.asarray([x, y]) for c in self.settings["rois"]])
            )
            in_roi = np.all((dist < roi_size // 2), axis=1)
            index = np.where(in_roi)[0]
            if len(index) > 0:
                self.settings["selected"] = index[0]

        if event == cv2.EVENT_MOUSEWHEEL:
            # zoom in or out
            if flags > 0:
                roi_size *= 1.3
            else:
                roi_size /= 1.3
            self._update_fov(width=roi_size)


if __name__ == "__main__":
    print("This is a module, not a standalone script.")
