import numpy as np
import threading
import time


class CameraApplication:
    def __init__(self, settings, root, data_queue):
        self.settings = settings
        self.root = root
        self.data_queue = data_queue
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._produce_frames)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        self.start()

    def _produce_frames(self):
        frame_nr = 0
        while self.running:
            frame = np.random.randint(
                0,
                256,
                (self.settings["camera (pix)"][0], self.settings["camera (pix)"][1]),
                dtype=np.uint8,
            )
            self.data_queue.put((frame_nr, frame))
            frame_nr += 1
            time.sleep(0.1)  # Simulate frame rate


# Example usage
if __name__ == "__main__":
    import queue
    import tkinter as tk

    settings = {"camera (pix)": np.asarray((1024, 1024))}
    root = tk.Tk()
    data_queue = queue.Queue()
    camera_app = CameraApplication(settings, root, data_queue)
    camera_app.start()

    def update():
        while not data_queue.empty():
            frame_nr, frame = data_queue.get()
            print(f"Frame {frame_nr}: {frame.shape}")
        root.after(100, update)

    root.after(100, update)
    root.mainloop()
    camera_app.stop()
