import tkinter as tk
from tkinter import messagebox
import threading
from time import sleep
import numpy as np
import os
from datetime import datetime
from icecream import ic
from pathlib import Path

from AlliedVision import CameraApplication
import DummyCamera
from MT_settings import SettingsEditor
from MT_steppers import to_gcode, StepperApplication
import TraceIO


def increment_filename(old_filename=None, base_dir=None):
    if old_filename:
        current_file_nr = int(old_filename[-7:-4]) + 1
    else:
        current_file_nr = 0

    if not current_file_nr:
        current_file_nr = 0

    if not base_dir:
        date = datetime.now().strftime("%Y%m%d")
        base_dir = Path(f"d:/users/{os.getlogin()}/data/{date}")

    base_dir.mkdir(parents=True, exist_ok=True)

    filename = base_dir / f"data_{current_file_nr:03d}.bin"

    # check if file exists, otherwise increase number
    while filename.exists():
        current_file_nr += 1
        filename = base_dir / f"data_{current_file_nr:03d}.bin"

    return filename


class MainApp:
    def __init__(self, root):

        self.initialize_settings()

        self.root = root
        self.root.geometry("400x200")
        self.root.title("Magnetic Tweezers")
        self.root.iconbitmap("MagnetIcon.ico")

        self.dummy_camera = DummyCamera.CameraApplication(dummy=True)
        self.stepper_app = StepperApplication(port="COM5")

        menubar = tk.Menu(root)
        root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.exit_application)
        menubar.add_cascade(label="File", menu=file_menu)

        measure_menu = tk.Menu(menubar, tearoff=0)
        measure_menu.add_command(label="Change settings", command=self.change_settings)
        measure_menu.add_separator()
        measure_menu.add_command(label="Stop", command=self.stop_camera)
        measure_menu.add_command(label="Start", command=self.start_camera)
        measure_menu.add_separator()
        measure_menu.add_command(label="Trajectory", command=self.create_trajectory)
        measure_menu.add_command(label="Go", command=self.go_trajectory)
        menubar.add_cascade(label="Measure", menu=measure_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Test", command=self.test)
        menubar.add_cascade(label="Help", menu=help_menu)

    def initialize_settings(self):
        self.settings = {}
        self.settings["roi_size (pix)"] = 50
        self.settings["rois"] = [(50, 50), (100, 200), (198, 150)]
        self.settings["selected"] = 0

        self.settings["window (pix)"] = 800
        self.settings["camera (pix)"] = np.asarray((1024, 1024))
        self.settings["fov_size (pix)"] = min(self.settings["camera (pix)"])
        self.settings["fov_center (pix)"] = self.settings["camera (pix)"] // 2

        self.settings["frames"] = 0

        self.settings["_filename"] = increment_filename()
        self.settings["_aquisition mode"] = "calibrate"
        self.settings["_trajectory"] = None

    def start_camera(self):
        self.camera = CameraApplication(settings=self.settings)
        threading.Thread(target=self.camera.run).start()

    def stop_camera(self):
        self.camera.stop()

    def exit_application(self):
        try:
            self.stop_camera()
        except AttributeError:
            pass
        self.root.quit()

    def change_settings(self):
        settings = {
            "roi_size (pix)": (self.settings["roi_size (pix)"], 2, 8, 1, "2log"),
            "frames": (self.settings["frames"], 0, 4, 1, "10log"),
            # "File": (Path("c:/tmp/image.bin"), "*.bin", "*.hdf"),
        }

        settings_editor = SettingsEditor(self.root, settings, "Adjust settings ...")
        self.root.wait_window(settings_editor)

        if settings_editor.settings:
            for key, value in settings_editor.settings.items():
                self.settings[key] = value

    def show_about(self):
        messagebox.showinfo(
            "About", "Camera Control Application v0.1\n(c) 2024 by John van Noort"
        )

    def test(self):
        print("Test")
        data = TraceIO.hdf_data(filename=self.settings["_filename"].with_suffix(".hdf"))
        data.settings = self.settings
        data.save(settings=True)
        ic(data.filename)

    def create_trajectory(self):
        settings = {
            "axis": ["Z (mm)", "X (mm)", "Y (mm)", "Z (mm)"],
            "start": [0, 0, 10, 0.1, "linear"],
            "target": [0.1, 0, 10, 0.001, "linear"],
            "wait (s)": [1, 0, 10, 0.1, "linear"],
            "move (s)": [5, 0, 10, 0.1, "linear"],
            "dwell (s)": [0, 0, 10, 0.1, "linear"],
            "repeat": [1, 1, 4, 1, "linear"],
        }
        if self.settings["_trajectory"]:
            for key, value in self.settings["_trajectory"].items():
                settings[key][0] = value

        settings_editor = SettingsEditor(self.root, settings, "Create trajectory ...")
        self.root.wait_window(settings_editor)
        if settings_editor.settings:
            self.settings["_trajectory"] = settings_editor.settings

    def go_trajectory(self):
        if self.settings["_trajectory"]:
            gcode = to_gcode(self.settings["_trajectory"])
            self.stepper_app.stop()  # Stop the stepper application before starting a new connection
            threading.Thread(target=self.stepper_app.run, args=(gcode,)).start()
        else:
            messagebox.showinfo("Error", "No trajectory defined")


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
