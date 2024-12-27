import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import threading
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from TraceIO import increment_filename, create_hdf
from MT_steppers import StepperApplication, to_gcode, to_profile
from MT_settings import SettingsEditor
from time import sleep
from collections import deque
import queue
import pandas as pd

# try:
from AlliedVision import CameraApplication

# except ImportError:
# from DummyCamera2 import CameraApplication

SHIFT_KEY = 0x0001
CTRL_KEY = 0x0004
SENTINEL = None


class MainApp:
    def __init__(self, root):
        self.initialize_settings()

        self.root = root
        self.root.geometry("1000x600+0+0")
        self.root.title("Magnetic Tweezers")
        self.root.iconbitmap("MagnetIcon.ico")

        self.stepper_queue = queue.Queue()
        self.stepper_data = deque()
        self.stepper_done = True
        self.stepper_app = StepperApplication(
            port="COM5", data_queue=self.stepper_queue
        )
        self.stepper_app.start()

        self.tracker_queue = queue.Queue()
        self.tracker_data = deque()
        self.tracker_done = True
        self.start_camera()

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
        measure_menu.add_command(label="Calibrate LUT", command=self.calibrate_lut)
        measure_menu.add_command(label="Trajectory", command=self.create_trajectory)
        measure_menu.add_command(label="Go", command=self.go_trajectory)
        menubar.add_cascade(label="Measure", menu=measure_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Test", command=self.test)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.bind("<Alt-Up>", self.handle_keyboard)
        self.root.bind("<Alt-Down>", self.handle_keyboard)
        self.root.bind("<Alt-Left>", self.handle_keyboard)
        self.root.bind("<Alt-Right>", self.handle_keyboard)
        self.root.bind("<Alt-Prior>", self.handle_keyboard)
        self.root.bind("<Alt-Next>", self.handle_keyboard)

        # Create a matplotlib figure and axis
        self.fig, self.axes = plt.subplots(2, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Start the update loop
        self.update_plot()

    def initialize_settings(self):
        self.settings = {}
        self.settings["roi_size (pix)"] = 100
        self.settings["rois"] = []  # [(50, 50), (100, 200), (198, 150)]
        self.settings["selected"] = 0

        self.settings["window (pix)"] = 800
        self.settings["camera (pix)"] = np.asarray((1024, 1024))
        self.settings["fov_size (pix)"] = min(self.settings["camera (pix)"])
        self.settings["fov_center (pix)"] = self.settings["camera (pix)"] // 2
        self.settings["pixel_size (um)"] = 0.71
        self.settings["frame rate (Hz)"] = 20

        self.settings["frames"] = 0

        self.settings["_filename"] = increment_filename()
        self.settings["_aquisition mode"] = "idle"

    def start_camera(self):
        self.camera_app = CameraApplication(
            settings=self.settings, root=self.root, data_queue=self.tracker_queue
        )
        threading.Thread(target=self.camera_app.run).start()

    def stop_camera(self):
        self.camera_app.stop()
        ic("Camera stopped")

    def exit_application(self):
        try:
            self.stepper_app.stop()
            self.camera_app.stop()
        except:
            pass
        self.root.quit()

    def change_settings(self):
        pass

    def handle_keyboard(self, event):
        step_size = 0.05  # mm
        if event.state & SHIFT_KEY:
            step_size *= 10
        if event.state & CTRL_KEY:
            step_size *= 0.1

        # adjust direction depending on camera orientation
        xdir = 1
        ydir = -1

        gcode = ["G91"]
        if event.keysym == "Up":
            gcode.append(f"G1 Y{ydir*step_size:.3f} F1000")
        elif event.keysym == "Down":
            gcode.append(f"G1 Y{-ydir*step_size:.3f} F1000")
        elif event.keysym == "Left":
            gcode.append(f"G1 X{-xdir*step_size:.3f} F1000")
        elif event.keysym == "Right":
            gcode.append(f"G1 X{xdir*step_size:.3f} F1000")
        elif event.keysym == "Prior":
            gcode.append(f"G1 Z{step_size:.3f} F1000")
        elif event.keysym == "Next":
            gcode.append(f"G1 Z{-step_size:.3f} F1000")
        gcode.append("G90")
        gcode.append("M400")
        gcode.append("G93 N0")

        self.stepper_app.command_queue.put(gcode)

    def show_about(self):
        messagebox.showinfo(
            "About", "Camera Control Application v0.1\n(c) 2024 by John van Noort"
        )

    def test(self):
        print("Test")
        # df = self.stepper_app.get_dataframe()
        # ic(df)
        # plt.plot(df["Z"])
        # plt.show()
        # create_hdf(self.settings["_filename"])
        print(self.stepper_app.get_current_position())

    def init_plot(self, profile=None):
        self.tracker_data = deque()
        self.stepper_data = deque()

        self.axes[0].clear()
        self.axes[1].clear()

        self.axes[1].set_ylabel("Z (um)")
        self.axes[1].set_xlabel("Time (s)")
        self.axes[1].set_ylim(-10, 10)

        if profile is not None:
            profile.plot(
                ax=self.axes[0], linestyle="--", color="lightgray", legend=False
            )

        (self.ax0_line,) = self.axes[0].plot([], [], "o", color="blue", alpha=0.3)
        (self.ax1_line,) = self.axes[1].plot([], [], "o", color="red", alpha=0.3)

        self.axes[0].set_ylabel("Focus (mm)")
        self.axes[0].set_xlabel("Time (s)")

        if profile is not None:
            profile.plot(
                ax=self.axes[0], linestyle="--", color="lightgray", legend=False
            )
            self.axes[0].set_xlim(0, profile.index[-1])
            self.axes[1].set_xlim(0, profile.index[-1])

        self.fig.tight_layout()
        self.canvas.draw()

    def update_plot(self):

        while not self.stepper_queue.empty():
            data = self.stepper_queue.get()
            if data is SENTINEL:
                self.stepper_done = True
            else:
                self.stepper_data.append(data)

        if len(self.stepper_data) > 1:
            time_index = self.stepper_app.axes.index("Time (s)")
            focus_index = self.stepper_app.axes.index("Focus (mm)")
            t = [data[time_index] for data in self.stepper_data]
            z = [data[focus_index] for data in self.stepper_data]

            self.ax0_line.set_data(t, z)
            self.axes[0].set_xlabel("Time (s)")  # Ensure the x-axis label is set
            self.canvas.draw()

        while not self.tracker_queue.empty():
            data = self.tracker_queue.get()
            if data is SENTINEL:
                self.tracker_done = True
            else:
                self.tracker_data.append(data)

        if len(self.tracker_data) > 1:
            time_index = 0
            position_index = 1
            t = [data[time_index] for data in self.tracker_data]
            t = (np.asarray(t) - t[0]) / self.settings["frame rate (Hz)"]
            z = [data[position_index] for data in self.tracker_data]

            self.ax1_line.set_data(t, z)
            self.axes[1].set_xlabel("Time (s)")  # Ensure the x-axis label is set
            self.axes[1].relim()
            self.axes[1].autoscale_view()
            self.canvas.draw()

        if self.stepper_done and self.tracker_done and self.stepper_data:
            # Save data to file
            self.settings["_aquisition mode"] == "idle"
            stepper_df = pd.DataFrame(self.stepper_data, columns=self.stepper_app.axes)

            cols = ["Frame"]
            for i, _ in enumerate(self.settings["rois"]):
                cols.append(f"A{i} (a.u.)")
                cols.append(f"X{i} (pix)")
                cols.append(f"Y{i} (pix)")
                cols.append(f"Z{i} (um)")
            tracker_df = pd.DataFrame(self.tracker_data, columns=cols)

            print(f"Data saved in {create_hdf(self.settings, stepper_df, tracker_df)}")

            self.stepper_data, self.tracker_data = [], []

        self.root.after(100, self.update_plot)

    def calibrate_lut(self):
        if len(self.settings["rois"]) == 0:
            messagebox.showinfo("Error", "No ROIs defined")
            return

        range = 0.06
        current_focus = self.stepper_app.get_current_position()[2]
        gcode = [
            f"G1 Z{current_focus:.3f} F10",
            "G93 S0.1",
            f"G1 Z{current_focus + range:.3f} F1",
            "G4 S0.05",
            "G93",
            f"G1 Z{current_focus:.3f} F10",
            f"M400",
        ]

        self.init_plot(to_profile(gcode))
        self.settings["_aquisition mode"] = "calibrate"
        self.settings["_filename"] = increment_filename()
        self.stepper_done, self.tracker_done = False, False
        self.stepper_app.command_queue.put(gcode + ["G93 N0"])

    def create_trajectory(self):
        settings = {
            "axis": [
                "X (mm)",
                "X (mm)",
                "Y (mm)",
                "Focus (mm)",
                "Shift (mm)",
                "Rotation (turns)",
            ],
            "relative": ["True", "False", "True"],
            "start": [0, -10, 10, 0.1, "linear"],
            "target": [0.02, -10, 10, 0.001, "linear"],
            "wait (s)": [1, 0, 10, 0.1, "linear"],
            "move (s)": [3, 0, 10, 0.1, "linear"],
            "dwell (s)": [-6, -100, 100, 0.1, "linear"],
            "repeat": [1, 1, 4, 1, "linear"],
        }
        if self.settings["_trajectory"]:
            for key, value in self.settings["_trajectory"].items():
                settings[key][0] = value

        settings_editor = SettingsEditor(self.root, settings, "Create trajectory ...")
        self.root.wait_window(settings_editor)

        if settings_editor.settings:
            self.settings["_trajectory"] = settings_editor.settings

            gcode = to_gcode(
                self.settings["_trajectory"],
                self.stepper_app.get_current_position(),
            )
            self.settings["_profile"] = to_profile(gcode)

            self.axes[0].clear()
            self.settings["_profile"].plot(
                ax=self.axes, linestyle="--", legend=False, color="lightgray"
            )

            self.axes.set_ylabel(self.settings["_trajectory"]["axis"])
            self.axes.set_xlabel("Time (s)")
            self.fig.tight_layout()
            self.canvas.draw()

    def go_trajectory(self):
        if self.settings["_trajectory"]:
            gcode = to_gcode(
                self.settings["_trajectory"],
                self.stepper_app.get_current_position(),
            )
            self.settings["_profile"] = to_profile(gcode)
            self.axes[0].clear()

            self.settings["_aquisition mode"] = "measure"
            self.settings["_filename"] = increment_filename()
            self.stepper_app.command_queue.put(
                to_gcode(
                    self.settings["_trajectory"],
                    self.stepper_app.get_current_position(),
                )
            )
        else:
            messagebox.showinfo("Error", "No trajectory defined")


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
