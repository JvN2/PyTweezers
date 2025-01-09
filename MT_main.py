import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from TraceIO import increment_filename, create_hdf, hdf_data
from MT_steppers import StepperApplication, to_gcode, to_profile, to_axis
from MT_settings import SettingsEditor
from MT_Tracker import Tracker
from time import sleep
from collections import deque
import queue
import pandas as pd

from AlliedVision import CameraApplication

# Constants
SHIFT_KEY = 0x0001
CTRL_KEY = 0x0004
SENTINEL = None
DEFAULT_PLOT_TITLE = " "
CHANNELS = ["X (um)", "Y (um)", "Z (um)", "A (a.u.)"]


class Settings:
    def __init__(self):
        self.roi_size__pix = 100
        self.rois = []  # [(50, 50), (100, 200), (198, 150)]
        self.selected = 0
        self.window__pix = 800
        self.camera__pix = np.asarray((1024, 1024))
        self.fov_size__pix = min(self.camera__pix)
        self.fov_center__pix = self.camera__pix // 2
        self.pixel_size__um = 0.71
        self.exposure_time__us = 5000

        self._trajectory = []
        self._filename = increment_filename()
        self._aquisition_mode = "idle"
        self._last_measured_file = None
        self._tracker = Tracker(r"data\data_153.hdf")

        self._plot_range = 10
        self._plot_offset = 0
        self._plot_channel = "Z (um)"

    def to_dict(self):
        old_dict = self.__dict__.copy()
        dict = {}
        for key, value in old_dict.items():
            if key[0] != "_":
                if "__" in key:
                    key = key.replace("__", " (") + ")"
                    dict[key] = value

        return dict


def plot_adjust_y(plot_range, plot_offset, axis):
    if plot_range > 0:
        yrange = plot_offset + np.asarray([-plot_range, plot_range]) / 2
        axis.set_ylim(yrange)


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x600+0+0")
        self.root.title("Magnetic Tweezers")
        self.root.iconbitmap("MagnetIcon.ico")

        self.settings = Settings()

        self.stepper_queue = queue.Queue()
        self.stepper_data = deque()
        self.stepper_done = True
        self.stepper_app = StepperApplication(
            port="COM3", data_queue=self.stepper_queue
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
        measure_menu.add_command(label="Go", command=self.go_trajectory)
        measure_menu.add_command(
            label="Change trajectory...", command=self.create_trajectory
        )

        measure_menu.add_separator()
        lut_menu = tk.Menu(measure_menu, tearoff=0)
        lut_menu.add_command(label="New", command=self.calibrate_lut)
        lut_menu.add_command(label="From file ...", command=self.change_lut)
        measure_menu.add_cascade(label="Look Up Table", menu=lut_menu)

        measure_menu.add_separator()
        measure_menu.add_command(
            label="Change settings...", command=self.change_settings
        )
        measure_menu.add_separator()
        camera_menu = tk.Menu(measure_menu, tearoff=0)
        camera_menu.add_command(label="Start", command=self.start_camera)
        camera_menu.add_command(label="Stop", command=self.stop_camera)
        measure_menu.add_cascade(label="Camera", menu=camera_menu)
        menubar.add_cascade(label="Measure", menu=measure_menu)

        plot_menu = tk.Menu(menubar, tearoff=0)
        plot_menu.add_command(label="Adjust", command=self.adjust_plot_range)
        menubar.add_cascade(label="Plot", menu=plot_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Test", command=self.test)
        menubar.add_cascade(label="Help", menu=help_menu)

        for key in [
            "<Alt-Up>",
            "<Alt-Down>",
            "<Alt-Left>",
            "<Alt-Right>",
            "<Alt-Prior>",
            "<Alt-Next>",
            "<F11>",
            "<F12>",
        ]:
            self.root.bind(key, self.handle_keyboard)

        # Create a matplotlib figure and axis
        self.fig, self.plt_axes = plt.subplots(2, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Start the update loop
        self.update_plot()

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
        if event.keysym in ["Up", "Down", "Left", "Right", "Prior", "Next"]:
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

        elif event.keysym == "F11":
            self.settings.selected = np.max([0, self.settings.selected - 1])
        elif event.keysym == "F12":
            self.settings.selected = np.min(
                [len(self.settings.rois) - 1, self.settings.selected + 1]
            )

    def show_about(self):
        messagebox.showinfo(
            "About", "Camera Control Application v0.1\n(c) 2024 by John van Noort"
        )

    def test(self):
        print("Test function")
        # df = self.stepper_app.get_dataframe()
        # ic(df)
        # plt.plot(df["Z"])
        # plt.show()
        # create_hdf(self.settings["_filename"])
        # print(self.stepper_app.get_current_position())

    def adjust_plot_range(self):

        plot_settings = {
            "channel": [self.settings._plot_channel] + CHANNELS,
            "range": [self.settings._plot_range, -1, 2, 0.2, "10log"],
            "offset": [self.settings._plot_offset, -100, 100, 0.1, "linear"],
        }

        plot_editor = SettingsEditor(
            self.root, plot_settings, "Modify plot ...", axis=self.plt_axes[1]
        )
        self.root.wait_window(plot_editor)

        if plot_editor.settings:
            self.settings._plot_range = plot_editor.settings["range"]
            self.settings._plot_offset = plot_editor.settings["offset"]
            self.settings._plot_channel = plot_editor.settings["channel"]

        self.plt_axes[1].set_ylabel(self.settings._plot_channel)
        plot_adjust_y(
            self.settings._plot_range, self.settings._plot_offset, self.plt_axes[1]
        )
        self.plt_axes[1].set_ylabel(self.settings._plot_channel)
        self.canvas.draw()

    def change_lut(self):
        if self.tracker_queue.empty():
            lut_filename = tk.filedialog.askopenfilename(
                title="Open LUT file",
                filetypes=[("Binary files", "*.bin")],
                initialfile=r"data\data_006.bin",
            )
            if lut_filename:
                self.stop_camera()
                sleep(0.5)
                self.settings._tracker = Tracker(lut_filename)
                self.start_camera()
                print(f"New LUT: {self.settings._tracker.filename}")
        else:
            messagebox.showinfo("Error", "Wait for acquisition to finish")

    def init_plot(self, profile=None):
        self.tracker_data = deque()
        self.stepper_data = deque()

        self.plt_axes[0].clear()
        self.plt_axes[1].clear()

        self.plt_axes[1].set_xlabel("Time (s)")
        self.plt_axes[1].set_ylabel(self.settings._plot_channel)

        if profile is not None:
            profile.plot(
                ax=self.plt_axes[0], linestyle="--", color="lightgray", legend=False
            )
            self.plt_axes[0].set_ylabel(profile.columns[0])

        (self.ax0_line,) = self.plt_axes[0].plot([], [], "o", color="blue", alpha=0.3)
        (self.ax1_line,) = self.plt_axes[1].plot([], [], "o", color="red", alpha=0.3)

        self.plt_axes[0].set_xlabel("Time (s)")
        plot_adjust_y(
            self.settings._plot_range, self.settings._plot_offset, self.plt_axes[1]
        )

        if profile is not None:
            profile.plot(
                ax=self.plt_axes[0], linestyle="--", color="lightgray", legend=False
            )
            self.plt_axes[0].set_xlim(0, profile.index[-1])
            self.plt_axes[1].set_xlim(0, profile.index[-1])

        self.plt_axes[0].set_title(DEFAULT_PLOT_TITLE)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_plot(self):
        # Update stepper log
        while not self.stepper_queue.empty():
            data = self.stepper_queue.get()
            if data is SENTINEL:
                self.stepper_done = True
            else:
                self.stepper_data.append(data)
                data = None

        if len(self.stepper_data) > 1:
            time_index = 0
            y_index, _ = to_axis(self.plt_axes[0].get_ylabel())
            t = [data[time_index] for data in self.stepper_data]
            plot_data = [data[y_index + 1] for data in self.stepper_data]

            self.ax0_line.set_data(t, plot_data)
            self.plt_axes[0].set_xlabel("Time (s)")  # Ensure the x-axis label is set
            self.plt_axes[0].set_title(self.settings._filename)
            self.canvas.draw()

        while not self.tracker_queue.empty():
            data = self.tracker_queue.get()
            if data is SENTINEL:
                self.tracker_done = True
            else:
                self.tracker_data.append(data)
                data = None

        if len(self.tracker_data) > 1:
            # get colum of z (um) of selected roi from tracker data
            column_index = CHANNELS.index(self.settings._plot_channel)
            index = self.settings.selected * 4 + column_index
            t = [data[0] for data in self.tracker_data]
            t = (np.asarray(t) - t[0]) * self.settings.exposure_time__us * 1e-6
            plot_data = [data[index] for data in self.tracker_data]
            self.ax1_line.set_data(t, plot_data)
            self.plt_axes[1].relim()
            self.plt_axes[1].autoscale_view()

        elif self.settings._last_measured_file:
            # Read data from hdf file
            ic(self.settings._plot_channel, self.settings._plot_range)
            data = hdf_data(self.settings._filename)
            if data.list_channels():
                data.read(self.settings._filename, label=str(self.settings.selected))
                t = data.traces["Time (s)"]
                plot_data = data.traces[self.settings._plot_channel]
                self.ax1_line.set_data(t, plot_data)
                self.plt_axes[1].relim()
                self.plt_axes[1].autoscale_view()
                self.plt_axes[0].set_title(self.settings._last_measured_file)
                self.canvas.draw()
        else:
            self.plt_axes[1].clear()
            self.plt_axes[0].set_title(DEFAULT_PLOT_TITLE)
            self.canvas.draw()

        if self.stepper_done and self.tracker_done and self.stepper_data:
            # Save data to file
            stepper_df = pd.DataFrame(
                self.stepper_data,
                columns=["Time (s)"] + list(self.stepper_app.axes.values()),
            )

            cols = ["Frame"]
            for i, _ in enumerate(self.settings.rois):
                for c in CHANNELS:
                    cols.append(f"{c[0]}{i}{c[1:]}")
            tracker_df = pd.DataFrame(self.tracker_data, columns=cols)

            print(f"Data saved in {create_hdf(self.settings, stepper_df, tracker_df)}")

            self.stepper_data.clear()
            self.tracker_data.clear()

            self.settings._aquisition_mode == "idle"

        self.root.after(100, self.update_plot)

    def calibrate_lut(self):
        if len(self.settings.rois) == 0:
            messagebox.showinfo("Error", "No ROIs defined")
            return

        self.settings.rois = [self.settings.rois[self.settings.selected]]
        self.settings.selected = 0

        range = 0.06
        current_focus = self.stepper_app.get_current_position()[2]
        gcode = [
            f"G1 Z{current_focus:.3f} F10",
            "G93 S0.1",
            f"G1 Z{current_focus + range:.3f} F0.1",
            "G4 S0.05",
            "G93",
            f"G1 Z{current_focus:.3f} F10",
            f"M400",
        ]

        self.go_trajectory(gcode, "calibrate")

    def create_trajectory(self):
        traject_settings = {
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
            "target": [0.02, -100, 100, 0.001, "linear"],
            "wait (s)": [1, 0, 10, 0.1, "linear"],
            "move (s)": [3, 0, 120, 0.1, "linear"],
            "dwell (s)": [0, -100, 100, 0.1, "linear"],
            "repeat": [1, 1, 4, 1, "linear"],
        }
        if self.settings._trajectory:
            for key, value in self.settings._trajectory.items():
                traject_settings[key][0] = value

        settings_editor = SettingsEditor(
            self.root, traject_settings, "Create trajectory ..."
        )
        self.root.wait_window(settings_editor)

        if settings_editor.settings:
            self.settings._trajectory = settings_editor.settings
            gcode = to_gcode(
                self.settings._trajectory,
                self.stepper_app.get_current_position(),
            )
            self.init_plot(to_profile(gcode))
            self.settings._settings_changed = True

    def go_trajectory(self, gcode=None, mode="measure"):
        if len(self.settings.rois) == 0:
            messagebox.showinfo("Error", "No ROIs defined")
            return
        else:
            if gcode is None:
                if self.settings._trajectory:
                    gcode = to_gcode(
                        self.settings._trajectory,
                        self.stepper_app.get_current_position(),
                    )
                else:
                    messagebox.showinfo("Error", "No trajectory defined")
                    return

            self.init_plot(to_profile(gcode))
            self.settings._aquisition_mode = mode
            self.settings._filename = increment_filename()
            self.stepper_done, self.tracker_done = False, False
            self.stepper_app.command_queue.put(gcode + ["G93 N0"])
            self.settings._last_measured_file = self.settings._filename


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
