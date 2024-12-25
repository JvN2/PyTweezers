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

# try:
# from AlliedVision import CameraApplication
# except ImportError:
from DummyCamera2 import CameraApplication

SHIFT_KEY = 0x0001
CTRL_KEY = 0x0004


class MainApp:
    def __init__(self, root):
        self.initialize_settings()

        self.root = root
        self.root.geometry("800x600+0+0")
        self.root.title("Magnetic Tweezers")
        self.root.iconbitmap("MagnetIcon.ico")

        self.data_queue = queue.Queue()
        self.data_list = deque()

        self.stepper_app = StepperApplication(port="COM5")
        self.stepper_app.start()
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

        self.settings["frames"] = 0

        self.settings["_filename"] = increment_filename()
        self.settings["_aquisition mode"] = "idle"
        self.settings["_trajectory"] = None
        self.settings["_traces"] = None
        self.settings["_logging"] = False

    def start_camera(self):
        self.camera_app = CameraApplication(
            settings=self.settings, root=self.root, data_queue=self.data_queue
        )
        threading.Thread(target=self.camera_app.run).start()

    def stop_camera(self):
        self.camera_app.stop()

    def exit_application(self):
        try:
            self.stepper_app.stop()
            self.camera_app.stop()
        except:
            pass
        self.root.quit()

    def change_settings(self):
        pass

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

        self.settings["_profile"] = to_profile(gcode)
        self.axes[0].clear()
        self.settings["_profile"].plot(
            ax=self.axes[0], linestyle="--", legend=False, color="lightgray"
        )

        self.axes[0].set_ylabel("Focus (mm)")
        self.axes[0].set_xlabel("Time (s)")
        self.settings["_trajectory"] = {"axis": "Focus (mm)"}
        self.fig.tight_layout()
        self.canvas.draw()

        self.settings["_aquisition mode"] = "calibrate"
        self.settings["_filename"] = increment_filename()
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

    def update_plot(self):
        stepper_df = self.stepper_app.get_dataframe()
        self.settings["_logging"] = self.stepper_app.get_logging_status()

        if not stepper_df.empty:
            self.axes[0].clear()

            label = self.settings["_trajectory"]["axis"]
            self.settings["_profile"].plot(
                ax=self.axes[0], linestyle="--", legend=False, color="lightgray"
            )
            stepper_df[label].plot(ax=self.axes[0], legend=False, color="blue")
            self.axes[0].set_ylabel(label)
            self.fig.tight_layout()
            self.canvas.draw()

            while not self.data_queue.empty():
                data = self.data_queue.get()
                self.data_list.append(data)

                tmp = np.asarray(self.data_list)
                # Update the plot data
                self.axes[1].clear()
                self.axes[1].plot(
                    tmp[:, 0] - tmp[0, 0], tmp[:, 1], "o", color="red", alpha=0.3
                )
                ic(self.settings["_aquisition mode"], self.settings["_logging"])

                #### to be fixed
                if (
                    self.settings["_aquisition mode"] == "done processing"
                    and not self.settings["_logging"]
                ):
                    create_hdf(self.settings, stepper_df)
                    ic(self.settings["_aquisition mode"], self.settings["_filename"])
                    self.stepper_app.clear_dataframe()
                    self.data_list.clear()
                    self.settings["_aquisition mode"] = "idle"

        self.root.after(500, self.update_plot)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
