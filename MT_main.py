import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import threading
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from AlliedVision import CameraApplication
from TraceIO import increment_filename, create_hdf
from MT_steppers import StepperApplication, to_gcode
from MT_settings import SettingsEditor

SHIFT_KEY = 0x0001
CTRL_KEY = 0x0004


class MainApp:
    def __init__(self, root):
        self.initialize_settings()

        self.root = root
        self.root.geometry("800x400+0+0")
        self.root.title("Magnetic Tweezers")
        self.root.iconbitmap("MagnetIcon.ico")

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
        self.fig, self.ax = plt.subplots()
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

        self.settings["frames"] = 0

        self.settings["_filename"] = increment_filename()
        self.settings["_aquisition mode"] = "idle"
        self.settings["_trajectory"] = None
        self.settings["_traces"] = None

    def start_camera(self):
        self.camera_app = CameraApplication(settings=self.settings)
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
        range = 0.04
        gcode = [
            "G91",
            "G93 S0.1",
            f"G1 Z{range:.3f} F0.5",
            "G4 S0.25",
            "G93",
            f"G1 Z-{range:.3f} F10",
            "G90",
        ]
        self.settings["_aquisition mode"] = "calibrate"
        self.settings["_filename"] = increment_filename()
        self.stepper_app.command_queue.put(gcode)

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
            self.settings["_aquisition mode"] = "measure"
            self.settings["_filename"] = increment_filename()
            self.stepper_app.command_queue.put(to_gcode(self.settings["_trajectory"]))
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
        create_hdf(self.settings["_filename"])

    def update_plot(self):
        stepper_df = self.stepper_app.get_dataframe()
        if not stepper_df.empty:
            label = "Focus (mm)"
            self.ax.clear()
            stepper_df[label].plot(ax=self.ax)
            self.ax.set_ylabel(label)
            self.fig.tight_layout()
            self.canvas.draw()

            if self.settings["_aquisition mode"] == "idle":
                create_hdf(self.settings, stepper_df, traces=self.settings["_traces"])
                self.stepper_app.clear_dataframe()

        self.root.after(500, self.update_plot)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
