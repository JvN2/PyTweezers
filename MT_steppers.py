# FILE: MT_steppers.py

import threading
import serial
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from icecream import ic
import queue


class StepperApplication(threading.Thread):
    def __init__(self, port, baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.running = False
        self.df = pd.DataFrame()
        self.stop_event = threading.Event()
        self.command_queue = queue.Queue()  # Queue for G-code commands

    def connect(self):
        self.serial_connection = serial.Serial(self.port, self.baudrate)
        time.sleep(2)  # Wait for the connection to establish

    def disconnect(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()

    def send_gcode(self, gcode):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.write((gcode + "\n").encode("utf-8"))
            time.sleep(0.05)  # Wait for the command to be processed

    def run(self):
        self.connect()
        self.running = True
        while self.running:
            self.read_response()
            try:
                gcodes = self.command_queue.get(timeout=1)
                if isinstance(gcodes, list):
                    for gcode in gcodes:
                        self.send_gcode(gcode)
                else:
                    self.send_gcode(gcodes)
            except queue.Empty:
                continue

    def read_response(self):
        while self.serial_connection and self.serial_connection.in_waiting > 0:
            response = self.serial_connection.readline().decode("utf-8").strip()
            if response[:4] == "log:":
                data = response[4:].split()
                try:
                    data = [float(x) for x in data]
                    self.df.loc[data[0]] = data[1:]
                except:
                    data = [
                        "Time (s)",
                        "X (mm)",
                        "Y (mm)",
                        "Focus (mm)",
                        "Shift (mm)",
                        "Rotation (turns)",
                    ]
                    self.df = pd.DataFrame(columns=data)
                    self.df.set_index(self.df.columns[0], inplace=True)

    def stop(self):
        self.running = False
        self.disconnect()

    def get_dataframe(self):
        return self.df.copy()


def update_plot(frame, stepper_app, lines):
    if not stepper_app.df.empty:
        for line, column in zip(lines, stepper_app.df.columns):
            line.set_data(stepper_app.df.index, stepper_app.df[column])
        plt.gca().relim()
        plt.gca().autoscale_view()
    return lines


def on_close(event, stepper_app):
    stepper_app.stop()
    plt.close("all")
    print("Plot closed and application stopped.")


def convert_to_trace(gcodes, a=8, axes=["X", "Y", "Z", "A", "B"]):
    df = pd.DataFrame(columns=["t", "dt"] + axes + [f"v_{axis}" for axis in axes])
    df.loc[0] = np.zeros(len(df.columns))
    print(df)

    for line in gcodes:
        gcode = line.split()
        if gcode[0] == "G1":
            df.loc[len(df), gcode[1][0]] = float(gcode[1][1:])
            df.loc[len(df) - 1, "v_" + gcode[1][0]] = float(gcode[2][1:])
        elif gcode[0] == "G4":
            df.loc[len(df), "dt"] = float(gcode[1][1:])
    df.loc[len(df), "dt"] = 0

    # Replace NaN values in columns that start with 'v_' with 0
    df.loc[:, df.columns.str.startswith("v_")] = df.loc[
        :, df.columns.str.startswith("v_")
    ].fillna(0)

    for row in range(1, len(df)):
        if np.isfinite(df.loc[row, "dt"]):
            df.loc[row, "t"] = df.loc[row - 1, "t"] + df.loc[row, "dt"]
            df.loc[row, axes] = df.loc[row - 1, axes]
        # else

    print(df)
    return df

    df[0] = current_position.values()
    current_time = 0

    for gcode in gcodes:
        parts = gcode.split()
        command = parts[0]
        if command in ["G0", "G1"]:
            for part in parts[1:]:
                if part[0] in axes:
                    current_position[part[0]] = float(part[1:])
            df.loc[current_time] = [current_position[axis] for axis in axes]
        elif command == "G4":
            for part in parts[1:]:
                if part[0] == "S":
                    current_time += float(part[1:])
                    df.loc[current_time] = [current_position[axis] for axis in axes]

    print(df)
    return df


def convert_to_section(t0, x0, v0, xe, ve, v, a, time, position):
    dt1 = (v - v0) / a
    dx1 = v0 * dt1 + 0.5 * a * dt1**2

    dt3 = (ve - v) / a
    dx3 = ve * dt3 - 0.5 * a * dt3**2

    dx2 = xe - x0 - dx1 - dx3
    dt2 = dx2 / v

    # while

    return


def to_gcode(trajectory):

    gcode = []
    axis = trajectory["axis"][0]
    velocity = (
        np.abs(trajectory["target"] - trajectory["start"]) / trajectory["move (s)"]
    )

    gcode.append(f"G1 {axis}{trajectory['start']} F1000")
    gcode.append(f"G93 S0.1")
    for i in range(trajectory["repeat"]):
        gcode.append(f"G4 S{trajectory['wait (s)']}")
        gcode.append(f"G1 {axis}{trajectory['target']} F{velocity*60}")
        gcode.append(f"G4 S{trajectory['dwell (s)']}")
        gcode.append(f"G1 {axis}{trajectory['start']} F{velocity*60}")
        gcode.append(f"G4 S{trajectory['wait (s)']}")
    gcode.append(f"G93")

    return gcode


if __name__ == "__main__":

    # print(convert_to_section(0, 0, 10, 0, 1, 8, 0.1))

    stepper_app = StepperApplication(port="COM5")
    stepper_app.start()

    trajectory = {
        "axis": "Z (mm)",
        "start": 0,
        "target": 0.1,
        "wait (s)": 1.0,
        "move (s)": 3.0,
        "dwell (s)": 0.0,
        "repeat": 1,
    }

    gcode = to_gcode(trajectory)
    # convert_to_trace(gcode)

    if True:

        # Start the stepper application with the G-code commands
        stepper_app.command_queue.put(gcode)

        # Set up the plot
        fig, ax = plt.subplots()
        lines = [
            ax.plot([], [], label=col)[0] for col in ["X", "Y", "Z", "A", "B"]
        ]  # Adjust columns as needed
        ax.legend()
        ani = FuncAnimation(
            fig,
            update_plot,
            fargs=(stepper_app, lines),
            interval=100,
            cache_frame_data=False,
        )

        # Connect the close event to the on_close function
        fig.canvas.mpl_connect(
            "close_event", lambda event: on_close(event, stepper_app)
        )

        plt.show()
