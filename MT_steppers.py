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

###### to do: insert the case that logging starts later than the first line of the gcode


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


def to_section(x0, xe, v=30, t0=0, v0=0, ve=0, dt=0.01, a=5, vmax=3, axis="X"):

    v = min(v, vmax)
    v0 = min(v0, vmax)
    ve = min(ve, vmax)

    if np.sign(v0) != np.sign(ve):
        ve = 0
    v = np.sign(xe - x0) * abs(v)

    dt1 = abs((v - v0) / a)
    dx1 = v0 * dt1 + np.sign(v - v0) * 0.5 * a * dt1**2

    dt3 = abs((ve - v) / a)
    dx3 = ve * dt3 - np.sign(ve - v) * 0.5 * a * dt3**2

    dx2 = (xe - x0) - dx1 - dx3
    dt2 = abs(dx2 / v)

    if np.sign(dx2) != np.sign(xe - x0):
        # not enough time to get to full speed
        v = np.sqrt(a * np.abs(xe - x0) + 0.5 * (v0**2 + ve**2)) * np.sign(xe - x0)
        dt1 = abs((v - v0) / a)
        dx1 = v0 * dt1 + np.sign(v - v0) * 0.5 * a * dt1**2

        dt3 = abs((v - ve) / a)
        dx3 = v * dt3 - np.sign(ve - v) * 0.5 * a * dt3**2

        dx2 = 0
        dt2 = 0

    t1 = np.linspace(0, dt1, int(dt1 / dt) + 1)
    x1 = x0 + v0 * t1 + np.sign(v - v0) * 0.5 * a * t1**2
    t1 = t1 + t0
    v1 = np.diff(x1) / np.diff(t1)
    v1 = np.append(v0, v1)

    t3 = np.linspace(0, dt3, int(dt3 / dt) + 1)
    x3 = x0 + dx1 + dx2 + v * t3 + np.sign(ve - v) * 0.5 * a * t3**2
    t3 = t3 + t0 + dt1 + dt2
    v3 = np.diff(x3) / np.diff(t3)
    v3 = np.append(v3, ve)

    t = np.append(t1[1:], t3)
    x = np.append(x1[1:], x3)
    v = np.append(v1[1:], v3)

    df = pd.DataFrame(np.asarray([t, x, v]).T, columns=["t", axis, f"v_{axis}"])
    return df


def to_profile(gcodes, a=8, axes=["X", "Y", "Z", "A", "B"], start_position=np.zeros(5)):

    dt = None
    logging = False
    relative_move = False

    for line in gcodes:
        gcode = line.upper().split()
        if gcode[0] == "G93":
            if len(gcode) == 2:
                df = pd.DataFrame(
                    [[0] + list(start_position) + [0] * len(axes)],
                    columns=["t"] + axes + [f"v_{axis}" for axis in axes],
                )
                dt = float(gcode[1][1:])
                logging = True
            else:
                logging = False

        elif gcode[0] == "G1":

            axis = gcode[1][0]
            end_position = float(gcode[1][1:])
            start_position = df.iloc[-1][axis]

            if relative_move:
                start_position = df.iloc[-1][axis]
                end_position += start_position

            start_time = df["t"].max()
            try:
                velocity = float(gcode[2][1:]) / 60
            except IndexError:
                velocity = 1000

            df1 = to_section(
                start_position, end_position, velocity, start_time, axis=axis, dt=dt
            )

            for col in df.columns:
                if col not in df1.columns:
                    df1[col] = df[col].iloc[-1]
            df = pd.concat([df, df1], axis=0, ignore_index=True)

        elif gcode[0] == "G4" and logging:
            df.loc[len(df)] = df.iloc[-1].values.copy()
            if gcode[1][0] == "S":
                df.loc[len(df) - 1, "t"] += float(gcode[1][1:])
            elif gcode[1][0] == "M":
                df.loc[len(df) - 1, "t"] += float(gcode[1][1:]) / 1000

        elif gcode[0] == "G91":
            relative_move = True
        elif gcode[0] == "G90":
            relative_move = False

    df.set_index("t", inplace=True)

    # drop all columns with constant values
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_columns)

    # drop all rows with velocities
    df = df.drop(columns=[col for col in df.columns if col.startswith("v_")])
    return df


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
        self.current_position = None

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
            elif response[:2] == "X:":
                tmp = response.split(" ")
                self.current_position = {x[0]: float(x[2:]) for x in tmp[:5]}
            else:
                pass

    def stop(self):
        self.running = False
        self.disconnect()

    def get_dataframe(self):
        return self.df.copy()

    def clear_dataframe(self):
        self.df = pd.DataFrame()

    def get_current_position(self):
        self.current_position = None
        self.send_gcode("M114")
        while self.current_position is None:
            time.sleep(0.1)
        return list(self.current_position.values())


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
