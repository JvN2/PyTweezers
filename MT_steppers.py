# FILE: MT_steppers.py

import threading
import serial
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class StepperApplication:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.running = False
        self.df = pd.DataFrame()

    def connect(self):
        self.serial_connection = serial.Serial(self.port, self.baudrate)
        time.sleep(2)  # Wait for the connection to establish

    def disconnect(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()

    def send_gcode(self, gcode):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.write((gcode + '\n').encode('utf-8'))
            time.sleep(0.1)  # Wait for the command to be processed

    def send_gcodes(self, gcodes):
        for gcode in gcodes:
            self.send_gcode(gcode)

    def read_response(self):
        while self.running:
            if self.serial_connection and self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode('utf-8').strip()
                if response[:4] == 'log:':
                    data = response[4:].split()
                    try:
                        data = [float(x) for x in data]
                        self.df.loc[data[0]] = data[1:] 
                    except:
                        self.df = pd.DataFrame(columns=data)
                        self.df.set_index(self.df.columns[0], inplace=True)
                if 'Stopped logging' in response:
                    self.running = False

    def run(self, gcodes):
        self.running = True
        self.connect()
        threading.Thread(target=self.read_response).start()
        if self.running:
            self.send_gcodes(gcodes)

    def stop(self):
        self.running = False
        self.disconnect()

def update_plot(frame, stepper_app, lines):
    if not stepper_app.df.empty:
        for line, column in zip(lines, stepper_app.df.columns):
            line.set_data(stepper_app.df.index, stepper_app.df[column])
        plt.gca().relim()
        plt.gca().autoscale_view()
    return lines

def on_close(event, stepper_app):
    stepper_app.stop()
    plt.close('all')
    print('Plot closed and application stopped.')

def convert_to_trace(gcodes, a=8, axes=['X', 'Y', 'Z', 'A', 'B']):
    df = pd.DataFrame(columns=['t', 'dt'] + axes + [f'v_{axis}' for axis in axes])
    df.loc[0] = np.zeros(len(df.columns))
    print(df)



    for line in gcodes:
        gcode = line.split()
        if gcode[0] == 'G1':
            df.loc[len(df), gcode[1][0]] = float(gcode[1][1:])
            df.loc[len(df)-1, 'v_' + gcode[1][0]] = float(gcode[2][1:])
        elif gcode[0] == 'G4':
            df.loc[len(df), 'dt'] = float(gcode[1][1:])
    df.loc[len(df), 'dt'] = 0

    # Replace NaN values in columns that start with 'v_' with 0
    df.loc[:, df.columns.str.startswith('v_')] = df.loc[:, df.columns.str.startswith('v_')].fillna(0)

    for row in range(1, len(df)):
        if np.isfinite(df.loc[row, 'dt']):
            df.loc[row, 't'] = df.loc[row - 1, 't'] + df.loc[row, 'dt']
            df.loc[row, axes] = df.loc[row - 1, axes]
        # else

    print(df)
    return df

    df[0] = current_position.values()
    current_time = 0

    for gcode in gcodes:
        parts = gcode.split()
        command = parts[0]
        if command in ['G0', 'G1']:
            for part in parts[1:]:
                if part[0] in axes:
                    current_position[part[0]] = float(part[1:])
            df.loc[current_time] = [current_position[axis] for axis in axes]
        elif command == 'G4':
            for part in parts[1:]:
                if part[0] == 'S':
                    current_time += float(part[1:])
                    df.loc[current_time] = [current_position[axis] for axis in axes]

    print(df)
    return df

def convert_to_section(t0, x0, v0, xe, ve, v, a, time, position):
    dt1 = (v - v0) / a
    dx1 = v0 * dt1 + 0.5 * a * dt1 ** 2

    dt3 = (ve - v) / a
    dx3 = ve * dt3 - 0.5 * a * dt3 ** 2

    dx2 = xe - x0 - dx1 - dx3
    dt2 = dx2 / v

    while 

    return n

if __name__ == "__main__":

    print(convert_to_section(0, 0, 10, 0, 1, 8, 0.1))

    stepper_app = StepperApplication(port='COM3')

    # Example G-code commands
    gcodes = [
        "G93 S0.01",
        "G4 S1",
        "G1 Y0.1 F30",
        "G1 X0.25 F30",
        "G4 S1",
        # "G1 X0 F10",
        # "G4 S0.4",
        # "G1 Y0.0 F10",
        # "G4 S0.4",
        "G94"
    ]
    # convert_to_trace(gcodes)

 


    if False:

        # Start the stepper application with the G-code commands
        threading.Thread(target=stepper_app.run, args=(gcodes,)).start()

        # Set up the plot
        fig, ax = plt.subplots()
        lines = [ax.plot([], [], label=col)[0] for col in ["X", "Y", "Z", "A", "B"]]  # Adjust columns as needed
        ax.legend()
        ani = FuncAnimation(fig, update_plot, fargs=(stepper_app, lines), interval=100, cache_frame_data=False)

        # Connect the close event to the on_close function
        fig.canvas.mpl_connect('close_event', lambda event: on_close(event, stepper_app))

        plt.show()