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

if __name__ == "__main__":
    stepper_app = StepperApplication(port='COM3')

    # Example G-code commands
    gcodes = [
        "G93 S0.01",
        "G4 S1",
        "G1 Y0.1 F30",
        "G1 X0.25 F30",
        "G4 S1",
        "G1 X0 F10",
        "G4 S0.4",
        "G1 Y0.0 F10",
        "G4 S0.4",
        "G94"
    ]
    print(gcodes)

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