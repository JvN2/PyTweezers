# FILE: MT_steppers.py

import threading
import serial
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class StepperApplication:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.running = False
        self.df = None

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
        positions = []
        while self.running:
            if self.serial_connection and self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode('utf-8').strip()
                if response[:4] == 'log:':
                    positions.append(response[4:].split())
                if 'Stopped logging' in response:
                    self.running = False
                    df = pd.DataFrame(positions[1:], columns=positions[0])
                    df = df.astype(float)
                    df.set_index(df.columns[0], inplace=True)
                    self.df = df



    def run(self, gcodes):
        self.running = True
        self.connect()
        threading.Thread(target=self.read_response).start()
        if self.running:
            self.send_gcodes(gcodes)


    def stop(self):
        self.running = False
        self.disconnect()

if __name__ == "__main__":
    stepper_app = StepperApplication(port='COM3')

        # Example G-code commands
    gcodes = [
        # "G28 X",  # Home all axes
        "G93 S0.1",
        "G4 S1"]
    for i in np.linspace(0, 1  , 100):
        gcodes.append(f"G1 X{i:3.3f} F{0.5*(i+1)**6:3.3f}")
    gcodes += [
        "G4 S1",
        "G1 X0 F20",  # Move to position
        "G4 S1",
        "G94"
    ]

    for g in gcodes:
        if "G1" in g:
            print(g)


    threading.Thread(target=stepper_app.run, args=(gcodes,)).start()
    while stepper_app.df is None:
        time.sleep(0.5)
    plt.plot(stepper_app.df['X'])
    plt.show()