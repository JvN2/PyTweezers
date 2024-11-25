# FILE: MT_steppers.py

import threading
import serial
import time

class StepperApplication:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.running = False

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

    def run(self):
        self.running = True
        self.connect()
        if self.running:
            # Example G-code commands
            gcodes = [
                "G28 X",  # Home all axes
                "G1 X1 F30",  # Move to position
                "M400",
                "G4 S1",
                "G1 X0 F30"
            ]
            self.send_gcodes(gcodes)
            print("Completed sending G-code commands")

    def stop(self):
        self.running = False
        self.disconnect()

if __name__ == "__main__":
    stepper_app = StepperApplication(port='COM3')
    threading.Thread(target=stepper_app.run).start()