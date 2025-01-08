import serial
import time

gcode = 'M114'
with serial.Serial("COM3", 115200) as serial_connection:
    time.sleep(2)  # Wait for the connection to establish
    if serial_connection.is_open:
        serial_connection.write((gcode + "\n").encode("utf-8"))
        while serial_connection.in_waiting > 0:
            response = serial_connection.readline().decode("utf-8").strip()
            print(response)
