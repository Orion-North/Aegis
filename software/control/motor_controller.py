#!/usr/bin/env python3
import serial
import time

class MotorController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None

    def connect(self):
        self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
        time.sleep(2)
        # TODO: send any init commands to the driver

    def move_to(self, pan_angle, tilt_angle):
        """
        Send pan and tilt angles (degrees) to the motor driver.
        """
        cmd = f'PA{pan_angle:.2f}YA{tilt_angle:.2f}\n'
        self.serial.write(cmd.encode('utf-8'))

    def cleanup(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
