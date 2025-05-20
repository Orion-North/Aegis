#!/usr/bin/env python3
import cv2
import time
import os, sys
# ensure common and control folders are on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import load_config, setup_logging
from control.motor_controller import MotorController

def main():
    config = load_config('../config.yaml')
    setup_logging()
    cap = cv2.VideoCapture(config['camera']['device'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    cap.set(cv2.CAP_PROP_FPS, config['camera']['fps'])

    motor = MotorController(
        port=config['motor']['port'],
        baudrate=config['motor']['baudrate']
    )
    motor.connect()

    # TODO: load your YOLO model here, e.g. with cv2.dnn or a framework of your choice

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # TODO: detect and track target, compute pan/tilt angles
        # pan_angle, tilt_angle = detect_and_compute(frame)
        # motor.move_to(pan_angle, tilt_angle)

        cv2.imshow('Aegis View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    motor.cleanup()

if __name__ == '__main__':
    main()
