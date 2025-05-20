# Aegis Turret Architecture

## Overview
Auto‑aiming turret that finds and tracks a human target using computer vision.

## Components
- **Vision**: software/vision/main.py (OpenCV + YOLO)
- **Control**: software/control/motor_controller.py (serial to stepper drivers)
- **Common**: software/common/utils.py (config, logging)
- **Hardware**: Raspberry Pi 5, Pi 3B, PiCam v3, Hailo accelerator, stepper motors, Neopixel ring

## Data Flow
1. Capture frame from camera  
2. Run detection model  
3. Compute target centroid  
4. Send pan/tilt commands to MotorController  
5. Update status LED (Neopixel)

## Deployment
Use `scripts/deploy.sh` to sync code and restart the systemd service.
