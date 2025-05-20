# Aegis Turret Setup

## Prerequisites
- Python 3.8+  
- Raspberry Pi OS on both Pi 5 and Pi 3B  
- PiCam v3 on Pi 5  
- Motor driver to Pi 3B via USB/UART  
- Hailo AI accelerator on Pi 5  
- Stepper motors mounted  

## Installation
```bash
git clone <your‑repo‑url>
cd Aegis-Turret
bash scripts/setup.sh
```

## Configuration
Edit `config.yaml` for camera, model and motor settings.

## Running
Activate env and start vision node:
```bash
source venv/bin/activate
python software/vision/main.py
```
