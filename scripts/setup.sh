#!/usr/bin/env bash
set -e

# Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install vision deps
pip install opencv-python torch pyyaml
echo -e "opencv-python\ntorch\npyyaml" > software/vision/requirements.txt

# Install control deps
pip install pyserial RPi.GPIO
echo -e "pyserial\nRPi.GPIO" > software/control/requirements.txt

# Create placeholders
touch software/vision/main.py \
      software/control/motor_controller.py \
      software/common/utils.py \
      scripts/deploy.sh \
      scripts/aegis.service

echo "Setup complete. Activate with: source venv/bin/activate"
