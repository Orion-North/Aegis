#!/usr/bin/env bash
set -e

REMOTE_USER=pi
REMOTE_HOST=raspberrypi.local
REMOTE_PATH=/home/pi/Aegis-Turret

# Sync code
rsync -avz --delete ./ $REMOTE_USER@${REMOTE_HOST}:${REMOTE_PATH}

# Install requirements on Pi
ssh $REMOTE_USER@${REMOTE_HOST} << 'EOSSH'
  cd $REMOTE_PATH
  source venv/bin/activate
  pip install -r software/vision/requirements.txt
  pip install -r software/control/requirements.txt
  sudo cp scripts/aegis.service /etc/systemd/system/aegis.service
  sudo systemctl daemon-reload
  sudo systemctl enable aegis.service
  sudo systemctl restart aegis.service
EOSSH

echo "Deployment complete"
