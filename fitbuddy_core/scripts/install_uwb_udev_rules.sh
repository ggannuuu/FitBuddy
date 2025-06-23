#!/bin/bash

# Install UWB udev rules script

echo "Installing UWB anchor udev rules..."

# Copy the udev rules file
sudo cp /home/acrl/fitbuddy_ws/60-uwb-anchors.rules /etc/udev/rules.d/

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "UWB udev rules installed successfully!"
echo "Device symlinks should now be available as:"
echo "  /dev/uwb_anchor_a1 (A1)"
echo "  /dev/uwb_anchor_a2 (A2)" 
echo "  /dev/uwb_anchor_a3 (A3)"

# Check if symlinks were created
sleep 2
echo "Checking device symlinks:"
ls -la /dev/uwb_anchor_* 2>/dev/null || echo "Symlinks not found - you may need to unplug/replug the UWB devices"