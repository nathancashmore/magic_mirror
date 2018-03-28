#!/bin/bash
source /home/pi/.profile
echo "Running with the following ENV:"
env

echo "Shutting down motion service"
sudo service motion stop
sudo service --status-all|grep motion

echo "Starting the magic mirror"
/usr/bin/python /home/pi/magic_mirror/face-recognition.py

#
# Add the following to /home/pi/.config/lxsession/LXDE-pi
# @sh /home/pi/magic_mirror/start.sh
#
