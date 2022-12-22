#!/bin/bash
# Starts dependency service ROS containers
podman run --net=host --env ROS_HOSTNAME=fedora.local --env ROS_PORT=11311 -d ros:noetic-ros-core roscore
podman run --net=host --env ROS_MASTER_URI="https://fedora.local:11311" --privileged -d localhost/daheng_camera:latest
echo -e "\n(Re-)Run 'source devel/setup.bash && roslaunch franka_control franka_control.launch --wait' on the RT control node"
