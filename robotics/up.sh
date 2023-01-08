#!/bin/bash
set -e
# Starts dependency service ROS containers
podman run --name=roscore --net=host --env ROS_HOSTNAME=fedora.local --env ROS_PORT=11311 -d ros:noetic-ros-core roscore
podman run --name=daheng_camera --net=host --env ROS_MASTER_URI="http://fedora.local:11311" --privileged -d localhost/daheng_camera:latest
echo -e "\n(Re-)Run 'source devel/setup.bash && ROS_MASTER_URI=http://fedora.local:11311 roslaunch franka_control franka_control.launch robot_ip:=172.16.0.2 load_gripper:=false --wait' on the RT control node"
