#!/bin/bash
set -e
# Run the controller container
podman run --name=spd_controller --net=host --env ROS_MASTER_URI=http://fedora.local:11311 -it spd_controller

