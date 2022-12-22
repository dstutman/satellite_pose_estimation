#!/bin/bash
# Builds all subproject containers
podman build --tag=daheng_camera daheng_camera
podman build --tag=spd_controller spd_controller
