#!/bin/bash
set -e
# Stop all dependency containers
podman rm --force daheng_camera
podman rm --force roscore
