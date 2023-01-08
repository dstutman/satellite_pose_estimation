#!/usr/bin/env python
from time import sleep

import numpy as np

import rospy
import moveit_commander
import moveit_msgs
import geometry_msgs
import spd_controller.trajectories as trajectories

# Configuration
trajectory = trajectories.line(np.array([0, 0, 0]), np.array([0, 0, 1], np.array([1, 0, 0, 0])))
num_trajectory_points = 10

rospy.init_node("logger")
rospy.loginfo("SPD test simulation starting...")

panda = moveit_commander.robot.RobotCommander()
scene = moveit_commander.planning_scene_interface.PlanningSceneInterface()
group = moveit_commander.move_group.MoveGroupCommander("panda_arm")
display_trajectory_publisher = rospy.Publisher(
    "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20
)

# Increase the speed of operation to 250%
group.set_max_acceleration_scaling_factor(0.25)
group.set_max_velocity_scaling_factor(0.25)
# Set a 5cm tolerance
group.set_goal_position_tolerance(0.05)

positions = [
    {"x": 0.5, "y": 0, "z": 0.5},
    {"x": 0.0, "y": 0.5, "z": 0.5},
    {"x": -0.5, "y": 0, "z": 0.5},
    {"x": 0.0, "y": -0.5, "z": 0.5},
    {"x": 0.5, "y": 0, "z": 0.5},
]

for pose in trajectories.discrete_points(num_trajectory_points, trajectory):
    rospy.loginfo(f"Starting move to: {pose.position}...")

    # Set target
    group.set_pose_target(pose.to_geom_msg())

    # Execute the motion
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
    rospy.loginfo(f"Move finished")
    sleep(1)
