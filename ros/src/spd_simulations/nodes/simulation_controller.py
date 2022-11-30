#!/usr/bin/env python
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from time import sleep

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

for position in positions:
    rospy.loginfo(f"Starting move to {position}...")

    # Set target
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = position["x"]
    target_pose.position.y = position["y"]
    target_pose.position.z = position["z"]

    group.set_pose_target(target_pose)

    # Execute the motion
    plan = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
    rospy.loginfo(f"Move finished")
    sleep(1)
