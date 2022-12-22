cmake_minimum_required(VERSION 3.5)
project(galaxy_camera_launch)

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

#find_package(catkin REQUIRED COMPONENTS galaxy_camera)
