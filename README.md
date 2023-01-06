# satellite_pose_estimation
HITL chase and pose estimation of uncooperative satellites

There are two main files, "PE_main" and "main _test"

PE_main
-------------------------
The primary function that runs the pose estimation. Pose inputs are taken from the inputs folder in json format, which needs to be added manually after initial download.
Be very careful of the input format to the EPnp function if making changes. 

Outputs plot visualizations and pose information

PE_main_test
-------------------------
The orinigal file made by EPFL, has a separate dataset and outputs error. Use for reference and as a baseline.

All other code: needed but don't fuck with it.
