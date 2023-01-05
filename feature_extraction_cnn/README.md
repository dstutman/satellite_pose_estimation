### How to use the CNN for pose estimation ###

* Setup
- Pull the `dev_fe` branch.
- Go to the Google Drive and download  `Images Series/CNN Testing Dataset/`. 
- Save the folders `annotations/` and `images/` to a folder called `test_dataset`
  in your root folder (i.e. `satellite_pose_estimation/test_dataset` is the 
  correct path.)
- Back in the Google Drive, download `keypointsrcnn_weights.pth`. Save it again 
  to the root folder of `satellite_pose_estimation`. 

* Evaluation
- In `evaluate.py`, you can just run the code. The evaluator actually evaluates every
  image in the testing dataset because I am trying to measure precision and recall
  scores, but that feature is not currently working.
- What is working is the output of the testing image and its keypoints. 
- Right now I have it set to shuffle the dataset, and whatever the last image is is what
  will be uploaded.
    - If you need a specific image you could put it in another folder for now and change 
      the 'KEYPOINTS_FOLDER_TEST' variable on line 214 to that folder name. There will 
      be a better method for the online evaluation, this file is more for diagnostics 
      at the moment. 
