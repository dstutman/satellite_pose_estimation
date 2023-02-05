### Feature Extraction CNN Implementation ###

This folder contains the CNN implementation for feature extraction of the Delfi-n3xt satellite.

The files `coco_eval.py`, `coco_utils.py`, `engine.py`, `group_by_aspect_ratio.py`, `presets.py`, `transforms.py`, and `utils.py` are all sourced from the Python COCO tools repository with some minor changes for the Delfi-n3xt dataset.

- `keypointrcnn_evaluate.py`: This script predicts keypoint and bounding box locations on an inputted testing image, including detection metrics.
- `keypointrcnn_train.py`: This script trains the CNN on an inputted training dataset.
- `dataset_annotation_keypoint.py`: This script generates the manually generated dataset annotations.
- `synthetic_generator.py`: This script generates random synthetic images of the satellite on various space environment backgrounds.

The CNN requires a training and testing dataset to run, which are not provided in this repository. It could also be run on a custom dataset with some modifications.
