########## normalize.py ##########
# This file normalizes the dataset so that features can be extracted.
# TODO: - add normalizer for bounding boxes

import os, json, numpy as np

training_path = "train_dataset\\annotations\\"
testing_path = "test_dataset\\annotations\\"

training_count = 0
testing_count = 0
for filename in os.listdir(training_path):
    training_count += 1
for filename in os.listdir(testing_path):
    testing_count += 1

train_keypoints_total = np.zeros((training_count, 8, 3))
for file_num, filename in enumerate(os.listdir(training_path)):
    keypoints = []
    with open(os.path.join(training_path, filename), 'r') as f:
        data = json.load(f)
        keypoints_current = data["keypoints"]
        for kp_num1 in range(0, len(keypoints_current[0])):
            keypoints.append(keypoints_current[0][kp_num1])
        for kp_num2 in range(0, len(keypoints_current[1])): 
            keypoints.append(keypoints_current[1][kp_num2])
        train_keypoints_total[file_num] = keypoints
    f.close()

test_keypoints_total = np.zeros((testing_count, 8, 3))
for file_num, filename in enumerate(os.listdir(testing_path)):
    keypoints = []
    with open(os.path.join(testing_path, filename), 'r') as f:
        data = json.load(f)
        keypoints_current = data["keypoints"]
        for kp_num1 in range(0, len(keypoints_current[0])):
            keypoints.append(keypoints_current[0][kp_num1])
        for kp_num2 in range(0, len(keypoints_current[1])): 
            keypoints.append(keypoints_current[1][kp_num2])
        test_keypoints_total[file_num] = keypoints
    f.close()

for i in range(0, len(test_keypoints_total)):
    for j in range(0, len(test_keypoints_total[i])):
        if test_keypoints_total[i][j][2] == 0:
            test_keypoints_total[i][j][0], test_keypoints_total[i][j][1] = np.nan, np.nan

for i in range(0, len(train_keypoints_total)):
    for j in range(0, len(train_keypoints_total[i])):
        if train_keypoints_total[i][j][2] == 0:
            train_keypoints_total[i][j][0], train_keypoints_total[i][j][1] = np.nan, np.nan

train_mean = np.zeros((8, 2))
train_std = np.zeros((8, 2))
keypoints_calc = []
for i in range(0, len(train_keypoints_total[0])):
    for j in range(0, len(train_keypoints_total)):
        keypoints_calc.append(train_keypoints_total[j][i])
    train_mean[i][0], train_mean[i][1], _ = np.nanmean(keypoints_calc, axis=0) 
    train_std[i][0], train_std[i][1], _ = np.nanstd(keypoints_calc, axis=0)
    keypoints_calc = []

test_mean = np.zeros((8, 2))
test_std = np.zeros((8, 2))
for i in range(0, len(test_keypoints_total[0])):
    for j in range(0, len(test_keypoints_total)):
        keypoints_calc.append(test_keypoints_total[j][i])
    test_mean[i][0], test_mean[i][1], _ = np.nanmean(keypoints_calc, axis=0) 
    test_std[i][0], test_std[i][1], _ = np.nanstd(keypoints_calc, axis=0)
    keypoints_calc = []

for i in range(0, np.shape(train_mean)[0]):
    print("TRAIN: Mean - Keypoint " + str(i + 1) + ": " + str(train_mean[i]) + " || Std - Keypoint " + str(i + 1)+ ": " + str(train_std[i]))
    print("TEST: Mean - Keypoint " + str(i + 1) + ": " + str(test_mean[i]) + " || Std - Keypoint " + str(i + 1)+ ": " + str(test_std[i]))



        



