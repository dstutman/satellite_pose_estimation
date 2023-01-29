#################################################################################################################
######################################### FEATURE EXTRACTION EVALUATION #########################################
#################################################################################################################  

'''
keypointrcnn_evaluate.py

Primary evaluation code for the keypoint CNN. Run this file to visualize outputs and record metrics from a network
that has already been trained.

Adjustable parameters are found at the end of the file in a clearly labelled section. 

Author: Adyn Miles

Some code and ideas taken from Alex P in the following blog post: 
https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da

'''
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, json
import PIL
import time

from heapq import nlargest
from utils import collate_fn
from engine import evaluate

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A


class ClassDataset(Dataset):
    """
    Class for creating dataset of images in the right format for training.
  
    Parameters:
    root (str): The root folder for training.
    transform (A.Compose()): sequence of transforms from Albumentations
    demo (bool): True or False to indicate whether demo mode is desired.
  
    Returns:
    img (np.array): The image to be evaluated or trained upon by the model.
    target (np.array): The ground truth bounding boxes, labels, and keypoint
                       locations for the image of interest.
  
    """
    def __init__(self, root, transform=None, demo=False, original=False):                
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.original = original  

        # Ensure that your testing dataset images are stored under respective folders called "images" and "annotations".
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        if not os.path.exists(os.path.join(root, "annotations")):
                os.mkdir(os.path.join(root, "annotations"))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
    
    # Function to get and prepare an image for use in the model.
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        self.img_name = self.imgs_files[idx]
        if not self.demo: 
            annotations_name = img_path.split("\\")[2].split(".")[0] + ".json"
            annotations_path = os.path.join(self.root, "annotations", annotations_name)
            with open(annotations_path, 'w') as f:
                kp_out = {'img_name': annotations_name, 'bboxes': [[0, 0, 1, 1], [0, 0, 1, 1]], 
                        'keypoints': [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]}
                json.dump(kp_out, f)

        if self.demo: 
            annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])
            

        # Python Image Library is used here to open the image of choice rather than OpenCV because OpenCV flips the RGB color
        # channels to BGR and flips H and W of the image. Opening those files was inconsistent and would throw errors.
        img_original = np.array(PIL.Image.open(img_path))
        
        with open(annotations_path) as f:
            # Load bounding box and keypoint coordinates from the dataset.
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']

            # All labelled bounding boxes correspond to the Delfi-n3xt satellite.
            bboxes_labels_original = ['Delfi-n3xt' for _ in bboxes_original]            

        if self.transform:   
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints            
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format            
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            

            # Apply augmentations to images, bounding box coordinates, and keypoint coordinates.
            transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']
            
            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1,4,2)).tolist()

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)

        
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original     
        
        # Convert everything into a torch tensor so that the network can understand and use it.         
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        # The labels of the bounding boxes are not particularly important for keypoint detection
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        # Area of the bounding box
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # If there is crowding (multiple detectable objects that are much too close to each other), then setting
        # this flag to 1 will tell the keypoint CNN to ignore it. However, based on the geometry of the satellite,
        # this will not be necessary.
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)
        
        # Do the same for the image prior to the transformations (for debugging purposes)
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64) # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)        
        img_original = F.to_tensor(img_original)

        if self.original:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs_files)

def test_transform():
    """
    Function for composing the transforms for testing images. This is just resizing each image to be
    1440 x 1080 in this case.
  
    Parameters:
    None
  
    Returns:
    A.Compose(): returns sequence of transformations consisting of normalization, random rotation, and the 
                 associated transformations of keypoints and bounding boxes.
  
    """   
    return A.Compose([
        A.Sequential([
            A.augmentations.geometric.resize.LongestMaxSize(max_size=1440, interpolation=cv2.INTER_LINEAR, always_apply=True, p=1),         
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )   


def get_model(num_keypoints, weights_path=None):
    """
    Builds the network, which uses the weights from a pretrained backbone applied to IMAGENET and performs
    transfer learning to find the keypoints in these satellite images.
  
    Parameters:
    num_keypoints (int): The number of keypoints can change the network architecture, so it is important to 
                         specify.
    training_loss (np.array, dtype=float): Keypoint loss from each epoch as measured by Keypoint CNN.
  
    Optionals:
    weights_path (str): If there is already a trained network, passing in the path of the network will load
                        its dictionary directly so that it can be trained further or fine-tuned.
    Returns:
    model (torch model): Returns the neural network model
  
    """
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=None,
                                                                   weights_backbone=torchvision.models.get_weight('ResNet50_Weights.IMAGENET1K_V1'),
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model

def visualize(image, bboxes, keypoints, img_name, image_original=None, bboxes_original=None, keypoints_original=None):
    """
    Function for visualizing a selected photo from the dataset.
  
    This function produces an image of the satellite along with the bounding box locations in green and the 
    keypoint coordinates (and labels) in red. This can be used for debugging the transformation code, as well 
    as showing how well the predictions have worked.
  
    Parameters:
    image (np.array): The base image that will be displayed.
    bboxes (np.array, dtype=int): The bounding box coordinates, inputted as a list with [x1, y1, x2, y2] format.
    keypoints (np.array, dtype=int): The keypoint coordinates, inputted as a list with [x, y, visibility] format.
                                     If visibility = 0, the keypoint will not be shown.
    
    Optionals:
    image_original (np.array): If you wish to show a transformed image, pass in the original image.
    bboxes_original (np.array): Passing in the original bounding box coordinates, in the same format as bboxes.
    keypoints_original (np.array): Passing in the original keypoint coordinates, in the same format as 
                                   keypoints.
  
    Returns:
    None
  
    """
    # Plotting parameters
    fontsize = 18
    keypoints_classes_ids2names = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8'} # Keypoint labels

    # Draw bounding boxes based on their coordinates
    for bbox in bboxes:
         start_point = (bbox[0], bbox[1])
         end_point = (bbox[2], bbox[3])
         image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    # Draw visible keypoints based on their coordinates
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            if kp[2] != 0:
                image = cv2.circle(image.copy(), tuple(kp[:2]), 5, (255,0,0), 10)
                image = cv2.putText(img=image.copy(), text=" " + keypoints_classes_ids2names[idx], org=tuple(kp[:2]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,0,0), thickness=3, lineType=cv2.LINE_AA)

    # This is done if the visualize() method is not being used to track the transformation of an image.
    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40,40), clear=True)
        plt.imshow(image)
        plt.clf()

    # If the visualize() method is being used to track the transformation of an image, the original image is
    # also annotated and displayed side-by-side.
    else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                if kp[2] != 0:
                    image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
                    image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20), clear=True)

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)
        plt.clf()

    outpath = 'outputs/' + str(img_name)
    plt.imsave(outpath, image)

def boxOverlap(box1, box2):
    """
    Function for testing if two bounding boxes overlap.
  
    Parameters:
    box1 (np.array): Bounding box 1 top left and bottom right coordinates written as [x1, y1, x2, y2].
    box2 (np.array): Bounding box 2 top left and bottom right coordinates written as [x1, y1, x2, y2].
    
    Returns:
    isOverlapping (bool): True if boxes overlap, and False if not.
  
    """
    isOverlapping = (box1[0] < box2[2] and box2[0] < box1[2] and box1[1] < box2[3] and box2[1] < box1[3])
    return isOverlapping

def eval_metrics(bboxes, keypoints, root, img_name):
    """
    Function for computing IOU, OKS, and distance metrics
  
    Parameters:
    bboxes (np.array): Bounding box prediction coordinates, each in format [x1, y1, x2, y2].
    keypoints (np.array): Keypoint prediction coordinates, each in format [x, y, visibility]
    root (str): Root of the file being evaluated
    img_name (str): Name of the image being evaluated.
        
    Returns:
    isOverlapping (bool): True if boxes overlap, and False if not.
  
    """
    # From the OKS formula, coeff (kappa_i) and scale (S).
    coeff = 0.5
    scale = 1.0

    # Load in ground truth image along with bounding boxes and keypoints
    if img_name[-4] == ".":
        ground_truth = root + "/annotations/" + str(img_name[:-4]) + ".json"
    else: 
        ground_truth = root + "/annotations/" + str(img_name[:-5]) + ".json"    
    with open(ground_truth, 'r') as f:
        data = json.load(f)
        bbox_gt = data['bboxes']
        kp_gt = data['keypoints']
    f.close()

    # Initialize arrays
    vis_gt = []
    dist = []
    iou = []

    # Bounding boxes and keypoints (shuffled), as sometimes the network predicts
    # the antenna side to be side 1, and some time it is the solar panels.
    bboxes_shuffled = []
    keypoints_shuffled = []
    dist_shuffle = []
    bboxes_shuffled.append(bboxes[1])
    bboxes_shuffled.append(bboxes[0])
    keypoints_shuffled.append(keypoints[1])
    keypoints_shuffled.append(keypoints[0])

    for idx in range(0, len(bboxes)):
        for gt_idx in range(0, len(bbox_gt)):
            # If the boxes do not overlap, IOU will return nonsense results.
            if boxOverlap(bbox_gt[gt_idx], bboxes[idx]):
                x1, y1, x2, y2 = bbox_gt[gt_idx]
                x3, y3, x4, y4 = bboxes[idx]
                # Calculate the coordinates of the intersection rectangle.
                inter_left_x = max(x1, x3)
                inter_left_y = max(y1, y3)
                inter_right_x = min(x2, x4)
                inter_right_y = min(y2, y4)
                # Area of the intersection rectangle.
                width_inter = abs(inter_left_x - inter_right_x)
                length_inter = abs(inter_left_y - inter_right_y)
                area_inter = width_inter * length_inter
                width_gt = abs(x2 - x1)
                height_gt = abs(y2 - y1)
                width_pre = abs(x4 - x3)
                height_pre = abs(y4 - y3)
                area_1 = width_gt * height_gt
                area_2 = width_pre * height_pre
                # Area of the union of the two bounding boxes, being careful not
                # to count the intersection of the two rectangles twice.
                area_union = area_1 + area_2 - area_inter
                iou.append(area_inter/area_union)
            else:
                # If the boxes don't overlap, IOU is 0.
                iou.append(0)
        for kp in range(0, len(keypoints[idx])):
            # Add the distances for the current iteration of keypoints. OKS is not 
            # calculated here because it is a weighted average of all the keypoints
            # (therefore it is done outside of the loop).
            vis_gt.append(kp_gt[idx][kp][2])
            dist.append(np.linalg.norm(np.array(keypoints[idx][kp][0:2]) - np.array(kp_gt[idx][kp][0:2])))
            dist_shuffle.append(np.linalg.norm(np.array(keypoints_shuffled[idx][kp][0:2]) - np.array(kp_gt[idx][kp][0:2])))
    
    # Populate visibility and distance arrays for the OKS calculation.
    vis_gt = np.array(vis_gt)
    dist = np.array(dist)
    dist_shuffle = np.array(dist_shuffle)
    gaussian = np.exp(-(dist**2) / (2 * (scale**2) * (coeff**2)))
    gaussian_shuffle = np.exp(-(dist_shuffle**2) / (2 * (scale**2) * (coeff**2)))
    numerator = np.dot(gaussian, vis_gt)
    numerator_shuffle = np.dot(gaussian_shuffle, vis_gt)
    denominator = np.sum(vis_gt)
    denominator_shuffle = np.sum(vis_gt)

    # Store the distances and shuffled distances (for reversed bounding boxes) for analysis as well.
    dist = np.multiply(dist, vis_gt)
    dist_shuffle = np.multiply(dist_shuffle, vis_gt)
    dist = [np.nan if x == 0 else x for x in dist]
    dist_shuffle = [np.nan if x == 0 else x for x in dist_shuffle]
    oks = [numerator/denominator, numerator_shuffle/denominator_shuffle]
    
    return iou, oks, dist, dist_shuffle     

def save_output(keypoints, img_name):
    """
    Function to save keypoint coordinates in a .JSON file in order to be used by 
    the pose estimation algorithm.
  
    Parameters:
    keypoints (np.array): Keypoint prediction coordinates, each in format [x, y, visibility]
    img_name (str): Name of the image being evaluated.
    
    Returns:
    None
  
    """
    camera_name = 'Daheng' if img_name[-5] == "." else 'Phone'
    kp_out = {'img_name': img_name, 'camera_name': camera_name, 
    'keypoints': keypoints}
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    outfile = 'outputs/' + str(img_name) + '.json'
    with open(outfile, 'w') as f:
        json.dump(kp_out, f)

iou_list = []
oks_list = []
dist_list = []
time_total_list = []

#################################################################################################################
######################################### ADJUSTABLE PARAMETERS #################################################
################################################################################################################# 

num_iter = 10 
holistic_metrics = False # Set to true if you want holistic metrics for OKS and IOU.
demo = False   # Set to False if you want to run the program without requiring annotation inputs.
original = False # This is for the training code. Keeping this false is the safer option.  

for idx in range(0, num_iter):
    # Testing dataset root folder path
    KEYPOINTS_FOLDER_TEST = 'jan 26 ground truth-20230129T230051Z-001'
    dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=test_transform(), demo=demo, original=original)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Change the weights_path based on which one you would like to load in for evaluation.
    model = get_model(num_keypoints = 4, weights_path='keypointsrcnn_weights.pth')
    model.to(device)

#################################################################################################################
###################################### END OF ADJUSTABLE PARAMETERS #############################################
################################################################################################################# 

    iterator = iter(data_loader_test)
    images, targets = next(iterator)
    images = list(image.to(device) for image in images)

    if holistic_metrics == True:
        evaluate(model, data_loader_test, device)

    time_start = time.time()
    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model(images)    
    time_end = time.time()
    time_total = time_end - time_start
    time_total_list.append(time_total) 
    print('Elapsed Time: ' + str(time_total))   

    print("Predictions: \n", output)
    image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    scores = output[0]['scores'].detach().cpu().numpy()

    high_scores_idxs = nlargest(2, range(len(scores)), key=lambda idx: scores[idx])

    distinct = False
    while distinct == False:
        if not boxOverlap(np.array(output[0]['boxes'][high_scores_idxs[0]].detach().cpu().numpy()), np.array(output[0]['boxes'][high_scores_idxs[1]].detach().cpu().numpy())):
            distinct = True

        else:
            high_scores_idxs[1] += 1
            if (high_scores_idxs[1] == len(output[0]['boxes'])) or (output[0]['scores'][high_scores_idxs[1]] < 0.2):
                high_scores_idxs[1] = 0
                break

    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp)) for kp in kps])

    for bb_num, bb in enumerate(keypoints):
        for i in range(0, len(bb)-1):
            for j in range(i+1, len(bb)):
                kp_diff = np.subtract(np.array(bb[i][:2]), np.array(bb[j][:2]))
                if np.linalg.norm(kp_diff) < 25:
                    if abs(output[0]['keypoints_scores'][high_scores_idxs][bb_num][i]) > abs(output[0]['keypoints_scores'][high_scores_idxs][bb_num][j]):
                        bb[j][0], bb[j][1], bb[j][2] = 0, 0, 0
                    else:
                        bb[i][0], bb[i][1], bb[i][2] = 0, 0, 0


    bboxes = []
    for bbox in output[0]['boxes'][high_scores_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))

    visualize(image, bboxes, keypoints, dataset_test.img_name)

    iou, oks, dist, dist_shuffle = eval_metrics(bboxes, keypoints, root=KEYPOINTS_FOLDER_TEST, img_name=dataset_test.img_name)

    iou_list.append(iou)
    oks_list.append(oks)
    dist_list.append(min(np.nanmean(dist), np.nanmean(dist_shuffle)))

    save_output(keypoints, dataset_test.img_name)

for i in range(0, num_iter):
    print("IOU " + str(i) + ": " + str(iou_list[i]))
    print("OKS " + str(i) + ": " + str(oks_list[i]))
    print("Mean distance " + str(i) + ": " + str(dist_list[i]))