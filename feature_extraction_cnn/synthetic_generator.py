#################################################################################################################
######################################### SYNTHETIC IMAGE GENERATOR #############################################
#################################################################################################################  


'''
synthetic_generator.py

Primary training code for the keypoint CNN. Run this file to train a new network architecture, which evaluates it
as you progress. The output is saved into an output .pth file which can then be used for evaluation with
evaluate.py.

Adjustable parameters are found at the end of the file in a clearly labelled section. 

Author: Adyn Miles

Majority of code and ideas taken from Alex P in the following blog post: 
https://medium.com/@alexppppp/how-to-create-synthetic-dataset-for-computer-vision-keypoint-detection-78ba481cdafd

'''

import albumentations as A
import numpy as np
import PIL
import cv2
import matplotlib.pyplot as plt
import os, json
import time
from tqdm import tqdm
from random import randint

def get_img_and_mask(img_path, mask_path):
    """
    Function for composing the transforms for training images. This is a process called dataset augmentation,
    to expose the network to new images upon each dataset iteration.
  
    Parameters:
    img_path (str): path to the image
    mask_path (str): path to the corresponding mask of the image
  
    Returns:
    img (np.array): return the image as an array
    mask (np.array, dtype=np.uint8): return the image mask as an array
  
    """
    img = np.array(PIL.Image.open(img_path))
    mask = np.array(PIL.Image.open(mask_path))
    # Pull the coordinates in the image that are black (the masked coordinates)
    mask_b = mask[:,:,0] == 0
    mask = mask_b.astype(np.uint8)

    return img, mask

def visualize_single_img(img, keypoints, bboxes):
    """
    Function for visualizing a single image from the dataset. Used primarily for
    debugging purposes.
  
    Parameters:
    img (np.array): the input image of the satellite
    keypoints (np.array): array containing keypoint coordinates in [x, y, visibility] 
                          format.
    bboxes (np.array): array containing bounding box coordinates in [x1, y1, x2, y2]
                       format, corresponding to the top left and bottom right corners.
  
    Returns:
    None
  
    """
    keypoints_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    title="Keypoints and bounding boxes"

    # Draw bounding boxes
    for bbox in bboxes:
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        img = cv2.rectangle(img.copy(), start_point, end_point, (0,255,0), 2)
    # Draw keypoints
    for idx, kps in enumerate(keypoints):
        for idx, kp in enumerate(kps):
            img = cv2.circle(img.copy(), tuple(kp[:2]), 3, (255,0,0), 6)
            img = cv2.putText(img.copy(), " " + keypoints_names[idx], tuple(kp[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    
    plt.figure(figsize=(16,16))
    plt.title(title, fontsize=18)
    plt.imshow(img)
    plt.clf()

def resize_img(img, desired_max=4032, desired_min=3024):
    """
    Function for resizing a single background image from the dataset.
  
    Parameters:
    img (np.array): the input image of the satellite
    desired_max (int): maximum allowable size of image background
    desired_min (int): minimum allowable size of image background
  
    Returns:
    img_r (np.array): return the resized image
  
    """
   
    h, w = img.shape[0], img.shape[1]
    
    # We want to keep a portrait image as portrait, and vice versa.
    longest, shortest = max(h, w), min(h, w)
    longest_new = desired_max
    if desired_min:
        shortest_new = desired_min
    else:
        shortest_new = int(shortest * (longest_new / longest))
    
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    # Compose the resizing transform.
    transform_resize = A.Compose([
        A.Sequential([
        A.Resize(h_new, w_new, interpolation=1, always_apply=True, p=1)
        ], p=1)
    ])

    # Apply the transformation
    transformed = transform_resize(image=img)
    img_r = transformed["image"]
        
    return img_r

def resize_transform_obj(img,
                         mask,
                         longest_min,
                         longest_max,
                         keypoints, bboxes,
                         transforms=False):

    """
    Function for resizing a single satellite image/mask from the dataset.
  
    Parameters:
    img (np.array): the input image of the satellite
    mask (np.array): the mask of the satellite, created through Photoshop.
    longest_min (int): minimum allowable size of the longest side of the 
                       mask.
    longest_max (int): maximum allowable size of the longest side of the 
                       mask.
    keypoints (np.array): keypoint coordinates in [x, y, visibility] format.
    bboxes (np.array): bounding box coordinates in [x1, y1, x2, y2] format.
    transforms (A.Compose): send in composition of transforms for the 
                            satellite mask.

    Returns:
    img_t (np.array): transformed image
    mask_t (np.array): transformed image mask
    keypoints_t (np.array): transformed keypoint coordinates
    bboxes_t (np.array): transformed bounding box coordinates
  
    """
   
    # Preserve proportional dimensions of the mask
    h, w = mask.shape[0], mask.shape[1]
    
    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new / longest))
    
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    keypoints_2 = [el[0:2] for kp in keypoints for el in kp]

    # Compose the resized image transformations, along with keypoints
    # and bounding box coordinates. 
    transform_resize = A.Compose([A.Resize(h_new,
                                           w_new,
                                           interpolation=1,
                                           always_apply=False,
                                           p=1)],
                                keypoint_params=A.KeypointParams(format='xy'),
                                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']))
    
    bboxes_labels = ['Delfi-n3xt' for _ in bboxes]

    # Apply transformation
    transformed_resized = transform_resize(image=img,
                                           mask=mask,
                                           keypoints=keypoints_2,
                                           bboxes=bboxes,
                                           bboxes_labels=bboxes_labels)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]
    keypoints_2_t = transformed_resized["keypoints"]   
    bboxes_t = transformed_resized["bboxes"]     

    # Apply additional transformations if required.
    if transforms:    
        transformed = transforms(image=img_t,
                                 mask=mask_t,
                                 keypoints=keypoints_2_t,
                                 bboxes=bboxes_t,
                                 bboxes_labels=bboxes_labels)
        img_t = transformed["image"]
        mask_t = transformed["mask"]
        keypoints_2_t = transformed["keypoints"]
        bboxes_t = transformed["bboxes"]

    keypoints_t = []
    
    keypoints_transformed_unflattened = np.reshape(np.array(keypoints_2_t), (-1,8,2)).tolist()

    # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
    for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
        obj_keypoints = []
        for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
            # kp - coordinates of keypoint
            # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
            obj_keypoints.append(kp + [keypoints[o_idx][k_idx][2]])
        keypoints_t.append(obj_keypoints)   
    
    keypoints_t = np.array(keypoints_t).astype(np.int32)

    return img_t, mask_t, keypoints_t, bboxes_t

def add_obj(img_comp, mask_comp, keypoints_comp, bboxes_comp, img, mask, keypoints, bboxes, x, y, idx):
    """
    Function for adding a single object to the image composition
  
    Parameters:
    img_comp (np.array): the input image of the background and satellite
    mask_comp (np.array): the mask of the satellite and current image composition
    keypoints_comp (np.array): keypoints of the current image composition
    img (np.array): image of object that you want to add
    mask (np.array): mask of object that you want to add
    keypoints (np.array): keypoint coordinates of object in [x, y, visibility] 
                          format. 
    x (int): x coordinate of the top left corner of the image being added
    y (int): y coordinate of the top left corner of the image being added
    idx (int): index of object number you wish to add to the current image.

    Returns:
    img_comp (np.array): composed image with added object
    mask_comp (np.array): mask of the object on the currently composed image
    keypoints_comp (np.array): keypoint coordinates relative to the current
                               image composition
    bboxes_t (np.array): bounding box coordinates relative to the current 
                         image composition
  
    """

    # collect composition and mask parameters
    h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]    
    h, w = img.shape[0], img.shape[1]   
    mask_b = mask == 1
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)
    
    # add image and mask to the composition
    img_comp[y:y+h, x:x+w, :] = img_comp[y:y+h, x:x+w, :] * ~mask_rgb_b + (img * mask_rgb_b)
    mask_comp[y:y+h, x:x+w] = mask_comp[y:y+h, x:x+w] * ~mask_b + (idx * mask_b)
    keypoints_2 = [el[0:2] for kp in keypoints for el in kp]
    # add the transformed keypoint (relative to composition grid)
    keypoints_comp.append([[kp[0] + x, kp[1] + y] for kp in keypoints_2])

    keypoints_comp_unflattened = np.reshape(np.array(keypoints_comp), (-1,8,2)).tolist()
    keypoints_comp = []

    # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
    for o_idx, obj in enumerate(keypoints_comp_unflattened): # Iterating over objects
        obj_keypoints = []
        for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
            # kp - coordinates of keypoint
            # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
            obj_keypoints.append(kp + [keypoints[o_idx][k_idx][2]])
        keypoints_comp.append(obj_keypoints)   

    # add the transformed bounding box (relative to composition grid)
    bboxes_comp.append([[bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y] for bbox in bboxes])
    
    
    return img_comp, mask_comp, keypoints_comp, bboxes_comp[0]

def visualize_composition(img_comp, keypoints_comp, bboxes_comp):

    """
    Function for visualizing the composition of the image with all added objects and
    associated bounding boxes and keypoints.
  
    Parameters:
    img_comp (np.array): the input image of the background and satellite
    mask_comp (np.array): the mask of the satellite and current image composition
    bboxes_comp (np.array): coordinates of bounding boxes in [x1, y1, x2, y2] given in
                            the composition grid.

    Returns:
    None
  
    """

    keypoints_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    if bboxes_comp:
        for bbox in bboxes_comp:
            start_point, end_point = tuple([int(bbox[0]), int(bbox[1])]), tuple([int(bbox[2]), int(bbox[3])])
            img_comp = cv2.rectangle(img_comp.copy(), start_point, end_point, (255,0,0), 2)
    
    for keypoints in keypoints_comp:
        for idx, kp in enumerate(keypoints):
            if kp[2] != 0:
                img_comp = cv2.circle(img_comp.copy(), tuple(kp[:2]), 3, (255,0,0), 6)
                img_comp = cv2.putText(img_comp.copy(), " " + keypoints_names[idx], tuple(kp[:2]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4, cv2.LINE_AA)
        
    plt.figure(figsize=(40,40))
    plt.imshow(img_comp)

def create_composition(img_comp_bg,
                       max_objs=1,
                       longest_min=300,
                       longest_max=700,
                       overlap_degree=0,
                       max_attempts_per_obj=1):

    """
    Function for creating a composition by resizing and adding images/masks in a loop
  
    Parameters:
    img_comp_bg (np.array): the background desired for the image composition
    max_objs (int): the maximum number of objects to add to the composition
    longest_min (int): minimum allowable size of the longest side of the 
                       mask.
    longest_max (int): maximum allowable size of the longest side of the 
                       mask.
    overlap_degree (int): how much are multiple objects allowed to overlap in the composition
    max_attempts_per_obj (int): how many times the image can be randomly added
                                before breaking the loop (due to too many overlaps)

    Returns:
    img_comp (np.array): image composition
    mask_comp (np.array): mask of the satellite in the composition
    keypoints_comp (np.array): keypoint coordinates relative to the current
                               image composition
    bboxes_comp (np.array): bounding box coordinates relative to the current
                            image composition
  
    """

    img_comp = img_comp_bg.copy()
    h, w = img_comp.shape[0], img_comp.shape[1]
    mask_comp = np.zeros((h,w), dtype=np.uint8)
    keypoints_comp = []
    bboxes_comp = []
    
    obj_areas = []
    num_objs = max_objs
    
    i = 1
    
    for _ in range(0, num_objs):
        imgs_number = len(image_files)
        idx = np.random.randint(imgs_number)
        
        img_path = image_files[idx]
        mask_path = masks_files[idx]
        keypoints_path = annotations_files[idx]
        
        # Keypoints and bounding boxes from satellite masks you wish to add to the image.
        img, mask = get_img_and_mask(img_path, mask_path)
        with open(keypoints_path) as f:
            data = json.load(f)
            keypoints = data['keypoints']
            bboxes = data['bboxes']
            
        # Resize the satellite mask that you wish to add to the image.
        img_t, mask_t, keypoints_t, bboxes_t = resize_transform_obj(img,
                                                            mask,
                                                            longest_min,
                                                            longest_max,
                                                            keypoints=keypoints,
                                                            bboxes=bboxes,
                                                            transforms=transforms_obj)
        x_max, y_max = img_comp.shape[1] - img_t.shape[1], img_comp.shape[0] - img_t.shape[0]
        x, y = np.random.randint(x_max), np.random.randint(y_max)

        # Add the resized mask to the background image composition
        img_comp, mask_comp, keypoints_comp, bboxes_comp = add_obj(img_comp,
                                                        mask_comp,
                                                        keypoints_comp,
                                                        bboxes_comp,
                                                        img_t,
                                                        mask_t,
                                                        keypoints_t,
                                                        bboxes_t,
                                                        x,
                                                        y,
                                                        i)
        
    return img_comp, mask_comp, keypoints_comp, bboxes_comp

def generate_bg_img(bg_img_files):
    """
    Function for generating a background image
  
    Parameters:
    bg_img_files (str): list of background image file paths

    Returns:
    img_comp_bg (np.array): resized background image
  
    """
    idx = randint(0, len(bg_img_files)-1)
    img_bg = np.array(PIL.Image.open(bg_img_files[idx]))
    img_comp_bg = resize_img(img_bg)
    return img_comp_bg

def generate_dataset(imgs_number, folder, split='train'):
    """
    Loop that randomly generates image composition and associated annotations,
    and saves them to a training folder.
  
    Parameters:
    imgs_number (int): number of images you wish to generate
    folder (str): desired output path of the synthetic generator
    split (str): subfolder in the synthetic dataset

    Returns:
    None
  
    """
    time_start = time.time()
    for j in tqdm(range(imgs_number)):
        # Create background image
        img_comp_bg = generate_bg_img(bg_files)

        # Create total composition
        img_comp, mask_comp, keypoints_comp, bboxes_comp = create_composition(img_comp_bg, max_objs=1,
                                                                 overlap_degree=0, longest_max=2500, longest_min=2000,
                                                                 max_attempts_per_obj=10)

        # Save the image composition
        img_comp = PIL.Image.fromarray(img_comp)
        img_comp.save(os.path.join(folder, split, 'images/{}.jpg').format(j))

        # Generate annotations for the image composition
        annotations = {}
        name = "SIMG_" + str(j)
        keypoints_comp = np.array(keypoints_comp, dtype=int)
        annotations['name'], annotations['bboxes'], annotations['keypoints'] = name, bboxes_comp, keypoints_comp.tolist()
        with open(os.path.join(folder, split, 'annotations/{}.json').format(j), "w") as f:
            json.dump(annotations, f)    
        
    time_end = time.time()
    time_total = round(time_end - time_start)
    time_per_img = round((time_end - time_start) / imgs_number, 1)
    
    print("Generation of {} synthetic images is completed. It took {} seconds, or {} seconds per image".format(imgs_number, time_total, time_per_img))
    print("Images are stored in '{}'".format(os.path.join(folder, split, 'images')))
    print("Annotations are stored in '{}'".format(os.path.join(folder, split, 'annotations')))

#################################################################################################################
######################################### ADJUSTABLE PARAMETERS #################################################
#################################################################################################################  

# Specify MAIN_PATH relative to your local root.
MAIN_PATH = "synthetic_dataset"

image_files = sorted(os.listdir(os.path.join(MAIN_PATH, "images")))
image_files = [os.path.join(MAIN_PATH, 'images', f) for f in image_files]
annotations_files = sorted(os.listdir(os.path.join(MAIN_PATH, "annotations")))
annotations_files = [os.path.join(MAIN_PATH, 'annotations', f) for f in annotations_files]
masks_files = sorted(os.listdir(os.path.join(MAIN_PATH, "masks")))
masks_files = [os.path.join(MAIN_PATH, 'masks', f) for f in masks_files]
bg_files = os.listdir(os.path.join(MAIN_PATH, 'backgrounds'))
bg_files = [os.path.join(MAIN_PATH, 'backgrounds', f) for f in bg_files]

transforms_obj = A.Compose([
    A.RandomRotate90(p=1),
],
keypoint_params=A.KeypointParams(format='xy'),
bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']))

generate_dataset(100, folder='synthetic_dataset', split='train')

#################################################################################################################
###################################### END OF ADJUSTABLE PARAMETERS #############################################
#################################################################################################################  