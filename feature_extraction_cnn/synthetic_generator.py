#################################################################################
########################## SYNTHETIC IMAGE GENERATOR ############################
#################################################################################


# This file generates synthetically transformed versions of the original dataset
# images, and stores their keypoint transforms into corresponding .json objects 
# for addition into the training dataset.

import albumentations as A
import numpy as np
import PIL
import cv2
import matplotlib.pyplot as plt
import os, json
import time
from tqdm import tqdm
from random import randint

MAIN_PATH = "synthetic_dataset"

image_files = sorted(os.listdir(os.path.join(MAIN_PATH, "images")))
image_files = [os.path.join(MAIN_PATH, 'images', f) for f in image_files]
annotations_files = sorted(os.listdir(os.path.join(MAIN_PATH, "annotations")))
annotations_files = [os.path.join(MAIN_PATH, 'annotations', f) for f in annotations_files]
masks_files = sorted(os.listdir(os.path.join(MAIN_PATH, "masks")))
masks_files = [os.path.join(MAIN_PATH, 'masks', f) for f in masks_files]
bg_files = os.listdir(os.path.join(MAIN_PATH, 'backgrounds'))
bg_files = [os.path.join(MAIN_PATH, 'backgrounds', f) for f in bg_files]

def get_img_and_mask(img_path, mask_path):
    img = np.array(PIL.Image.open(img_path))
    mask = np.array(PIL.Image.open(mask_path))
    mask_b = mask[:,:,0] == 0
    mask = mask_b.astype(np.uint8)

    return img, mask

def visualize_single_img(img, mask, keypoints, keypoints_names, bboxes, title):
    for bbox in bboxes:
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        img = cv2.rectangle(img.copy(), start_point, end_point, (0,255,0), 2)
    for idx, kps in enumerate(keypoints):
        for idx, kp in enumerate(kps):
            img = cv2.circle(img.copy(), tuple(kp[:2]), 3, (255,0,0), 6)
            img = cv2.putText(img.copy(), " " + keypoints_names[idx], tuple(kp[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    
    plt.figure(figsize=(16,16))
    plt.title(title, fontsize=18)
    plt.imshow(img)
    plt.clf()

def resize_img(img, desired_max=4032, desired_min=3024):
   
    h, w = img.shape[0], img.shape[1]
    
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
        
    transform_resize = A.Compose([
        A.Sequential([
        A.Resize(h_new, w_new, interpolation=1, always_apply=True, p=1)
        ], p=1)
    ])

    transformed = transform_resize(image=img)
    img_r = transformed["image"]
        
    return img_r

def resize_transform_obj(img,
                         mask,
                         longest_min,
                         longest_max,
                         keypoints, bboxes,
                         transforms=False):
   
    h, w = mask.shape[0], mask.shape[1]
    
    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new / longest))
    
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    keypoints_2 = [el[0:2] for kp in keypoints for el in kp]
        
    transform_resize = A.Compose([A.Resize(h_new,
                                           w_new,
                                           interpolation=1,
                                           always_apply=False,
                                           p=1)],
                                keypoint_params=A.KeypointParams(format='xy'),
                                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']))
    
    bboxes_labels = ['Delfi-n3xt' for _ in bboxes]
    transformed_resized = transform_resize(image=img,
                                           mask=mask,
                                           keypoints=keypoints_2,
                                           bboxes=bboxes,
                                           bboxes_labels=bboxes_labels)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]
    keypoints_2_t = transformed_resized["keypoints"]   
    bboxes_t = transformed_resized["bboxes"]     

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
    
    keypoints_transformed_unflattened = np.reshape(np.array(keypoints_2_t), (-1,4,2)).tolist()

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
    '''
    img_comp - composition of objects
    mask_comp - composition of objects` masks
    keypoints_comp - composition of keypoints
    img - image of object
    mask - mask of object
    keypoints - keypoints of object
    x, y - coordinates where left top corner of img is placed
    Function returns img_comp in CV2 RGB format + mask_comp + keypoints_comp as a list
    '''
    h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]    
    h, w = img.shape[0], img.shape[1]   
    mask_b = mask == 1
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)
    
    img_comp[y:y+h, x:x+w, :] = img_comp[y:y+h, x:x+w, :] * ~mask_rgb_b + (img * mask_rgb_b)
    mask_comp[y:y+h, x:x+w] = mask_comp[y:y+h, x:x+w] * ~mask_b + (idx * mask_b)
    keypoints_2 = [el[0:2] for kp in keypoints for el in kp]
    keypoints_comp.append([[kp[0] + x, kp[1] + y] for kp in keypoints_2])

    keypoints_comp_unflattened = np.reshape(np.array(keypoints_comp), (-1,4,2)).tolist()
    keypoints_comp = []
    # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
    for o_idx, obj in enumerate(keypoints_comp_unflattened): # Iterating over objects
        obj_keypoints = []
        for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
            # kp - coordinates of keypoint
            # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
            obj_keypoints.append(kp + [keypoints[o_idx][k_idx][2]])
        keypoints_comp.append(obj_keypoints)   

    # keypoints_comp.append([[kp[0] + x, kp[1] + y, kp[2]] for kp in keypoints])
    bboxes_comp.append([[bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y] for bbox in bboxes])
    
    
    return img_comp, mask_comp, keypoints_comp, bboxes_comp[0]

def visualize_composition(img_comp, keypoints_comp, keypoints_names, bboxes_comp):
    
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
                       max_objs=15,
                       longest_min=300,
                       longest_max=700,
                       overlap_degree=0,
                       max_attempts_per_obj=1):

    img_comp = img_comp_bg.copy()
    h, w = img_comp.shape[0], img_comp.shape[1]
    mask_comp = np.zeros((h,w), dtype=np.uint8)
    keypoints_comp = []
    bboxes_comp = []
    
    obj_areas = []
    num_objs = 1
    
    i = 1
    
    for _ in range(0, num_objs):
        imgs_number = len(image_files)
        idx = np.random.randint(imgs_number)
        
        img_path = image_files[idx]
        mask_path = masks_files[idx]
        keypoints_path = annotations_files[idx]
        
        img, mask = get_img_and_mask(img_path, mask_path)
        with open(keypoints_path) as f:
            data = json.load(f)
            keypoints = data['keypoints']
            bboxes = data['bboxes']
            
        img_t, mask_t, keypoints_t, bboxes_t = resize_transform_obj(img,
                                                            mask,
                                                            longest_min,
                                                            longest_max,
                                                            keypoints=keypoints,
                                                            bboxes=bboxes,
                                                            transforms=transforms_obj)
        x_max, y_max = img_comp.shape[1] - img_t.shape[1], img_comp.shape[0] - img_t.shape[0]
        x, y = np.random.randint(x_max), np.random.randint(y_max)

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
    idx = randint(0, len(bg_img_files)-1)
    img_bg = np.array(PIL.Image.open(bg_img_files[idx]))
    img_comp_bg = resize_img(img_bg)
    return img_comp_bg

def generate_dataset(imgs_number, folder, split='train'):
    time_start = time.time()
    for j in tqdm(range(imgs_number)):
        img_comp_bg = generate_bg_img(bg_files)
        img_comp, mask_comp, keypoints_comp, bboxes_comp = create_composition(img_comp_bg, max_objs=1,
                                                                 overlap_degree=0, longest_max=2500, longest_min=2000,
                                                                 max_attempts_per_obj=10)

        img_comp = PIL.Image.fromarray(img_comp)
        img_comp.save(os.path.join(folder, split, 'images/{}.jpg').format(j))

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

transforms_obj = A.Compose([
    A.RandomRotate90(p=1),
    # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2),
    #                            contrast_limit=0.1,
    #                            brightness_by_max=True,
    #                            always_apply=False,
    #                            p=1)
],
keypoint_params=A.KeypointParams(format='xy'),
bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']))

keypoints_names = ['1', '2', '3', '4']

img_num = randint(0, len(image_files)-1)
img_path = image_files[img_num]
mask_path = masks_files[img_num]
img, mask = get_img_and_mask(img_path, mask_path)
with open(annotations_files[img_num]) as f:
    data = json.load(f)
    keypoints = np.array(data['keypoints'])
    bboxes = np.array(data['bboxes'])
f.close()
    
print("Keypoints:", keypoints)
print("Bounding Boxes", bboxes)

visualize_single_img(img, mask, keypoints, keypoints_names, bboxes, title="Keypoints and bounding boxes")

bg_num = randint(0, len(bg_files)-1)
img_bg_path = bg_files[bg_num]
img_bg = np.array(PIL.Image.open(img_bg_path))

img_bg_resized_1 = resize_img(img_bg, desired_max=4032, desired_min=3024)

print("Shape of the original background image:", img_bg.shape)

print("Shape of the resized background image (desired_max=4032, desired_min=3024):", img_bg_resized_1.shape)

plt.figure(figsize=(40,40))
plt.imshow(img_bg_resized_1)
plt.title('Resized (desired_max=4032, desired_min=3024)', fontsize=18)
plt.clf()

# img_path = image_files[img_num]
# mask_path = masks_files[img_num]
# img, mask = get_img_and_mask(img_path, mask_path)

# with open(annotations_files[img_num]) as f:
#     data = json.load(f)
#     keypoints = data['keypoints']
#     bboxes = data['bboxes']
# f.close()

img_t, mask_t, keypoints_t, bboxes_t = resize_transform_obj(img,
                                                  mask,
                                                  longest_min=2000,
                                                  longest_max=2500,
                                                  keypoints=keypoints,
                                                  bboxes=bboxes,
                                                  transforms=transforms_obj)

print("\nShape of the image of the transformed object:", img_t.shape)
print("Shape of the transformed binary mask:", img_t.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 7))
ax[0].imshow(img_t)
ax[0].set_title('Transformed object', fontsize=18)
ax[1].imshow(mask_t)
ax[1].set_title('Transformed binary mask', fontsize=18)

visualize_single_img(img_t,
                         mask_t,
                         keypoints_t,
                         keypoints_names,
                         bboxes_t,
                         title="Keypoints of the transformed object")

h, w = img_bg_resized_1.shape[0], img_bg_resized_1.shape[1]
mask_comp = np.zeros((h,w), dtype=np.uint8)
keypoints_comp = []
bboxes_comp = []

img_comp, mask_comp, keypoints_comp, bboxes_comp = add_obj(img_bg_resized_1,
                                              mask_comp,
                                              keypoints_comp,
                                              bboxes_comp,
                                              img_t,
                                              mask_t,
                                              keypoints_t,
                                              bboxes_t,
                                              x=100,
                                              y=100,
                                              idx=1)

fig, ax = plt.subplots(1, 2, figsize=(16, 7))
ax[0].imshow(img_comp)
ax[0].set_title('Composition', fontsize=18)
ax[1].imshow(mask_comp)
ax[1].set_title('Composition mask', fontsize=18)

print("Keypoints:", keypoints_comp)
print("Bounding boxes:", bboxes_comp)
visualize_composition(img_comp, keypoints_comp, keypoints_names, bboxes_comp)

generate_dataset(500, folder='synthetic_dataset', split='train')