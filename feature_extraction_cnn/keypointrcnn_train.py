#################################################################################################################
######################################### FEATURE EXTRACTION TRAINING ###########################################
#################################################################################################################  

'''
keypointrcnn_train.py

Primary training code for the keypoint CNN. Run this file to train a new network architecture, which evaluates it
as you progress. The output is saved into an output .pth file which can then be used for evaluation with
evaluate.py.

Adjustable parameters are found at the end of the file in a clearly labelled section. 

Author: Adyn Miles

Majority of code and ideas taken from Alex P in the following blog post: 
https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da

'''

import os, json, cv2, numpy as np, matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A

import PIL
import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate
from heapq import nlargest

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
    def __init__(self, root, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)

        # Ensure that your training and testing dataset images are stored under respective folders called "images" and "annotations".
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
    
    # Function to get and prepare an image for use in the model.
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
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

            # This snippet of code is meant to error handle transformations to bounding boxes that are [0, 0, 1, 1], the convention we
            # used to label a bounding box that was not present in the image. This code may be removed in future iterations because it
            # was revealed that these bounding boxes would intensely skew the output data.
            for idx in range(0, len(bboxes)):
                bboxes[idx] = list(bboxes[idx])
                if abs(bboxes[idx][0] - bboxes[idx][2]) <= 2 or abs(bboxes[idx][1] - bboxes[idx][3]) <= 2:
                    bboxes[idx][0] = 0.0
                    bboxes[idx][1] = 0.0
                    bboxes[idx][2] = 1.0
                    bboxes[idx][3] = 1.0
                bboxes[idx] = tuple(bboxes[idx])

            
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

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs_files)

def train_transform():
    """
    Function for composing the transforms for training images. This is a process called dataset augmentation,
    to expose the network to new images upon each dataset iteration.
  
    Parameters:
    None
  
    Returns:
    A.Compose(): returns sequence of transformations consisting of normalization, random rotation, and the 
                 associated transformations of keypoints and bounding boxes.
  
    """
    return A.Compose([
        A.Sequential([
            A.augmentations.geometric.resize.LongestMaxSize(max_size=1440, interpolation=cv2.INTER_LINEAR, always_apply=True, p=1),
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0) # Normalizing the dataset using ImageNet statistics.
            # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
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

        ax[0].imshow(image_original.astype(np.uint8))
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image.astype(np.uint8))
        ax[1].set_title('Transformed image', fontsize=fontsize)
        plt.clf()


def plot_results(num_epochs, training_loss):
    """
    Function for producing and saving a plot of training loss over each training epoch, a useful tool in 
    network debugging and systematic improvement. The keypoint loss is the loss parameter being tracked.
  
    Parameters:
    num_epochs (int): The number of epochs used for training
    training_loss (np.array, dtype=float): Keypoint loss from each epoch as measured by Keypoint CNN.
  
    Returns:
    None
  
    """
    plt.plot(range(0, num_epochs), training_loss)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss")
    plt.savefig("outputs/test_loss.png")
    plt.show()
    plt.clf()

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

if __name__ == '__main__':
    # Some preamble to ensure there is enough GPU memory (RAM) allocated for Pytorch.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'
    torch.cuda.device_count()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Use multi-processing if available to reduce training time
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    #################################################################################################################
    ######################################### ADJUSTABLE PARAMETERS #################################################
    #################################################################################################################  

    # Specify the relative path of the training and testing datasets.
    KEYPOINTS_FOLDER_TRAIN = 'train_dataset'
    KEYPOINTS_FOLDER_TEST = 'test_dataset'

    # Initialize the datasets for the training and testing sets. Set demo=True if transform visualization features are
    # desired.
    dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)
    dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

    # Load in images to train and evaluate the network. Here, the batch_size (the number of images loaded into the
    # network at a time) can be adjusted here if desired.
    # The datasets can also be not shuffled if that is desired by using shuffle=False.
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Initialize the model with the correct number of keypoints (counted per bounding box). This number can be adjusted 
    # based on the dataset being passed in.
    model = get_model(num_keypoints = 4)
    model.to(device)

    # Network parameters that can be varied to adjust performance of the optimizer and the learning rate scheduler.
    # lr: The learning rate, in stochastic gradient descent, refers to how drastically the model will adjust 
    #     parameters to reach a certain optimum. A low learning rate may never reach optimum or get trapped in a
    #     local optimum, while a high one may pass over optima and provide unpredictable behaviour.
    # momentum: Momentum takes advantage of the model's progression towards a minima to help it reach that target
    #           faster and make the descent more predictable. This value likely does not need to be changed.
    # weight_decay: Penalizes the model for getting too complex and overfitting the training data. This likely
    #               does not need to be changed.
    # step_size: The LR scheduler method adjusts the learning rate after a certain amount of epochs to better adapt
    #            to the problem as its knowledge gets more refined and it nears an optimum. Step size is the number
    #            of epochs after which the learning rate should be dropped.
    # gamma: This refers to the proportion at which the learning rate should be dropped.
    # num_epochs: Total number of epochs to train the model. An epoch is complete once the network has done one iteration
    #             of learning on the training set. More epochs is generally good for better performance, but takes
    #             much more time to train.

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    num_epochs = 20
    #################################################################################################################
    ###################################### END OF ADJUSTABLE PARAMETERS #############################################
    #################################################################################################################  

    training_loss = []

    # Training and evaluation loop. Most of the neural network magic happens in these four lines, with a lot under the hood.
    for epoch in range(num_epochs):
        metric_logger = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        training_loss.append(metric_logger.meters["loss_keypoint"].global_avg)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device)

        
    # Save model weights after training. Change this path or the currently saved file if you do not wish to overwrite the 
    # previously saved weights.
    torch.save(model.state_dict(), 'keypointsrcnn_weights.pth')

    # Plot the training loss
    plot_results(num_epochs, training_loss)

    # This code can also be used to get an initial view of how your network will evaluate images. A better analysis can be
    # done in evaluate.py.
    iterator = iter(data_loader_test)
    images, targets = next(iterator)
    images = list(image.to(device) for image in images)

    # Evaluate the images using the model and record its output.
    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model(images)

    print("Predictions: \n", output)
    image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    scores = output[0]['scores'].detach().cpu().numpy()

    # Choose the bounding boxes corresponding to the two highest confidence scores of the network.
    high_scores_idxs = nlargest(2, range(len(scores)), key=lambda idx: scores[idx])

    keypoints = []
    # Save the keypoints in a format so that they can be visualized.
    for kps in output[0]['keypoints'][high_scores_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp)) for kp in kps])

    bboxes = []
    # Save the bounding boxes in a format so that they can be visualized.
    for bbox in output[0]['boxes'][high_scores_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))
        
    # Visualize a sample network prediction.
    visualize(image, bboxes, keypoints)