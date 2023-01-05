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

# Image augmentations
def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0) # Try with ImageNet statistics. If it doesn't work, recalculate means using our dataset.
            # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )

# Dataset class
class ClassDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])

        img_original = np.array(PIL.Image.open(img_path))
        # if np.shape(img_original)[1] == 3024:
        #     img_original = cv2.flip(cv2.transpose(img_original), 0)
            
        # I want to be able to visualize the incoming image here, to make sure everything is annotated correctly.   
        
        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']

            # cols = 4032
            # rows = 3024
            # for j in range(0, len(bboxes_original)):
            #     for k in range(0, len(bboxes_original[j])):
            #         if k % 2 == 0:
            #             bboxes_original[j][k] /= cols
            #         elif k % 2 == 1:
            #             bboxes_original[j][k] /= rows
            
            # for j in range(0, len(keypoints_original)):
            #     for k in range(0, len(keypoints_original[j])):
            #         keypoints_original[j][k][0] /= cols
            #         keypoints_original[j][k][1] /= rows

            
            # All objects are glue tubes
            bboxes_labels_original = ['Delfi-n3xt' for _ in bboxes_original]            

        if self.transform:   
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints            
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format            
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            

            # Apply augmentations
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

            vis_original = img

            for bbox in bboxes:
                bbox = list(bbox)
                start_point = (int(bbox[0]), int(bbox[1]))
                end_point = (int(bbox[2]), int(bbox[3]))
                vis_original = cv2.rectangle(vis_original.copy(), start_point, end_point, (0,255,0), 2)

            keypoints_classes_ids2names = {0: '1', 1: '2', 2: '3', 3: '4'}
            for kps in keypoints:
                 for idx, kp in enumerate(kps):
                    vis_original = cv2.circle(vis_original.copy(), tuple(kp[0:2]), 5, (255,0,0), 10)
                    vis_original = cv2.putText(img=vis_original.copy(), text=" " + keypoints_classes_ids2names[idx], org=tuple(kp[:2]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,0,0), thickness=3, lineType=cv2.LINE_AA)
            
            # plt.figure(figsize=(40,40), clear=True)
            # plt.imshow(vis_original)
            # plt.clf()

            # img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        
        else:
            img_original = cv2.normalize(img_original, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original   
            # img, keypoints = img_original, keypoints_original     
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are glue tubes
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)
        
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


# Visualizing a random item from dataset
KEYPOINTS_FOLDER_TRAIN = 'train_dataset'
dataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader)
batch = next(iterator)

print("Original targets:\n", batch[3], "\n\n")
print("Transformed targets:\n", batch[1])

keypoints_classes_ids2names = {0: '1', 1: '2', 2: '3', 3: '4'}


def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
         start_point = (bbox[0], bbox[1])
         end_point = (bbox[2], bbox[3])
         image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            if kp[2] != 0:
                image = cv2.circle(image.copy(), tuple(kp[:2]), 5, (255,0,0), 10)
                image = cv2.putText(img=image.copy(), text=" " + keypoints_classes_ids2names[idx], org=tuple(kp[:2]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,0,0), thickness=3, lineType=cv2.LINE_AA)

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40,40), clear=True)
        plt.imshow(image)
        plt.clf()

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

    plt.imsave('outputs/test.png', image)

def output(keypoints):
    kp_out = {'keypoint_1': keypoints[0], 
            'keypoint_2': keypoints[1],
            'keypoint_3': keypoints[2],
            'keypoint_4': keypoints[3]}
    outfile = 'outputs/test.json'
    with open(outfile, 'w') as f:
        json.dump(f, kp_out)

def plot_results(num_epochs, training_loss):
    plt.plot(range(0, num_epochs), training_loss)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss")
    plt.savefig("outputs/test_loss.png")
    plt.show()
    plt.clf()
        
image = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

keypoints = []
for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
# for kp in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    # keypoints.append([kp[:2]])
    keypoints.append([kp[:2] for kp in kps])

image_original = (batch[2][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

keypoints_original = []
for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
# for kp in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    # keypoints.append([kp[:2]])
    keypoints_original.append([kp[:2] for kp in kps])

# visualize(image, bboxes, keypoints, image_original, bboxes_original, keypoints_original)
# visualize(image, keypoints, image_original, keypoints_original)

# Training model

def get_model(num_keypoints, weights_path=None):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=False,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

KEYPOINTS_FOLDER_TRAIN = 'train_dataset'
KEYPOINTS_FOLDER_TEST = 'test_dataset'

dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)
dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=5, shuffle=True, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, collate_fn=collate_fn)

model = get_model(num_keypoints = 4)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
num_epochs = 5

training_loss = []

for epoch in range(num_epochs):
    metric_logger = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
    training_loss.append(metric_logger.meters["loss_keypoint"].global_avg)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device)

    
# Save model weights after training
torch.save(model.state_dict(), 'keypointsrcnn_weights_scratchtrain.pth')

plot_results(num_epochs, training_loss)

# Visualizing model predictions
iterator = iter(data_loader_test)
images, targets = next(iterator)
images = list(image.to(device) for image in images)

with torch.no_grad():
    model.to(device)
    model.eval()
    output = model(images)

print("Predictions: \n", output)
image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
scores = output[0]['scores'].detach().cpu().numpy()

high_scores_idxs = nlargest(2, range(len(scores)), key=lambda idx: scores[idx])

# high_scores_idxs = np.where(scores > 0)[0].tolist() # Indexes of boxes with scores > 0.7
# post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

# Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
# Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
# Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

keypoints = []
# for kp in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    # keypoints.append([list(map(int, kp[:2]))])
for kps in output[0]['keypoints'][high_scores_idxs].detach().cpu().numpy():
    keypoints.append([list(map(int, kp)) for kp in kps])
# for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    # keypoints.append([list(map(int, kp[:2])) for kp in kps])

bboxes = []
for bbox in output[0]['boxes'][high_scores_idxs].detach().cpu().numpy():
    bboxes.append(list(map(int, bbox.tolist())))
# for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
#     bboxes.append(list(map(int, bbox.tolist())))
    

visualize(image, bboxes, keypoints)

output(keypoints[0])



