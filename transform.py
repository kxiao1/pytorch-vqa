"""
Generate additional training data by introducing modifications to the images

Modifications:
- Blurring (slight)
- Rotation
- Scaling
- Gaussian salt/pepper
"""

from PIL import Image
from time import time
import random

import torchvision
import numpy as np
import os
from torchvision import transforms
from torchvision.utils import save_image
from skimage.transform import rotate, AffineTransform, warp
from skimage.filters import gaussian
from skimage.util import random_noise, crop
import json
import skimage.io as io
import torch

from multiprocessing import Process

def load_image_from_file(file):
    image = Image.open(file)
    tensor = image
    # tensor = transforms.ToTensor()(image)
    # print(tensor.shape)
    return tensor

# def rotate(image):
#     pass

def all_transforms():
    return transforms.RandomApply([
        # transforms.RandomHorizontalFlip(p=1.0),
        transforms.GaussianBlur(kernel_size=5, sigma=10.0),
        # CustomRescalingTransform([-5, 5], [0.85, 0.95], [0.85, 0.95])
    ], p=1.0)

class CustomRescalingTransform(torch.nn.Module):
    """
    The purpose of this transformation is to crop an image with a given
    _relative_ scale on the x and y dimensions. Pytorch only natively supports
    scaling based upon the _absolute_ x and y dimensions.

    This is useful to avoid black bars in rotation.
    """

    def __init__(self, rotation, scale_x, scale_y):
        super().__init__()
        self.rotation = rotation
        self.scale_x = scale_x
        self.scale_y = scale_y

    def forward(self, img):
        rotation = random.uniform(self.rotation[0], self.rotation[1])
        rotated = transforms.RandomRotation([rotation, rotation + 0.00001])(img)

        scale_x = random.uniform(self.scale_x[0], self.scale_y[1])
        scale_y = random.uniform(self.scale_y[0], self.scale_y[1])

        rotation *= .01

        if abs(rotation) > 1 - scale_x:
            scale_x -= abs(rotation)
        if abs(rotation) > 1 - scale_y:
            scale_y -= abs(rotation)

        width, height = img.width, img.height
        return transforms.CenterCrop([int(height * scale_x),
            int(width * scale_y)])(rotated)

def random_augmentation(image, blur_percent=0.5, pepper_percent=0.5, crop_percent=0.5, rotation_percent=0.5):
    pepper_sigma = np.random.normal(0.100, 0.010)
    blur_sigma = np.random.normal(1, 0.1)
    rotate_angle = np.random.normal(0, 5)

    image_x = image.shape[0]
    image_y = image.shape[1]
    delta_x_percent = tuple([int(image_x * abs(np.random.normal(0, 0.03))) for _ in range(2)])
    delta_y_percent = tuple([int(image_y * abs(np.random.normal(0, 0.03))) for _ in range(2)])

    if random.random() < pepper_percent:
        image = random_noise(image, var=pepper_sigma**2)
    if random.random() < blur_percent:
        image = gaussian(image, sigma=blur_sigma, multichannel=True)
    if random.random() < rotation_percent:
        image = rotate(image, angle=rotate_angle, mode='reflect')
    if random.random() < crop_percent:
        image = crop(image, (delta_x_percent, delta_y_percent, (0, 0)), copy=False)
    
    return image


def main():
    # m = 4 # NUMBER OF THREADS TO SPAWN
    # n = 3 # NUMBER OF AUGMENTATIONS TO GENERATE
    # suffix = "augmented"
    # processes = []

    # start = time()

    # for i in range(m):
    #     p = Process(target=run_thread, args=(i, m, n, suffix))
    #     processes.append(p)
    #     p.start()
    
    # for p in processes:
    #     p.join()

    # end = time()

    # elapsed = end - start
    # print(f"Time: {elapsed}")
        
    generate_augmentation(3)
    # test_image = load_image_from_file("vizwiz/train/VizWiz_train_00000228.jpg")
    # test_image_transformed = all_transforms()(test_image)
    # test_image_transformed = transforms.ToTensor()(test_image_transformed)
    # save_image(test_image_transformed, "transform_tmp.jpg")
    # image = io.imread("vizwiz/train/VizWiz_train_00000228.jpg")


    # (1) Salt and pepper
    # sigma = 0.155
    # sigma = 0.3
    # transformed = random_noise(image, var=sigma ** 2)

    # (2)
    # transformed = gaussian(image, sigma=5, multichannel=True)

    # (3)
    # transformed = rotate(image, angle=-10, mode='reflect')

    # (4)
    # transform = AffineTransform(translation=(25,25))
    # transformed = warp(image, transform, mode='wrap')
    # print(image.shape)

    # transformed = random_augmentation(image)
    # io.imsave("transform_tmp.jpg", transformed)

# Generate augmented images with provided image number and suffix.
# Will make original image plus n augmented ones.
# Uses numbers (n + 1)image_number -> (n + 1)image_number + n
# Writes to train_{suffix}
def generate_augmented_image(image_number, n, suffix="augmented"):
    print(f"Augmenting image {image_number}")
    image_file = f"VizWiz_train_{image_number:08}.jpg"
    image = io.imread(f"vizwiz/train_all/{image_file}")

    for i in range(n + 1):
        curr_index = (n + 1) * image_number + i

        # New file
        aug_image_file = f'VizWiz_train_{curr_index:08}.jpg'

        # Create and save an augmented or original image
        if i == 0:
            aug_image = image
        else:
            aug_image = random_augmentation(image)
        io.imsave(f"vizwiz/train_{suffix}/{aug_image_file}", aug_image)

# image number in train goes up to 23953
# run thread with index i in {0, 1, ..., m - 1} modulo m
# make n augmentations
def run_thread(i, m, n, suffix):
    for j in range(i, 23954, m):
        generate_augmented_image(j, n, suffix)

# Generate n new training examples for every image, along with the original image.
# Save the output to vizwiz/Annotations_{suffix}
# Only training data will be changed; validation and test data are not augmented.
def generate_augmentation(n=5, suffix="augmented"):
    with open("vizwiz/Annotations_all/train.json") as f:
        train_data = f.read()
        train_json = json.loads(train_data)
    with open("vizwiz/Annotations_all/test.json") as f:
        test_data = f.read()
    with open("vizwiz/Annotations_all/val.json") as f:
        val_data = f.read()

    for prefix in ["train", "test", "val", "Annotations"]:
        if not os.path.exists(f"vizwiz/{prefix}_{suffix}"):
            os.mkdir(f"vizwiz/{prefix}_{suffix}")
    
    res_json = []
    curr_index = 0
    
    for image_number, example in enumerate(train_json):
        # Format:
        # {
        #   "image": "VizWiz_val_00000000.jpg",
        #   "question": "Ok. There is another picture I hope it is a better one.",
        #   "answers": [
        #     {
        #     "answer": "unanswerable",
        #     "answer_confidence": "yes"
        #     },
        #     ...
        #   ]
        # }

        for i in range(n + 1):
            # New file
            aug_image_file = f'VizWiz_train_{curr_index:08}.jpg'
            # generate_augmented_image(image_number, n, suffix)

            # Create and save a corresponding annotation
            curr_json = {}
            curr_json["image"] = aug_image_file
            curr_json["question"] = example["question"]
            curr_json["answers"] = example["answers"]
            res_json.append(curr_json)
            curr_index += 1
        
    # Write the new train json file to disk, and copy over the old
    # test and validation files to the new directory.
    res_json_str = json.dumps(res_json)
    with open(f"vizwiz/Annotations_{suffix}/train.json", "w+") as f:
        f.write(res_json_str)
    with open(f"vizwiz/Annotations_{suffix}/test.json", "w+") as f:
        f.write(test_data)
    with open(f"vizwiz/Annotations_{suffix}/val.json", "w+") as f:
        f.write(val_data)

if __name__ == "__main__":
    main()