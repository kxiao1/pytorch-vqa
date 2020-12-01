"""
Generate additional training data by introducing modifications to the images

Modifications:
- Blurring (slight)
- Rotation
- Scaling
- Gaussian salt/pepper
"""

from PIL import Image
import random

import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torch

def load_image_from_file(file):
    image = Image.open(file)
    tensor = image
    # tensor = transforms.ToTensor()(image)
    # print(tensor.shape)
    return tensor

def rotate(image):
    pass

def all_transforms():
    return transforms.RandomApply([
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        CustomRescalingTransform([-5, 5], [0.85, 0.95], [0.85, 0.95])
    ], p=0.2)

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

def main():
    test_image = load_image_from_file("vizwiz/train/VizWiz_train_00000228.jpg")
    test_image_transformed = all_transforms()(test_image)
    test_image_transformed = transforms.ToTensor()(test_image_transformed)
    save_image(test_image_transformed, "transform_tmp.jpg")

if __name__ == "__main__":
    main()