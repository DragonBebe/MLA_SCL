import numpy as np
import torch
from torchvision import transforms  # 添加这行以导入 transforms

class Cutout:
    """
    A custom implementation of Cutout data augmentation.

    Parameters:
        n_holes (int): Number of holes to cut out from the image.
        length (int): Length of each square hole.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Apply Cutout to the input image.

        Parameters:
            img (torch.Tensor or PIL.Image): Input image.

        Returns:
            torch.Tensor: Augmented image with holes cut out.
        """
        # If the input is a PIL.Image, convert to torch.Tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)  # Convert to Tensor for processing

        # Get height and width of the image
        _, h, w = img.size()

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            img[:, y1:y2, x1:x2] = 0

        return img
