from torchvision import transforms
import numpy as np
import random
from scipy import signal

# data transform for image
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# generate a gaussion on points in a map with im_shap
def get_paste_kernel(im_shape, points, kernel, shape=(224 // 4, 224 // 4)):
    # square kernel
    k_size = kernel.shape[0] // 2
    x, y = points
    image_height, image_width = im_shape[:2]
    x, y = int(round(image_width * x)), int(round(y * image_height))
    x1, y1 = x - k_size, y - k_size
    x2, y2 = x + k_size, y + k_size
    h, w = shape
    if x2 >= w:
        w = x2 + 1
    if y2 >= h:
        h = y2 + 1
    heatmap = np.zeros((h, w))
    left, top, k_left, k_top = x1, y1, 0, 0
    if x1 < 0:
        left = 0
        k_left = -x1
    if y1 < 0:
        top = 0
        k_top = -y1

    heatmap[top:y2+1, left:x2+1] = kernel[k_top:, k_left:]
    return heatmap[0:shape[0], 0:shape[0]]



def gkern(kernlen=51, std=9):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

kernel_map = gkern(21, 3)


