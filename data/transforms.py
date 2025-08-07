import cv2
from PIL import ImageOps
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

class GaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()).mul_(self.std).add_(self.mean)
        return tensor.add_(noise)

class GrayscaleToRGB:
    def __call__(self, img):
        if len(img.getbands()) == 1:  # If image is grayscale
            img = transforms.functional.to_pil_image(np.stack([img] * 3, axis=-1))
        return img

class ResizeAndPad:
    def __init__(self, output_size, fill=0):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size

        # Determine the longer side and calculate 5% of its length
        trim_percent = 0.05
        if h > w:
            trim_size = int(h * trim_percent)
            # Trim 5% from the top and bottom
            img = img.crop((0, trim_size, w, h - trim_size))
        else:
            trim_size = int(w * trim_percent)
            # Trim 5% from the left and right
            img = img.crop((trim_size, 0, w - trim_size, h))

        # Update new image size
        w, h = img.size

        # Resize the image
        if h > w:
            new_h, new_w = self.output_size, int(self.output_size * (w / h))
        else:
            new_h, new_w = int(self.output_size * (h / w)), self.output_size
        img = img.resize((new_w, new_h))

        # Calculate padding
        diff = self.output_size - new_w if h > w else self.output_size - new_h
        padding = [diff // 2, diff // 2]

        # If the difference is odd, add the extra padding to the end
        if diff % 2 != 0:
            padding[1] += 1

        # Apply padding
        padding = (padding[0], 0, padding[1], 0) if h > w else (0, padding[0], 0, padding[1])
        
        # Adjust fill value based on image mode
        if img.mode == 'RGB':
            fill_value = (self.fill, self.fill, self.fill)
        else:
            fill_value = self.fill
            
        img = ImageOps.expand(img, border=padding, fill=fill_value)

        return img
    
    
    
class ResizeAndStretch:
    def __init__(self, output_size, fill=0):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size

        # Determine which dimension (width or height) is smaller
        if w < h:  # Width is smaller
            stretched_w = self.output_size
            stretched_h = int(h * (stretched_w / w))
        else:  # Height is smaller
            stretched_h = self.output_size
            stretched_w = int(w * (stretched_h / h))

        # Stretch the image
        img = img.resize((stretched_w, stretched_h), Image.ANTIALIAS)

        # Final resize to ensure the image is output_size x output_size
        img = img.resize((self.output_size, self.output_size), Image.ANTIALIAS)

        return img
    
class HistogramEqualization(object):
    def __call__(self, image):
        # Must be a PIL Image
        image_np = np.array(image)

        # Check if the image is grayscale or color and apply histogram equalization accordingly
        if len(image_np.shape) == 2:
            # Grayscale image
            image_eq = cv2.equalizeHist(image_np)
        else:
            # Color image
            image_eq = np.zeros_like(image_np)
            for i in range(image_np.shape[2]):
                image_eq[..., i] = cv2.equalizeHist(image_np[..., i])

        # Convert back to PIL Image
        image_eq = Image.fromarray(image_eq)
        return image_eq
    



class CLAHETransform(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        img = np.array(img)
        if len(img.shape) == 2:
            img = clahe.apply(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img[:, :, 0] = clahe.apply(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)