import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_image(row, root_dir, output_dir, resize_and_pad):
    patient_id = row['Patient_ID']
    img_name = row['ImageName']
    input_path = os.path.join(f'{root_dir}images/', img_name)
    output_path = os.path.join(output_dir, img_name)

    if os.path.exists(output_path):  # Skip images that are already processed
        return

    image = Image.open(input_path)
    image = resize_and_pad(image)
    image.save(output_path)

def preprocess_and_save_images(data, root_dir, output_dir, image_size, fill=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Preprocessing Data")

    resize_and_pad = ResizeAndPad(image_size, fill)
    data_rows = [row for _, row in data.iterrows()]  # Convert the DataFrame to a list of rows

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, row, root_dir, output_dir, resize_and_pad): row for row in data_rows}

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                future.result()  # We don't actually use the result, but this will raise any exceptions
                pbar.update()

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
        if h > w:
            new_h, new_w = self.output_size, int(self.output_size * (w / h))
        else:
            new_h, new_w = int(self.output_size * (h / w)), self.output_size
        img = transforms.functional.resize(img, (new_h, new_w))

        diff = self.output_size - new_w if h > w else self.output_size - new_h
        padding = [diff // 2, diff // 2]

        # If the difference is odd, add the extra padding to the end
        if diff % 2 != 0:
            padding[1] += 1

        # Use the padding values for the left/right or top/bottom
        padding = (padding[0], 0, padding[1], 0) if h > w else (0, padding[0], 0, padding[1])
        img = transforms.functional.pad(img, padding, fill=self.fill)
        return img
    
    

def upsample_bag_to_min_count(group, min_count):
    num_needed = min_count - len(group)
    
    if num_needed > 0:
        # Duplicate random samples within the same group
        random_rows = group.sample(num_needed, replace=True)
        group = pd.concat([group, random_rows], axis=0).reset_index(drop=True)
        
    return group