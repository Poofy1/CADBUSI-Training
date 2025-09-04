import pandas as pd
import os, sys
from storage_adapter import *
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ast

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import *

config = build_config()
export_path = f"{config['export_location']}/{config['dataset_name']}"
target_model = "08_19_ABMIL_ultrasound01"

# Read all three CSV files
train_data = f"{export_path}/TrainData.csv"
worst_instances = f"{parent_dir}/models/{target_model}/evaluation/bag_metrics_val/worst_instances.csv"
breast_data = f"{export_path}/BreastData.csv"
image_dir = f"{export_path}/images/"
output_path = f"{current_dir}/worst_instances/"
os.makedirs(output_path, exist_ok=True)

train_df = read_csv(train_data)
worst_df = pd.read_csv(worst_instances)
breast_df = read_csv(breast_data)

# First merge: TrainData + worst_instances on ID = id
merged_step1 = pd.merge(
    train_df, 
    worst_df, 
    left_on='ID', 
    right_on='id', 
    how='inner'
)

# Second merge: result + BreastData on ID = Patient_ID
final_merged = pd.merge(
    merged_step1, 
    breast_df, 
    left_on='Accession_Number', 
    right_on='Accession_Number', 
    how='inner'
)


import textwrap

def add_text_to_image(image_path, output_path, text_info):
    
    image = read_image(image_path, use_pil=True)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
        
    draw = ImageDraw.Draw(image)
    
    # Try to use a larger font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Prepare text lines with wrapping and truncation
    text_lines = []
    for key, value in text_info.items():
        text_value = str(value)
        full_line = f"{key}: {text_value}"
        
        # Wrap text to fit on screen (adjust width as needed)
        wrapped_lines = textwrap.wrap(full_line, width=80)  # Adjust width as needed
        text_lines.extend(wrapped_lines)
    
    # Position text (top-left corner with some margin)
    x, y = 10, 10
    line_height = 15
    
    # Draw text lines
    for i, line in enumerate(text_lines):
        draw.text((x, y + i * line_height), line, fill=(255, 255, 255), font=font)
    
    # Convert back to RGB and save locally
    image = image.convert('RGB')
    image.save(output_path)
    return True



# Process each row in the merged dataframe
for idx, row in final_merged.iterrows():
    # Create folder for this Accession_Number
    accession_folder = os.path.join(output_path, str(row['Accession_Number']))
    os.makedirs(accession_folder, exist_ok=True)
    
    # Parse the Images column (convert string representation of list to actual list)
    try:
        if pd.isna(row['Images']) or row['Images'] == '[]':
            continue
            
        # Handle different possible formats
        if isinstance(row['Images'], str):
            # Remove brackets and quotes, then split
            images_str = row['Images'].strip('[]').replace("'", "").replace('"', '')
            image_list = [img.strip() for img in images_str.split(',') if img.strip()]
        else:
            image_list = row['Images']
            
    except Exception as e:
        print(f"Error parsing images for row {idx}: {e}")
        continue
    
    # Prepare text information to overlay
    text_info = {
        'Target': row.get('targets', 'N/A'),
        'Prediction': row.get('predictions', 'N/A'),
        'BI-RADS': str(row.get('BI-RADS', 'N/A')), 
        'Description': str(row.get('DESCRIPTION', 'N/A')), 
        'Density': row.get('Density_Desc', 'N/A'),
        #'Impression': str(row.get('rad_impression', 'N/A'))
    }
    
    # Process each image in the list
    for image_name in image_list:
        if not image_name:  # Skip empty strings
            continue
            
        # Construct the path for your custom read_image function
        input_image_path = os.path.join(image_dir, image_name).replace('\\', '/')
        # Save in the accession-specific folder
        output_image_path = os.path.join(accession_folder, f"annotated_{image_name}")
        
        success = add_text_to_image(input_image_path, output_image_path, text_info)
        if success:
            print(f"Processed: {row['Accession_Number']}/{image_name}")
        else:
            print(f"Failed to process: {row['Accession_Number']}/{image_name}")

print(f"Finished processing. Annotated images saved to: {output_path}")