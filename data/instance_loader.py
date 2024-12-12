from fastai.vision.all import *
import torch.utils.data as TUD
from torch.utils.data import Sampler
import cv2
from storage_adapter import *

class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, warmup=True, 
                 dual_output=False, only_negative=False, max_positive=None):
        self.transform = transform
        self.warmup = warmup
        self.dual_output = dual_output
        self.only_negative = only_negative
        self.max_positive = max_positive

        self.images = []
        self.final_labels = []
        self.unique_ids = []
        
        # Keep track of positive instances
        positive_count = 0
        temp_positive_data = []
        
        for bag_id, bag_info in bags_dict.items():
            images = bag_info['images']
            image_labels = bag_info['image_labels']
            bag_label = bag_info['bag_labels'][0]
            accession_number = bag_info['Accession_Number']
            
            acc_number_key = accession_number.item() if isinstance(accession_number, torch.Tensor) else accession_number
            
            if acc_number_key in selection_mask:
                selection_mask_labels, _ = selection_mask[acc_number_key]
            else: 
                selection_mask_labels = None

            for idx, (img, labels) in enumerate(zip(images, image_labels)):
                image_label = None
                
                if self.only_negative:
                    if labels[0] is not None and labels[0] == 0:
                        image_label = 0
                    elif bag_label == 0:
                        image_label = 0
                        
                elif self.warmup:
                    if labels[0] is not None:
                        image_label = labels[0]
                    elif bag_label == 0:
                        image_label = 0
                    elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                        image_label = selection_mask_labels[idx]
                else:
                    if labels[0] is not None:
                        image_label = labels[0]
                    elif bag_label == 0:
                        image_label = 0
                    elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                        image_label = selection_mask_labels[idx]
                    else:
                        image_label = -1
                
                if image_label is not None:
                    unique_id = f"{acc_number_key}_{idx}"
                    
                    if image_label == 1 and self.max_positive is not None:
                        # Store positive instances temporarily
                        temp_positive_data.append((img, image_label, unique_id))
                    else:
                        # Add negative instances directly
                        self.images.append(img)
                        self.final_labels.append(image_label)
                        self.unique_ids.append(unique_id)

        # Handle positive instances with cap
        if self.max_positive is not None and temp_positive_data:
            # Randomly shuffle and select up to max_positive instances
            random.shuffle(temp_positive_data)
            selected_positive = temp_positive_data[:self.max_positive]
            
            for img, label, unique_id in selected_positive:
                self.images.append(img)
                self.final_labels.append(label)
                self.unique_ids.append(unique_id)
            
            print(f"Selected {len(selected_positive)} positive instances out of {len(temp_positive_data)} total positive instances")

        print(f"Dataset created with {len(self.images)} images")
        if self.only_negative:
            print("Dataset contains only negative (label 0) images")
        if self.max_positive is not None:
            positive_count = sum(1 for label in self.final_labels if label == 1)
            print(f"Dataset contains {positive_count} positive images (capped at {self.max_positive})")

        
    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.final_labels[index]
        unique_id = self.unique_ids[index]
        
        img = read_image(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        if self.dual_output:
            image_data_q = self.transform(img)
            image_data_k = self.transform(img)
            return (image_data_q, image_data_k), instance_label, unique_id
        else:
            image_data = self.transform(img)
            return image_data, instance_label, unique_id


    def __len__(self):
        return len(self.images)
    
    
    
    
def collate_instance(batch):
    if isinstance(batch[0][0], tuple):  # Check if it's dual output
        batch_data_q = []
        batch_data_k = []
        batch_labels = []
        batch_ids = []

        for (image_data_q, image_data_k), bag_label, unique_id in batch:
            batch_data_q.append(image_data_q)
            batch_data_k.append(image_data_k)
            batch_labels.append(bag_label)
            batch_ids.append(unique_id)

        batch_data_q = torch.stack(batch_data_q)
        batch_data_k = torch.stack(batch_data_k)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return (batch_data_q, batch_data_k), batch_labels, batch_ids
    else:
        batch_data = []
        batch_labels = []
        batch_ids = []

        for image_data, bag_label, unique_id in batch:
            batch_data.append(image_data)
            batch_labels.append(bag_label)
            batch_ids.append(unique_id)

        batch_data = torch.stack(batch_data)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return batch_data, batch_labels, batch_ids


class InstanceSampler(Sampler):
    def __init__(self, dataset, batch_size, strategy=1):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get indices for each class
        self.indices_positive = [i for i, label in enumerate(self.dataset.final_labels) if label == 1]
        self.indices_negative = [i for i, label in enumerate(self.dataset.final_labels) if label == 0]
        self.indices_unknown = [i for i, label in enumerate(self.dataset.final_labels) if label == -1]
        self.indices_non_positive = self.indices_negative + self.indices_unknown
        
        # Number of positive samples determines the number of samples per class
        self.samples_per_class = len(self.indices_positive)
        
        # Calculate total number of batches possible with balanced classes
        self.total_samples = self.samples_per_class * 2  # multiply by 2 for pos and neg
        self.total_batches = self.total_samples // self.batch_size
        
        print("Sampler statistics:")
        print(f"Number of positive samples: {len(self.indices_positive)}")
        print(f"Number of negative samples: {len(self.indices_negative)}")
        print(f"Number of unknown samples: {len(self.indices_unknown)}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {self.total_batches}")

    def __iter__(self):
        # Randomly sample from majority class to match minority class size
        selected_non_positive = random.sample(self.indices_non_positive, self.samples_per_class)
        
        # Create balanced dataset
        all_indices = self.indices_positive + selected_non_positive
        
        # Shuffle initially
        random.shuffle(all_indices)
        
        # Create batches ensuring at least one positive sample per batch
        for i in range(self.total_batches):
            batch = []
            remaining_size = self.batch_size
            
            # Ensure at least one positive sample
            pos_sample = random.choice(self.indices_positive)
            batch.append(pos_sample)
            remaining_size -= 1
            
            # Fill rest of batch from shuffled indices, excluding the chosen positive sample
            available_indices = [idx for idx in all_indices if idx != pos_sample]
            batch.extend(random.sample(available_indices, remaining_size))
            
            # Final shuffle of the batch
            random.shuffle(batch)
            
            yield batch

    def __len__(self):
        return self.total_batches
    