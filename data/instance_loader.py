from fastai.vision.all import *
import torch.utils.data as TUD
from torch.utils.data import Sampler
import cv2

class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, warmup=True, dual_output=False):
        self.transform = transform
        self.warmup = warmup
        self.dual_output = dual_output

        self.images = []
        self.final_labels = []
        self.unique_ids = []
        
        for bag_id, bag_info in bags_dict.items():
            images = bag_info['images']
            image_labels = bag_info['image_labels']
            bag_label = bag_info['bag_labels'][0]  # Assuming each bag has a single label
            
            bag_id_key = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
            
            
            if bag_id_key in selection_mask:
                selection_mask_labels, _ = selection_mask[bag_id_key]
            else: 
                selection_mask_labels = None

            for idx, (img, label) in enumerate(zip(images, image_labels)):
                image_label = None
                
                if self.warmup:
                    # Only include confident instances (selection_mask) or negative bags or instance labels
                    if label[0] is not None:
                        image_label = label[0]
                    elif bag_label == 0:
                        image_label = 0
                    elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                        image_label = selection_mask_labels[idx]
                else:
                    # Return all images with unknown possiblity 
                    if label[0] is not None:
                        image_label = label[0]
                    elif bag_label == 0:
                        image_label = 0
                    elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                        image_label = selection_mask_labels[idx]
                    else:
                        image_label = -1
                
                if image_label is not None:
                    self.images.append(img)
                    self.final_labels.append(image_label)
                    # Create a unique ID combining bag_id and image index
                    unique_id = f"{bag_id_key}_{idx}"
                    self.unique_ids.append(unique_id)

    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.final_labels[index]
        unique_id = self.unique_ids[index]
        
        img = cv2.imread(img_path)
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
        self.strategy = strategy
        self.indices_positive = [i for i, label in enumerate(self.dataset.final_labels) if label == 1]
        self.indices_negative = [i for i, label in enumerate(self.dataset.final_labels) if label == 0]
        self.indices_unknown = [i for i, label in enumerate(self.dataset.final_labels) if label == -1]
        self.indices_non_positive = self.indices_negative + self.indices_unknown

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size

        for _ in range(total_batches):
            if self.strategy == 1:
                # Ensure at least one positive sample
                num_positives = random.randint(1, max(1, min(len(self.indices_positive), self.batch_size - 1)))
                num_others = self.batch_size - num_positives

                batch_positives = random.sample(self.indices_positive, num_positives)
                batch_others = random.sample(self.indices_non_positive, num_others)

                batch = batch_positives + batch_others
            elif self.strategy == 2:
                # Aim for 50/50 balance between positive and non-positive (including unknown)
                num_positives = self.batch_size // 2
                num_others = self.batch_size - num_positives

                batch_positives = random.sample(self.indices_positive, num_positives)
                batch_others = random.sample(self.indices_non_positive, num_others)

                batch = batch_positives + batch_others
            else:
                raise ValueError("Invalid strategy. Choose 1 or 2")

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size
    