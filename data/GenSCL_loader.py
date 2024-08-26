from fastai.vision.all import *
import torch.utils.data as TUD
from torch.utils.data import Sampler
import cv2


class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, warmup=True):
        self.transform = transform
        self.warmup = warmup 
        self.images = []
        self.final_labels = []
        self.warmup_mask = []
        
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
                warmup_mask_value = 0
                
                if not self.warmup:
                    # Only include confident instances (selection_mask) or negative bags or instance labels
                    if label[0] is not None:
                        image_label = label[0]
                    elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                        image_label = selection_mask_labels[idx]
                    elif bag_label == 0:
                        image_label = 0
                else:
                    # Include all data but replace with image_labels if present
                    if label[0] is not None:
                        image_label = label[0]
                    else:
                        image_label = bag_label  # Use bag label if instance label is not present
                        if bag_label == 1:
                            warmup_mask_value = 1 # Set warmup mask to 1 for instances without label[0] and bag_label is 1
                
                if image_label is not None:
                    self.images.append(img)
                    self.final_labels.append(image_label)
                    self.warmup_mask.append(warmup_mask_value)

    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.final_labels[index]
        warmup_unconfident = self.warmup_mask[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads in BGR, so convert to RGB
        img = Image.fromarray(img)

        image_data_q = self.transform(img)
        #image_data_k = self.transform(img)
        
        return (image_data_q, image_data_q), instance_label, warmup_unconfident


    def __len__(self):
        return len(self.images)


class WarmupSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_0 = [i for i, mask in enumerate(self.dataset.warmup_mask) if mask == 0]
        self.indices_1 = [i for i, mask in enumerate(self.dataset.warmup_mask) if mask == 1]

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size

        for _ in range(total_batches):
            # Randomly decide how many mask 0s to include, ensuring at least 1
            num_mask_0 = random.randint(1, max(1, min(len(self.indices_0), self.batch_size - 1)))
            batch_mask_0 = random.sample(self.indices_0, num_mask_0)

            # Fill the rest of the batch with mask 1s, if any space left
            num_mask_1 = self.batch_size - num_mask_0 
            batch_mask_1 = random.sample(self.indices_1, num_mask_1) if num_mask_1 > 0 else []

            batch = batch_mask_0 + batch_mask_1
            random.shuffle(batch)
            yield batch
            
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
def collate_instance(batch):
    batch_data_q = []
    batch_data_k = [] 
    batch_labels = []
    batch_unconfident = []

    for (image_data_q, image_data_k), bag_label, warmup_unconfident in batch:
        batch_data_q.append(image_data_q)
        batch_data_k.append(image_data_k)
        batch_labels.append(bag_label)
        batch_unconfident.append(warmup_unconfident)

    # Stack the images and labels
    batch_data_q = torch.stack(batch_data_q)
    batch_data_k = torch.stack(batch_data_k)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    batch_unconfident = torch.tensor(batch_unconfident, dtype=torch.long)

    return (batch_data_q, batch_data_k), batch_labels, batch_unconfident