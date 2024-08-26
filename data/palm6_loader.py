from fastai.vision.all import *
import torch.utils.data as TUD
from torch.utils.data import Sampler
import cv2

class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, warmup=False):
        self.transform = transform
        self.warmup = warmup

        self.images = []
        self.final_labels = []

        
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

    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.final_labels[index]
        
        #img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads in BGR, so convert to RGB
        img = Image.fromarray(img)
        image_data = self.transform(img)

        return image_data, instance_label


    def __len__(self):
        return len(self.images)


class WarmupSampler(Sampler):
    def __init__(self, dataset, batch_size, strategy=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.indices_0 = [i for i, label in enumerate(self.dataset.final_labels) if label == 0]
        self.indices_1 = [i for i, label in enumerate(self.dataset.final_labels) if label == 1]
        self.indices_unlabeled = [i for i, label in enumerate(self.dataset.final_labels) if label == -1]

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size

        for _ in range(total_batches):
            if self.strategy == 1:
                # Ensure at least one positive sample
                num_positives = random.randint(1, max(1, min(len(self.indices_1), self.batch_size - 1)))
                num_negatives = self.batch_size - num_positives
                batch_positives = random.sample(self.indices_1, num_positives)
                batch_negatives = random.sample(self.indices_0, num_negatives)
                batch = batch_positives + batch_negatives
            elif self.strategy == 2:
                # Aim for 50/50 balance
                num_positives = self.batch_size // 2
                num_negatives = self.batch_size - num_positives
                batch_positives = random.sample(self.indices_1, num_positives)
                batch_negatives = random.sample(self.indices_0, num_negatives)
                batch = batch_positives + batch_negatives
            elif self.strategy == 3:
                # At least one positive, one negative, and possibility of -1 labels
                num_positives = max(1, random.randint(1, self.batch_size // 2))
                num_negatives = max(1, random.randint(1, self.batch_size // 2))
                num_unlabeled = self.batch_size - num_positives - num_negatives

                batch_positives = random.sample(self.indices_1, num_positives)
                batch_negatives = random.sample(self.indices_0, num_negatives)
                
                if num_unlabeled > 0 and self.indices_unlabeled:
                    batch_unlabeled = random.sample(self.indices_unlabeled, min(num_unlabeled, len(self.indices_unlabeled)))
                    batch = batch_positives + batch_negatives + batch_unlabeled
                else:
                    # If no unlabeled samples or not enough space, fill with more labeled samples
                    remaining = self.batch_size - len(batch_positives) - len(batch_negatives)
                    additional_samples = random.sample(self.indices_0 + self.indices_1, remaining)
                    batch = batch_positives + batch_negatives + additional_samples
            else:
                raise ValueError("Invalid strategy. Choose 1, 2, or 3")

            random.shuffle(batch)
            yield batch
            
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    
def collate_instance(batch):
    batch_data = []
    batch_labels = []

    for image_data, bag_label in batch:
        batch_data.append(image_data)
        batch_labels.append(bag_label)

    # Stack the images and labels
    batch_data = torch.stack(batch_data)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return batch_data, batch_labels
