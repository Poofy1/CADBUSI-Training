from fastai.vision.all import *
import torch.utils.data as TUD
from torch.utils.data import Sampler
import cv2

class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, transform=None, known_only = False, pos_only = False):
        self.transform = transform
        self.known_only = known_only
        self.pos_only = pos_only
        
        self.images = []
        self.partialY = []
        self.partialY_pos = []
        self.partialY_cls = []
        self.unique_ids = []
        self.bag_labels = []
        
        instance_counter = 0  # Counter for generating unique IDs
        
        for bag_id, bag_info in bags_dict.items():
            images = bag_info['images']
            image_labels = bag_info['image_labels']
            bag_label = bag_info['bag_labels'][0]  # Assuming each bag has a single label
            
            for img, labels in zip(images, image_labels):
                partial_label = None
                cls_label = None

                if bag_label == 1:
                    self.partialY_pos.append([1, 1])
                    partial_label = [1, 1]
                    cls_label = [0, 1]
                elif not pos_only:
                    partial_label = [1, 0]
                    cls_label = [1, 0]
                
                if bag_label == 1 or not pos_only:
                    self.images.append(img)
                    self.partialY.append(partial_label)
                    self.partialY_cls.append(cls_label)
                    self.unique_ids.append(instance_counter)
                    self.bag_labels.append(bag_label)
                    instance_counter += 1
                
        self.partialY = torch.tensor(self.partialY, dtype=torch.float)
        self.partialY_pos = torch.tensor(self.partialY_pos, dtype=torch.float)
        self.partialY_cls = torch.tensor(self.partialY_cls, dtype=torch.float)
        self.bag_labels = torch.tensor(self.bag_labels, dtype=torch.long)

        
    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.partialY[index]
        bag_label = self.bag_labels[index]
        unique_id = self.unique_ids[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        point = bag_label # Weird thing from the R-MIL paper, idk
        
        image_data_q = self.transform(img)
        image_data_k = self.transform(img)
        return image_data_q, image_data_k, instance_label, bag_label, point, unique_id
    
    def __len__(self):
        return len(self.images)
    
    
    
def collate_instance(batch):
    batch_data_q = []
    batch_data_k = []
    batch_partial_labels = []
    batch_true_labels = []
    batch_positive_labels = []
    batch_ids = []

    for image_data_q, image_data_k, partial_label, true_label, positive_label, unique_id in batch:
        batch_data_q.append(image_data_q)
        batch_data_k.append(image_data_k)
        batch_partial_labels.append(partial_label)
        batch_true_labels.append(true_label)
        batch_positive_labels.append(positive_label)
        batch_ids.append(unique_id)

    batch_data_q = torch.stack(batch_data_q)
    batch_data_k = torch.stack(batch_data_k)
    batch_partial_labels = torch.stack(batch_partial_labels)
    batch_true_labels = torch.stack(batch_true_labels)
    batch_positive_labels = torch.stack(batch_positive_labels)

    return batch_data_q, batch_data_k, batch_partial_labels, batch_true_labels, batch_positive_labels, torch.tensor(batch_ids)


import numpy as np
from torch.utils.data import Sampler

class BalancedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_indices = []
        self.neg_indices = []
        
        for i, label in enumerate(dataset.partialY):
            if label[1] == 1:  # Positive class
                self.pos_indices.append(i)
            elif label[0] == 1:  # Negative class
                self.neg_indices.append(i)
        
        self.num_pos = len(self.pos_indices)
        self.num_neg = len(self.neg_indices)
    
    def __iter__(self):
        pos_sample = self.pos_indices.copy()
        neg_sample = self.neg_indices.copy()
        np.random.shuffle(pos_sample)
        np.random.shuffle(neg_sample)
        
        # Ensure even number of samples for both classes
        min_samples = min(len(pos_sample), len(neg_sample))
        pos_sample = pos_sample[:min_samples]
        neg_sample = neg_sample[:min_samples]
        
        batch = []
        for i in range(0, len(pos_sample), self.batch_size // 2):
            batch.extend(pos_sample[i:i + self.batch_size // 2])
            batch.extend(neg_sample[i:i + self.batch_size // 2])
            
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        
        # Handle the last batch
        if batch:
            while len(batch) < self.batch_size:
                # Randomly sample from positive or negative class
                if np.random.random() < 0.5:
                    batch.append(np.random.choice(self.pos_indices))
                else:
                    batch.append(np.random.choice(self.neg_indices))
            yield batch
    
    def __len__(self):
        return (min(self.num_pos, self.num_neg) * 2 + self.batch_size - 1) // self.batch_size