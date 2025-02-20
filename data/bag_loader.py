from fastai.vision.all import *
import torch.utils.data as TUD
from torch.utils.data.distributed import DistributedSampler
from storage_adapter import * 

class BagOfImagesDataset(TUD.Dataset):
    def __init__(self, bags_dict, transform=None, save_processed=False):
        self.bags_dict = bags_dict
        self.unique_bag_ids = list(bags_dict.keys())
        self.save_processed = save_processed
        self.transform = transform
        
        # Pre-process bag labels at initialization
        self.bag_labels = {
            bag_id: torch.tensor(info['bag_labels'], dtype=torch.float32)
            for bag_id, info in bags_dict.items()
        }

    def __getitem__(self, index):
        actual_id = self.unique_bag_ids[index]
        bag_info = self.bags_dict[actual_id]
        
        # Use list comprehension for image loading
        images = [read_image(fn, use_pil=True) for fn in bag_info['images'] + bag_info['videos']]
        
        # Batch process transformations
        if self.transform:
            images = [self.transform(img.convert('RGB') if img.mode != 'RGB' else img) 
                     for img in images]
        
        # Stack all images at once
        image_data = torch.stack(images)
        
        # Process instance labels more efficiently
        instance_labels = (bag_info['image_labels'] + 
                         [[None]] * len(bag_info['videos']))
        
        # Use list comprehension for instance labels
        instance_labels_tensors = [
            torch.tensor([-1] if labels == [None] else labels, dtype=torch.float32)
            for labels in instance_labels
        ]
        
        return (
            image_data,
            self.bag_labels[actual_id],
            instance_labels_tensors,
            actual_id
        )
    
    def __len__(self):
        return len(self.unique_bag_ids)
    
    def n_features(self):
        return self.data.size(1)



def collate_bag(batch, pad_bags=False, fixed_bag_size=25):
    batch_bag_labels = []
    batch_instance_labels = []
    batch_ids = []

    if pad_bags:
        # Use fixed padding size instead of dynamic max size
        _, C, H, W = batch[0][0].shape
        padded_images = torch.zeros((len(batch), fixed_bag_size, C, H, W), dtype=torch.float32)

        for i, (image_data, bag_labels, instance_labels, bag_id) in enumerate(batch):
            num_images = min(image_data.shape[0], fixed_bag_size)  # Limit to fixed size
            padded_images[i, :num_images] = image_data[:num_images]  # Truncate if needed
            batch_bag_labels.append(bag_labels)
            batch_instance_labels.append(instance_labels[:num_images] if num_images < len(instance_labels) else instance_labels)
            batch_ids.append(bag_id)

        out_bag_labels = torch.stack(batch_bag_labels)
        out_ids = torch.tensor(batch_ids, dtype=torch.long)
        return padded_images, out_bag_labels, batch_instance_labels, out_ids

    else:
        # No padding mode - return original bags without padding
        batch_images = [sample[0] for sample in batch]
        batch_bag_labels = [sample[1] for sample in batch]
        batch_instance_labels = [sample[2] for sample in batch]
        batch_ids = [sample[3] for sample in batch]
        
        out_bag_labels = torch.stack(batch_bag_labels)
        out_ids = torch.tensor(batch_ids, dtype=torch.long)
        return batch_images, out_bag_labels, batch_instance_labels, out_ids




class BalancedBagSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get positive and negative indices
        self.pos_indices = []
        self.neg_indices = []
        
        for idx in range(len(dataset)):
            bag_labels = dataset.bags_dict[dataset.unique_bag_ids[idx]]['bag_labels']
            if 1 in bag_labels:
                self.pos_indices.append(idx)
            else:
                self.neg_indices.append(idx)
                
        self.n_pos = len(self.pos_indices)
        self.n_neg = len(self.neg_indices)
        
        # Determine minority class size
        self.min_class_size = min(self.n_pos, self.n_neg)
        
        # Calculate number of batches
        self.n_samples = 2 * self.min_class_size  # Total samples will be 2 * number of minority samples
        self.n_batches = self.n_samples // batch_size
        if self.n_samples % batch_size != 0:
            self.n_batches += 1
    
    def __iter__(self):
        # Sample from majority class to match minority class size
        if self.n_pos > self.n_neg:
            sampled_pos = np.random.choice(self.pos_indices, size=self.min_class_size, replace=False)
            all_indices = np.concatenate([sampled_pos, self.neg_indices])
        else:
            sampled_neg = np.random.choice(self.neg_indices, size=self.min_class_size, replace=False)
            all_indices = np.concatenate([self.pos_indices, sampled_neg])
        
        # Shuffle the combined indices
        np.random.shuffle(all_indices)
        
        # Create batches
        batches = []
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.n_samples)
            if start_idx < self.n_samples:
                batches.append(all_indices[start_idx:end_idx])
        
        return iter(batches)
    
    def __len__(self):
        return self.n_batches
    
    
    
    
    

class SyntheticBagDataset(TUD.Dataset):
    def __init__(self, bags_dict, transform=None, min_bag_size=3, max_bag_size=20):
        self.transform = transform
        self.min_bag_size = min_bag_size
        self.max_bag_size = max_bag_size
        self.num_bags = len(bags_dict)
        
        # Initialize lists for positive and negative instances
        self.pos_images = []
        self.neg_images = []
        
        # Process original bags based on single bag label
        for bag_id, bag_info in bags_dict.items():
            bag_label = bag_info['bag_labels'][0]  # Single label per bag
            images = bag_info['images']
            
            if bag_label == 1:
                self.pos_images.extend(images)
            elif bag_label == 0:
                self.neg_images.extend(images)
        
        print(f"Found {len(self.pos_images)} positive images and {len(self.neg_images)} negative images")
        print(f"Will generate {self.num_bags} bags")
        
    def __getitem__(self, index):
        bag_size = random.randint(self.min_bag_size, self.max_bag_size)
        is_positive_bag = random.random() < 0.5
        
        if is_positive_bag:
            bag_images = [random.choice(self.pos_images)]
            for _ in range(bag_size - 1):
                if random.random() < 0.5:
                    bag_images.append(random.choice(self.pos_images))
                else:
                    bag_images.append(random.choice(self.neg_images))
            bag_label = [1]
            instance_labels = [[-1] if img in self.pos_images else [0] for img in bag_images]
        else:
            bag_images = [random.choice(self.neg_images) for _ in range(bag_size)]
            bag_label = [0]
            instance_labels = [[0] for _ in range(bag_size)]
        
        image_data = torch.stack([self.transform(read_image(fn, use_pil=True).convert("RGB")) 
                                for fn in bag_images])
        
        bag_labels_tensor = torch.tensor(bag_label, dtype=torch.float32)
        instance_labels_tensors = [torch.tensor(labels, dtype=torch.float32) 
                                 for labels in instance_labels]
        
        return image_data, bag_labels_tensor, instance_labels_tensors, index
    
    def __len__(self):
        return self.num_bags