from fastai.vision.all import *
import torch.utils.data as TUD
from storage_adapter import * 

class BagOfImagesDataset(TUD.Dataset):

    def __init__(self, bags_dict, transform=None, save_processed=False):
        self.bags_dict = bags_dict
        self.unique_bag_ids = list(bags_dict.keys())
        self.save_processed = save_processed
        self.transform = transform
    
    def __getitem__(self, index):
        actual_id = self.unique_bag_ids[index]
        bag_info = self.bags_dict[actual_id]

        # Extract labels, image file paths, instance-level labels, and accession number
        bag_labels = bag_info['bag_labels']
        images_this_bag = bag_info['images']
        videos_this_bag = bag_info['videos']
        instance_labels = bag_info['image_labels'].copy() # Make a copy of instance labels to avoid modifying the original
        accession_number = actual_id #bag_info['Accession_Number']  # Accession number is not unique!!! :C

        # Process regular images
        image_data = [self.transform(read_image(fn, use_pil=True).convert("RGB")) for fn in images_this_bag]
        
        # Process video images if they exist
        if videos_this_bag:
            video_data = [self.transform(read_image(fn, use_pil=True).convert("RGB")) for fn in videos_this_bag]
            # Add video frames to image data
            image_data.extend(video_data)
            # Add None labels for video frames (same length as video_data)
            instance_labels.extend([[None]] * len(video_data))
            
            
        # Stack all images together
        image_data = torch.stack(image_data)

        # Convert bag labels list to a tensor
        bag_labels_tensor = torch.tensor(bag_labels, dtype=torch.float32)

        # Convert instance labels to a tensor, using -1 for None
        instance_labels_tensors = [torch.tensor(labels, dtype=torch.float32) if labels != [None] else torch.tensor([-1], dtype=torch.float32) for labels in instance_labels]

        return image_data, bag_labels_tensor, instance_labels_tensors, accession_number

    
    def __len__(self):
        return len(self.unique_bag_ids)
    
    def n_features(self):
        return self.data.size(1)



def collate_bag(batch):
    batch_data = []
    batch_bag_labels = []
    batch_instance_labels = []
    batch_ids = []

    for sample in batch:
        image_data, bag_labels, instance_labels, bag_id = sample  # Updated to unpack four items
        batch_data.append(image_data)
        batch_bag_labels.append(bag_labels)
        batch_instance_labels.append(instance_labels)
        batch_ids.append(bag_id)

    # Use torch.stack for bag labels to handle multiple labels per bag
    out_bag_labels = torch.stack(batch_bag_labels).cuda()

    # Converting to a tensor
    out_ids = torch.tensor(batch_ids, dtype=torch.long).cuda()

    return batch_data, out_bag_labels, batch_instance_labels, out_ids



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