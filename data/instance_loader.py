from fastai.vision.all import *
import torch.utils.data as TUD
from torch.utils.data import Sampler
import cv2
from torch.utils.data.distributed import DistributedSampler
from storage_adapter import *
from config import train_transform, val_transform

class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, warmup=True, use_bag_labels=False,
                 dual_output=False, only_negative=False, max_positive=None):
        self.transform = transform
        self.warmup = warmup
        self.dual_output = dual_output
        self.only_negative = only_negative
        self.max_positive = max_positive
        self.use_bag_labels = use_bag_labels

        self.images = []
        self.output_image_labels = []
        self.unique_ids = []
        
        # Keep track of positive instances
        temp_positive_data = []
        
        for bag_id, bag_info in bags_dict.items():
            images = bag_info['images'].copy()  # Create copies to avoid modifying original
            image_labels = bag_info['image_labels'].copy()
            videos = bag_info['videos']
            bag_label = bag_info['bag_labels'][0]
            accession_number = bag_info['Accession_Number']
            
             #Debug prints
            """print(f"\nBag {bag_id}:")
            print(f"Number of images: {len(images)}")
            print(f"Number of videos: {len(videos)}")
            print(f"Total instances after combine: {len(images) + len(videos)}")
            """

            # Extend images and labels with videos
            if not warmup: # do not include video data in warmup (more noise)
                images.extend(videos)
                image_labels.extend([[None]] * len(videos))  # Add empty labels for videos
            
            acc_number_key = accession_number.item() if isinstance(accession_number, torch.Tensor) else accession_number
            
            if bag_id in selection_mask:
                selection_mask_labels, _ = selection_mask[bag_id]
                #print(f"Selection mask size for bag {bag_id}: {len(selection_mask_labels)}")
            else: 
                selection_mask_labels = None
                #print(f"No selection mask for bag {bag_id}")
                
            
            
            
            # Process all instances (images + videos)
            for idx, (img, labels) in enumerate(zip(images, image_labels)):
                image_label = None
                is_video = idx >= len(bag_info['images'])  # Check if this is a video frame
                
            
                if self.only_negative:
                    if labels[0] is not None and labels[0] == 0:
                        image_label = 0
                    elif bag_label == 0:
                        image_label = 0
                
                elif self.use_bag_labels:   
                    image_label = bag_label    
                    
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
                    unique_id = f"{acc_number_key}_{idx}_{'vid' if is_video else 'img'}"

                    if image_label == 1 and self.max_positive is not None and not is_video:
                        # Store positive instances temporarily (only for images)
                        temp_positive_data.append((img, image_label, unique_id))
                    else:
                        # Add instances directly
                        self.images.append(img)
                        self.output_image_labels.append(image_label)
                        self.unique_ids.append(unique_id)
                        
        # Handle positive instances with cap
        if self.max_positive is not None and temp_positive_data:
            random.shuffle(temp_positive_data)
            selected_positive = temp_positive_data[:self.max_positive]
            
            for img, label, unique_id in selected_positive:
                self.images.append(img)
                self.output_image_labels.append(label)
                self.unique_ids.append(unique_id)
            
            print(f"Selected {len(selected_positive)} positive instances out of {len(temp_positive_data)} total positive instances")

        print(f"Dataset created with {len(self.images)} instances")
        if self.only_negative:
            print("Dataset contains only negative (label 0) instances")
        if self.max_positive is not None:
            positive_count = sum(1 for label in self.output_image_labels if label == 1)
            print(f"Dataset contains {positive_count} positive instances (capped at {self.max_positive})")

        
    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.output_image_labels[index]
        unique_id = self.unique_ids[index]
        
        """img = read_image(img_path, local_override=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)"""
        
        img = Image.open(img_path).convert("RGB")
        
        if self.dual_output:
            image_data_q = self.transform(img)
            image_data_k = self.transform(img)
            return (image_data_q, image_data_k), instance_label, unique_id
        else:
            image_data = self.transform(img)
            return image_data, instance_label, unique_id


    def __len__(self):
        return len(self.images)
    
    
def get_instance_loaders(bags_train, bags_val, state, config, warmup=True, use_bag_labels=False, dual_output=False, only_negative=False, max_positive=None):
    if not config['use_sudo_labels']:
        state['selection_mask'] = []
    
    # Used the instance predictions from bag training to update the Instance Dataloader
    instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=warmup, 
                                              use_bag_labels=use_bag_labels,
                                              dual_output=dual_output,
                                              only_negative=only_negative,
                                              max_positive=max_positive)
    instance_dataset_val = Instance_Dataset(bags_val, [], transform=val_transform, warmup=True,
                                            use_bag_labels=use_bag_labels,
                                            dual_output=dual_output,
                                            only_negative=only_negative,
                                            max_positive=max_positive)
    train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'], strategy=1)
    val_sampler = InstanceSampler(instance_dataset_val, config['instance_batch_size'], seed=1)
    
    if platform.system() == 'Windows': #Windows works better on its own
        instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, collate_fn = collate_instance)
    else:
        instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, collate_fn = collate_instance, num_workers=8, pin_memory=True, persistent_workers=True)
    instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
    
    return instance_dataloader_train, instance_dataloader_val



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
        batch_labels = torch.tensor(batch_labels, dtype=torch.float)

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
        batch_labels = torch.tensor(batch_labels, dtype=torch.float)

        return batch_data, batch_labels, batch_ids


class InstanceSampler(Sampler):
    def __init__(self, dataset, batch_size, strategy=1, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        
        # Get indices for each class
        self.indices_positive = [i for i, label in enumerate(self.dataset.output_image_labels) if label == 1]
        self.indices_negative = [i for i, label in enumerate(self.dataset.output_image_labels) if label == 0]
        self.indices_unknown = [i for i, label in enumerate(self.dataset.output_image_labels) if label != 0 and label != 1]
        self.indices_non_positive = self.indices_negative + self.indices_unknown
        
        # Number of positive samples determines the number of samples per class
        self.samples_per_class = len(self.indices_positive)
        
        # Calculate total number of batches possible with balanced classes
        self.total_samples = self.samples_per_class * 2  # multiply by 2 for pos and neg
        self.total_batches = self.total_samples // self.batch_size
        
        """print("Sampler statistics:")
        print(f"Number of positive samples: {len(self.indices_positive)}")
        print(f"Number of negative samples: {len(self.indices_negative)}")
        print(f"Number of unknown samples: {len(self.indices_unknown)}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {self.total_batches}")"""

    def __iter__(self):
        # If a seed is specified, set it for reproducible sampling
        if self.seed is not None:
            random.seed(self.seed)

        # Randomly sample from non-positive class to match the number of positive samples
        # Safely sample from non-positive class
        if len(self.indices_non_positive) < self.samples_per_class:
            print(f"Warning: Not enough non-positive samples. Requested {self.samples_per_class} but only have {len(self.indices_non_positive)} available.")
            selected_non_positive = self.indices_non_positive
        else:
            selected_non_positive = random.sample(self.indices_non_positive, self.samples_per_class)
        
        # Create balanced dataset
        all_indices = self.indices_positive + selected_non_positive
        
        # Shuffle once before creating batches
        random.shuffle(all_indices)
        
        # Create batches ensuring at least one positive sample per batch
        for i in range(self.total_batches):
            batch = []
            remaining_size = self.batch_size
            
            # Ensure at least one positive sample
            pos_sample = random.choice(self.indices_positive)
            batch.append(pos_sample)
            remaining_size -= 1
            
            # Fill the rest of the batch from the shuffled indices, excluding the chosen positive sample
            available_indices = [idx for idx in all_indices if idx != pos_sample]
            batch.extend(random.sample(available_indices, remaining_size))
            
            # Shuffle the final batch order
            random.shuffle(batch)
            
            yield batch

    def __len__(self):
        return self.total_batches



class DistributedInstanceSampler(DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, seed=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.batch_size = batch_size
        self.seed = seed
        
        self.indices_positive = [i for i, label in enumerate(dataset.output_image_labels) if label == 1]
        self.indices_negative = [i for i, label in enumerate(dataset.output_image_labels) if label == 0]
        self.indices_unknown = [i for i, label in enumerate(dataset.output_image_labels) if label == -1]
        self.indices_non_positive = self.indices_negative + self.indices_unknown
        
        self.samples_per_class = len(self.indices_positive)
        self.total_samples = self.samples_per_class * 2
        self.total_batches = (self.total_samples // self.batch_size) // self.num_replicas

    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed + self.epoch)

        selected_non_positive = random.sample(self.indices_non_positive, self.samples_per_class)
        all_indices = self.indices_positive + selected_non_positive
        random.shuffle(all_indices)
        
        # Distribute indices across GPUs
        indices_for_rank = all_indices[self.rank:self.total_samples:self.num_replicas]
        batches = []
        
        for i in range(self.total_batches):
            batch = []
            remaining_size = self.batch_size
            
            pos_indices = [idx for idx in indices_for_rank if idx in self.indices_positive]
            if pos_indices:
                pos_sample = random.choice(pos_indices)
                batch.append(pos_sample)
                remaining_size -= 1
            
            available_indices = [idx for idx in indices_for_rank if idx != pos_sample]
            batch.extend(random.sample(available_indices, remaining_size))
            random.shuffle(batch)
            batches.extend(batch)
            
        return iter(batches)

    def __len__(self):
        return self.total_batches * self.batch_size