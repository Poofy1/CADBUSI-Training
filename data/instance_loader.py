from fastai.vision.all import *
import torch.utils.data as TUD
from torch.utils.data import Sampler
import cv2
from torch.utils.data.distributed import DistributedSampler
from storage_adapter import *
from config import train_transform, val_transform

class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, pseudo_dict, config, transform=None, only_known=True, only_pseudo = False, use_bag_labels=False,
                 dual_output=False, only_negative=False, subset=None):
        self.transform = transform
        self.only_known = only_known
        self.only_pseudo = only_pseudo
        self.dual_output = dual_output
        self.only_negative = only_negative
        self.use_bag_labels = use_bag_labels
        self.target_column = config['instance_columns'][0]
        
        # Handle subset selection - filter bags_dict first
        if subset is None:
            filtered_bags_dict = bags_dict
        else:
            # Create a local random state for consistency
            rng = np.random.RandomState(42)
            
            if isinstance(subset, (int, float)) and 0 < subset <= 1.0:
                # Percentage of the data
                all_bag_ids = list(bags_dict.keys())
                subset_size = int(len(all_bag_ids) * subset)
                selected_bag_ids = rng.choice(all_bag_ids, size=subset_size, replace=False).tolist()
                filtered_bags_dict = {bag_id: bags_dict[bag_id] for bag_id in selected_bag_ids}
            else:
                raise ValueError("subset must be None or a float between 0 and 1")
            
        self.images = []
        self.output_image_labels = []
        self.output_pseudo_labels = []
        self.unique_ids = []
        
        for bag_id, bag_info in filtered_bags_dict.items():
            images = bag_info['images'].copy()  # Create copies to avoid modifying original
            image_labels = bag_info['image_labels'].copy()
            videos = bag_info['videos']
            bag_label = bag_info['bag_labels'][0]
            accession_number = bag_info['Accession_Number']

            # Extend images and labels with videos
            if not self.only_known: # do not include video data in warmup (more noise)
                images.extend(videos)
                image_labels.extend([{}] * len(videos))  # Changed from [[None]] to [{}]
            
            # ID
            acc_number_key = accession_number.item() if isinstance(accession_number, torch.Tensor) else accession_number

            if bag_id in pseudo_dict:
                selection_mask_labels, _ = pseudo_dict[bag_id]
            else: 
                selection_mask_labels = None
                
            # Helper function to extract target label
            def get_target_label(labels):
                if isinstance(labels, dict) and self.target_column in labels:
                    return labels[self.target_column]
                return None
            
            # Process all instances (images + videos)
            for idx, (img, labels) in enumerate(zip(images, image_labels)):
                image_label = None
                pseudo_label = None
                is_video = idx >= len(bag_info['images'])  # Check if this is a video frame
                unique_id = f"{acc_number_key}_{idx}_{'vid' if is_video else 'img'}"
                
                # Extract the target label
                target_label = get_target_label(labels)
                
                # true labels
                if self.only_negative:
                    if target_label is not None and target_label == 0:
                        image_label = 0
                    elif bag_label == 0:
                        image_label = 0
                
                elif self.use_bag_labels:   
                    image_label = bag_label    
                    
                elif self.only_known:
                    if target_label is not None:
                        image_label = target_label
                    elif bag_label == 0:
                        image_label = 0
                        
                else:
                    if target_label is not None:
                        image_label = target_label
                    elif bag_label == 0:
                        image_label = 0
                    else:
                        image_label = -1 # unknown
                
                # pseudo labels
                if selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                    pseudo_label = selection_mask_labels[idx] 
                if image_label != -1:
                    pseudo_label = image_label
                        
                # if label exists
                use_instance = False
                if self.only_known:
                    if image_label is not None:
                        use_instance = True
                elif self.only_pseudo:
                    if pseudo_label is not None:
                        use_instance = True
                else:
                    use_instance = True
                        
                if use_instance:
                    self.images.append(img)
                    self.output_image_labels.append(-1 if image_label is None else image_label)
                    self.output_pseudo_labels.append(-1 if pseudo_label is None else pseudo_label)
                    self.unique_ids.append(unique_id)
                        
        print(f"Dataset created with {len(self.images)} instances")
        print(f"Using target column: {self.target_column}")
        if self.only_negative:
            print("Dataset contains only negative (label 0) instances")

    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.output_image_labels[index]
        pseudo_label = self.output_pseudo_labels[index]
        unique_id = self.unique_ids[index]
        
        img = Image.open(img_path).convert("RGB")
        
        if self.dual_output:
            image_data_q = self.transform(img)
            image_data_k = self.transform(img)
            return (image_data_q, image_data_k), instance_label, pseudo_label, unique_id
        else:
            image_data = self.transform(img)
            
            return image_data, instance_label, pseudo_label, unique_id

    def __len__(self):
        return len(self.images)
    
    
def get_instance_loaders(bags_train, bags_val, state, config, only_known=False, only_pseudo = False, use_bag_labels=False, dual_output=False, only_negative=False):
    
    pseudo_labels = state['selection_mask'].copy()
    if not config['use_pseudo_labels']:
        pseudo_labels = []
    
    # Used the instance predictions from bag training to update the Instance Dataloader
    instance_dataset_train = Instance_Dataset(bags_train, pseudo_labels, config, transform=train_transform, only_known=only_known, 
                                              only_pseudo = only_pseudo,
                                              use_bag_labels=use_bag_labels,
                                              dual_output=dual_output,
                                              only_negative=only_negative,
                                              subset=config["data_subset_ratio"])
    instance_dataset_val = Instance_Dataset(bags_val, [], config, transform=val_transform, only_known=only_known,
                                            only_pseudo = only_pseudo,
                                            use_bag_labels=use_bag_labels,
                                            dual_output=dual_output,
                                            only_negative=only_negative,
                                            subset=config["data_subset_ratio"])
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
        batch_instance_labels = []
        batch_pseudo_labels = []
        batch_ids = []

        for (image_data_q, image_data_k), instance_label, pseudo_label, unique_id in batch:
            batch_data_q.append(image_data_q)
            batch_data_k.append(image_data_k)
            batch_instance_labels.append(instance_label)
            batch_pseudo_labels.append(pseudo_label)
            batch_ids.append(unique_id)

        batch_data_q = torch.stack(batch_data_q)
        batch_data_k = torch.stack(batch_data_k)
        batch_instance_labels = torch.tensor(batch_instance_labels, dtype=torch.float)
        batch_pseudo_labels = torch.tensor(batch_pseudo_labels, dtype=torch.float)

        return (batch_data_q, batch_data_k), batch_instance_labels, batch_pseudo_labels, batch_ids
    else:
        batch_data = []
        batch_instance_labels = []
        batch_pseudo_labels = []
        batch_ids = []

        for image_data, instance_label, pseudo_label, unique_id in batch:
            batch_data.append(image_data)
            batch_instance_labels.append(instance_label)
            batch_pseudo_labels.append(pseudo_label)
            batch_ids.append(unique_id)

        batch_data = torch.stack(batch_data)
        batch_instance_labels = torch.tensor(batch_instance_labels, dtype=torch.float)
        batch_pseudo_labels = torch.tensor(batch_pseudo_labels, dtype=torch.float)

        return batch_data, batch_instance_labels, batch_pseudo_labels, batch_ids


class InstanceSampler(Sampler):
    def __init__(self, dataset, batch_size, strategy=1, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        
        # Use pseudo labels when available, otherwise use image labels
        effective_labels = []
        for i in range(len(dataset.output_image_labels)):
            pseudo_label = dataset.output_pseudo_labels[i]
            image_label = dataset.output_image_labels[i]
            
            # Use pseudo label if it's not -1, otherwise use image label
            if pseudo_label != -1:
                effective_labels.append(pseudo_label)
            else:
                effective_labels.append(image_label)
        
        # Get indices for each class based on effective labels
        self.indices_positive = [i for i, label in enumerate(effective_labels) if label == 1]
        self.indices_negative = [i for i, label in enumerate(effective_labels) if label == 0]
        self.indices_unknown = [i for i, label in enumerate(effective_labels) if label != 0 and label != 1]
        self.indices_non_positive = self.indices_negative + self.indices_unknown
        
        # Number of positive samples determines the number of samples per class
        self.samples_per_class = len(self.indices_positive)
        
        # Calculate total number of batches possible with balanced classes
        self.total_samples = self.samples_per_class * 2  # multiply by 2 for pos and neg
        self.total_batches = self.total_samples // self.batch_size
        
        if len(self.indices_non_positive) < self.samples_per_class:
            print(f"Warning: Not enough non-positive samples. Requested {self.samples_per_class} but only have {len(self.indices_non_positive)} available.")
        
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