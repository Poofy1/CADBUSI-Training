import torchvision.transforms as T
from data.transforms import CLAHETransform
from storage_adapter import * 
from archs.factory import get_model

class BaseConfig:
    def to_dict(self):
        """Convert config to a dictionary with only serializable types"""
        return {k: v for k, v in self.__dict__.items()}


####### ITS2CLR CONFIG #######

class ITS2CLRConfig(BaseConfig):
    def __init__(self):
        self.feature_extractor_train_count = 25
        self.MIL_train_count = 50
        self.initial_ratio = 0.1
        self.final_ratio = .5
        self.total_epochs = 500
        self.warmup_epochs = 0 
        self.learning_rate = 0.001
        self.use_pseudo_labels = True  # Whether or not the instance loader will incorperate instance sudo labels
        self.reset_aggregator = False




####### DATASETS #######
""" Common models
efficientnet_b0   - images 224
efficientnet_b3   - images 300
efficientnet_v2_s   - images 384
convnextv2_tiny   - images 224
resnet18
"""

class LesionDataConfig(BaseConfig):
    def __init__(self):
        self.dataset_name = 'export_oneLesions' #export_12_12_2024_17_35_49' 'export_oneLesions'
        self.label_columns = ['Has_Malignant']
        self.instance_columns = ['Malignant Lesion Present']
        self.img_size = 224 #224
        self.bag_batch_size = 5
        self.min_bag_size = 2
        self.max_bag_size = 50
        self.instance_batch_size = 128
        self.encoder = 'resnet18'
        self.arch = "model_MIL"
        self.pretrained_arch = False 
        self.use_videos = False

class FishDataConfig(BaseConfig):
    def __init__(self):
        self.dataset_name = 'imagenette2_hard' #'imagenette2_hard'
        self.label_columns = ['Has_Fish']
        self.instance_columns = ['Has_Fish']
        self.img_size = 128
        self.bag_batch_size = 5
        self.min_bag_size = 2
        self.max_bag_size = 25
        self.instance_batch_size = 128
        self.encoder = 'resnet18'
        self.arch = "model_MIL"
        self.pretrained_arch = True
        self.use_videos = False
        
class DogDataConfig(BaseConfig):
    def __init__(self):
        self.dataset_name = 'imagenette_dog' #'imagenette_dog_hard'
        self.label_columns = ['Has_Highland']
        self.instance_columns = ['Has_Highland']
        self.img_size = 128
        self.bag_batch_size = 5
        self.min_bag_size = 2
        self.max_bag_size = 25
        self.instance_batch_size = 128
        self.encoder = 'resnet18'
        self.arch = "model_MIL"
        self.pretrained_arch = True
        self.use_videos = False




####### MODEL CONFIG #######

head_name = "TEST_Dog_1"
model_version = ""
data_subset_ratio = 1 # 1.0 uses 100% of the data
data_config_class = DogDataConfig # FishDataConfig / LesionDataConfig / DogDataConfig
bucket = "" # optional - enables GCP
export_location = "D:/DATA/CASBUSI/exports/"

encoder_head_name = ""
encoder_model_version = ""
        
####### Augmentations #######
        
train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            T.RandomAffine(degrees=(-90, 90), translate=(0.05, 0.05), scale=(1, 1.2),),
            CLAHETransform(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

val_transform = T.Compose([
            CLAHETransform(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

##############


def build_config():
    """Combines configs into a single dictionary"""
    its2clr_config = ITS2CLRConfig().to_dict()
    data_config = data_config_class().to_dict()
    
    # Direct path config as dictionary
    path_config = {
        "bucket": bucket,
        "export_location": export_location,
    }
    
    print(f'Selected model: {head_name}, version: {model_version}')
    
    config = {
        "head_name": head_name,
        "model_version": model_version,
        "data_subset_ratio": data_subset_ratio,
        "encoder_head_name": encoder_head_name,
        "encoder_model_version": encoder_model_version,
        **its2clr_config,
        **data_config,
        **path_config,
        "num_classes": len(data_config['label_columns']) + 1,
        "num_labels": len(data_config['label_columns']),
    }
    
    # Determine storage client
    StorageClient.get_instance(None, config['bucket'])
    
    return config



def build_model(config):
    model = get_model(
        model_type= config['arch'],
        arch=config['encoder'],
        pretrained_arch=config['pretrained_arch'],
        num_classes = config['num_labels'],
    )
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")   
    return model.cuda()