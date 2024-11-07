import torchvision.transforms as T
from data.transforms import CLAHETransform


class BaseConfig:
    def to_dict(self):
        """Convert config to a dictionary with only serializable types"""
        return {k: v for k, v in self.__dict__.items()}



####### CONFIGURE #######

class ITS2CLRConfig(BaseConfig):
    def __init__(self):
        self.feature_extractor_train_count = 1
        self.MIL_train_count = 1 #6
        self.initial_ratio = 0.3
        self.final_ratio = 1
        self.total_epochs = 100
        self.warmup_epochs = 10
        self.learning_rate = 0.001
        self.reset_aggregator = False

class LesionDataConfig(BaseConfig):
    def __init__(self):
        self.dataset_name = 'export_oneLesions'
        self.label_columns = ['Has_Malignant']
        self.instance_columns = ['Malignant Lesion Present']
        self.img_size = 224
        self.bag_batch_size = 8
        self.min_bag_size = 2
        self.max_bag_size = 25
        self.instance_batch_size = 25
        self.arch = 'efficientnet'
        self.pretrained_arch = False

class FishDataConfig(BaseConfig):
    def __init__(self):
        self.dataset_name = 'imagenette2_hard'
        self.label_columns = ['Has_Fish']
        self.instance_columns = ['Has_Fish']
        self.img_size = 128
        self.bag_batch_size = 5
        self.min_bag_size = 2
        self.max_bag_size = 25
        self.instance_batch_size = 25
        self.arch = 'efficientnet'
        self.pretrained_arch = False

class PathConfig(BaseConfig):
    def __init__(self):
        self.export_location = "D:/DATA/CASBUSI/exports/"
        self.cropped_images = "F:/Temp_SSD_Data/"
        
        
        
        
# Augmentations 
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


def build_config(model_version, head_name, data_config_class):
    """Combines configs into a single dictionary"""
    its2clr_config = ITS2CLRConfig().to_dict()
    data_config = data_config_class().to_dict()
    
    config = {
        "head_name": head_name,
        "model_version": model_version,
        **its2clr_config,
        **data_config
    }
    return config
        
def load_paths():
    """Returns a dictionary of paths using PathConfig"""
    path_config = PathConfig()

    # Convert PathConfig to dictionary
    paths = path_config.to_dict()
    return paths

