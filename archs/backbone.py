from timm import create_model
from fastai.vision.learner import _update_first_layer
from torch import nn
from fastai.vision.all import *

# this function is used to cut off the head of a pretrained timm model and return the body
def create_timm_body(arch:str, pretrained=False, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or function")
    
    
    
def get_num_features(model):
    # Keep track of the last Conv2d layer found
    last_conv_channels = None
    
    def recursive_search(layer):
        nonlocal last_conv_channels
        
        if isinstance(layer, nn.Conv2d):
            last_conv_channels = layer.out_channels
        
        # Continue searching through children
        for child in layer.children():
            recursive_search(child)
        
        return last_conv_channels

    # Start the recursive search from the model
    nf = recursive_search(model)
    
    if nf is None:
        raise ValueError("No Conv2d layer found in the feature extractor!")
    
    print("Output channels (nf) via layer inspection:", nf)
    return nf
    

class _Hook:
    def __init__(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.handle.remove()


class ResnetPool(nn.Module):
    def __init__(self, arch:str, pretrained=True, n_in=3, num_hooks=4, upsample_to='largest'):
        super().__init__()
        
        # 1) Create the model
        self.model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
        _update_first_layer(self.model, n_in, pretrained)

        # 2) Find submodules, register hooks as you did
        all_children = dict(self.model.named_children())
        layer_keys = sorted([k for k in all_children.keys() if k.startswith('layer')],
                            key=lambda x: int(x.replace('layer','')))
        chosen_layer_keys = layer_keys[-num_hooks:] if num_hooks <= len(layer_keys) else layer_keys

        self.hooks = {}
        for k in chosen_layer_keys:
            self.hooks[k] = _Hook(all_children[k])

        self.chosen_layer_keys = chosen_layer_keys
        self.upsample_to = upsample_to

    def forward(self, x):
        # Normal forward
        out = self.model(x)

        # Grab features from hooks
        feats = []
        shapes = []
        for layer_name in self.chosen_layer_keys:
            f = self.hooks[layer_name].features
            # Force the feature to be on the same device as x
            if f is not None:
                f = f.to(x.device)
            feats.append(f)
            shapes.append(f.shape[-2:])

        # Upsample and concat
        if self.upsample_to == 'largest':
            target_h = max(s[0] for s in shapes)
            target_w = max(s[1] for s in shapes)
            up_feats = [
                F.interpolate(f, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
                for f in feats
            ]
            concat_feats = torch.cat(up_feats, dim=1)
        else:
            concat_feats = torch.cat(feats, dim=1)

        return out, concat_feats
    

def create_pooled_resnet(arch:str, pretrained=False, n_in=3, num_hooks=4):
    body = ResnetPool(arch, pretrained, n_in, num_hooks)

    dummy = torch.zeros(1, n_in, 224, 224)
    with torch.no_grad():
        out, multi_pooled = body(dummy)
    
    pooled_feat_size = multi_pooled.shape[1]
    return body, pooled_feat_size


class ConvNextPool(nn.Module):
    def __init__(self, arch:str, pretrained=True, n_in=3, num_hooks=4, upsample_to='largest'):
        super().__init__()
        
        self.model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
        _update_first_layer(self.model, n_in, pretrained)

        # For ConvNeXt, we need to look at the stages within downsample_layers
        self.hooks = {}
        
        # ConvNeXt stages are in the 'stages' attribute
        stages = self.model.stages
        num_stages = len(stages)
        chosen_stages = range(num_stages)[-num_hooks:] if num_hooks <= num_stages else range(num_stages)
        
        for i in chosen_stages:
            self.hooks[f'stage_{i}'] = _Hook(stages[i])

        self.chosen_stage_keys = [f'stage_{i}' for i in chosen_stages]
        self.upsample_to = upsample_to

    def forward(self, x):
        out = self.model(x)

        feats = []
        shapes = []
        for stage_name in self.chosen_stage_keys:
            f = self.hooks[stage_name].features
            if f is not None:
                f = f.to(x.device)
            feats.append(f)
            shapes.append(f.shape[-2:])

        if self.upsample_to == 'largest':
            target_h = max(s[0] for s in shapes)
            target_w = max(s[1] for s in shapes)
            up_feats = [
                F.interpolate(f, size=(target_h, target_w),
                            mode='bilinear', align_corners=False)
                for f in feats
            ]
            concat_feats = torch.cat(up_feats, dim=1)
        else:
            concat_feats = torch.cat(feats, dim=1)

        return out, concat_feats

def create_pooled_convnext(arch:str, pretrained=False, n_in=3, num_hooks=4):
    body = ConvNextPool(arch, pretrained, n_in, num_hooks)

    dummy = torch.zeros(1, n_in, 224, 224)
    with torch.no_grad():
        out, multi_pooled = body(dummy)
    
    pooled_feat_size = multi_pooled.shape[1]
    return body, pooled_feat_size




class EfficientNetPooled(nn.Module): 
    def __init__(self, encoder, num_hooks=4, upsample_to='largest'):
        super().__init__()
        self.encoder = encoder  # typically a Sequential container (model.features)
        self.upsample_to = upsample_to

        # The total number of top-level modules in the encoder
        num_modules = len(self.encoder)

        # We only want to hook the last `num_hooks` modules
        # e.g. if num_modules = 9 and num_hooks = 4 => hook indices = [5, 6, 7, 8]
        num_hooks = min(num_hooks, num_modules)
        self.hook_indices = list(range(num_modules - num_hooks, num_modules))

        # Register hooks
        self.hooks = {}
        for idx in self.hook_indices:
            self.hooks[idx] = _Hook(self.encoder[idx])

    def forward(self, x):
        features_list = []
        # Run the encoder sequentially; collect features from the hooked modules
        for idx, module in enumerate(self.encoder):
            x = module(x)
            if idx in self.hook_indices:
                features_list.append(x)
                #print(x.shape)

        if self.upsample_to == 'largest':
            # Upsample all feature maps to the largest spatial resolution among them
            target_h = max(feat.shape[-2] for feat in features_list)
            target_w = max(feat.shape[-1] for feat in features_list)
            up_features = [
                F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                for feat in features_list
            ]
            multi_pooled = torch.cat(up_features, dim=1)
        else:
            # Or just concat them at whatever their current sizes are (must match though)
            multi_pooled = torch.cat(features_list, dim=1)

        return x, multi_pooled



def create_pooled_efficientnet(model, n_in=3, num_hooks=3):

    # Now create a body that registers hooks on the last num_hooks modules.
    body = EfficientNetPooled(model, num_hooks=num_hooks, upsample_to='largest')
    
    # Determine the channel count of the concatenated (pooled) features using a dummy forward.
    dummy = torch.zeros(1, n_in, 224, 224)
    with torch.no_grad():
        _, multi_pooled = body(dummy)
    pooled_feat_size = multi_pooled.shape[1]
    return body, pooled_feat_size



def get_efficientnet_model(arch, pretrained_arch):
    # Mapping of EfficientNet model names to their corresponding torchvision functions
    efficientnet_models = {
        # EfficientNet V1
        'efficientnet_b0': (efficientnet_b0, EfficientNet_B0_Weights),
        'efficientnet_b1': (efficientnet_b1, EfficientNet_B1_Weights),
        'efficientnet_b2': (efficientnet_b2, EfficientNet_B2_Weights),
        'efficientnet_b3': (efficientnet_b3, EfficientNet_B3_Weights),
        'efficientnet_b4': (efficientnet_b4, EfficientNet_B4_Weights),
        'efficientnet_b5': (efficientnet_b5, EfficientNet_B5_Weights),
        'efficientnet_b6': (efficientnet_b6, EfficientNet_B6_Weights),
        'efficientnet_b7': (efficientnet_b7, EfficientNet_B7_Weights),
        
        # EfficientNet V2
        'efficientnet_v2_s': (efficientnet_v2_s, EfficientNet_V2_S_Weights),
        'efficientnet_v2_m': (efficientnet_v2_m, EfficientNet_V2_M_Weights),
        'efficientnet_v2_l': (efficientnet_v2_l, EfficientNet_V2_L_Weights),
    }
    
    # Find the matching model function and weights
    model_func = None
    weights_class = None
    
    for model_name, (func, weights) in efficientnet_models.items():
        if model_name.lower() in arch.lower():
            model_func = func
            weights_class = weights
            break
    
    if model_func is None:
        raise ValueError(f"Unsupported EfficientNet architecture: {arch}")
    
    # Select weights based on pretrained flag
    weights = weights_class.DEFAULT if pretrained_arch else None
    
    # Create the model and extract features
    model = model_func(weights=weights)
    model_features = model.features
    
    
    return model_features