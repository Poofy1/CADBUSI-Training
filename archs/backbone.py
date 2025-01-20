from timm import create_model
from fastai.vision.learner import _update_first_layer
from torch import nn
from fastai.vision.all import *

# this function is used to cut off the head of a pretrained timm model and return the body
def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or function")
    
    
    
    
    
    
class _Hook:
    def __init__(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output):
        self.features = output
    def remove(self):
        self.handle.remove()

class TimmBodyMultiAuto(nn.Module):
    def __init__(self, arch:str, pretrained=True, n_in=3, num_hooks=4, upsample_to='largest'):
        super().__init__()
        
        # 1) Create the model
        self.model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
        _update_first_layer(self.model, n_in, pretrained)

        # 2) Find submodules that look like "layerX" 
        #    (For a typical resnet in timm, that’s what they’re named.)
        #    Collect them in a list sorted by X, e.g. ['layer1','layer2','layer3','layer4'].
        all_children = dict(self.model.named_children())
        # Filter for keys that start with 'layer' 
        # or do some more robust logic for your architecture
        layer_keys = sorted([k for k in all_children.keys() if k.startswith('layer')],
                            key=lambda x: int(x.replace('layer','')))
        
        # If user wants num_hooks=2 but the model has 4 "layers", we pick the last 2, etc.
        chosen_layer_keys = layer_keys[-num_hooks:] if num_hooks <= len(layer_keys) else layer_keys

        # Register hooks on those chosen layers
        self.hooks = {}
        for k in chosen_layer_keys:
            self.hooks[k] = _Hook(all_children[k])

        self.chosen_layer_keys = chosen_layer_keys
        self.upsample_to = upsample_to

    def forward(self, x):
        # Regular forward
        out = self.model(x)

        # Grab features
        feats = []
        shapes = []
        for layer_name in self.chosen_layer_keys:
            f = self.hooks[layer_name].features
            feats.append(f)
            shapes.append(f.shape[-2:])

        # Optional upsample so we can concat
        if self.upsample_to == 'largest':
            target_h = max(s[0] for s in shapes)
            target_w = max(s[1] for s in shapes)
            up_feats = [F.interpolate(f, size=(target_h,target_w), 
                                      mode='bilinear', align_corners=False)
                        for f in feats]
            concat_feats = torch.cat(up_feats, dim=1)
        else:
            # or skip interpolation for a demo
            concat_feats = torch.cat(feats, dim=1)

        # Return final features + multi-scale
        return out, concat_feats

def create_timm_body_multi(arch:str, pretrained=True, n_in=3, num_hooks=4):
    """
    Creates a multi-hook timm body that returns:
       out, multi_pooled = body(x)
    Where:
       out is the final spatial feature (B, C, H, W)
       multi_pooled is the (B, sum_of_channels, 1, 1) from the chosen layers
    We also return the shape of multi_pooled's channel dim as `pooled_feat_size`.
    """
    body = TimmBodyMultiAuto(arch, pretrained, n_in, num_hooks)

    # 1) Do a dummy forward pass to figure out the shape of multi_pooled
    dummy = torch.zeros(1, n_in, 224, 224)
    with torch.no_grad():
        out, multi_pooled = body(dummy)
    # e.g. multi_pooled might be shape (1, 960, 1, 1)
    pooled_feat_size = multi_pooled.shape[1]

    # 2) Return the encoder AND the channel count
    return body, pooled_feat_size
