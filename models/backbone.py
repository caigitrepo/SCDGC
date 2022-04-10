import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

        
class BackboneBase(nn.Module):
    
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layers: Dict,
    # return_interm_layers: bool
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, input: Tensor):
        xs = self.body(input)
        out: Dict[str, Tensor] = {}
        for name, x in xs.items():
            out[name] = x
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrained: bool = True):
        if name in ['resnet18', 'resnet50', 'resnet34', 'resnet101']:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True,
                norm_layer=FrozenBatchNorm2d)
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
        else:
            raise NotImplementedError("Unknow name: %s" % name)
            
        NCDICT = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
        }
        num_channels = NCDICT[name]
        super().__init__(backbone, train_backbone, num_channels, return_layers)

def build_backbone():
    
    class Joiner(nn.Module):
        
        def __init__(self, backbone):
            super(Joiner, self).__init__()
            self.backbone = backbone
        
        def forward(self, input):
            return self.backbone(input)['0']
        
    return Joiner(Backbone('resnet101', True, False, False, True))

# model = build_backbone()
# x = torch.rand(1, 3, 576, 576)
# x = model(x)
# print(model.backbone.body.layer4)
# print(x.shape)