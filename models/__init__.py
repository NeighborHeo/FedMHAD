from .model import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
import timm

def get_vit_model(model_name : str, num_classes : int, pretrained : bool = False):
    if model_name == "vit_tiny_patch16_224":
        model = vit_tiny_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vit_small_patch16_224":
        model = vit_small_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vit_base_patch16_224":
        model = vit_base_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name in timm.list_models():
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        print(f"Using {model_name} from timm.")
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")
    return model

import segmentation_models_pytorch as smp
from .segformer import *

def get_network(model_name : str, num_classes : int, pretrained : bool = False, excluded_heads : list = []):
    if model_name == 'segformer': # segformer-b0
        network = SegFormerB0(num_classes=num_classes, encoder_weight=None) # binary classifier
        if pretrained:
            print("Loading pretrained model...")
            network.load_official_state_dict('segformer.b0.512x512.ade.160k.pth', strict=False) # the final prediction layer is not loaded
        if len(excluded_heads) > 0:
            network.setExcludedHeads(excluded_heads)
            
    elif model_name == 'unet':
        network = smp.Unet('resnet34', encoder_weights='imagenet', classes=num_classes)
        
    elif model_name == 'deeplabv3plus':
        network = smp.DeepLabV3Plus('resnet34', encoder_weights='imagenet', classes=num_classes)
        
    elif model_name == 'deeplabv3plus_mobile':
        network = smp.DeepLabV3Plus('mobilenet_v2', encoder_weights='imagenet', classes=num_classes)
    
    return network
