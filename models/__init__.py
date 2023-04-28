from .model import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224

def get_vit_model(model_name : str, num_classes : int, pretrained : bool = False):
    print(f"get_vit_model : {model_name}, num_classes : {num_classes}, pretrained : {pretrained}")
    if model_name == "vit_tiny_patch16_224":
        model = vit_tiny_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vit_small_patch16_224":
        model = vit_small_patch16_224(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "vit_base_patch16_224":
        model = vit_base_patch16_224(num_classes=num_classes, pretrained=pretrained)
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")
    return model