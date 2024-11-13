import torch
from torchvision.models.detection import retinanet_resnet50_fpn


def load_model_retinanet(checkpoint_path: str = None) -> torch.nn.Module:
    """
    ?checkpoint_path: checkpoint file path for the retinenet weights, leave None to not load
    Uses retinaNet model
    :return: torch.Module
    """

    model = retinanet_resnet50_fpn(num_classes=2)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    return model
