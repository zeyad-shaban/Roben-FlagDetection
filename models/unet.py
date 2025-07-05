import torch

def load_model_unet(checkpoint_path: str = "") -> torch.nn.Module:
    """
    ?checkpoint_path: checkpoint file path for the retinenet weights, leave None to not load
    Uses retinaNet model
    :return: torch.Module
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                           init_features=32, pretrained=False).to(device)


    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    return model