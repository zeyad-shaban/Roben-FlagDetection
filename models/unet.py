import torch


def load_model_retinanet(checkpoint_path: str = None) -> torch.nn.Module:
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1,
                                   init_features=32, pretrained=False)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    return model
