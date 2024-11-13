import torch

from .base_trainer import BaseTrainer


class RetinaNetTrainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, device=None):
        super().__init__(model, device)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5)


    def train_epoch(self, dataloader) -> float:
        if self.optimizer is None:
            raise Exception("Optimizer not implemented, please invoke set_optimizers with model as a parameter first")

        self.model.train()
        for images, targets in dataloader:
            targets_dict = [{
                'boxes': t.unsqueeze(0).to(self.device),
                'labels': torch.tensor([1], device=self.device)
            } for t in targets]

            loss_dict = self.model(images, targets_dict)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

        return losses.item()

