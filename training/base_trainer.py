import torch


class BaseTrainer:
    def __init__(self, model: torch.nn.Module, device=None):
        self.model = model
        self.optimizer = None
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.device = device
        self.lr_scheduler = None

    def train_epoch(self, dataloader: torch.utils.data.dataloader.DataLoader) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def save_model(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), save_path)

    def train_epochs(self, dataloader: torch.utils.data.DataLoader, save_path, num_epochs=9999) -> None:
        """
        :param dataloader: dataloader to train on
        :param save_path: path of the File to save at
        :param num_epochs: number of epochs...
        :return: None
        """
        print("here")
        for epoch in range(num_epochs):
            loss = self.train_epoch(dataloader)
            if self.lr_scheduler:
                self.lr_scheduler.step(loss)

            print(f"Epoch {epoch + 1}, Loss: {loss}, LR: {self.optimizer.param_groups[0]['lr']}")

            # Save the model checkpoint
            self.save_model(save_path)