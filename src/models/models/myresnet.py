import torch
from torch import nn

from src.models.models import ClassificationModel
from src.models.models.resnet import ResNetBlock


class MyResNet(ClassificationModel):
    """A PyTorch implementation of the ResNet Baseline
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "MyResNet"
        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
        ])
        self.final = nn.Linear(mid_channels, num_pred_classes)
        self.criterion = nn.CrossEntropyLoss()
        print(f"{self.__class__.__name__} created with {in_channels, mid_channels, num_pred_classes, kwargs}")

    def forward(self, inputs_embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = inputs_embeds.transpose(1, 2)
        x = self.layers(x)
        preds = self.final(x.mean(dim=-1))
        loss = self.criterion(preds, labels)
        return loss, preds
