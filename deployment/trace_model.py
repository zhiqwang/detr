import torch
from torch import nn, Tensor
from util.misc import nested_tensor_from_tensor_list
from hubconf import detr_resnet50

from typing import List


class WrappedDETR(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs: List[Tensor]):
        sample = nested_tensor_from_tensor_list(inputs)
        return self.model(sample)


if __name__ == "__main__":

    model = detr_resnet50(pretrained=False)
    wrapped_model = WrappedDETR(model)
    wrapped_model.eval()
    scripted_model = torch.jit.script(wrapped_model)

    scripted_model.save("./deployment/detr_resnet50.torchscript.pt")
