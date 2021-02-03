import torch.nn as nn
import torchvision.models as models


def get_resnet():
    # Create body of network
    model = models.resnet34(pretrained=True)
    children = list(model.children())[:6]
    return nn.Sequential(*children)


class ResHead(nn.Module):
    # Fully connected head
    def __init__(self, outclasses):
        super(ResHead, self).__init__()
        self.classes = outclasses
        self.av_pool = nn.AvgPool2d(kernel_size=8, stride=6, padding=2)
        self.linear = nn.Linear(512, outclasses)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class CustomResnet(nn.Module):
    # Combine body and head
    def __init__(self, outclasses):
        super(CustomResnet, self).__init__()
        self.classes = outclasses
        self.resnet = get_resnet()
        self.reshead = ResHead(outclasses)

    def forward(self, x):
        result = self.reshead(self.resnet(x))
        return result.view(-1, self.classes)

    def set_freeze(self, x):
        # Allow unfreezing of layers from head to body of model
        for i, layer in enumerate(self.resnet):
            needs_grad = i > (len(self.resnet) - 1) - x
            print(f'Layer {i} : Grad:{needs_grad}')
            for param in layer.parameters():
                param.requires_grad = needs_grad

    def lr_groups(self):
        # Different learning rate groups, Not using center group
        return self.resnet[:6], self.resnet[6:], self.reshead
