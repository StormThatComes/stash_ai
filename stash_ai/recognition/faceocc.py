"""Inference-only U-Net/ResNet18 for the pinned FaceOcc SafeTensors release.

Decoder/state-dict compatibility follows segmentation-models-pytorch v0.5.0
(MIT, Pavel Yakubovskiy); attribution is in docs/faceocc-attribution.md.
Uses the application's existing torch/torchvision, without SMP/timm downloads.
"""


def build_model():
    import torch
    from torch import nn
    from torchvision.models import resnet18

    def convolution(inputs, outputs):
        return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                             nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))

    class Block(nn.Module):
        def __init__(self, inputs, skip, outputs):
            super().__init__()
            self.conv1 = convolution(inputs + skip, outputs)
            self.conv2 = convolution(outputs, outputs)

        def forward(self, value, size, skip):
            value = nn.functional.interpolate(value, size=size, mode='nearest')
            if skip is not None:
                value = torch.cat((value, skip), dim=1)
            return self.conv2(self.conv1(value))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet18(weights=None)
            del self.encoder.fc, self.encoder.avgpool
            self.decoder = nn.Module()
            self.decoder.blocks = nn.ModuleList(Block(i, s, o) for i, s, o in
                ((512, 256, 256), (256, 128, 128), (128, 64, 64), (64, 64, 32), (32, 0, 16)))
            self.segmentation_head = nn.Sequential(nn.Conv2d(16, 1, 3, padding=1))

        def forward(self, value):
            features = [value]
            value = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(value)))
            features.append(value)
            value = self.encoder.maxpool(value)
            for layer in (self.encoder.layer1, self.encoder.layer2, self.encoder.layer3, self.encoder.layer4):
                value = layer(value)
                features.append(value)
            for index, block in enumerate(self.decoder.blocks):
                reference = features[-index - 2]
                value = block(value, reference.shape[2:], reference if index < 4 else None)
            return self.segmentation_head(value)

    return Model()
