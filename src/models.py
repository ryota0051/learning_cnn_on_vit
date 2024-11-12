import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer
from timm.models.resnet import resnet10t, ResNet


class VitFeatureExtractor(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        embed_dim=512,
        num_heads=8,
        num_layers=3,
        **vit_args,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            img_size,
            patch_size,
            num_heads=num_heads,
            depth=num_layers,
            embed_dim=embed_dim,
            **vit_args
        )

    def forward(self, x):
        x = self.vit.forward_features(x)
        x = x.mean(dim=1)
        return x


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn: ResNet = resnet10t()

    def forward(self, x):
        return self.cnn.forward_features(x).squeeze((2, 3))


class Predictor(nn.Module):
    def __init__(self, num_class=10, inc=4096, temp=0.05):
        super().__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x):
        x = self.fc1(x)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


def get_model(
    model_type,
    num_feature_extractor_out=512,
):
    if model_type == 'vit':
        G = VitFeatureExtractor()
    elif model_type == 'cnn':
        G = CNNFeatureExtractor()

    F = Predictor(inc=num_feature_extractor_out)
    return G, F
