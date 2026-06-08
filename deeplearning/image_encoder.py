import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.makedirs("./model_cache", exist_ok=True) # Store the downloaded model
os.environ["TORCH_HOME"] = "./model_cache"

class ImageEncoderResnNet50(nn.Module):
    """
    基于 ResNet50 的图像编码器，输出论文中的 F_img
    输入:  [B, 3, 244, 244]
    输出:  [B, m, emb_dim]

    说明：
    - ResNet50 先提取图像 feature map。
    - 去掉 ResNet50 原始的 avgpool 和 fc 分类层。
    - 将 feature map 展平为空间 token 序列，其中 m = h * w。
    - 再用线性层把每个 image token 从 2048 维映射到 emb_dim 维。
    - 最终输出 F_img，形状为 [B, m, emb_dim]。
    """
    def __init__(self, emb_dim=64, device='cuda', freeze_backbone=True):
        super().__init__()
        import torchvision.models as models
        self.emb_dim = emb_dim
        self.device = device

        # 加载预训练的 ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # ResNet50 最后一层卷积输出的通道数通常是 2048
        in_features = self.resnet50.fc.in_features

        # 去掉 ResNet50 原始的 avgpool 和 fc，保留卷积 backbone
        self.resnet50.avgpool = nn.Identity()
        self.resnet50.fc = nn.Identity()

        # 将每个 image token 从 2048 维映射到 emb_dim 维
        self.proj = nn.Linear(in_features, self.emb_dim)

        if freeze_backbone:
            for param in self.resnet50.parameters():
                param.requires_grad = False

        self.resnet50 = self.resnet50.to(self.device)
        self.proj = self.proj.to(self.device)

    def forward(self, x):
        """
        x: [B, 3, 244, 244]
        return: [B, m, emb_dim], where m = h * w
        """
        x = x.to(self.device)

        # 手动执行 ResNet50 的卷积部分，避免 avgpool 把空间维度压成 1x1
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)  # [B, 2048, h, w]

        # 将 CNN feature map 转成论文里的 image token 序列 F_img
        x = x.flatten(2)             # [B, 2048, h*w]
        x = x.transpose(1, 2)        # [B, h*w, 2048]
        x = self.proj(x)             # [B, m, emb_dim]

        return x