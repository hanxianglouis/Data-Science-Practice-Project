import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import re
from collections import Counter

class ImageContextDataset(Dataset) :
    """
    图像描述数据集
    image_names: 图像文件相对文件地址列表
    context: 图像描述文字列表
    labels: 图像分类，整数型id
    max_len: 序列最大长度
    """
    def __init__(self, image_names: list[str], tokenized_id_context: list[list[int]], labels: list[int], transform: transforms.Compose, max_len: int =10) :
        self.image_names = image_names
        self.tokenized_id_context = tokenized_id_context
        self.labels = labels
        self.loader = Image.open
        self.transform = transform

        self.max_len = max_len


    def __len__(self)  -> int:
        return len(self.image_names)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, str, int]:
        image_file_path = self.image_names[index]
        img = self.transform(self.loader(image_file_path))

        tokens = torch.tensor(self.tokenized_id_context[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return img, tokens, label