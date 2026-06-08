import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class ImageCaptionDataset(Dataset):
    def __init__(self, df: pd.DataFrame,image_col: str = "image_path",caption_col: str = "caption", prompt: str = "Please generate a concise and accurate caption for this image."):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.caption_col = caption_col
        self.prompt = prompt

        # 清理缺失值
        self.df = self.df.dropna(subset=[self.image_col, self.caption_col]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = str(row[self.image_col])
        caption = str(row[self.caption_col])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": self.prompt,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": caption,
            },
        ]

        return {
            "messages": messages,
            "caption": caption,
        }
