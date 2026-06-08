import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, f1_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

from dataset import ImageContextDataset


def calc_metrics_train(logits: torch.Tensor, labels: torch.Tensor):
    """
    通过模型输出的logit和真实label tensor计算相关评价指标
    """
    logits = logits.detach().to("cpu").numpy()
    labels = labels.detach().to("cpu").numpy()

    preds = logits.argmax(axis=1)

    result = {
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

    return result

def calc_metrics_final(logits: torch.Tensor, labels: torch.Tensor, id2label: dict[int, str]):
    """
    根据模型输出的 logits 和真实 labels 计算：
    1. 每个类别的 Precision / Recall / F1
    2. 总体的 macro-F1

    参数
    ----
    logits : torch.Tensor
        形状 [B, num_classes]
    labels : torch.Tensor
        形状 [B]
    id2label : dict[int, str] or None
        类别id到类别名称的映射，例如 {0: "cat", 1: "dog"}

    返回
    ----
    result : dict
        {
            "per_class": {
                "cat": {"precision": ..., "recall": ..., "f1": ..., "support": ...},
                "dog": {...},
                ...
            },
            "macro_f1": ...
        }
    """
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    preds = logits.argmax(axis=1)

    classes = np.unique(np.concatenate([labels, preds]))
    classes = np.sort(classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=classes,
        average=None,
        zero_division=0
    )

    per_class = {}
    for i, class_id in enumerate(classes):
        class_name = id2label[int(class_id)]

        per_class[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    result = {
        "per_class": per_class,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0))
    }

    return result

def evaluate_model(model, df_test: pd.DataFrame, id2label: dict[int, str], max_len=10, batch_size=128) :
    """
    测试集评估模型
    """
    test_transform = transforms.Compose([
        transforms.Resize(256), # 缩放短边至256
        transforms.CenterCrop(224), # 取中心224乘224区域
        transforms.ToTensor(),
        ])
    test_set = ImageContextDataset(df_test['image_path'].tolist(),df_test['tokenized_id_context'].tolist(), labels=df_test['label_id'].tolist(), transform=test_transform, max_len=max_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


    model.eval()
    with torch.no_grad() :
        logits = []
        labels = []
        for test_img_batch, test_token_batch, test_label_batch in test_loader :
            logit, _ = model(test_img_batch, test_token_batch, pertubed=False)
            logit = logit.cpu()
            test_label_batch = test_label_batch.cpu()

            logits.append(logit)
            labels.append(test_label_batch)

        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(labels, dim=0)

        metrics = calc_metrics_final(all_logits, all_labels, id2label=id2label)

    return metrics
