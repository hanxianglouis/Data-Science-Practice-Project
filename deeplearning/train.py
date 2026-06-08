import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import os
import pandas as pd

from model import MultiviewModel
from dataset import ImageContextDataset
from evaluate import calc_metrics_train

def train_model(model, df_train, df_valid, max_len=10, num_epochs=10, batch_size=256, lr=1e-3, device='cuda', early_stop_patience=10) :
    """
    训练模型，包含早停机制，返回最优模型
    """
    os.makedirs('./log', exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize(256), # Resize成规定尺寸
        transforms.RandomCrop(224, padding=8), # 随机裁剪
        transforms.RandomHorizontalFlip(), # 水平翻转
        transforms.ToTensor()
        ])
    valid_transform = transforms.Compose([
        transforms.Resize(256), # 缩放短边至256
        transforms.CenterCrop(224), # 取中心224乘224区域
        transforms.ToTensor(),
        ])
    train_set = ImageContextDataset(df_train['image_path'].tolist(),df_train['tokenized_id_context'].tolist(), labels=df_train['label_id'].tolist(), transform=train_transform, max_len=max_len)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    valid_set = ImageContextDataset(df_valid['image_path'].tolist(),df_valid['tokenized_id_context'].tolist(), labels=df_valid['label_id'].tolist(), transform=valid_transform, max_len=max_len)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_macro_f1 = 0.0
    no_improve_count = 0

    print(f"Train Samples: {len(df_train)}")
    print(f"Valid Samples: {len(df_valid)}")

    for epoch in range(num_epochs) :
        model.train()

        total_loss = 0.0
        total_other_loss = 0.0  # Total Contrastive Learning Loss
        total_pred_loss = 0.0  # Total Prediction Loss

        for img_batch, token_batch, label_batch in tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} Training"
        ) :
            optimizer.zero_grad()
            
            logit, other_loss = model(img_batch, token_batch, pertubed=model.CL)

            label_batch = label_batch.to(device)

            pred_loss = F.cross_entropy(logit, label_batch)

            loss = pred_loss + model.CL_loss_weight * other_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_other_loss += other_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_other_loss = total_other_loss / len(train_loader)
        avg_pred_loss = total_pred_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - prediction loss={avg_pred_loss:.4f}, other loss={avg_other_loss:.4f}, total Loss={avg_loss:.4f}")

        # ------ 验证集评估 -----
        model.eval()
        with torch.no_grad() :
            logits = []
            labels = []
            for val_img_batch, val_token_batch, val_label_batch in valid_loader :
                logit, _ = model(val_img_batch, val_token_batch, pertubed=False)
                logit = logit.cpu()
                val_label_batch = val_label_batch.cpu()

                logits.append(logit)
                labels.append(val_label_batch)

            all_logits = torch.concat(logits, dim=0)
            all_labels = torch.concat(labels, dim=0)

            metrics = calc_metrics_train(all_logits, all_labels)
            valid_macro_f1 = metrics['macro_f1']

            print(f"Epoch {epoch+1}/{num_epochs} - Valid precision={metrics['precision']:.4f}, macro-F1={metrics['macro_f1']:.4f}")

        # Early Stopping

        if valid_macro_f1 > best_macro_f1 :
            best_macro_f1 = valid_macro_f1
            no_improve_count = 0
            torch.save(model.state_dict(), "./log/best_model.pt")
        else :
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs")
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(torch.load("./log/best_model.pt"))
    return model