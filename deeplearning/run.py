import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime
import os
import pandas as pd
import json

from load_and_process import load_data, process_data
from model import CNNTransformer
from train import train_model
from evaluate import evaluate_model


def main(has_image_encoder=True, has_text_encoder=True, fusion='att', CL=True, freeze_image_backbone=True, completed=False) :
    if torch.cuda.is_available() :
        device = 'cuda'
    elif torch.backends.mps.is_available() :
        device = 'mps'
    else :
        device = 'cpu'

    print(f"Using device: {device}")
    
    df = load_data(completed=completed)

    max_len = 10
    min_freq = 3

    df, label2id, id2label, vocab_size = process_data(df,max_len=max_len,min_freq=min_freq,completed=completed)
    cat_number = len(label2id)

    # 按照7:1:2划分train，valid，test
    df_train, df_temp = train_test_split(df,test_size=0.3,random_state=42,shuffle=True)

    df_valid, df_test = train_test_split(df_temp,test_size=2/3,random_state=42,shuffle=True)

    # 初始化模型
    model = CNNTransformer(cat_number=cat_number, vocab_size=vocab_size, max_len=max_len, emb_dim=64, freeze_image_backbone=freeze_image_backbone, device=device, has_image_encoder=has_image_encoder, has_text_encoder=has_text_encoder)

    # ----- 训练模型 -----

    batch_size = 128

    model = train_model(model=model, df_train=df_train, df_valid=df_valid, max_len=max_len, 
                        num_epochs=500, batch_size=batch_size, lr=1e-3, device=device, early_stop_patience=10)
    
    # ------ 测试集评估 -----
    test_metrics = evaluate_model(model=model, df_test=df_test, max_len=max_len, batch_size=batch_size, id2label=id2label)

    print("=== Test Set Metrics ===")
    for category in label2id :
        print(f"Category: {category}")
        for k, v in test_metrics['per_class'][category].items() :
            print(f"{k}: {v:.4f}")
        print()
    
    print(f"Macro-F1: {test_metrics['macro_f1']}")


    # Generate timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Path (Customizable)
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"CNNTransformer_image={has_image_encoder}_text={has_text_encoder}_completed={completed}_{ts}.json")

    with open(save_path, "w") as f:
        json.dump(test_metrics, fp=f, indent=4)

    print(f"Test results have been saved to: {save_path}")


if __name__ == '__main__' :
    #main(has_image_encoder=False, has_text_encoder=True, CL=False)
    #main(has_image_encoder=False, has_text_encoder=True, CL=True)
    #main(has_image_encoder=True, has_text_encoder=False, CL=False)
    #main(has_image_encoder=True, has_text_encoder=False, CL=True)
    #main(has_image_encoder=True, has_text_encoder=True, fusion='att', CL=False)
    #main(has_image_encoder=True, has_text_encoder=True, fusion='att', CL=True)
    #main(has_image_encoder=True, has_text_encoder=True, fusion='mlp', CL=False)
    #main(has_image_encoder=True, has_text_encoder=True, fusion='mlp', CL=True)
    main(has_image_encoder=False, has_text_encoder=True, completed=True)
    main(has_image_encoder=True, has_text_encoder=False, completed=True)
    main(has_image_encoder=True, has_text_encoder=True, completed=True)
    main(has_image_encoder=False, has_text_encoder=True, completed=False)
    main(has_image_encoder=True, has_text_encoder=False, completed=False)
    main(has_image_encoder=True, has_text_encoder=True, completed=False)