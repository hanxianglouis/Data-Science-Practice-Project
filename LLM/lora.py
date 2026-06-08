import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from qwen_vl_utils import process_vision_info

from dataset import ImageCaptionDataset
from collector import QwenVLCollator
from model_utils import load_model_and_processor

def add_lora(model):
    """
    Qwen 系列常见 LoRA target modules:
    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

    如果显存紧张，可以先只训练注意力层：
    ["q_proj", "v_proj"]
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def LoRA(df: pd.DataFrame, model, processor, output_dir: str ="./qwen_model_cache/qwen2_5_vl_3b_caption_lora"):

    print(">>> Begin LoRA")

    before = len(df)
    df = df[df['image_path'].apply(lambda x: os.path.exists(str(x)))].copy()
    after = len(df)
    print(f">>> Valid images: {after}/{before}")

    train_dataset = ImageCaptionDataset(df=df)

    model = add_lora(model)

    # It is recommended to enable gradient checkpointing during training to save GPU memory.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    data_collator = QwenVLCollator(processor)

    training_args = TrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,

        num_train_epochs=2,
        learning_rate=2e-4,

        logging_strategy="no", 
        save_steps=500,
        save_total_limit=2,

        optim="adamw_torch",

        remove_unused_columns=False,
        dataloader_num_workers=4,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(">>> Start training...")
    trainer.train()


    if trainer.is_world_process_zero():
    
        print(">>> Saving LoRA adapter...")
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
    
        print(f">>> LoRA adapter saved to: {output_dir}")

    return model, processor